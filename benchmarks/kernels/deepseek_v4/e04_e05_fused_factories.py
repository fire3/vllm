# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.input_batch import (
    _get_num_sampled_and_rejected_kernel,
    _post_update_kernel,
)

_I32_GUARD = -777_000

_CANDIDATE_GEOMETRY = {
    "fused-w1": 1,
    "fused-w2": 2,
    "fused-w4": 4,
    "fused-w8": 8,
}


@triton.jit
def _reset_i32_kernel(dst, src, count, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        dst + offset,
        tl.load(src + offset, mask=offset < count),
        mask=offset < count,
    )


@triton.jit
def _e04_e05_fused_kernel(
    num_sampled_ptr,
    num_rejected_ptr,
    seq_lens_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    prefill_len_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    sampled_tokens_ptr,
    sampled_tokens_stride,
    query_start_loc_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    total_len_ptr,
    HAS_OUTPUT_BIN_COUNTS: tl.constexpr,
    HAS_QUERY_START_LOC: tl.constexpr,
):
    req_id = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_id)

    seq_len = tl.load(seq_lens_ptr + req_id)
    prefill_len = tl.load(prefill_len_ptr + req_state_idx)
    is_chunked_prefilling = seq_len < prefill_len

    num_sampled = tl.load(num_sampled_ptr + req_id)
    num_sampled = tl.where(is_chunked_prefilling, 0, num_sampled)
    tl.store(num_sampled_ptr + req_id, num_sampled)

    logits_start = tl.load(cu_num_logits_ptr + req_id)
    logits_end = tl.load(cu_num_logits_ptr + req_id + 1)
    num_rejected = logits_end - logits_start - num_sampled
    num_rejected = tl.where(is_chunked_prefilling, 0, num_rejected)
    tl.store(num_rejected_ptr + req_id, num_rejected)

    if req_state_idx < 0:
        return

    total_len = tl.load(total_len_ptr + req_state_idx)
    if num_sampled > 0:
        token_id = tl.load(
            sampled_tokens_ptr + req_id * sampled_tokens_stride + num_sampled - 1
        )
        tl.store(last_sampled_tokens_ptr + req_state_idx, token_id)
        tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)

    for i in range(num_sampled):
        token_id = tl.load(sampled_tokens_ptr + req_id * sampled_tokens_stride + i)
        tl.store(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + total_len + i,
            token_id,
        )

        if HAS_OUTPUT_BIN_COUNTS:
            token_ptr = (
                output_bin_counts_ptr
                + req_state_idx * output_bin_counts_stride
                + token_id
            )
            count = tl.load(token_ptr)
            tl.store(token_ptr, count + 1)

    if HAS_QUERY_START_LOC:
        query_start = tl.load(query_start_loc_ptr + req_id)
        query_end = tl.load(query_start_loc_ptr + req_id + 1)
        query_len = query_end - query_start
    else:
        query_len = 0

    computed_delta = query_len - num_rejected
    if computed_delta != 0:
        num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
        tl.store(num_computed_tokens_ptr + req_state_idx, num_computed + computed_delta)


def _prefix(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _batch(args: Mapping[str, Any]) -> tuple[int, int]:
    num_reqs = int(args.get("num_reqs", 4))
    max_num_reqs = int(args.get("max_num_reqs", max(32, num_reqs)))
    if num_reqs < 1 or max_num_reqs < num_reqs:
        raise ValueError("E04/E05 fused case requires 1 <= num_reqs <= max_num_reqs")
    return num_reqs, max_num_reqs


def _draft_lengths(args: Mapping[str, Any], num_reqs: int) -> list[int]:
    value = args.get("draft_lengths")
    if value is None:
        base = [0, 1, 4, 7]
        return [base[index % len(base)] for index in range(num_reqs)]
    if not isinstance(value, list | tuple) or len(value) != num_reqs:
        raise ValueError("draft_lengths must be a per-request list")
    lengths = [int(item) for item in value]
    if any(length < 0 for length in lengths):
        raise ValueError("draft_lengths must be non-negative")
    return lengths


def _candidate_num_warps(args: Mapping[str, Any]) -> int:
    candidate_mode = str(args.get("candidate_mode", "fused-w1"))
    if candidate_mode not in _CANDIDATE_GEOMETRY:
        raise ValueError("unsupported E04/E05 fused candidate mode")
    return _CANDIDATE_GEOMETRY[candidate_mode]


def _reset_i32(dst: torch.Tensor, src: torch.Tensor) -> None:
    block_size = 1024
    _reset_i32_kernel[(triton.cdiv(dst.numel(), block_size),)](
        dst,
        src,
        dst.numel(),
        BLOCK_SIZE=block_size,
    )


def _reset_state(
    state: Mapping[str, torch.Tensor],
    snapshots: Mapping[str, torch.Tensor],
) -> None:
    for name, tensor in state.items():
        _reset_i32(tensor, snapshots[name])


def _exact_tensor_checks(
    candidate: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    return {
        name: torch.equal(candidate_tensor, baseline[name])
        for name, candidate_tensor in candidate.items()
    }


def _idx_mapping(args: Mapping[str, Any], num_reqs: int) -> list[int]:
    if "idx_mapping" in args:
        value = args["idx_mapping"]
        if not isinstance(value, list | tuple) or len(value) != num_reqs:
            raise ValueError("idx_mapping must be a per-request list")
        mapping = [int(item) for item in value]
    else:
        mapping = list(reversed(range(num_reqs)))
    if bool(args.get("include_negative_idx", False)) and num_reqs > 1:
        mapping[1::3] = [-1 for _ in mapping[1::3]]
    if any(index < -1 for index in mapping):
        raise ValueError("negative idx_mapping entries must be -1 in this benchmark")
    return mapping


def build_e04_e05_fused_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    draft_lengths = _draft_lengths(args, num_reqs)
    candidate_mode = str(args.get("candidate_mode", "fused-w1"))
    candidate_num_warps = _candidate_num_warps(args)
    with_bin_counts = bool(args.get("with_bin_counts", True))
    with_query_start_loc = bool(args.get("with_query_start_loc", True))
    chunked_prefill = bool(args.get("chunked_prefill", False))
    vocab_size = int(args.get("vocab_size", 257))
    max_model_len = int(args.get("max_model_len", 160))
    device = "cuda"

    query_lens = [length + 1 for length in draft_lengths]
    max_sampled_width = max(query_lens)
    mapping_values = _idx_mapping(args, num_reqs)
    active_state_indices = [index for index in mapping_values if index >= 0]
    if active_state_indices and max(active_state_indices) >= max_num_reqs:
        raise ValueError("idx_mapping references outside max_num_reqs")

    idx_mapping = torch.tensor(mapping_values, dtype=torch.int32, device=device)
    query_start_loc = (
        torch.tensor(_prefix(query_lens), dtype=torch.int32, device=device)
        if with_query_start_loc
        else None
    )
    cu_num_logits = torch.tensor(_prefix(query_lens), dtype=torch.int32, device=device)

    prefill_storage = torch.full(
        (max_num_reqs + 1,), _I32_GUARD, dtype=torch.int32, device=device
    )
    prefill_values = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
    if chunked_prefill:
        chunked_indices = {
            index for row, index in enumerate(mapping_values) if index >= 0 and row % 2
        }
        for index in chunked_indices:
            prefill_values[index] = query_lens[mapping_values.index(index)] + 64
    prefill_storage[1:] = prefill_values
    prefill_len = prefill_storage[1:]

    seq_lens = torch.tensor(
        [length + 32 for length in query_lens],
        dtype=torch.int32,
        device=device,
    )
    num_sampled_snapshot = torch.tensor(
        [((row * 2 + length) % (length + 2)) for row, length in enumerate(query_lens)],
        dtype=torch.int32,
        device=device,
    )
    num_sampled_snapshot = torch.minimum(
        num_sampled_snapshot,
        torch.tensor(query_lens, dtype=torch.int32, device=device),
    )
    sampled_tokens = (
        torch.arange(num_reqs * max_sampled_width, dtype=torch.int32, device=device)
        .reshape(num_reqs, max_sampled_width)
        .add_(17)
        .remainder_(vocab_size)
    )
    total_len_snapshot = torch.tensor(
        [8 + state_idx * 3 for state_idx in range(max_num_reqs)],
        dtype=torch.int32,
        device=device,
    )
    num_computed_snapshot = torch.tensor(
        [40 + state_idx * 5 for state_idx in range(max_num_reqs)],
        dtype=torch.int32,
        device=device,
    )
    if int(total_len_snapshot.max().item()) + max_sampled_width >= max_model_len:
        raise ValueError("max_model_len is too small for this E04/E05 fused case")

    def allocate_state() -> dict[str, torch.Tensor]:
        state = {
            "num_sampled": torch.empty((num_reqs,), dtype=torch.int32, device=device),
            "num_rejected": torch.empty((num_reqs,), dtype=torch.int32, device=device),
            "num_computed": torch.empty(
                (max_num_reqs,), dtype=torch.int32, device=device
            ),
            "last_sampled": torch.empty(
                (max_num_reqs,), dtype=torch.int32, device=device
            ),
            "all_token_ids": torch.empty(
                (max_num_reqs, max_model_len), dtype=torch.int32, device=device
            ),
            "total_len": torch.empty((max_num_reqs,), dtype=torch.int32, device=device),
        }
        if with_bin_counts:
            state["output_bin_counts"] = torch.empty(
                (max_num_reqs, vocab_size), dtype=torch.int32, device=device
            )
        return state

    snapshots = {
        "num_sampled": num_sampled_snapshot,
        "num_rejected": torch.full(
            (num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
        ),
        "num_computed": num_computed_snapshot,
        "last_sampled": torch.full(
            (max_num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
        ),
        "all_token_ids": torch.full(
            (max_num_reqs, max_model_len), _I32_GUARD, dtype=torch.int32, device=device
        ),
        "total_len": total_len_snapshot,
    }
    if with_bin_counts:
        snapshots["output_bin_counts"] = torch.zeros(
            (max_num_reqs, vocab_size), dtype=torch.int32, device=device
        )

    baseline_state = allocate_state()
    candidate_state = allocate_state()

    def launch_baseline() -> torch.Tensor:
        _reset_state(baseline_state, snapshots)
        output_bin_counts = baseline_state.get("output_bin_counts")
        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            baseline_state["num_sampled"],
            baseline_state["num_rejected"],
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            baseline_state["num_computed"],
            baseline_state["last_sampled"],
            output_bin_counts,
            output_bin_counts.stride(0) if output_bin_counts is not None else 0,
            sampled_tokens,
            sampled_tokens.stride(0),
            baseline_state["num_sampled"],
            baseline_state["num_rejected"],
            query_start_loc,
            baseline_state["all_token_ids"],
            baseline_state["all_token_ids"].stride(0),
            baseline_state["total_len"],
            num_warps=1,
        )
        return baseline_state["num_computed"]

    def launch_candidate() -> torch.Tensor:
        _reset_state(candidate_state, snapshots)
        output_bin_counts = candidate_state.get("output_bin_counts")
        _e04_e05_fused_kernel[(num_reqs,)](
            candidate_state["num_sampled"],
            candidate_state["num_rejected"],
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
            candidate_state["num_computed"],
            candidate_state["last_sampled"],
            output_bin_counts,
            output_bin_counts.stride(0) if output_bin_counts is not None else 0,
            sampled_tokens,
            sampled_tokens.stride(0),
            query_start_loc,
            candidate_state["all_token_ids"],
            candidate_state["all_token_ids"].stride(0),
            candidate_state["total_len"],
            HAS_OUTPUT_BIN_COUNTS=output_bin_counts is not None,
            HAS_QUERY_START_LOC=query_start_loc is not None,
            num_warps=candidate_num_warps,
        )
        return candidate_state["num_computed"]

    shape = {
        "name": f"e04-e05-fused-b{num_reqs}-draft{'_'.join(map(str, draft_lengths))}",
        "chain": "E04-E05-fused",
        "num_reqs": num_reqs,
        "max_num_reqs": max_num_reqs,
        "draft_lengths": draft_lengths,
        "idx_mapping": mapping_values,
        "candidate_mode": candidate_mode,
        "with_bin_counts": with_bin_counts,
        "with_query_start_loc": with_query_start_loc,
        "chunked_prefill": chunked_prefill,
        "ragged_num_sampled": True,
        "reset": "symmetric-gpu",
    }

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = _exact_tensor_checks(candidate_state, baseline_state)
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    return ChainCase(
        baseline=Provider(
            "production-e04-then-e05-with-gpu-reset",
            launch_baseline,
            {
                "operator_launches": 2,
                "reset": "symmetric-gpu",
                "candidate_active": False,
                "e04": "_get_num_sampled_and_rejected_kernel",
                "e05": "_post_update_kernel",
            },
        ),
        candidate=Provider(
            f"triton-single-launch-e04-e05-{candidate_mode}",
            launch_candidate,
            {
                "operator_launches": 1,
                "reset": "symmetric-gpu",
                "candidate_active": True,
                "num_warps": candidate_num_warps,
                "has_output_bin_counts": with_bin_counts,
                "has_query_start_loc": with_query_start_loc,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


__all__ = ["build_e04_e05_fused_case"]
