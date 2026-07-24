# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla.indexer import (
    _build_prefill_chunk_metadata_kernel,
)

BASELINE_BLOCK_SIZE = 1024
GUARD_VALUE = -123456789


def _load_parallel_candidate() -> Callable[..., Any] | None:
    from vllm.v1.attention.backends.mla import indexer

    return getattr(indexer, "_build_prefill_chunk_metadata_parallel_kernel", None)


def _split_lengths(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _local_length(global_length: int, rank: int, world: int, interleave: int) -> int:
    base = global_length // interleave // world * interleave
    remainder = global_length - base * world
    return base + min(max(remainder - rank * interleave, 0), interleave)


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    num_query_tokens = int(args["num_query_tokens"])
    request_batch = int(args.get("request_batch", 1))
    compress_ratio = int(args.get("compress_ratio", 4))
    context_jitter = int(args.get("context_jitter", 0))
    query_slice_start = int(args.get("query_slice_start", 0))
    dcp_rank = int(args.get("dcp_rank", 0))
    dcp_world = int(args.get("dcp_world", 1))
    dcp_interleave = int(args.get("dcp_interleave", 1))
    seed = int(args.get("seed", 0))

    if context_tokens <= 0:
        raise ValueError("C06 context_tokens must be positive")
    if num_query_tokens <= 0:
        raise ValueError("C06 num_query_tokens must be positive")
    if request_batch not in (1, 4):
        raise ValueError("C06 request_batch must be 1 or 4")
    if compress_ratio not in (4, 128):
        raise ValueError("C06 compress_ratio must be 4 or 128")
    if context_jitter < 0:
        raise ValueError("C06 context_jitter must be non-negative")
    if dcp_world not in (1, 2, 4):
        raise ValueError("C06 dcp_world must be 1, 2, or 4")
    if not 0 <= dcp_rank < dcp_world:
        raise ValueError("C06 dcp_rank must be within dcp_world")
    if dcp_interleave <= 0:
        raise ValueError("C06 dcp_interleave must be positive")

    query_lens = _split_lengths(num_query_tokens, request_batch)
    seq_lens = [
        context_tokens - index * context_jitter for index in range(request_batch)
    ]
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("C06 context jitter produced a non-positive sequence")
    if any(query_len > seq_len for query_len, seq_len in zip(query_lens, seq_lens)):
        raise ValueError("C06 query length exceeds its request context")

    query_slice_tokens = int(
        args.get("query_slice_tokens", num_query_tokens - query_slice_start)
    )
    query_slice_stop = query_slice_start + query_slice_tokens
    if not 0 <= query_slice_start < num_query_tokens:
        raise ValueError("C06 query_slice_start must select a query token")
    if query_slice_tokens <= 0 or query_slice_stop > num_query_tokens:
        raise ValueError("C06 query slice must be non-empty and in bounds")

    compressed_seq_lens = [seq_len // compress_ratio for seq_len in seq_lens]
    if not any(compressed_seq_lens):
        raise ValueError("C06 requires at least one compressed token")

    query_starts = [0]
    global_row_starts = [0]
    local_row_starts = [0]
    for query_len, compressed_len in zip(query_lens, compressed_seq_lens):
        query_starts.append(query_starts[-1] + query_len)
        global_row_starts.append(global_row_starts[-1] + compressed_len)
        local_row_starts.append(
            local_row_starts[-1]
            + _local_length(compressed_len, dcp_rank, dcp_world, dcp_interleave)
        )

    expected_ks = [0] * query_slice_tokens
    expected_ke = [0] * query_slice_tokens
    expected_token_to_seq: list[int] = []
    for request_index, (query_len, seq_len, compressed_len) in enumerate(
        zip(query_lens, seq_lens, compressed_seq_lens)
    ):
        query_start = query_starts[request_index]
        prefix_len = seq_len - query_len
        row_start = local_row_starts[request_index]
        for offset in range(query_len):
            absolute_query = query_start + offset
            if query_slice_start <= absolute_query < query_slice_stop:
                output_index = absolute_query - query_slice_start
                global_context = (prefix_len + 1 + offset) // compress_ratio
                local_context = _local_length(
                    global_context,
                    dcp_rank,
                    dcp_world,
                    dcp_interleave,
                )
                expected_ks[output_index] = row_start
                expected_ke[output_index] = row_start + local_context
        expected_token_to_seq.extend([request_index] * compressed_len)

    set_random_seed(seed)
    device = torch.device("cuda")
    tensors = {
        "query_start_loc": torch.tensor(query_starts, dtype=torch.int32, device=device),
        "uncompressed_seq_lens": torch.tensor(
            seq_lens, dtype=torch.int32, device=device
        ),
        "cu_compressed_seq_lens": torch.tensor(
            global_row_starts, dtype=torch.int32, device=device
        ),
        "row_start_cu_compressed_seq_lens": torch.tensor(
            local_row_starts, dtype=torch.int32, device=device
        ),
        "expected_token_to_seq": torch.tensor(
            expected_token_to_seq, dtype=torch.int32, device=device
        ),
        "expected_ks": torch.tensor(expected_ks, dtype=torch.int32, device=device),
        "expected_ke": torch.tensor(expected_ke, dtype=torch.int32, device=device),
    }
    max_query_len = max(query_lens)
    max_compressed_seq_len = max(compressed_seq_lens)
    return {
        **tensors,
        "num_reqs": request_batch,
        "query_slice_start": query_slice_start,
        "query_slice_stop": query_slice_stop,
        "dcp_rank": dcp_rank,
        "dcp_world": dcp_world,
        "dcp_interleave": dcp_interleave,
        "compress_ratio": compress_ratio,
        "max_work_items": max(max_query_len, max_compressed_seq_len),
        "shape": {
            "name": (
                f"ctx{context_tokens}-q{num_query_tokens}-b{request_batch}"
                f"-c{compress_ratio}-slice{query_slice_start}_{query_slice_stop}"
                f"-dcp{dcp_world}r{dcp_rank}i{dcp_interleave}-j{context_jitter}"
            ),
            "context_tokens": context_tokens,
            "context_jitter": context_jitter,
            "request_batch": request_batch,
            "num_query_tokens": num_query_tokens,
            "query_lens": query_lens,
            "query_slice_start": query_slice_start,
            "query_slice_stop": query_slice_stop,
            "compressed_seq_lens": compressed_seq_lens,
            "compress_ratio": compress_ratio,
            "dcp_rank": dcp_rank,
            "dcp_world": dcp_world,
            "dcp_interleave": dcp_interleave,
            "max_work_items": max(max_query_len, max_compressed_seq_len),
        },
    }


def _guarded_output(
    length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    storage = torch.full((length + 2,), GUARD_VALUE, dtype=torch.int32, device=device)
    return storage, storage[1:-1]


def _guards_intact(storage: torch.Tensor) -> bool:
    return bool((storage[[0, -1]] == GUARD_VALUE).all().item())


def _allocate_outputs(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    device = inputs["query_start_loc"].device
    token_storage, token_to_seq = _guarded_output(
        inputs["expected_token_to_seq"].numel(), device
    )
    ks_storage, ks = _guarded_output(inputs["expected_ks"].numel(), device)
    ke_storage, ke = _guarded_output(inputs["expected_ke"].numel(), device)
    return {
        "token_storage": token_storage,
        "token_to_seq": token_to_seq,
        "ks_storage": ks_storage,
        "ks": ks,
        "ke_storage": ke_storage,
        "ke": ke,
    }


def _launch_baseline(
    outputs: Mapping[str, torch.Tensor], inputs: Mapping[str, Any]
) -> None:
    _build_prefill_chunk_metadata_kernel[(inputs["num_reqs"],)](
        inputs["query_start_loc"],
        inputs["uncompressed_seq_lens"],
        inputs["cu_compressed_seq_lens"],
        inputs["row_start_cu_compressed_seq_lens"],
        outputs["token_to_seq"],
        outputs["ks"],
        outputs["ke"],
        inputs["query_slice_start"],
        inputs["query_slice_stop"],
        inputs["dcp_rank"],
        inputs["dcp_world"],
        inputs["dcp_interleave"],
        BLOCK_SIZE=BASELINE_BLOCK_SIZE,
        COMPRESS_RATIO=inputs["compress_ratio"],
    )


def _launch_parallel_candidate(
    kernel: Callable[..., Any],
    outputs: Mapping[str, torch.Tensor],
    inputs: Mapping[str, Any],
    *,
    block_size: int,
    num_warps: int,
) -> None:
    grid = (
        inputs["num_reqs"],
        triton.cdiv(inputs["max_work_items"], block_size),
    )
    kernel[grid](
        inputs["query_start_loc"],
        inputs["uncompressed_seq_lens"],
        inputs["cu_compressed_seq_lens"],
        inputs["row_start_cu_compressed_seq_lens"],
        outputs["token_to_seq"],
        outputs["ks"],
        outputs["ke"],
        inputs["query_slice_start"],
        inputs["query_slice_stop"],
        inputs["dcp_rank"],
        inputs["dcp_world"],
        inputs["dcp_interleave"],
        BLOCK_SIZE=block_size,
        COMPRESS_RATIO=inputs["compress_ratio"],
        num_warps=num_warps,
    )


def _is_sm120() -> bool:
    return torch.cuda.get_device_capability() == (12, 0)


def build_c06_prefill_chunk_metadata_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    candidate_block_size = int(args.get("candidate_block_size", 256))
    candidate_num_warps = int(args.get("candidate_num_warps", 4))
    if candidate_block_size not in (128, 256, 512, 1024):
        raise ValueError("C06 candidate_block_size must be 128, 256, 512, or 1024")
    if candidate_num_warps not in (4, 8):
        raise ValueError("C06 candidate_num_warps must be 4 or 8")

    baseline = _allocate_outputs(inputs)
    candidate = _allocate_outputs(inputs)
    candidate_kernel = _load_parallel_candidate()
    candidate_active = (
        candidate_kernel is not None
        and _is_sm120()
        and inputs["dcp_world"] == 1
        and inputs["max_work_items"] > BASELINE_BLOCK_SIZE
    )

    def run_baseline() -> torch.Tensor:
        _launch_baseline(baseline, inputs)
        return baseline["token_to_seq"]

    def run_candidate() -> torch.Tensor:
        if candidate_active:
            assert candidate_kernel is not None
            _launch_parallel_candidate(
                candidate_kernel,
                candidate,
                inputs,
                block_size=candidate_block_size,
                num_warps=candidate_num_warps,
            )
        else:
            _launch_baseline(candidate, inputs)
        return candidate["token_to_seq"]

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "baseline_token_to_seq": torch.equal(
                baseline["token_to_seq"], inputs["expected_token_to_seq"]
            ),
            "baseline_ks": torch.equal(baseline["ks"], inputs["expected_ks"]),
            "baseline_ke": torch.equal(baseline["ke"], inputs["expected_ke"]),
            "candidate_token_to_seq": torch.equal(
                candidate["token_to_seq"], inputs["expected_token_to_seq"]
            ),
            "candidate_ks": torch.equal(candidate["ks"], inputs["expected_ks"]),
            "candidate_ke": torch.equal(candidate["ke"], inputs["expected_ke"]),
            "baseline_token_guards": _guards_intact(baseline["token_storage"]),
            "baseline_ks_guards": _guards_intact(baseline["ks_storage"]),
            "baseline_ke_guards": _guards_intact(baseline["ke_storage"]),
            "candidate_token_guards": _guards_intact(candidate["token_storage"]),
            "candidate_ks_guards": _guards_intact(candidate["ks_storage"]),
            "candidate_ke_guards": _guards_intact(candidate["ke_storage"]),
        }
        return {"passed": all(checks.values()), "exact": checks}

    shape = dict(inputs["shape"])
    shape.update(
        chain="prefill-indexer-chunk-metadata",
        candidate_block_size=candidate_block_size,
        candidate_num_warps=candidate_num_warps,
        candidate_dispatch_expected=(
            _is_sm120()
            and inputs["dcp_world"] == 1
            and inputs["max_work_items"] > BASELINE_BLOCK_SIZE
        ),
    )
    return ChainCase(
        baseline=Provider(
            "triton-one-program-per-request",
            run_baseline,
            {"grid": [inputs["num_reqs"]], "block_size": BASELINE_BLOCK_SIZE},
        ),
        candidate=Provider(
            "sm120-tiled-request-metadata" if candidate_active else "triton-fallback",
            run_candidate,
            {
                "candidate_active": candidate_active,
                "block_size": candidate_block_size,
                "num_warps": candidate_num_warps,
                "dispatch_threshold": BASELINE_BLOCK_SIZE,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
