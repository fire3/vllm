# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)
from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel
from vllm.v1.worker.gpu.input_batch import (
    _combine_sampled_and_draft_tokens_kernel,
    _expand_idx_mapping_kernel,
    _get_num_sampled_and_rejected_kernel,
    _post_update_kernel,
    _post_update_num_computed_tokens_kernel,
    _prepare_pos_seq_lens_kernel,
    _prepare_prefill_inputs_kernel,
)

_I32_GUARD = -777_000
_I64_GUARD = -888_000
_BOOL_GUARD = True

_GEOMETRY = {
    "mirror": (1024, None),
    "block8-w1": (8, 1),
    "block8-w2": (8, 2),
    "block8-w4": (8, 4),
    "block16-w1": (16, 1),
    "block32-w1": (32, 1),
    "block64-w2": (64, 2),
    "block128-w4": (128, 4),
    "block256-w4": (256, 4),
    "block512-w8": (512, 8),
    "block1024-w4": (1024, 4),
}


@triton.jit
def _reset_i32_kernel(dst, src, count, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        dst + offset, tl.load(src + offset, mask=offset < count), mask=offset < count
    )


def _geometry(mode: str) -> tuple[int, int | None]:
    if mode not in _GEOMETRY:
        raise ValueError("unsupported E candidate mode")
    return _GEOMETRY[mode]


def _launch_kwargs(block_size: int, num_warps: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"BLOCK_SIZE": block_size}
    if num_warps is not None:
        kwargs["num_warps"] = num_warps
    return kwargs


def _prefix(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _mode(args: Mapping[str, Any]) -> str:
    return str(args.get("candidate_mode", "mirror"))


def _batch(args: Mapping[str, Any]) -> tuple[int, int]:
    num_reqs = int(args.get("num_reqs", 4))
    max_num_reqs = int(args.get("max_num_reqs", max(4, num_reqs)))
    if num_reqs < 1 or max_num_reqs < num_reqs:
        raise ValueError("E factories require 1 <= num_reqs <= max_num_reqs")
    return num_reqs, max_num_reqs


def _draft_lengths(args: Mapping[str, Any], num_reqs: int) -> list[int]:
    value = args.get("draft_lengths")
    if value is None:
        return [0, 1, 4, 7][:num_reqs]
    if not isinstance(value, list | tuple) or len(value) != num_reqs:
        raise ValueError("draft_lengths must be a per-request list")
    lengths = [int(item) for item in value]
    if any(length < 0 for length in lengths):
        raise ValueError("draft_lengths must be non-negative")
    return lengths


def _ptr_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.uint64,
        device="cuda",
    )


def _exact_tensor_checks(
    candidate: Mapping[str, torch.Tensor],
    baseline: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    return {
        name: torch.equal(candidate_tensor, baseline[name])
        for name, candidate_tensor in candidate.items()
    }


def _reset_i32(dst: torch.Tensor, src: torch.Tensor) -> None:
    count = dst.numel()
    block = 1024
    _reset_i32_kernel[(triton.cdiv(count, block),)](
        dst,
        src,
        count,
        BLOCK_SIZE=block,
    )


def _case(
    *,
    baseline: Provider,
    candidate: Provider,
    shape: Mapping[str, Any],
) -> ChainCase:
    return ChainCase(
        baseline=baseline,
        candidate=candidate,
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_e01_prefill_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    prefill_len = int(args.get("prefill_len", 256))
    query_lens = [prefill_len for _ in range(num_reqs)]
    total_tokens = sum(query_lens)
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    device = "cuda"

    mapping_values = list(reversed(range(num_reqs)))
    idx_mapping = torch.tensor(mapping_values, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    all_token_ids = torch.arange(
        max_num_reqs * (prefill_len + 1),
        dtype=torch.int32,
        device=device,
    ).reshape(max_num_reqs, prefill_len + 1)
    prefill_lens = torch.full(
        (max_num_reqs,), prefill_len, dtype=torch.int32, device=device
    )
    num_computed_tokens = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
    expected = torch.cat(
        [all_token_ids[req_state_idx, :prefill_len] for req_state_idx in mapping_values]
    )

    def allocate() -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.full(
                (total_tokens,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "next_prefill_tokens": torch.full(
                (max_num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
            ),
        }

    baseline_state = allocate()
    candidate_state = allocate()

    def launch(
        state: Mapping[str, torch.Tensor], block: int, warps: int | None
    ) -> torch.Tensor:
        _prepare_prefill_inputs_kernel[(num_reqs,)](
            state["input_ids"],
            state["next_prefill_tokens"],
            idx_mapping,
            query_start_loc,
            all_token_ids,
            all_token_ids.stride(0),
            prefill_lens,
            num_computed_tokens,
            **_launch_kwargs(block, warps),
        )
        return state["input_ids"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = _exact_tensor_checks(candidate_state, baseline_state)
        checks["input_ids_match_reference"] = torch.equal(
            candidate_state["input_ids"], expected
        )
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": f"e01-b{num_reqs}-prefill{prefill_len}-{candidate_mode}",
        "chain": "E01-prefill",
        "num_reqs": num_reqs,
        "prefill_len": prefill_len,
        "candidate_mode": candidate_mode,
    }
    return _case(
        baseline=Provider(
            "E01-production-block1024",
            lambda: launch(baseline_state, 1024, None),
            {"block_size": 1024, "timed_provider_excludes_setup": True},
        ),
        candidate=Provider(
            f"E01-{candidate_mode}",
            lambda: launch(candidate_state, candidate_block, candidate_warps),
            {
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
    )


def build_e02_e03_e07_input_spec_chain_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    draft_lengths = _draft_lengths(args, num_reqs)
    num_new_sampled_tokens = int(args.get("num_new_sampled_tokens", 1))
    num_speculative_steps = int(args.get("num_speculative_steps", 7))
    decode_query_len = int(
        args.get("decode_query_len", num_speculative_steps + num_new_sampled_tokens)
    )
    if max(draft_lengths) > num_speculative_steps:
        raise ValueError("draft length exceeds num_speculative_steps")
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    candidate_target = str(args.get("candidate_target", "E02"))
    if candidate_target not in {"E02", "E03", "E07"}:
        raise ValueError("candidate_target must be E02, E03, or E07")
    query_lens = [length + num_new_sampled_tokens for length in draft_lengths]
    total_tokens = sum(query_lens)
    device = "cuda"

    idx_mapping = torch.tensor(
        list(reversed(range(num_reqs))), dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    num_computed_tokens = torch.arange(
        11,
        11 + max_num_reqs,
        dtype=torch.int32,
        device=device,
    )
    last_sampled_tokens = torch.arange(
        100,
        100 + max_num_reqs,
        dtype=torch.int32,
        device=device,
    )
    prefill_len = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
    draft_tokens = torch.arange(
        max_num_reqs * num_speculative_steps,
        dtype=torch.int32,
        device=device,
    ).reshape(max_num_reqs, num_speculative_steps)
    cu_num_logits = query_start_loc.clone()

    def allocate() -> dict[str, torch.Tensor]:
        return {
            "positions": torch.full(
                (total_tokens,), _I64_GUARD, dtype=torch.int64, device=device
            ),
            "seq_lens": torch.full(
                (max_num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "is_padding": torch.full(
                (total_tokens,), _BOOL_GUARD, dtype=torch.bool, device=device
            ),
            "input_ids": torch.full(
                (total_tokens,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "logits_indices": torch.full(
                (total_tokens,), _I64_GUARD, dtype=torch.int64, device=device
            ),
            "expanded_idx_mapping": torch.full(
                (total_tokens,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "expanded_local_pos": torch.full(
                (total_tokens,), _I32_GUARD, dtype=torch.int32, device=device
            ),
        }

    baseline_state = allocate()
    candidate_state = allocate()

    production_e03_block = triton.next_power_of_2(
        num_speculative_steps + num_new_sampled_tokens
    )
    production_e07_block = triton.next_power_of_2(decode_query_len)

    def launch(state: Mapping[str, torch.Tensor], target: str | None) -> torch.Tensor:
        tuned_block = candidate_block
        tuned_warps = candidate_warps
        if candidate_mode == "mirror":
            tuned_block = {
                "E02": 1024,
                "E03": production_e03_block,
                "E07": production_e07_block,
            }.get(target, 1024)
            tuned_warps = None
        e02_block = tuned_block if target == "E02" else 1024
        e03_block = tuned_block if target == "E03" else production_e03_block
        e07_block = tuned_block if target == "E07" else production_e07_block
        e02_warps = tuned_warps if target == "E02" else None
        e03_warps = tuned_warps if target == "E03" else None
        e07_warps = tuned_warps if target == "E07" else None
        _prepare_pos_seq_lens_kernel[(num_reqs + 1,)](
            state["positions"],
            state["is_padding"],
            state["seq_lens"],
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            max_num_reqs,
            CLEAR_PADDING=True,
            **_launch_kwargs(e02_block, e02_warps),
        )
        _combine_sampled_and_draft_tokens_kernel[(num_reqs,)](
            state["input_ids"],
            idx_mapping,
            last_sampled_tokens,
            query_start_loc,
            state["seq_lens"],
            prefill_len,
            draft_tokens,
            draft_tokens.stride(0),
            cu_num_logits,
            state["logits_indices"],
            NUM_NEW_SAMPLED_TOKENS=num_new_sampled_tokens,
            **_launch_kwargs(e03_block, e03_warps),
        )
        _expand_idx_mapping_kernel[(num_reqs,)](
            idx_mapping,
            state["expanded_idx_mapping"],
            state["expanded_local_pos"],
            cu_num_logits,
            **_launch_kwargs(e07_block, e07_warps),
        )
        return state["input_ids"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = _exact_tensor_checks(candidate_state, baseline_state)
        checks["padding_cleared"] = bool((~candidate_state["is_padding"]).all())
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": f"e02-e03-e07-b{num_reqs}-draft{'_'.join(map(str, draft_lengths))}",
        "chain": "E02-E03-E07-input-spec",
        "num_reqs": num_reqs,
        "draft_lengths": draft_lengths,
        "num_speculative_steps": num_speculative_steps,
        "candidate_mode": candidate_mode,
        "candidate_target": candidate_target,
    }
    return _case(
        baseline=Provider(
            "E02-E03-E07-production",
            lambda: launch(baseline_state, None),
            {
                "e02_block_size": 1024,
                "e03_block_size": production_e03_block,
                "e07_block_size": production_e07_block,
                "timed_provider_excludes_setup": True,
            },
        ),
        candidate=Provider(
            f"E02-E03-E07-{candidate_target}-{candidate_mode}",
            lambda: launch(candidate_state, candidate_target),
            {
                "candidate_target": candidate_target,
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
    )


def build_e04_e05_post_chain_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    draft_lengths = _draft_lengths(args, num_reqs)
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    query_lens = [length + 1 for length in draft_lengths]
    vocab_size = int(args.get("vocab_size", 128))
    device = "cuda"

    idx_mapping = torch.tensor(
        list(reversed(range(num_reqs))), dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor(
        [32 + length for length in query_lens],
        dtype=torch.int32,
        device=device,
    )
    prefill_len = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
    cu_num_logits = query_start_loc.clone()
    num_sampled_snapshot = torch.ones((num_reqs,), dtype=torch.int32, device=device)
    sampled_tokens = torch.arange(
        num_reqs * (max(draft_lengths) + 1),
        dtype=torch.int32,
        device=device,
    ).reshape(num_reqs, max(draft_lengths) + 1)
    sampled_tokens %= vocab_size
    total_len_snapshot = torch.full(
        (max_num_reqs,), 10, dtype=torch.int32, device=device
    )
    num_computed_snapshot = torch.full(
        (max_num_reqs,), 20, dtype=torch.int32, device=device
    )

    def allocate() -> dict[str, torch.Tensor]:
        return {
            "num_sampled": num_sampled_snapshot.clone(),
            "num_rejected": torch.full(
                (num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "num_computed": num_computed_snapshot.clone(),
            "last_sampled": torch.full(
                (max_num_reqs,), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "output_bin_counts": torch.zeros(
                (max_num_reqs, vocab_size), dtype=torch.int32, device=device
            ),
            "all_token_ids": torch.full(
                (max_num_reqs, 64), _I32_GUARD, dtype=torch.int32, device=device
            ),
            "total_len": total_len_snapshot.clone(),
        }

    baseline_state = allocate()
    candidate_state = allocate()

    def reset(state: Mapping[str, torch.Tensor]) -> None:
        _reset_i32(state["num_sampled"], num_sampled_snapshot)
        _reset_i32(state["num_computed"], num_computed_snapshot)
        _reset_i32(state["total_len"], total_len_snapshot)
        state["output_bin_counts"].zero_()

    def launch(state: Mapping[str, torch.Tensor], warps: int) -> torch.Tensor:
        reset(state)
        _get_num_sampled_and_rejected_kernel[(num_reqs,)](
            state["num_sampled"],
            state["num_rejected"],
            seq_lens,
            cu_num_logits,
            idx_mapping,
            prefill_len,
        )
        _post_update_kernel[(num_reqs,)](
            idx_mapping,
            state["num_computed"],
            state["last_sampled"],
            state["output_bin_counts"],
            state["output_bin_counts"].stride(0),
            sampled_tokens,
            sampled_tokens.stride(0),
            state["num_sampled"],
            state["num_rejected"],
            query_start_loc,
            state["all_token_ids"],
            state["all_token_ids"].stride(0),
            state["total_len"],
            num_warps=warps,
        )
        return state["num_computed"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = _exact_tensor_checks(candidate_state, baseline_state)
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": f"e04-e05-b{num_reqs}-draft{'_'.join(map(str, draft_lengths))}",
        "chain": "E04-E05-post",
        "num_reqs": num_reqs,
        "draft_lengths": draft_lengths,
        "candidate_mode": candidate_mode,
        "reset": "symmetric",
    }
    return _case(
        baseline=Provider(
            "E04-E05-production",
            lambda: launch(baseline_state, 1),
            {"reset": "symmetric", "timed_provider_excludes_setup": True},
        ),
        candidate=Provider(
            f"E04-E05-{candidate_mode}",
            lambda: launch(candidate_state, candidate_warps or 1),
            {
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "reset": "symmetric",
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
    )


def build_e06_pp_update_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    query_lens = [int(args.get("query_len", 3)) + req for req in range(num_reqs)]
    device = "cuda"
    idx_mapping = torch.tensor(
        list(reversed(range(num_reqs))), dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    snapshot = torch.arange(100, 100 + max_num_reqs, dtype=torch.int32, device=device)

    def allocate() -> dict[str, torch.Tensor]:
        return {"num_computed": snapshot.clone()}

    baseline_state = allocate()
    candidate_state = allocate()

    def launch(
        state: Mapping[str, torch.Tensor], _block: int, warps: int | None
    ) -> torch.Tensor:
        _reset_i32(state["num_computed"], snapshot)
        kwargs: dict[str, Any] = {}
        if warps is not None:
            kwargs["num_warps"] = warps
        _post_update_num_computed_tokens_kernel[(num_reqs,)](
            idx_mapping,
            state["num_computed"],
            query_start_loc,
            **kwargs,
        )
        return state["num_computed"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = _exact_tensor_checks(candidate_state, baseline_state)
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": f"e06-b{num_reqs}-{candidate_mode}",
        "chain": "E06-pp-update",
        "num_reqs": num_reqs,
        "candidate_mode": candidate_mode,
    }
    return _case(
        baseline=Provider(
            "E06-production",
            lambda: launch(baseline_state, 1024, None),
            {"timed_provider_excludes_setup": True},
        ),
        candidate=Provider(
            f"E06-{candidate_mode}",
            lambda: launch(candidate_state, candidate_block, candidate_warps),
            {"num_warps": candidate_warps, "timed_provider_excludes_setup": True},
            correctness_comparator=compare,
        ),
        shape=shape,
    )


def build_e08_e09_block_slot_chain_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, max_num_reqs = _batch(args)
    num_groups = int(args.get("num_groups", 1))
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    candidate_target = str(args.get("candidate_target", "E08"))
    if candidate_target not in {"E08", "E09"}:
        raise ValueError("candidate_target must be E08 or E09")
    block_size = int(args.get("block_size", 16))
    max_tokens = int(args.get("max_num_batched_tokens", 64))
    query_lens = [3 + req for req in range(num_reqs)]
    total_tokens = sum(query_lens)
    device = "cuda"

    idx_mapping = torch.tensor(
        list(reversed(range(num_reqs))), dtype=torch.int32, device=device
    )
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    positions = torch.arange(total_tokens, dtype=torch.int64, device=device) % 48
    is_padding = torch.zeros((max_tokens,), dtype=torch.bool, device=device)
    is_padding[1:total_tokens:3] = bool(args.get("padding", False))
    src_tables = [
        (
            torch.arange(max_num_reqs * 16, dtype=torch.int32, device=device).reshape(
                max_num_reqs, 16
            )
            + group * 100
        )
        for group in range(num_groups)
    ]
    block_table_ptrs = _ptr_tensor(src_tables)
    strides = torch.tensor(
        [table.stride(0) for table in src_tables], dtype=torch.int64, device=device
    )
    num_blocks = torch.full(
        (num_groups, max_num_reqs), 8, dtype=torch.int32, device=device
    )
    block_sizes = torch.full(
        (num_groups,), block_size, dtype=torch.int32, device=device
    )

    def allocate() -> dict[str, torch.Tensor]:
        dst_tables = [torch.full_like(table, _I32_GUARD) for table in src_tables]
        state = {
            "slot_mappings": torch.full(
                (num_groups, max_tokens), _I64_GUARD, dtype=torch.int64, device=device
            ),
            "block_table_ptrs": _ptr_tensor(dst_tables),
        }
        state.update(
            {f"block_table_{group}": table for group, table in enumerate(dst_tables)}
        )
        state.update(
            {
                f"source_block_table_{group}": table
                for group, table in enumerate(src_tables)
            }
        )
        return state

    baseline_state = allocate()
    candidate_state = allocate()

    def launch(state: Mapping[str, torch.Tensor], target: str | None) -> torch.Tensor:
        e08_block = candidate_block if target == "E08" else 1024
        e09_block = candidate_block if target == "E09" else 1024
        e08_warps = candidate_warps if target == "E08" else None
        e09_warps = candidate_warps if target == "E09" else None
        _gather_block_tables_kernel[(num_groups, max_num_reqs)](
            idx_mapping,
            block_table_ptrs,
            state["block_table_ptrs"],
            strides,
            num_blocks,
            num_blocks.stride(0),
            num_reqs,
            **_launch_kwargs(e08_block, e08_warps),
        )
        e09_kwargs: dict[str, Any] = {}
        if e09_warps is not None:
            e09_kwargs["num_warps"] = e09_warps
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_tokens,
            idx_mapping,
            query_start_loc,
            positions,
            is_padding,
            state["block_table_ptrs"],
            strides,
            block_sizes,
            state["slot_mappings"],
            state["slot_mappings"].stride(0),
            0,
            CP_SIZE=1,
            CP_INTERLEAVE=1,
            PAD_ID=PAD_SLOT_ID,
            APPLY_PADDING_MASK=bool(args.get("padding", False)),
            TRITON_BLOCK_SIZE=e09_block,
            **e09_kwargs,
        )
        return state["slot_mappings"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = {
            "slot_mappings": torch.equal(
                candidate_state["slot_mappings"], baseline_state["slot_mappings"]
            ),
            **{
                f"block_table_{group}": torch.equal(
                    candidate_state[f"block_table_{group}"],
                    baseline_state[f"block_table_{group}"],
                )
                for group in range(num_groups)
            },
        }
        checks["padding_tail"] = bool(
            (candidate_state["slot_mappings"][:, total_tokens:] == PAD_SLOT_ID).all()
        )
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": (
            f"e08-e09-b{num_reqs}-g{num_groups}"
            f"-pad{int(bool(args.get('padding', False)))}"
        ),
        "chain": "E08-E09-block-table-slot-mapping",
        "num_reqs": num_reqs,
        "num_groups": num_groups,
        "padding": bool(args.get("padding", False)),
        "candidate_mode": candidate_mode,
        "candidate_target": candidate_target,
    }
    return _case(
        baseline=Provider(
            "E08-E09-production-block1024",
            lambda: launch(baseline_state, None),
            {"block_size": 1024, "timed_provider_excludes_setup": True},
        ),
        candidate=Provider(
            f"E08-E09-{candidate_target}-{candidate_mode}",
            lambda: launch(candidate_state, candidate_target),
            {
                "candidate_target": candidate_target,
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
    )


def build_e10_staged_write_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, _ = _batch(args)
    num_groups = int(args.get("num_groups", 1))
    candidate_mode = _mode(args)
    candidate_block, candidate_warps = _geometry(candidate_mode)
    content_len = int(args.get("content_len", 5))
    if content_len < 1:
        raise ValueError("content_len must be positive")
    starts_host = [(req_idx * 3) % 7 for req_idx in range(num_reqs)]
    required_row_width = max(starts_host) + content_len
    row_width = int(args.get("row_width", max(32, required_row_width)))
    if row_width < required_row_width:
        raise ValueError("row_width must cover every staged write")
    device = "cuda"

    group_ids = torch.arange(num_reqs, dtype=torch.int32, device=device) % num_groups
    indices = torch.arange(num_reqs, dtype=torch.int32, device=device)
    starts = torch.tensor(starts_host, dtype=torch.int32, device=device)
    lens = torch.full((num_reqs,), content_len, dtype=torch.int32, device=device)
    cu_lens = torch.cumsum(lens, dim=0)
    contents = (
        torch.arange(int(cu_lens[-1].item()), dtype=torch.int32, device=device) + 9
    )
    strides = torch.full((num_groups,), row_width, dtype=torch.int64, device=device)

    def allocate() -> dict[str, torch.Tensor]:
        tensors = [
            torch.full(
                (num_reqs, row_width), _I32_GUARD, dtype=torch.int32, device=device
            )
            for _ in range(num_groups)
        ]
        return {
            "flat": torch.empty((), dtype=torch.int32, device=device),
            "ptrs": _ptr_tensor(tensors),
            **{f"group_{idx}": tensor for idx, tensor in enumerate(tensors)},
        }

    baseline_state = allocate()
    candidate_state = allocate()

    def launch(
        state: Mapping[str, torch.Tensor], block: int, warps: int | None
    ) -> torch.Tensor:
        kwargs = _launch_kwargs(block, warps)
        if num_groups == 1:
            _apply_write_kernel[(num_reqs,)](
                state["group_0"],
                row_width,
                indices,
                starts,
                contents,
                cu_lens,
                group_ids,
                MULTI_GROUP=False,
                **kwargs,
            )
            return state["group_0"]
        _apply_write_kernel[(num_reqs,)](
            state["ptrs"],
            strides,
            indices,
            starts,
            contents,
            cu_lens,
            group_ids,
            MULTI_GROUP=True,
            **kwargs,
        )
        return state["group_0"]

    def compare(*_: Any) -> dict[str, Any]:
        names = [f"group_{idx}" for idx in range(num_groups)]
        checks = {
            name: torch.equal(candidate_state[name], baseline_state[name])
            for name in names
        }
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": (f"e10-b{num_reqs}-g{num_groups}-len{content_len}-{candidate_mode}"),
        "chain": "E10-staged-write",
        "num_reqs": num_reqs,
        "num_groups": num_groups,
        "content_len": content_len,
        "row_width": row_width,
        "candidate_mode": candidate_mode,
    }
    return _case(
        baseline=Provider(
            "E10-production-block1024",
            lambda: launch(baseline_state, 1024, None),
            {"block_size": 1024, "timed_provider_excludes_setup": True},
        ),
        candidate=Provider(
            f"E10-{candidate_mode}",
            lambda: launch(candidate_state, candidate_block, candidate_warps),
            {
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
    )
