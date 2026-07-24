# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import (
    _compute_slot_mappings_kernel,
    _gather_block_tables_and_compute_slot_mappings_kernel,
    _gather_block_tables_kernel,
)

_I32_GUARD = -777_000
_I64_GUARD = -888_000

_CANDIDATE_GEOMETRY = {
    "fused-b256-w4": (256, 4),
    "fused-b512-w4": (512, 4),
    "fused-b1024-w4": (1024, 4),
}


def _prefix(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + length)
    return out


def _ptr_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [tensor.data_ptr() for tensor in tensors],
        dtype=torch.uint64,
        device="cuda",
    )


def _batch(args: Mapping[str, Any]) -> tuple[int, int, int]:
    num_reqs = int(args.get("num_reqs", 4))
    num_reqs_padded = int(args.get("num_reqs_padded", num_reqs))
    max_num_reqs = int(args.get("max_num_reqs", max(num_reqs_padded, num_reqs)))
    if num_reqs < 1:
        raise ValueError("E08/E09 fused case requires at least one request")
    if num_reqs_padded < num_reqs:
        raise ValueError("num_reqs_padded must be >= num_reqs")
    if max_num_reqs < num_reqs_padded:
        raise ValueError("max_num_reqs must be >= num_reqs_padded")
    return num_reqs, num_reqs_padded, max_num_reqs


def _candidate_geometry(args: Mapping[str, Any]) -> tuple[str, int, int]:
    candidate_mode = str(args.get("candidate_mode", "fused-b1024-w4"))
    if candidate_mode not in _CANDIDATE_GEOMETRY:
        raise ValueError("unsupported E08/E09 fused candidate mode")
    block_size, num_warps = _CANDIDATE_GEOMETRY[candidate_mode]
    return candidate_mode, block_size, num_warps


def _query_lens(args: Mapping[str, Any], num_reqs: int) -> list[int]:
    value = args.get("query_lens")
    if value is None:
        base = [1, 3, 5, 7]
        return [base[index % len(base)] for index in range(num_reqs)]
    if not isinstance(value, list | tuple) or len(value) != num_reqs:
        raise ValueError("query_lens must be a per-request list")
    query_lens = [int(item) for item in value]
    if any(length < 1 for length in query_lens):
        raise ValueError("query_lens entries must be positive")
    return query_lens


def _idx_mapping(
    args: Mapping[str, Any],
    num_reqs: int,
    max_num_reqs: int,
) -> list[int]:
    value = args.get("idx_mapping")
    if value is None:
        mapping = list(reversed(range(num_reqs)))
    else:
        if not isinstance(value, list | tuple) or len(value) != num_reqs:
            raise ValueError("idx_mapping must be a per-request list")
        mapping = [int(item) for item in value]
    if min(mapping) < 0 or max(mapping) >= max_num_reqs:
        raise ValueError("idx_mapping must reference valid request-state rows")
    return mapping


def build_e08_e09_fused_case(args: Mapping[str, Any]) -> ChainCase:
    num_reqs, num_reqs_padded, max_num_reqs = _batch(args)
    num_groups = int(args.get("num_groups", 1))
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    block_size = int(args.get("block_size", 16))
    if block_size < 1:
        raise ValueError("block_size must be positive")
    cp_size = int(args.get("cp_size", 1))
    cp_rank = int(args.get("cp_rank", 0))
    cp_interleave = int(args.get("cp_interleave", 1))
    if cp_size < 1 or not 0 <= cp_rank < cp_size or cp_interleave < 1:
        raise ValueError("invalid CP_SIZE/CP_RANK/CP_INTERLEAVE")
    candidate_mode, candidate_block, candidate_warps = _candidate_geometry(args)
    padding = bool(args.get("padding", False))
    query_lens = _query_lens(args, num_reqs)
    total_tokens = sum(query_lens)
    max_tokens = int(args.get("max_num_batched_tokens", max(64, total_tokens + 17)))
    if max_tokens < total_tokens:
        raise ValueError("max_num_batched_tokens must cover query_lens")
    row_width = int(args.get("row_width", 16))
    max_position = block_size * cp_size * max(row_width // 2, 1) - 1
    if max_position < 0:
        raise ValueError("row_width must be positive")
    device = "cuda"

    mapping_values = _idx_mapping(args, num_reqs, max_num_reqs)
    idx_mapping = torch.tensor(mapping_values, dtype=torch.int32, device=device)
    slot_idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        _prefix(query_lens), dtype=torch.int32, device=device
    )
    positions = (
        torch.arange(total_tokens, dtype=torch.int64, device=device) * 7
        + torch.arange(total_tokens, dtype=torch.int64, device=device).remainder(5)
    ).remainder_(max_position + 1)
    is_padding = torch.zeros((max_tokens,), dtype=torch.bool, device=device)
    if padding:
        is_padding[1:total_tokens:3] = True
        is_padding[total_tokens - 1 : total_tokens] = True

    src_tables = [
        (
            torch.arange(max_num_reqs * row_width, dtype=torch.int32, device=device)
            .reshape(max_num_reqs, row_width)
            .mul_(3)
            .add_(group * 10_000)
        )
        for group in range(num_groups)
    ]
    block_table_ptrs = _ptr_tensor(src_tables)
    block_table_strides = torch.tensor(
        [table.stride(0) for table in src_tables],
        dtype=torch.int64,
        device=device,
    )
    num_blocks = torch.full(
        (num_groups, max_num_reqs),
        row_width // 2,
        dtype=torch.int32,
        device=device,
    )
    block_sizes = torch.full(
        (num_groups,), block_size, dtype=torch.int32, device=device
    )

    def allocate() -> dict[str, torch.Tensor]:
        dst_tables = [
            torch.full_like(table, _I32_GUARD, device=device) for table in src_tables
        ]
        state = {
            "slot_mappings": torch.full(
                (num_groups, max_tokens),
                _I64_GUARD,
                dtype=torch.int64,
                device=device,
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

    def launch_baseline() -> torch.Tensor:
        _gather_block_tables_kernel[(num_groups, num_reqs_padded)](
            idx_mapping,
            block_table_ptrs,
            baseline_state["block_table_ptrs"],
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            num_reqs,
            BLOCK_SIZE=1024,
        )
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            max_tokens,
            slot_idx_mapping,
            query_start_loc,
            positions,
            is_padding,
            baseline_state["block_table_ptrs"],
            block_table_strides,
            block_sizes,
            baseline_state["slot_mappings"],
            baseline_state["slot_mappings"].stride(0),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            APPLY_PADDING_MASK=padding,
            TRITON_BLOCK_SIZE=1024,
        )
        return baseline_state["slot_mappings"]

    def launch_candidate() -> torch.Tensor:
        grid = (num_groups, max(num_reqs_padded, num_reqs + 1))
        _gather_block_tables_and_compute_slot_mappings_kernel[grid](
            idx_mapping,
            query_start_loc,
            positions,
            is_padding,
            block_table_ptrs,
            candidate_state["block_table_ptrs"],
            block_table_strides,
            num_blocks,
            num_blocks.stride(0),
            block_sizes,
            candidate_state["slot_mappings"],
            candidate_state["slot_mappings"].stride(0),
            max_tokens,
            num_reqs,
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            APPLY_PADDING_MASK=padding,
            NUM_REQS_PADDED=num_reqs_padded,
            BLOCK_SIZE=candidate_block,
            num_warps=candidate_warps,
        )
        return candidate_state["slot_mappings"]

    def compare(*_: Any) -> dict[str, Any]:
        checks = {
            "slot_mappings": torch.equal(
                candidate_state["slot_mappings"], baseline_state["slot_mappings"]
            ),
            "padding_tail": bool(
                (candidate_state["slot_mappings"][:, total_tokens:] == PAD_SLOT_ID)
                .all()
                .item()
            ),
            **{
                f"block_table_{group}": torch.equal(
                    candidate_state[f"block_table_{group}"],
                    baseline_state[f"block_table_{group}"],
                )
                for group in range(num_groups)
            },
        }
        if padding:
            token_padding = is_padding[:total_tokens]
            checks["padding_mask"] = bool(
                (
                    candidate_state["slot_mappings"][:, :total_tokens][:, token_padding]
                    == PAD_SLOT_ID
                )
                .all()
                .item()
            )
        if num_reqs_padded > num_reqs:
            checks["padded_block_table_rows_zero"] = all(
                bool(
                    (
                        candidate_state[f"block_table_{group}"][
                            num_reqs:num_reqs_padded
                        ]
                        == 0
                    )
                    .all()
                    .item()
                )
                for group in range(num_groups)
            )
        return {"passed": all(checks.values()), "exact": checks, "shape": shape}

    shape = {
        "name": (
            f"e08-e09-fused-b{num_reqs}-bp{num_reqs_padded}-g{num_groups}"
            f"-bs{block_size}-cp{cp_size}i{cp_interleave}"
            f"-pad{int(padding)}-{candidate_mode}"
        ),
        "chain": "E08-E09-fused-block-table-slot-mapping",
        "num_reqs": num_reqs,
        "num_reqs_padded": num_reqs_padded,
        "num_groups": num_groups,
        "block_size": block_size,
        "padding": padding,
        "cp_size": cp_size,
        "cp_rank": cp_rank,
        "cp_interleave": cp_interleave,
        "candidate_mode": candidate_mode,
    }
    return ChainCase(
        baseline=Provider(
            "E08-E09-production-two-launch",
            launch_baseline,
            {
                "operator_launches": 2,
                "block_size": 1024,
                "timed_provider_excludes_setup": True,
            },
        ),
        candidate=Provider(
            f"E08-E09-fused-{candidate_mode}",
            launch_candidate,
            {
                "operator_launches": 1,
                "block_size": candidate_block,
                "num_warps": candidate_warps,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


__all__ = ["build_e08_e09_fused_case"]
