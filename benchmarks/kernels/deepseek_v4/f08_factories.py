# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.utils.math_utils import cdiv
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)

_OUTPUT_GUARD = -777777
_PARALLEL_DRAFT_TOKEN = 128001


@dataclass
class _Outputs:
    input_ids: torch.Tensor
    context_positions: torch.Tensor
    query_positions: torch.Tensor
    context_slot_mapping: torch.Tensor
    query_slot_mapping: torch.Tensor
    token_indices: torch.Tensor
    return_value: torch.Tensor


def _split_lengths(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _int_list(
    args: Mapping[str, Any], key: str, default: list[int], expected: int
) -> list[int]:
    raw = args.get(key)
    values = default if raw is None else [int(value) for value in raw]
    if len(values) != expected:
        raise ValueError(f"F08 {key} must contain one value per request")
    return values


def _parse_candidate_config(args: Mapping[str, Any]) -> tuple[int | None, int | None]:
    block_size = args.get("candidate_block_size")
    num_warps = args.get("candidate_num_warps")
    parsed_block_size = None if block_size is None else int(block_size)
    parsed_num_warps = None if num_warps is None else int(num_warps)
    legal_block_sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    legal_num_warps = (1, 2, 4, 8)
    if parsed_block_size is not None and parsed_block_size not in legal_block_sizes:
        raise ValueError("F08 candidate_block_size must be a power of two <= 256")
    if parsed_num_warps is not None and parsed_num_warps not in legal_num_warps:
        raise ValueError("F08 candidate_num_warps must be one of 1, 2, 4, or 8")
    return parsed_block_size, parsed_num_warps


def _allocate_outputs(
    num_context: int,
    num_query_total: int,
    token_indices_count: int,
    device: torch.device,
) -> _Outputs:
    return _Outputs(
        input_ids=torch.full(
            (num_query_total,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        context_positions=torch.full(
            (num_context,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        query_positions=torch.full(
            (num_query_total,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        context_slot_mapping=torch.full(
            (num_context,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        query_slot_mapping=torch.full(
            (num_query_total,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        token_indices=torch.full(
            (token_indices_count,), _OUTPUT_GUARD, dtype=torch.int32, device=device
        ),
        return_value=torch.zeros(1, dtype=torch.int32, device=device),
    )


def _slot_for_position(
    position: int,
    req_idx: int,
    block_table_host: list[list[int]],
    block_size: int,
) -> int:
    block_table_width = len(block_table_host[req_idx])
    block_number = min(position // block_size, block_table_width - 1)
    return block_table_host[req_idx][block_number] * block_size + position % block_size


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    method = str(args.get("method", "dflash"))
    if method != "dflash":
        raise ValueError("F08 copy-and-expand factory only supports method=dflash")

    num_reqs = int(args.get("num_reqs", 1))
    total_context_tokens = int(args.get("total_context_tokens", num_reqs))
    num_speculative_tokens = int(args.get("num_speculative_tokens", 7))
    has_num_rejected = bool(args.get("has_num_rejected", False))
    cache_block_size = int(args.get("block_size", 16))
    max_model_len = int(args.get("max_model_len", 131072))
    context_tokens = int(args.get("context_tokens", 8192))
    context_jitter = int(args.get("context_jitter", 0))
    candidate_block_size, candidate_num_warps = _parse_candidate_config(args)

    if num_reqs < 1:
        raise ValueError("F08 num_reqs must be positive")
    if total_context_tokens < num_reqs:
        raise ValueError("F08 total_context_tokens must cover every request")
    if num_speculative_tokens < 1:
        raise ValueError("F08 num_speculative_tokens must be positive")
    if cache_block_size < 1:
        raise ValueError("F08 block_size must be positive")
    if max_model_len < cache_block_size:
        raise ValueError("F08 max_model_len must cover one cache block")

    context_lens = _int_list(
        args,
        "context_lens",
        _split_lengths(total_context_tokens, num_reqs),
        num_reqs,
    )
    if any(length < 1 for length in context_lens):
        raise ValueError("F08 every request needs at least one context token")
    if sum(context_lens) != total_context_tokens:
        raise ValueError("F08 context_lens must sum to total_context_tokens")

    num_rejected = _int_list(args, "num_rejected", [0] * num_reqs, num_reqs)
    if not has_num_rejected and any(num_rejected):
        raise ValueError("F08 num_rejected requires has_num_rejected=true")
    if any(
        rejected < 0 or rejected >= length
        for rejected, length in zip(num_rejected, context_lens, strict=True)
    ):
        raise ValueError("F08 rejected counts must leave one valid context token")

    seq_lens = _int_list(
        args,
        "sequence_tokens",
        [context_tokens - index * context_jitter for index in range(num_reqs)],
        num_reqs,
    )
    if any(
        seq_len < ctx_len
        for seq_len, ctx_len in zip(seq_lens, context_lens, strict=True)
    ):
        raise ValueError("F08 sequence lengths must cover context lengths")

    device = torch.device("cuda")
    query_start_host = [0]
    target_positions_host: list[int] = []
    for seq_len, ctx_len in zip(seq_lens, context_lens, strict=True):
        start = seq_len - ctx_len
        target_positions_host.extend(range(start, seq_len))
        query_start_host.append(query_start_host[-1] + ctx_len)

    target_positions = torch.tensor(
        target_positions_host, dtype=torch.int64, device=device
    )
    query_start_loc = torch.tensor(query_start_host, dtype=torch.int32, device=device)
    next_token_ids = torch.arange(
        10000, 10000 + num_reqs, dtype=torch.int64, device=device
    )
    num_rejected_tensor = torch.tensor(num_rejected, dtype=torch.int32, device=device)

    block_table_width = cdiv(max_model_len, cache_block_size)
    block_table_host = [
        [
            (req_idx * block_table_width + block_idx) * 5 + 11
            for block_idx in range(block_table_width)
        ]
        for req_idx in range(num_reqs)
    ]
    block_table = torch.tensor(block_table_host, dtype=torch.int32, device=device)

    num_query_per_req = num_speculative_tokens + 1
    num_query_total = num_reqs * num_query_per_req
    expected = _allocate_outputs(
        total_context_tokens,
        num_query_total,
        num_reqs * num_speculative_tokens,
        device,
    )
    next_token_ids_host = [10000 + req_idx for req_idx in range(num_reqs)]
    for req_idx, ctx_len in enumerate(context_lens):
        ctx_start = query_start_host[req_idx]
        valid_ctx_end = query_start_host[req_idx + 1] - num_rejected[req_idx]
        last_pos = target_positions_host[valid_ctx_end - 1]
        for ctx_offset in range(ctx_len):
            context_index = ctx_start + ctx_offset
            position = target_positions_host[context_index]
            expected.context_positions[context_index] = position
            expected.context_slot_mapping[context_index] = _slot_for_position(
                position, req_idx, block_table_host, cache_block_size
            )
        for query_offset in range(num_query_per_req):
            query_index = req_idx * num_query_per_req + query_offset
            position = last_pos + 1 + query_offset
            expected.input_ids[query_index] = (
                next_token_ids_host[req_idx]
                if query_offset == 0
                else _PARALLEL_DRAFT_TOKEN
            )
            expected.query_positions[query_index] = position
            expected.query_slot_mapping[query_index] = _slot_for_position(
                position, req_idx, block_table_host, cache_block_size
            )
            if query_offset > 0:
                expected.token_indices[
                    req_idx * num_speculative_tokens + query_offset - 1
                ] = query_index

    max_ctx_per_req = max(context_lens)
    max_tokens_per_req = max_ctx_per_req + num_query_per_req
    production_block_size = min(256, next_power_of_2(max_tokens_per_req))

    is_ragged = int(len(set(context_lens)) > 1)

    return {
        "method": method,
        "num_reqs": num_reqs,
        "context_lens": context_lens,
        "total_context_tokens": total_context_tokens,
        "num_speculative_tokens": num_speculative_tokens,
        "num_query_per_req": num_query_per_req,
        "num_query_total": num_query_total,
        "has_num_rejected": has_num_rejected,
        "num_rejected": num_rejected,
        "cache_block_size": cache_block_size,
        "max_model_len": max_model_len,
        "block_table_width": block_table_width,
        "production_block_size": production_block_size,
        "candidate_block_size": candidate_block_size,
        "candidate_num_warps": candidate_num_warps,
        "target_positions": target_positions,
        "query_start_loc": query_start_loc,
        "next_token_ids": next_token_ids,
        "num_rejected_tensor": num_rejected_tensor,
        "block_table": block_table,
        "expected": expected,
        "expected_sources": {
            "target_positions": target_positions.clone(),
            "query_start_loc": query_start_loc.clone(),
            "next_token_ids": next_token_ids.clone(),
            "num_rejected_tensor": num_rejected_tensor.clone(),
            "block_table": block_table.clone(),
        },
        "shape": {
            "name": (
                f"method{method}-b{num_reqs}-ctx{total_context_tokens}"
                f"-draft{num_speculative_tokens}-ragged{is_ragged}"
                f"-rejected{int(has_num_rejected)}-pblk{production_block_size}"
                f"-cblk{candidate_block_size or production_block_size}"
                f"-warps{candidate_num_warps or 'default'}"
            ),
            "method": method,
            "num_reqs": num_reqs,
            "total_context_tokens": total_context_tokens,
            "context_lens": context_lens,
            "num_speculative_tokens": num_speculative_tokens,
            "num_query_per_req": num_query_per_req,
            "has_num_rejected": has_num_rejected,
            "num_rejected": num_rejected,
            "block_size": cache_block_size,
            "max_model_len": max_model_len,
            "block_table_width": block_table_width,
            "production_block_size": production_block_size,
            "candidate_block_size": candidate_block_size,
            "candidate_num_warps": candidate_num_warps,
            "chain": "copy-and-expand-dflash-first-pass-inputs",
        },
    }


def _launch(
    outputs: _Outputs,
    inputs: Mapping[str, Any],
    *,
    kernel_block_size: int,
    num_warps: int | None = None,
) -> None:
    max_tokens_per_req = max(inputs["context_lens"]) + inputs["num_query_per_req"]
    num_blocks = cdiv(max_tokens_per_req, kernel_block_size)
    grid = (inputs["num_reqs"], num_blocks)
    launch_kwargs = {
        "BLOCK_SIZE": kernel_block_size,
        "HAS_NUM_REJECTED": inputs["has_num_rejected"],
    }
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps

    copy_and_expand_dflash_inputs_kernel[grid](
        next_token_ids_ptr=inputs["next_token_ids"],
        target_positions_ptr=inputs["target_positions"],
        out_input_ids_ptr=outputs.input_ids,
        out_context_positions_ptr=outputs.context_positions,
        out_query_positions_ptr=outputs.query_positions,
        out_context_slot_mapping_ptr=outputs.context_slot_mapping,
        out_query_slot_mapping_ptr=outputs.query_slot_mapping,
        out_token_indices_ptr=outputs.token_indices,
        block_table_ptr=inputs["block_table"],
        block_table_stride=inputs["block_table"].stride(0),
        query_start_loc_ptr=inputs["query_start_loc"],
        num_rejected_tokens_ptr=(
            inputs["num_rejected_tensor"] if inputs["has_num_rejected"] else 0
        ),
        parallel_drafting_token_id=_PARALLEL_DRAFT_TOKEN,
        block_size=inputs["cache_block_size"],
        num_query_per_req=inputs["num_query_per_req"],
        num_speculative_tokens=inputs["num_speculative_tokens"],
        total_input_tokens=inputs["total_context_tokens"],
        **launch_kwargs,
    )


def build_f08_copy_and_expand_dflash_inputs_case(
    args: Mapping[str, Any],
) -> ChainCase:
    inputs = _make_inputs(args)
    expected = inputs["expected"]
    baseline = _allocate_outputs(
        inputs["total_context_tokens"],
        inputs["num_query_total"],
        inputs["num_reqs"] * inputs["num_speculative_tokens"],
        torch.device("cuda"),
    )
    candidate = _allocate_outputs(
        inputs["total_context_tokens"],
        inputs["num_query_total"],
        inputs["num_reqs"] * inputs["num_speculative_tokens"],
        torch.device("cuda"),
    )

    def run_baseline() -> torch.Tensor:
        _launch(
            baseline,
            inputs,
            kernel_block_size=inputs["production_block_size"],
        )
        return baseline.return_value

    def run_candidate() -> torch.Tensor:
        _launch(
            candidate,
            inputs,
            kernel_block_size=(
                inputs["candidate_block_size"] or inputs["production_block_size"]
            ),
            num_warps=inputs["candidate_num_warps"],
        )
        return candidate.return_value

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        output_names = (
            "input_ids",
            "context_positions",
            "query_positions",
            "context_slot_mapping",
            "query_slot_mapping",
            "token_indices",
        )
        checks: dict[str, bool] = {}
        for provider_name, outputs in (
            ("baseline", baseline),
            ("candidate", candidate),
        ):
            for output_name in output_names:
                checks[f"{provider_name}_{output_name}"] = torch.equal(
                    getattr(outputs, output_name), getattr(expected, output_name)
                )
        for source_name, source_snapshot in inputs["expected_sources"].items():
            checks[f"source_{source_name}"] = torch.equal(
                inputs[source_name], source_snapshot
            )
        return {"passed": all(checks.values()), "exact": checks}

    return ChainCase(
        baseline=Provider(
            "dflash-production-first-pass-launch",
            run_baseline,
            {
                "launches": 1,
                "candidate_active": False,
                "BLOCK_SIZE": inputs["production_block_size"],
                "HAS_NUM_REJECTED": inputs["has_num_rejected"],
            },
        ),
        candidate=Provider(
            "dflash-copy-expand-benchmark-only-tuned-launch",
            run_candidate,
            {
                "launches": 1,
                "candidate_active": (
                    inputs["candidate_block_size"] is not None
                    or inputs["candidate_num_warps"] is not None
                ),
                "BLOCK_SIZE": (
                    inputs["candidate_block_size"] or inputs["production_block_size"]
                ),
                "num_warps": inputs["candidate_num_warps"],
                "HAS_NUM_REJECTED": inputs["has_num_rejected"],
            },
            correctness_comparator=compare,
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
