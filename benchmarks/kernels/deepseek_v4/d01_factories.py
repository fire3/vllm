# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _prepare_dflash_inputs_kernel,
    prepare_dflash_inputs,
)

_OUTPUT_GUARD = -777777
_PARALLEL_DRAFT_TOKEN = 128001


@dataclass
class _Outputs:
    input_ids: torch.Tensor
    query_positions: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    query_slot_mapping: torch.Tensor
    context_positions: torch.Tensor
    context_slot_mapping: torch.Tensor
    sample_indices: torch.Tensor
    sample_pos: torch.Tensor
    sample_idx_mapping: torch.Tensor
    return_value: torch.Tensor

    @property
    def input_buffers(self) -> SimpleNamespace:
        return SimpleNamespace(
            input_ids=self.input_ids,
            positions=self.query_positions,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
        )


def _split_lengths(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _int_list(
    args: Mapping[str, Any], key: str, default: list[int], expected: int
) -> list[int]:
    raw = args.get(key)
    values = default if raw is None else [int(value) for value in raw]
    if len(values) != expected:
        raise ValueError(f"D01 {key} must contain one value per request")
    return values


def _allocate_outputs(
    max_num_reqs: int,
    max_num_tokens: int,
    num_speculative_steps: int,
    device: torch.device,
) -> _Outputs:
    sample_capacity = max_num_reqs * num_speculative_steps
    return _Outputs(
        input_ids=torch.full(
            (max_num_tokens,), _OUTPUT_GUARD, dtype=torch.int32, device=device
        ),
        query_positions=torch.full(
            (max_num_tokens,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        query_start_loc=torch.full(
            (max_num_reqs + 1,),
            _OUTPUT_GUARD,
            dtype=torch.int32,
            device=device,
        ),
        seq_lens=torch.full(
            (max_num_reqs,), _OUTPUT_GUARD, dtype=torch.int32, device=device
        ),
        query_slot_mapping=torch.full(
            (max_num_tokens,),
            _OUTPUT_GUARD,
            dtype=torch.int64,
            device=device,
        ),
        context_positions=torch.full(
            (max_num_tokens,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        context_slot_mapping=torch.full(
            (max_num_tokens,),
            _OUTPUT_GUARD,
            dtype=torch.int64,
            device=device,
        ),
        sample_indices=torch.full(
            (sample_capacity,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        sample_pos=torch.full(
            (sample_capacity,), _OUTPUT_GUARD, dtype=torch.int64, device=device
        ),
        sample_idx_mapping=torch.full(
            (sample_capacity,), _OUTPUT_GUARD, dtype=torch.int32, device=device
        ),
        return_value=torch.zeros(1, dtype=torch.int32, device=device),
    )


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    num_reqs = int(args.get("num_reqs", 1))
    total_target_tokens = int(args.get("total_target_tokens", num_reqs))
    num_speculative_steps = int(args.get("num_speculative_steps", 7))
    sample_from_anchor = bool(args.get("sample_from_anchor", True))
    max_num_reqs = int(args.get("max_num_reqs", 128))
    max_num_tokens = int(args.get("max_num_tokens", 2048))
    block_size = int(args.get("block_size", 64))
    max_model_len = int(args.get("max_model_len", 131072))
    context_tokens = int(args.get("context_tokens", 8192))
    context_jitter = int(args.get("context_jitter", 0))
    remap_requests = bool(args.get("remap_requests", True))
    candidate_mode = str(args.get("candidate_mode", "mirror"))

    if num_reqs < 1 or num_reqs > max_num_reqs:
        raise ValueError("D01 num_reqs must fit max_num_reqs")
    if total_target_tokens < num_reqs or total_target_tokens > max_num_tokens:
        raise ValueError("D01 target token count must fit the graph token buffer")
    if num_speculative_steps < 1:
        raise ValueError("D01 speculative steps must be positive")
    if max_num_reqs < 1 or max_num_tokens < 1:
        raise ValueError("D01 graph capacities must be positive")
    if block_size not in (16, 32, 64, 128, 256):
        raise ValueError("D01 block_size must be a supported power of two")
    if max_model_len < block_size:
        raise ValueError("D01 max_model_len must cover one cache block")
    fixed_modes = {"fixed32", "fixed64", "fixed128", "fixed256"}
    if candidate_mode not in {"mirror", "dispatch", *fixed_modes}:
        raise ValueError(
            "D01 candidate_mode must be mirror, fixed32/64/128/256, or dispatch"
        )

    query_lens = _int_list(
        args,
        "target_query_lens",
        _split_lengths(total_target_tokens, num_reqs),
        num_reqs,
    )
    if any(length < 1 for length in query_lens):
        raise ValueError("D01 every request needs at least one target token")
    if sum(query_lens) != total_target_tokens:
        raise ValueError("D01 target query lengths must sum to total_target_tokens")

    num_rejected = _int_list(args, "num_rejected", [0] * num_reqs, num_reqs)
    if any(
        rejected < 0 or rejected >= length
        for rejected, length in zip(num_rejected, query_lens, strict=True)
    ):
        raise ValueError("D01 rejected counts must leave one valid target token")
    num_sampled = _int_list(args, "num_sampled", [1] * num_reqs, num_reqs)
    if any(value < 0 for value in num_sampled):
        raise ValueError("D01 sampled counts must be non-negative")

    seq_lens = _int_list(
        args,
        "sequence_tokens",
        [context_tokens - index * context_jitter for index in range(num_reqs)],
        num_reqs,
    )
    if any(
        seq_len < query_len
        for seq_len, query_len in zip(seq_lens, query_lens, strict=True)
    ):
        raise ValueError("D01 sequence lengths must cover target query lengths")

    num_query_per_req = num_speculative_steps + int(not sample_from_anchor)
    if num_reqs * num_query_per_req > max_num_tokens:
        raise ValueError("D01 query rows must fit max_num_tokens")

    device = torch.device("cuda")
    query_start_host = [0]
    target_positions_host: list[int] = []
    for seq_len, query_len in zip(seq_lens, query_lens, strict=True):
        target_positions_host.extend(range(seq_len - query_len, seq_len))
        query_start_host.append(query_start_host[-1] + query_len)

    target_positions = torch.full(
        (max_num_tokens,), _OUTPUT_GUARD, dtype=torch.int64, device=device
    )
    target_positions[:total_target_tokens] = torch.tensor(
        target_positions_host, dtype=torch.int64, device=device
    )
    target_query_start_loc = torch.tensor(
        query_start_host, dtype=torch.int32, device=device
    )

    if remap_requests:
        state_indices = [(index * 17 + 3) % max_num_reqs for index in range(num_reqs)]
        if len(set(state_indices)) != num_reqs:
            state_indices = list(range(num_reqs - 1, -1, -1))
    else:
        state_indices = list(range(num_reqs))
    idx_mapping = torch.tensor(state_indices, dtype=torch.int32, device=device)

    last_sampled = torch.arange(max_num_reqs, dtype=torch.int64, device=device).add_(
        10000
    )
    next_prefill_tokens = torch.arange(
        max_num_reqs, dtype=torch.int64, device=device
    ).add_(20000)
    num_sampled_tensor = torch.tensor(num_sampled, dtype=torch.int32, device=device)
    num_rejected_tensor = torch.tensor(num_rejected, dtype=torch.int32, device=device)

    block_table_width = cdiv(max_model_len, block_size)
    block_table = torch.arange(
        num_reqs * block_table_width, dtype=torch.int32, device=device
    ).view(num_reqs, block_table_width)
    block_table.mul_(5).add_(11)

    expected = _allocate_outputs(
        max_num_reqs, max_num_tokens, num_speculative_steps, device
    )
    expected.query_start_loc.fill_(num_reqs * num_query_per_req)
    expected.query_start_loc[:num_reqs] = torch.arange(
        num_reqs, dtype=torch.int32, device=device
    ).mul_(num_query_per_req)
    expected.seq_lens.zero_()
    expected.query_slot_mapping.fill_(PAD_SLOT_ID)
    expected.sample_indices.zero_()
    expected.sample_pos.zero_()
    expected.sample_idx_mapping.fill_(-1)

    for req_idx in range(num_reqs):
        ctx_start = query_start_host[req_idx]
        ctx_len = query_lens[req_idx]
        valid_ctx_len = ctx_len - num_rejected[req_idx]
        last_valid_pos = target_positions_host[ctx_start + valid_ctx_len - 1]
        expected.seq_lens[req_idx] = last_valid_pos + 1 + num_query_per_req
        req_state_idx = state_indices[req_idx]
        bonus_token = (
            10000 + req_state_idx if num_sampled[req_idx] > 0 else 20000 + req_state_idx
        )

        for ctx_offset in range(ctx_len):
            output_index = ctx_start + ctx_offset
            if ctx_offset < valid_ctx_len:
                position = target_positions_host[output_index]
                block_number = min(position // block_size, block_table_width - 1)
                block_id = (req_idx * block_table_width + block_number) * 5 + 11
                slot = block_id * block_size + position % block_size
                expected.context_positions[output_index] = position
                expected.context_slot_mapping[output_index] = slot
            else:
                expected.context_positions[output_index] = 0
                expected.context_slot_mapping[output_index] = PAD_SLOT_ID

        query_base = req_idx * num_query_per_req
        sample_offset = 0 if sample_from_anchor else 1
        for query_offset in range(num_query_per_req):
            query_index = query_base + query_offset
            query_position = last_valid_pos + 1 + query_offset
            expected.input_ids[query_index] = (
                bonus_token if query_offset == 0 else _PARALLEL_DRAFT_TOKEN
            )
            expected.query_positions[query_index] = min(
                query_position, max_model_len - 1
            )
            block_number = min(query_position // block_size, block_table_width - 1)
            block_id = (req_idx * block_table_width + block_number) * 5 + 11
            expected.query_slot_mapping[query_index] = (
                block_id * block_size + query_position % block_size
            )
            if query_offset >= sample_offset:
                sample_index = (
                    req_idx * num_speculative_steps + query_offset - sample_offset
                )
                expected.sample_indices[sample_index] = query_index
                expected.sample_pos[sample_index] = (
                    query_position + 1 if sample_from_anchor else query_position
                )
                expected.sample_idx_mapping[sample_index] = req_state_idx

    expected_sources = {
        "target_positions": target_positions.clone(),
        "target_query_start_loc": target_query_start_loc.clone(),
        "idx_mapping": idx_mapping.clone(),
        "last_sampled": last_sampled.clone(),
        "next_prefill_tokens": next_prefill_tokens.clone(),
        "num_sampled": num_sampled_tensor.clone(),
        "num_rejected": num_rejected_tensor.clone(),
        "block_table": block_table.clone(),
    }

    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.asarray(query_lens, dtype=np.int32),
        positions=target_positions,
        query_start_loc=target_query_start_loc,
        idx_mapping=idx_mapping,
    )
    return {
        "num_reqs": num_reqs,
        "query_lens": query_lens,
        "num_rejected_host": num_rejected,
        "num_sampled_host": num_sampled,
        "seq_lens_host": seq_lens,
        "num_speculative_steps": num_speculative_steps,
        "num_query_per_req": num_query_per_req,
        "sample_from_anchor": sample_from_anchor,
        "max_num_reqs": max_num_reqs,
        "max_num_tokens": max_num_tokens,
        "max_model_len": max_model_len,
        "block_size": block_size,
        "block_table_width": block_table_width,
        "target_positions": target_positions,
        "target_query_start_loc": target_query_start_loc,
        "idx_mapping": idx_mapping,
        "last_sampled": last_sampled,
        "next_prefill_tokens": next_prefill_tokens,
        "num_sampled": num_sampled_tensor,
        "num_rejected": num_rejected_tensor,
        "block_table": block_table,
        "expected": expected,
        "expected_sources": expected_sources,
        "input_batch": input_batch,
        "candidate_mode": candidate_mode,
        "shape": {
            "name": (
                f"b{num_reqs}-t{total_target_tokens}-q{num_query_per_req}"
                f"-s{num_speculative_steps}-ctx{context_tokens}"
                f"-maxr{max_num_reqs}-maxt{max_num_tokens}"
                f"-blk{block_size}-anchor{int(sample_from_anchor)}"
            ),
            "num_reqs": num_reqs,
            "total_target_tokens": total_target_tokens,
            "target_query_lens": query_lens,
            "sequence_tokens": seq_lens,
            "num_rejected": num_rejected,
            "num_sampled": num_sampled,
            "num_speculative_steps": num_speculative_steps,
            "num_query_per_req": num_query_per_req,
            "sample_from_anchor": sample_from_anchor,
            "max_num_reqs": max_num_reqs,
            "max_num_tokens": max_num_tokens,
            "max_model_len": max_model_len,
            "block_size": block_size,
            "block_table_width": block_table_width,
            "candidate_mode": candidate_mode,
            "chain": "prepare-dflash-dspark-context-query-and-sample-metadata",
        },
    }


def _launch_triton(
    outputs: _Outputs, inputs: Mapping[str, Any], triton_block_size: int
) -> None:
    max_tokens_per_req = max(inputs["query_lens"]) + inputs["num_query_per_req"]
    num_blocks = cdiv(max_tokens_per_req, triton_block_size)
    _prepare_dflash_inputs_kernel[(inputs["num_reqs"], num_blocks)](
        outputs.input_ids,
        outputs.query_positions,
        outputs.query_start_loc,
        outputs.seq_lens,
        outputs.query_slot_mapping,
        outputs.context_positions,
        outputs.context_slot_mapping,
        outputs.sample_indices,
        outputs.sample_pos,
        outputs.sample_idx_mapping,
        inputs["target_positions"],
        inputs["target_query_start_loc"],
        inputs["idx_mapping"],
        inputs["last_sampled"],
        inputs["next_prefill_tokens"],
        inputs["num_sampled"],
        inputs["num_rejected"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        _PARALLEL_DRAFT_TOKEN,
        inputs["block_size"],
        inputs["num_query_per_req"],
        inputs["num_speculative_steps"],
        inputs["max_num_reqs"],
        inputs["max_num_tokens"],
        inputs["max_model_len"],
        SAMPLE_FROM_ANCHOR=inputs["sample_from_anchor"],
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=triton_block_size,
    )


def _launch_baseline(outputs: _Outputs, inputs: Mapping[str, Any]) -> None:
    max_tokens_per_req = max(inputs["query_lens"]) + inputs["num_query_per_req"]
    triton_block_size = min(256, 1 << (max(1, max_tokens_per_req) - 1).bit_length())
    _launch_triton(outputs, inputs, triton_block_size)


def _launch_dispatch(outputs: _Outputs, inputs: Mapping[str, Any]) -> None:
    prepare_dflash_inputs(
        outputs.input_buffers,
        outputs.query_slot_mapping,
        outputs.context_positions,
        outputs.context_slot_mapping,
        outputs.sample_indices,
        outputs.sample_pos,
        outputs.sample_idx_mapping,
        inputs["input_batch"],
        inputs["num_sampled"],
        inputs["num_rejected"],
        inputs["last_sampled"],
        inputs["next_prefill_tokens"],
        inputs["block_table"],
        inputs["block_size"],
        _PARALLEL_DRAFT_TOKEN,
        inputs["num_query_per_req"],
        inputs["num_speculative_steps"],
        inputs["max_num_reqs"],
        inputs["max_num_tokens"],
        inputs["max_model_len"],
        inputs["sample_from_anchor"],
    )


def build_d01_prepare_dflash_inputs_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    expected = inputs["expected"]
    baseline = _allocate_outputs(
        inputs["max_num_reqs"],
        inputs["max_num_tokens"],
        inputs["num_speculative_steps"],
        torch.device("cuda"),
    )
    candidate = _allocate_outputs(
        inputs["max_num_reqs"],
        inputs["max_num_tokens"],
        inputs["num_speculative_steps"],
        torch.device("cuda"),
    )

    def run_baseline() -> torch.Tensor:
        _launch_baseline(baseline, inputs)
        return baseline.return_value

    def run_candidate() -> torch.Tensor:
        if inputs["candidate_mode"] == "dispatch":
            _launch_dispatch(candidate, inputs)
        elif inputs["candidate_mode"].startswith("fixed"):
            _launch_triton(
                candidate, inputs, int(inputs["candidate_mode"].removeprefix("fixed"))
            )
        else:
            _launch_baseline(candidate, inputs)
        return candidate.return_value

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        output_names = (
            "input_ids",
            "query_positions",
            "query_start_loc",
            "seq_lens",
            "query_slot_mapping",
            "context_positions",
            "context_slot_mapping",
            "sample_indices",
            "sample_pos",
            "sample_idx_mapping",
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
            "triton-2d-last-cta-padding",
            run_baseline,
            {"launches": 1, "candidate_active": False},
        ),
        candidate=Provider(
            (
                "sm120-production-dispatch"
                if inputs["candidate_mode"] == "dispatch"
                else (
                    f"triton-{inputs['candidate_mode']}-prototype"
                    if inputs["candidate_mode"].startswith("fixed")
                    else "triton-frozen-mirror"
                )
            ),
            run_candidate,
            {
                "launches": 1,
                "candidate_active": inputs["candidate_mode"] != "mirror",
            },
            correctness_comparator=compare,
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
