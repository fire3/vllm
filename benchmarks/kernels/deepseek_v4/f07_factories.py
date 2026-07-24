# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _flatten_sampled_kernel,
)

_SAMPLED_PAD = -1
_FLAT_INIT = 0
_CANDIDATE_CONFIGS = {
    "mirror": 1,
    "warps1": 1,
    "warps2": 2,
    "warps4": 4,
    "warps8": 8,
}


def _parse_int_list(value: Any, name: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple):
        raise ValueError(f"F07 {name} must be a list")
    return [int(item) for item in value]


def _default_accepted_lengths(num_reqs: int, max_sampled_width: int) -> list[int]:
    return [req_idx % (max_sampled_width + 1) for req_idx in range(num_reqs)]


def _prefix(values: list[int]) -> list[int]:
    output = [0]
    running = 0
    for value in values:
        running += value
        output.append(running)
    return output


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    num_reqs = int(args.get("num_reqs", 32))
    num_speculative_steps = int(args.get("num_speculative_steps", 7))
    max_sampled_width = num_speculative_steps + 1
    candidate_mode = str(args.get("candidate_mode", "mirror"))
    logprobs_enabled = bool(args.get("logprobs_enabled", True))

    if num_reqs < 1:
        raise ValueError("F07 num_reqs must be positive")
    if num_speculative_steps < 0:
        raise ValueError("F07 num_speculative_steps must be non-negative")
    if candidate_mode not in _CANDIDATE_CONFIGS:
        raise ValueError("unsupported F07 candidate mode")
    if not logprobs_enabled:
        raise ValueError("F07 flatten sampled is only active when logprobs are enabled")

    accepted_lengths = _parse_int_list(args.get("accepted_lengths"), "accepted_lengths")
    if accepted_lengths is None:
        accepted_lengths = _default_accepted_lengths(num_reqs, max_sampled_width)
    if len(accepted_lengths) != num_reqs:
        raise ValueError("F07 accepted_lengths must match num_reqs")
    if any(length < 0 or length > max_sampled_width for length in accepted_lengths):
        raise ValueError(
            "F07 accepted lengths must be in [0, num_speculative_steps + 1]"
        )

    device = "cuda"
    sampled = torch.full(
        (num_reqs, max_sampled_width),
        _SAMPLED_PAD,
        dtype=torch.int64,
        device=device,
    )
    for req_idx, length in enumerate(accepted_lengths):
        if length == 0:
            continue
        sampled[req_idx, :length] = (
            torch.arange(length, dtype=torch.int64, device=device) + 1000 + req_idx * 37
        )

    cu_num_logits = torch.tensor(
        _prefix([max_sampled_width] * num_reqs),
        dtype=torch.int32,
        device=device,
    )
    num_sampled = torch.tensor(accepted_lengths, dtype=torch.int32, device=device)
    total_logits = num_reqs * max_sampled_width
    expected_flat = torch.full(
        (total_logits,), _FLAT_INIT, dtype=torch.int64, device=device
    )
    for req_idx, length in enumerate(accepted_lengths):
        start = req_idx * max_sampled_width
        expected_flat[start : start + length] = sampled[req_idx, :length]

    shape = {
        "name": (
            f"f07-b{num_reqs}-spec{num_speculative_steps}"
            f"-accepted{'_'.join(map(str, accepted_lengths[:9]))}"
        ),
        "chain": "flatten-sampled-logprobs",
        "num_reqs": num_reqs,
        "num_speculative_steps": num_speculative_steps,
        "max_sampled_width": max_sampled_width,
        "accepted_lengths": accepted_lengths,
        "ragged_accepted_lengths_0_to_8": set(range(9)).issubset(accepted_lengths),
        "candidate_mode": candidate_mode,
        "logprobs_enabled": logprobs_enabled,
        "num_sampled_dtype": str(num_sampled.dtype),
        "sampled_padding_value": _SAMPLED_PAD,
        "flat_initial_value": _FLAT_INIT,
        "block_size": None,
        "block_size_reason": "production kernel has no BLOCK_SIZE constexpr",
    }
    return {
        "sampled": sampled,
        "sampled_snapshot": sampled.clone(),
        "num_sampled": num_sampled,
        "num_sampled_snapshot": num_sampled.clone(),
        "cu_num_logits": cu_num_logits,
        "expected_flat": expected_flat,
        "num_reqs": num_reqs,
        "candidate_mode": candidate_mode,
        "shape": shape,
    }


def _allocate_state(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {"flat_sampled": torch.empty_like(inputs["expected_flat"])}


def _launch(
    state: Mapping[str, torch.Tensor],
    inputs: Mapping[str, Any],
    *,
    num_warps: int,
) -> torch.Tensor:
    state["flat_sampled"].fill_(_FLAT_INIT)
    _flatten_sampled_kernel[(inputs["num_reqs"],)](
        state["flat_sampled"],
        inputs["sampled"],
        inputs["sampled"].stride(0),
        inputs["num_sampled"],
        inputs["cu_num_logits"],
        num_warps=num_warps,
    )
    return state["flat_sampled"]


def build_f07_flatten_sampled_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    baseline_state = _allocate_state(inputs)
    candidate_state = _allocate_state(inputs)
    candidate_num_warps = _CANDIDATE_CONFIGS[inputs["candidate_mode"]]

    def run_baseline() -> torch.Tensor:
        return _launch(baseline_state, inputs, num_warps=1)

    def run_candidate() -> torch.Tensor:
        return _launch(candidate_state, inputs, num_warps=candidate_num_warps)

    def compare(
        baseline_output: torch.Tensor,
        candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "flat_matches_expected": torch.equal(
                candidate_output, inputs["expected_flat"]
            ),
            "flat_matches_baseline": torch.equal(candidate_output, baseline_output),
            "baseline_matches_expected": torch.equal(
                baseline_output, inputs["expected_flat"]
            ),
            "sampled_padding_is_negative_one": bool(
                (
                    inputs["sampled"][inputs["sampled"] == _SAMPLED_PAD] == _SAMPLED_PAD
                ).all()
            ),
            "sampled_immutable": torch.equal(
                inputs["sampled"], inputs["sampled_snapshot"]
            ),
            "num_sampled_immutable": torch.equal(
                inputs["num_sampled"], inputs["num_sampled_snapshot"]
            ),
            "num_sampled_int32": inputs["num_sampled"].dtype == torch.int32,
        }
        return {
            "passed": all(checks.values()),
            "exact": checks,
            "shape": inputs["shape"],
        }

    return ChainCase(
        baseline=Provider(
            "production-flatten-sampled-num-warps1",
            run_baseline,
            {
                "operator_launches": 1,
                "candidate_active": False,
                "num_warps": 1,
                "direct_production_kernel": True,
                "logprobs_enabled": True,
            },
        ),
        candidate=Provider(
            f"benchmark-only-flatten-sampled-{inputs['candidate_mode']}",
            run_candidate,
            {
                "operator_launches": 1,
                "candidate_active": inputs["candidate_mode"] != "mirror",
                "num_warps": candidate_num_warps,
                "legal_num_warps": candidate_num_warps in (1, 2, 4, 8),
                "block_size": None,
                "block_size_legal": True,
                "direct_production_kernel": True,
                "logprobs_enabled": True,
            },
            correctness_comparator=compare,
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


__all__ = ["_CANDIDATE_CONFIGS", "build_f07_flatten_sampled_case"]
