# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.f07_factories import (
    _CANDIDATE_CONFIGS,
    build_f07_flatten_sampled_case,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="F07 flatten sampled benchmark harness requires CUDA",
)


def _assert_case_passes(factory_args: dict[str, object]) -> dict[str, Any]:
    case = build_f07_flatten_sampled_case(factory_args)
    baseline_output = case.baseline.fn()
    candidate_output = case.candidate.fn()
    torch.accelerator.synchronize()
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output,
        candidate_output,
        case.tolerances,
    )
    assert comparison["passed"], comparison
    assert comparison["shape"]["chain"] == "flatten-sampled-logprobs"
    assert comparison["shape"]["logprobs_enabled"] is True
    assert comparison["exact"]["num_sampled_int32"]
    return comparison


@pytest.mark.parametrize(
    ("num_reqs", "accepted_lengths"),
    [
        (1, [0]),
        (4, [0, 1, 4, 8]),
        (32, [idx % 9 for idx in range(32)]),
    ],
)
def test_f07_batch_sizes_and_ragged_lengths(
    num_reqs: int,
    accepted_lengths: list[int],
) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": num_reqs,
            "num_speculative_steps": 7,
            "accepted_lengths": accepted_lengths,
            "candidate_mode": "mirror",
            "logprobs_enabled": True,
        }
    )
    assert comparison["shape"]["num_reqs"] == num_reqs
    assert comparison["exact"]["sampled_padding_is_negative_one"]
    if num_reqs == 32:
        assert comparison["shape"]["ragged_accepted_lengths_0_to_8"]


@pytest.mark.parametrize("candidate_mode", tuple(_CANDIDATE_CONFIGS))
def test_f07_candidate_num_warps_modes(candidate_mode: str) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": 9,
            "num_speculative_steps": 7,
            "accepted_lengths": list(range(9)),
            "candidate_mode": candidate_mode,
            "logprobs_enabled": True,
        }
    )
    assert comparison["shape"]["candidate_mode"] == candidate_mode


def test_f07_rejects_inactive_logprobs_condition() -> None:
    with pytest.raises(ValueError, match="logprobs"):
        build_f07_flatten_sampled_case(
            {
                "num_reqs": 4,
                "accepted_lengths": [0, 1, 4, 8],
                "logprobs_enabled": False,
            }
        )


def test_f07_rejects_invalid_candidate_mode_before_cuda_work() -> None:
    with pytest.raises(ValueError, match="candidate mode"):
        build_f07_flatten_sampled_case({"candidate_mode": "invalid"})


def test_f07_cuda_graph_replay_is_repeatable() -> None:
    case = build_f07_flatten_sampled_case(
        {
            "num_reqs": 32,
            "num_speculative_steps": 7,
            "accepted_lengths": [idx % 9 for idx in range(32)],
            "candidate_mode": "mirror",
            "logprobs_enabled": True,
        }
    )
    runner = GraphRunner(case.candidate, repeats=7)
    runner.capture()
    for _ in range(5):
        runner.replay()
    torch.accelerator.synchronize()
    baseline_output = case.baseline.fn()
    torch.accelerator.synchronize()
    assert runner.output is not None
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output,
        runner.output,
        case.tolerances,
    )
    assert comparison["passed"], comparison
    assert case.candidate.metadata["legal_num_warps"]
