# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.e04_e05_fused_factories import (
    build_e04_e05_fused_case,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="E04/E05 fused benchmark harness requires CUDA",
)


def _assert_case_passes(factory_args: dict[str, object]) -> dict[str, Any]:
    case = build_e04_e05_fused_case(factory_args)
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
    return comparison


@pytest.mark.parametrize("num_reqs", [1, 4, 32])
def test_e04_e05_fused_batch_sizes(num_reqs: int) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": num_reqs,
            "max_num_reqs": max(32, num_reqs),
            "draft_lengths": [0, 1, 4, 7] * (num_reqs // 4)
            + [0, 1, 4, 7][: num_reqs % 4],
            "candidate_mode": "fused-w1",
        }
    )
    assert comparison["shape"]["num_reqs"] == num_reqs
    assert comparison["shape"]["ragged_num_sampled"]


def test_e04_e05_fused_negative_idx_and_chunked_prefill() -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "draft_lengths": [0, 1, 4, 7],
            "idx_mapping": [3, -1, 1, 0],
            "chunked_prefill": True,
            "candidate_mode": "fused-w2",
        }
    )
    assert comparison["shape"]["idx_mapping"][1] == -1
    assert comparison["exact"]["num_sampled"]
    assert comparison["exact"]["num_rejected"]


@pytest.mark.parametrize("with_query_start_loc", [False, True])
@pytest.mark.parametrize("with_bin_counts", [False, True])
def test_e04_e05_fused_optional_outputs(
    with_query_start_loc: bool,
    with_bin_counts: bool,
) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "draft_lengths": [7, 4, 1, 0],
            "with_query_start_loc": with_query_start_loc,
            "with_bin_counts": with_bin_counts,
            "candidate_mode": "fused-w4",
        }
    )
    assert comparison["shape"]["with_query_start_loc"] is with_query_start_loc
    assert comparison["shape"]["with_bin_counts"] is with_bin_counts


def test_e04_e05_fused_cuda_graph_replay() -> None:
    case = build_e04_e05_fused_case(
        {
            "num_reqs": 32,
            "max_num_reqs": 32,
            "draft_lengths": [0, 1, 4, 7] * 8,
            "include_negative_idx": True,
            "chunked_prefill": True,
            "with_query_start_loc": True,
            "with_bin_counts": True,
            "candidate_mode": "fused-w1",
        }
    )
    runner = GraphRunner(case.candidate, repeats=5)
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
    assert case.candidate.metadata["operator_launches"] == 1
