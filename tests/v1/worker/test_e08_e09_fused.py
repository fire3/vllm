# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.e08_e09_fused_factories import (
    build_e08_e09_fused_case,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="E08/E09 fused benchmark harness requires CUDA",
)


def _assert_case_passes(factory_args: dict[str, object]) -> dict[str, Any]:
    case = build_e08_e09_fused_case(factory_args)
    baseline_output = case.baseline.fn().detach().clone()
    candidate_output = case.candidate.fn().detach().clone()
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
@pytest.mark.parametrize("num_groups", [1, 3])
@pytest.mark.parametrize("block_size", [16, 64, 256])
def test_e08_e09_fused_exact_batches_groups_and_block_sizes(
    num_reqs: int,
    num_groups: int,
    block_size: int,
) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": num_reqs,
            "num_reqs_padded": num_reqs,
            "max_num_reqs": num_reqs,
            "num_groups": num_groups,
            "block_size": block_size,
            "query_lens": [1, 3, 5, 7] * (num_reqs // 4) + [1, 3, 5, 7][: num_reqs % 4],
            "candidate_mode": "fused-b1024-w4",
        }
    )
    assert comparison["shape"]["num_reqs"] == num_reqs
    assert comparison["shape"]["num_groups"] == num_groups
    assert comparison["shape"]["block_size"] == block_size
    assert comparison["exact"]["slot_mappings"]


def test_e08_e09_fused_padding_mask_and_padded_request_rows() -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": 4,
            "num_reqs_padded": 8,
            "max_num_reqs": 8,
            "num_groups": 3,
            "block_size": 64,
            "padding": True,
            "query_lens": [2, 4, 6, 8],
            "idx_mapping": [6, 2, 5, 1],
            "candidate_mode": "fused-b512-w4",
        }
    )
    assert comparison["exact"]["padding_mask"]
    assert comparison["exact"]["padding_tail"]
    assert comparison["exact"]["padded_block_table_rows_zero"]


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_e08_e09_fused_context_parallel_interleave(cp_rank: int) -> None:
    comparison = _assert_case_passes(
        {
            "num_reqs": 4,
            "num_reqs_padded": 6,
            "max_num_reqs": 6,
            "num_groups": 3,
            "block_size": 16,
            "cp_size": 2,
            "cp_rank": cp_rank,
            "cp_interleave": 2,
            "padding": True,
            "query_lens": [5, 6, 7, 8],
            "idx_mapping": [5, 1, 4, 0],
            "candidate_mode": "fused-b256-w4",
        }
    )
    assert comparison["shape"]["cp_size"] == 2
    assert comparison["shape"]["cp_interleave"] == 2
    assert comparison["exact"]["slot_mappings"]


def test_e08_e09_fused_cuda_graph_replay() -> None:
    case = build_e08_e09_fused_case(
        {
            "num_reqs": 32,
            "num_reqs_padded": 40,
            "max_num_reqs": 40,
            "num_groups": 3,
            "block_size": 256,
            "padding": True,
            "query_lens": [1, 3, 5, 7] * 8,
            "candidate_mode": "fused-b1024-w4",
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
    assert case.baseline.metadata["operator_launches"] == 2
    assert case.candidate.metadata["operator_launches"] == 1
