# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.e_factories import (
    build_e01_prefill_case,
    build_e02_e03_e07_input_spec_chain_case,
    build_e04_e05_post_chain_case,
    build_e06_pp_update_case,
    build_e08_e09_block_slot_chain_case,
    build_e10_staged_write_case,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Phase E bookkeeping harness requires CUDA",
)


Factory = Callable[[dict[str, object]], ChainCase]


def _assert_case_passes(
    factory: Factory,
    factory_args: dict[str, object],
) -> dict[str, Any]:
    case = factory(factory_args)
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


@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize("prefill_len", [256, 8192])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block128-w4"])
def test_e01_prefill_b1_b4_and_long_prefill(
    num_reqs: int,
    prefill_len: int,
    candidate_mode: str,
) -> None:
    comparison = _assert_case_passes(
        build_e01_prefill_case,
        {
            "num_reqs": num_reqs,
            "max_num_reqs": 4,
            "prefill_len": prefill_len,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["shape"]["chain"] == "E01-prefill"


@pytest.mark.parametrize("draft_length", [0, 1, 4, 7])
@pytest.mark.parametrize("candidate_target", ["E02", "E03", "E07"])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block64-w2"])
def test_e02_e03_e07_input_spec_draft_lengths(
    draft_length: int,
    candidate_target: str,
    candidate_mode: str,
) -> None:
    comparison = _assert_case_passes(
        build_e02_e03_e07_input_spec_chain_case,
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "draft_lengths": [draft_length] * 4,
            "candidate_target": candidate_target,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["exact"]["padding_cleared"]


@pytest.mark.parametrize("draft_lengths", [[0, 1, 4, 7], [7, 4, 1, 0]])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block32-w1"])
def test_e04_e05_post_chain_is_symmetric_and_repeatable(
    draft_lengths: list[int],
    candidate_mode: str,
) -> None:
    comparison = _assert_case_passes(
        build_e04_e05_post_chain_case,
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "draft_lengths": draft_lengths,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["shape"]["reset"] == "symmetric"


@pytest.mark.parametrize("num_reqs", [1, 4])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block16-w1"])
def test_e06_pp_update_b1_b4(num_reqs: int, candidate_mode: str) -> None:
    comparison = _assert_case_passes(
        build_e06_pp_update_case,
        {
            "num_reqs": num_reqs,
            "max_num_reqs": 4,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["shape"]["chain"] == "E06-pp-update"


@pytest.mark.parametrize("num_groups", [1, 3])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("candidate_target", ["E08", "E09"])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block256-w4"])
def test_e08_e09_block_table_slot_mapping_groups_and_padding(
    num_groups: int,
    padding: bool,
    candidate_target: str,
    candidate_mode: str,
) -> None:
    comparison = _assert_case_passes(
        build_e08_e09_block_slot_chain_case,
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "num_groups": num_groups,
            "padding": padding,
            "candidate_target": candidate_target,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["exact"]["padding_tail"]


@pytest.mark.parametrize("num_groups", [1, 3])
@pytest.mark.parametrize("candidate_mode", ["mirror", "block512-w8"])
@pytest.mark.parametrize("content_len", [1, 5, 256, 8192])
def test_e10_staged_write_single_and_multi_group(
    num_groups: int,
    candidate_mode: str,
    content_len: int,
) -> None:
    comparison = _assert_case_passes(
        build_e10_staged_write_case,
        {
            "num_reqs": 4,
            "max_num_reqs": 4,
            "num_groups": num_groups,
            "content_len": content_len,
            "candidate_mode": candidate_mode,
        },
    )
    assert comparison["shape"]["chain"] == "E10-staged-write"


@pytest.mark.parametrize(
    ("factory", "factory_args"),
    [
        (
            build_e01_prefill_case,
            {
                "num_reqs": 4,
                "max_num_reqs": 4,
                "prefill_len": 256,
                "candidate_mode": "block128-w4",
            },
        ),
        (
            build_e02_e03_e07_input_spec_chain_case,
            {
                "num_reqs": 4,
                "max_num_reqs": 4,
                "draft_lengths": [0, 1, 4, 7],
                "candidate_target": "E03",
                "candidate_mode": "block64-w2",
            },
        ),
        (
            build_e04_e05_post_chain_case,
            {
                "num_reqs": 4,
                "max_num_reqs": 4,
                "draft_lengths": [0, 1, 4, 7],
                "candidate_mode": "block32-w1",
            },
        ),
        (
            build_e08_e09_block_slot_chain_case,
            {
                "num_reqs": 4,
                "max_num_reqs": 4,
                "num_groups": 3,
                "padding": True,
                "candidate_target": "E09",
                "candidate_mode": "block256-w4",
            },
        ),
        (
            build_e10_staged_write_case,
            {
                "num_reqs": 4,
                "max_num_reqs": 4,
                "num_groups": 3,
                "candidate_mode": "block512-w8",
            },
        ),
    ],
)
def test_e_candidate_cuda_graph_replay(
    factory: Factory,
    factory_args: dict[str, object],
) -> None:
    case = factory(factory_args)
    runner = GraphRunner(case.candidate, repeats=3)
    runner.capture()
    for _ in range(3):
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
