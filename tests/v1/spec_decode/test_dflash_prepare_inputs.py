# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.d01_factories import (
    build_d01_prepare_dflash_inputs_case,
)
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlash input preparation requires CUDA"
)


@pytest.mark.parametrize(
    "factory_args",
    [
        {
            "num_reqs": 1,
            "total_target_tokens": 8,
            "num_speculative_steps": 7,
            "sample_from_anchor": True,
            "num_rejected": [3],
            "max_num_tokens": 2048,
        },
        {
            "num_reqs": 4,
            "total_target_tokens": 32,
            "target_query_lens": [1, 7, 8, 16],
            "num_rejected": [0, 2, 7, 3],
            "num_sampled": [1, 0, 2, 0],
            "num_speculative_steps": 7,
            "sample_from_anchor": True,
            "context_tokens": 32768,
            "context_jitter": 127,
            "max_num_tokens": 2048,
        },
        {
            "num_reqs": 4,
            "total_target_tokens": 64,
            "target_query_lens": [1, 15, 17, 31],
            "num_rejected": [0, 14, 3, 30],
            "num_speculative_steps": 16,
            "sample_from_anchor": False,
            "context_tokens": 131072,
            "context_jitter": 64,
            "max_num_tokens": 2048,
        },
        {
            "num_reqs": 4,
            "total_target_tokens": 2048,
            "target_query_lens": [1, 127, 512, 1408],
            "num_rejected": [0, 1, 7, 3],
            "num_speculative_steps": 7,
            "sample_from_anchor": True,
            "context_tokens": 131072,
            "context_jitter": 511,
            "max_num_tokens": 2048,
        },
        {
            "num_reqs": 32,
            "total_target_tokens": 256,
            "num_speculative_steps": 7,
            "sample_from_anchor": True,
            "max_num_reqs": 128,
            "max_num_tokens": 8192,
            "block_size": 256,
        },
    ],
)
def test_prepare_dflash_inputs_matches_independent_reference(factory_args) -> None:
    factory_args["candidate_mode"] = "dispatch"
    case = build_d01_prepare_dflash_inputs_case(factory_args)
    baseline_output = case.baseline.fn()
    candidate_output = case.candidate.fn()
    torch.accelerator.synchronize()
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output, candidate_output, case.tolerances
    )
    assert comparison["passed"], comparison


def test_prepare_dflash_inputs_cuda_graph_replay() -> None:
    case = build_d01_prepare_dflash_inputs_case(
        {
            "num_reqs": 4,
            "total_target_tokens": 32,
            "target_query_lens": [1, 7, 8, 16],
            "num_rejected": [0, 2, 7, 3],
            "num_sampled": [1, 0, 2, 0],
            "num_speculative_steps": 7,
            "sample_from_anchor": True,
            "max_num_reqs": 128,
            "max_num_tokens": 2048,
            "candidate_mode": "dispatch",
        }
    )
    runner = GraphRunner(case.candidate, repeats=3)
    runner.capture()
    runner.replay()
    torch.accelerator.synchronize()
    baseline_output = case.baseline.fn()
    torch.accelerator.synchronize()
    assert runner.output is not None
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output, runner.output, case.tolerances
    )
    assert comparison["passed"], comparison


@pytest.mark.parametrize(
    ("max_tokens_per_req", "baseline", "sm120"),
    [(1, 1, 128), (15, 16, 128), (48, 64, 128), (129, 256, 256), (4096, 256, 256)],
)
def test_prepare_dflash_block_size_dispatch(
    monkeypatch,
    max_tokens_per_req: int,
    baseline: int,
    sm120: int,
) -> None:
    monkeypatch.setattr(
        dflash_speculator.current_platform,
        "is_device_capability_family",
        lambda family: False,
    )
    assert (
        dflash_speculator._select_prepare_dflash_block_size(max_tokens_per_req)
        == baseline
    )
    monkeypatch.setattr(
        dflash_speculator.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    assert (
        dflash_speculator._select_prepare_dflash_block_size(max_tokens_per_req) == sm120
    )
