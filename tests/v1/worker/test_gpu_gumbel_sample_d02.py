# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.d02_factories import (
    _allocate_state,
    _launch_frozen_baseline,
    _make_inputs,
    build_d02_gumbel_sample_case,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="D02 Gumbel sampling requires CUDA"
)


@pytest.mark.parametrize(
    "factory_args",
    [
        {
            "num_tokens": 1,
            "num_reqs": 1,
            "vocab_size": 129280,
            "temperatures": [0.0],
            "candidate_mode": "torch-argmax",
        },
        {
            "num_tokens": 4,
            "num_reqs": 4,
            "vocab_size": 129280,
            "temperatures": [1.0],
            "output_mode": "scalar",
            "candidate_mode": "dispatch",
        },
        {
            "num_tokens": 4,
            "num_reqs": 4,
            "vocab_size": 4097,
            "temperatures": [0.0, 0.7, 1.0, 1.3],
            "output_mode": "per-token",
            "logits_stride_extra": 3,
            "candidate_mode": "dispatch",
        },
        {
            "num_tokens": 8,
            "num_reqs": 4,
            "valid_tokens": 4,
            "vocab_size": 129280,
            "temperatures": [1.0],
            "output_mode": "per-token",
            "candidate_mode": "dispatch",
        },
        {
            "num_tokens": 2,
            "num_reqs": 2,
            "vocab_size": 999,
            "temperatures": [0.5, 1.0],
            "use_fp64": True,
            "candidate_mode": "dispatch",
        },
    ],
)
def test_d02_candidate_matches_frozen_rng_contract(factory_args) -> None:
    case = build_d02_gumbel_sample_case(factory_args)
    baseline_output = case.baseline.fn()
    candidate_output = case.candidate.fn()
    torch.accelerator.synchronize()
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output, candidate_output, case.tolerances
    )
    assert comparison["passed"], comparison


@pytest.mark.parametrize("candidate_mode", ["dispatch", "torch-argmax"])
def test_d02_candidate_cuda_graph_replay(candidate_mode: str) -> None:
    all_greedy = candidate_mode == "torch-argmax"
    case = build_d02_gumbel_sample_case(
        {
            "num_tokens": 4,
            "num_reqs": 4,
            "vocab_size": 129280,
            "temperatures": [0.0 if all_greedy else 1.0],
            "output_mode": "none" if all_greedy else "scalar",
            "candidate_mode": candidate_mode,
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
    "pattern",
    ["all-tie", "cross-block-tie", "nan-first-block", "all-nan-blocks"],
)
def test_d02_dispatch_preserves_argmax_edge_semantics(pattern: str) -> None:
    inputs = _make_inputs(
        {
            "num_tokens": 1,
            "num_reqs": 1,
            "vocab_size": 2048,
            "temperatures": [0.0],
            "candidate_mode": "dispatch",
        }
    )
    logits = inputs["logits"]
    logits.fill_(-1.0)
    if pattern == "all-tie":
        logits.zero_()
    elif pattern == "cross-block-tie":
        logits[0, 5] = 3.0
        logits[0, 1029] = 3.0
    elif pattern == "nan-first-block":
        logits[0, :1024] = float("nan")
        logits[0, 1029] = 3.0
    else:
        logits.fill_(float("nan"))

    baseline_state = _allocate_state(inputs, 1024)
    baseline = _launch_frozen_baseline(baseline_state, inputs)
    candidate = gumbel_sample(
        logits,
        inputs["expanded_idx_mapping"],
        inputs["temperature"],
        inputs["seeds"],
        inputs["positions"],
        apply_temperature=True,
    )
    torch.accelerator.synchronize()
    assert torch.equal(candidate, baseline)
