# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.f08_factories import (
    build_f08_copy_and_expand_dflash_inputs_case,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DFlash copy-and-expand requires CUDA"
)


def _assert_exact(case) -> None:
    baseline_output = case.baseline.fn()
    candidate_output = case.candidate.fn()
    torch.accelerator.synchronize()
    assert case.candidate.correctness_comparator is not None
    comparison = case.candidate.correctness_comparator(
        baseline_output, candidate_output, case.tolerances
    )
    assert comparison["passed"], comparison


@pytest.mark.parametrize("num_reqs", [1, 4, 32])
@pytest.mark.parametrize("num_speculative_tokens", [1, 4, 7])
@pytest.mark.parametrize("has_num_rejected", [False, True])
def test_f08_copy_and_expand_dflash_inputs_exact(
    num_reqs: int,
    num_speculative_tokens: int,
    has_num_rejected: bool,
) -> None:
    context_lens = [index % 7 + 1 for index in range(num_reqs)]
    num_rejected = [
        min(index % 3, context_len - 1) if has_num_rejected else 0
        for index, context_len in enumerate(context_lens)
    ]
    case = build_f08_copy_and_expand_dflash_inputs_case(
        {
            "method": "dflash",
            "num_reqs": num_reqs,
            "total_context_tokens": sum(context_lens),
            "context_lens": context_lens,
            "num_speculative_tokens": num_speculative_tokens,
            "has_num_rejected": has_num_rejected,
            "num_rejected": num_rejected,
            "context_tokens": 32768,
            "context_jitter": 31,
            "candidate_block_size": 128,
            "candidate_num_warps": 4,
        }
    )
    _assert_exact(case)


def test_f08_copy_and_expand_dflash_inputs_ragged_large_query_lengths() -> None:
    case = build_f08_copy_and_expand_dflash_inputs_case(
        {
            "method": "dflash",
            "num_reqs": 4,
            "total_context_tokens": 176,
            "context_lens": [1, 7, 40, 128],
            "num_speculative_tokens": 7,
            "has_num_rejected": True,
            "num_rejected": [0, 2, 7, 31],
            "context_tokens": 131072,
            "context_jitter": 257,
            "block_size": 256,
            "candidate_block_size": 256,
            "candidate_num_warps": 8,
        }
    )
    _assert_exact(case)


def test_f08_copy_and_expand_dflash_inputs_cuda_graph_replay() -> None:
    case = build_f08_copy_and_expand_dflash_inputs_case(
        {
            "method": "dflash",
            "num_reqs": 4,
            "total_context_tokens": 32,
            "context_lens": [1, 7, 8, 16],
            "num_speculative_tokens": 7,
            "has_num_rejected": True,
            "num_rejected": [0, 2, 7, 3],
            "candidate_block_size": 64,
            "candidate_num_warps": 4,
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


def test_f08_copy_and_expand_requires_dflash_method() -> None:
    with pytest.raises(ValueError, match="method=dflash"):
        build_f08_copy_and_expand_dflash_inputs_case(
            {
                "method": "eagle",
                "num_reqs": 1,
                "total_context_tokens": 1,
                "num_speculative_tokens": 1,
            }
        )


@pytest.mark.parametrize(
    "factory_args",
    [
        {"candidate_block_size": 3},
        {"candidate_num_warps": 3},
    ],
)
def test_f08_copy_and_expand_screens_candidate_launch_config(factory_args) -> None:
    with pytest.raises(ValueError):
        build_f08_copy_and_expand_dflash_inputs_case(
            {
                "method": "dflash",
                "num_reqs": 1,
                "total_context_tokens": 1,
                "num_speculative_tokens": 1,
                **factory_args,
            }
        )
