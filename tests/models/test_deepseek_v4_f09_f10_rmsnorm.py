# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.f09_f10_factories import (
    DEEPSEEK_V4_BATCH_SIZES,
    DEEPSEEK_V4_DRAFT_TOKENS,
    DEEPSEEK_V4_DTYPE,
    DEEPSEEK_V4_HC_MULT,
    DEEPSEEK_V4_HIDDEN_SIZE,
    DEEPSEEK_V4_PREFILL_TOKENS,
    build_f09_triton_warps_case,
    build_f10_rmsnorm_case,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="DeepSeek-V4 F09/F10 RMSNorm factories require a CUDA/ROCm device",
)


def _assert_case_correct_and_graphable(case) -> None:
    with torch.inference_mode():
        baseline_out = case.baseline.fn()
        candidate_out = case.candidate.fn()
        torch.accelerator.synchronize()
        result = case.candidate.correctness_comparator(
            baseline_out,
            candidate_out,
            case.tolerances,
        )
        assert result["passed"], result

        runner = GraphRunner(case.candidate, repeats=2)
        runner.capture()
        runner.replay()
        torch.accelerator.synchronize()


@pytest.mark.parametrize("batch_size", DEEPSEEK_V4_BATCH_SIZES)
def test_f09_triton_warps_match_production_for_decode_batches(batch_size: int):
    case = build_f09_triton_warps_case(
        {
            "shape_kind": "decode",
            "batch_size": batch_size,
            "candidate_mode": "triton-w4",
        }
    )

    assert case.shape["num_tokens"] == batch_size
    assert case.shape["hidden_size"] == DEEPSEEK_V4_HIDDEN_SIZE
    assert case.shape["hc_mult"] == DEEPSEEK_V4_HC_MULT
    assert case.shape["draft_tokens"] == DEEPSEEK_V4_DRAFT_TOKENS
    assert case.shape["dtype"] == str(DEEPSEEK_V4_DTYPE)
    _assert_case_correct_and_graphable(case)


def test_f09_triton_warps_match_production_for_prefill_t256():
    case = build_f09_triton_warps_case(
        {"shape_kind": "prefill", "candidate_mode": "triton-w8"}
    )

    assert case.shape["num_tokens"] == DEEPSEEK_V4_PREFILL_TOKENS
    _assert_case_correct_and_graphable(case)


@pytest.mark.parametrize("candidate_mode", ["flashinfer-cute", "vllm-native"])
@pytest.mark.parametrize("batch_size", DEEPSEEK_V4_BATCH_SIZES)
def test_f10_direct_rmsnorm_candidates_match_production_for_decode_batches(
    candidate_mode: str,
    batch_size: int,
):
    case = build_f10_rmsnorm_case(
        {
            "shape_kind": "decode",
            "batch_size": batch_size,
            "candidate_mode": candidate_mode,
        }
    )

    assert case.shape["num_tokens"] == batch_size
    assert case.shape["hidden_size"] == DEEPSEEK_V4_HIDDEN_SIZE
    assert case.shape["hc_mult"] == DEEPSEEK_V4_HC_MULT
    assert case.shape["draft_tokens"] == DEEPSEEK_V4_DRAFT_TOKENS
    assert case.shape["dtype"] == str(DEEPSEEK_V4_DTYPE)
    _assert_case_correct_and_graphable(case)


@pytest.mark.parametrize("candidate_mode", ["flashinfer-cute", "triton-w8"])
def test_f10_direct_rmsnorm_candidates_match_production_for_prefill_t256(
    candidate_mode: str,
):
    case = build_f10_rmsnorm_case(
        {"shape_kind": "prefill", "candidate_mode": candidate_mode}
    )

    assert case.shape["num_tokens"] == DEEPSEEK_V4_PREFILL_TOKENS
    _assert_case_correct_and_graphable(case)


def test_f09_rejects_non_contract_candidate_modes():
    with pytest.raises(ValueError, match="legal Triton warp"):
        build_f09_triton_warps_case({"candidate_mode": "flashinfer-cute"})
