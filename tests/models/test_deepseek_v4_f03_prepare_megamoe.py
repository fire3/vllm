# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as HfDeepseekV4Config,
)

from benchmarks.kernels.deepseek_v4.common import GraphRunner
from benchmarks.kernels.deepseek_v4.f03_factories import (
    DEEPSEEK_V4_HIDDEN_SIZE,
    DEEPSEEK_V4_NUM_EXPERTS,
    DEEPSEEK_V4_TOPK,
    F03_DEFAULT_CASE_ARGS,
    _resolve_candidate_geometry,
    build_f03_prepare_megamoe_case,
    deepseek_v4_f03_config,
)
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
    _prepare_megamoe_num_warps,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="DeepSeek V4 F03 prepare_megamoe benchmarks require CUDA",
)


def test_f03_deepseek_v4_config_is_explicit() -> None:
    assert deepseek_v4_f03_config() == {
        "hidden_size": 4096,
        "top_k": HfDeepseekV4Config.num_experts_per_tok,
        "num_experts": HfDeepseekV4Config.n_routed_experts,
    }


def test_f03_production_warp_dispatch_matches_architecture() -> None:
    expected = 2 if current_platform.is_device_capability(120) else 4
    assert _prepare_megamoe_num_warps() == expected


def test_f03_default_cases_cover_decode_prefill_padding_matrix() -> None:
    assert {case["hidden_size"] for case in F03_DEFAULT_CASE_ARGS} == {
        DEEPSEEK_V4_HIDDEN_SIZE
    }
    assert {case["top_k"] for case in F03_DEFAULT_CASE_ARGS} == {DEEPSEEK_V4_TOPK}
    assert {case["num_experts"] for case in F03_DEFAULT_CASE_ARGS} == {
        DEEPSEEK_V4_NUM_EXPERTS
    }
    assert {
        (case["phase"], case["num_tokens"], case["is_padding"])
        for case in F03_DEFAULT_CASE_ARGS
    } == {
        ("decode", 1, False),
        ("decode", 1, True),
        ("decode", 4, False),
        ("decode", 4, True),
        ("decode", 32, False),
        ("decode", 32, True),
        ("prefill", 256, False),
        ("prefill", 256, True),
        ("prefill", 8192, False),
        ("prefill", 8192, True),
    }


def test_f03_candidate_rejects_block_k_search() -> None:
    with pytest.raises(ValueError, match="only sweep num_warps"):
        _resolve_candidate_geometry({"candidate_block_k": 256})


@pytest.mark.parametrize("is_padding", [False, True])
@pytest.mark.parametrize("candidate_num_warps", [1, 2, 4, 8])
def test_f03_factory_matches_production_wrapper_exactly(
    is_padding: bool,
    candidate_num_warps: int,
) -> None:
    case = build_f03_prepare_megamoe_case(
        {
            "num_tokens": 4,
            "hidden_size": 128,
            "top_k": DEEPSEEK_V4_TOPK,
            "num_experts": DEEPSEEK_V4_NUM_EXPERTS,
            "is_padding": is_padding,
            "candidate_num_warps": candidate_num_warps,
            "seed": 3,
        }
    )

    reference = case.baseline.fn().detach().clone()
    candidate = case.candidate.fn().detach().clone()
    torch.accelerator.synchronize()

    assert case.candidate.correctness_comparator is not None
    correctness = case.candidate.correctness_comparator(
        reference,
        candidate,
        case.tolerances,
    )
    assert correctness == {
        "passed": True,
        "comparison": "bitwise_exact_payload_scales_ids_weights",
        "mismatch_count": 0,
        "first_mismatch_byte": -1,
        "num_bytes": reference.numel(),
    }


def test_f03_factory_uses_cuda_graph_capture() -> None:
    case = build_f03_prepare_megamoe_case(
        {
            "num_tokens": 1,
            "hidden_size": 128,
            "is_padding": True,
            "candidate_num_warps": _prepare_megamoe_num_warps(),
            "candidate_provider": "production",
            "seed": 5,
        }
    )
    runner = GraphRunner(case.candidate, repeats=2)

    runner.capture()
    latency_us = runner.measure_us(graph_replays=1)

    assert runner.graph is not None
    assert latency_us > 0.0


def test_f03_factory_runs_actual_hidden_size() -> None:
    case = build_f03_prepare_megamoe_case(
        {
            "num_tokens": 1,
            "hidden_size": DEEPSEEK_V4_HIDDEN_SIZE,
            "top_k": DEEPSEEK_V4_TOPK,
            "num_experts": DEEPSEEK_V4_NUM_EXPERTS,
            "is_padding": False,
            "candidate_num_warps": 4,
            "seed": 7,
        }
    )

    reference = case.baseline.fn().detach().clone()
    candidate = case.candidate.fn().detach().clone()
    torch.accelerator.synchronize()

    assert case.candidate.correctness_comparator is not None
    correctness = case.candidate.correctness_comparator(
        reference,
        candidate,
        case.tolerances,
    )
    assert correctness["passed"]
    assert case.shape["H"] == DEEPSEEK_V4_HIDDEN_SIZE
