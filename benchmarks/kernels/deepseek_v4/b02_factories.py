# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    _fused_inv_rope_fp8_quant_per_head,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

HEAD_DIM = 512
HEADS_PER_GROUP = 8
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_GROUP_SIZE = 128
CHUNKS_PER_HEAD = HEAD_DIM // QUANT_GROUP_SIZE
SCALE_BLOCKS_PER_GROUP = HEADS_PER_GROUP * CHUNKS_PER_HEAD
FP8_MAX = 448.0
MIN_CUTE_TOKENS = 256


def _make_cos_sin_cache(max_position: int) -> torch.Tensor:
    half_rope = ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(
                half_rope,
                device="cuda",
                dtype=torch.float32,
            )
            / half_rope
        )
    )
    positions = torch.arange(max_position, device="cuda", dtype=torch.float32)
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


def _launch_triton_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    fp8_output: torch.Tensor,
    scale_output: torch.Tensor,
) -> None:
    num_tokens, num_heads, _ = o.shape
    scale_view = scale_output.permute(0, 2, 1)
    use_gdc = current_platform.is_arch_support_pdl()
    _fused_inv_rope_fp8_quant_per_head[(num_tokens, num_heads)](
        o,
        positions,
        cos_sin_cache,
        fp8_output,
        scale_view,
        num_tokens,
        heads_per_group=HEADS_PER_GROUP,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_group=fp8_output.stride(0),
        fp8_stride_token=fp8_output.stride(1),
        scale_stride_group=scale_view.stride(0),
        scale_stride_k=scale_view.stride(2),
        fp8_max=FP8_MAX,
        eps=1e-10,
        QUANT_GROUP_SIZE=QUANT_GROUP_SIZE,
        CHUNKS_PER_HEAD=CHUNKS_PER_HEAD,
        ROPE_START=NOPE_DIM % QUANT_GROUP_SIZE,
        HALF_ROPE=ROPE_DIM // 2,
        TMA_ALIGNED_SCALES=False,
        USE_GDC=use_gdc,
        launch_pdl=use_gdc,
        num_stages=1,
        num_warps=1,
    )


def _compare_quantized_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    baseline_scale: torch.Tensor,
    candidate_scale: torch.Tensor,
    tolerances: CorrectnessTolerances,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }

    scale_ratio = candidate_scale / baseline_scale.clamp_min(1e-30)
    ratio_min = float(scale_ratio.min().item())
    ratio_max = float(scale_ratio.max().item())
    scale_log2_residual = float(
        (torch.log2(candidate_scale) - torch.log2(candidate_scale).round())
        .abs()
        .max()
        .item()
    )

    total_abs = 0.0
    total_reference_abs = 0.0
    dot = 0.0
    reference_norm = 0.0
    candidate_norm = 0.0
    max_abs = 0.0
    max_rel = 0.0
    finite = True
    count = 0
    token_chunk = 128
    for token_start in range(0, reference.shape[1], token_chunk):
        token_end = min(token_start + token_chunk, reference.shape[1])
        reference_chunk = reference[:, token_start:token_end].float()
        candidate_chunk = candidate[:, token_start:token_end].float()
        baseline_scale_chunk = (
            baseline_scale[:, :, token_start:token_end]
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, QUANT_GROUP_SIZE)
            .reshape_as(reference_chunk)
        )
        candidate_scale_chunk = (
            candidate_scale[:, :, token_start:token_end]
            .permute(0, 2, 1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, QUANT_GROUP_SIZE)
            .reshape_as(candidate_chunk)
        )
        reference_dequant = reference_chunk * baseline_scale_chunk
        candidate_dequant = candidate_chunk * candidate_scale_chunk
        difference = (candidate_dequant - reference_dequant).abs()
        denominator = reference_dequant.abs().clamp_min(torch.finfo(torch.float32).eps)
        finite &= bool(
            torch.isfinite(reference_dequant).all()
            and torch.isfinite(candidate_dequant).all()
        )
        max_abs = max(max_abs, float(difference.max().item()))
        max_rel = max(max_rel, float((difference / denominator).max().item()))
        total_abs += float(difference.double().sum().item())
        total_reference_abs += float(reference_dequant.abs().double().sum().item())
        dot += float(
            (reference_dequant.double() * candidate_dequant.double()).sum().item()
        )
        reference_norm += float(reference_dequant.double().square().sum().item())
        candidate_norm += float(candidate_dequant.double().square().sum().item())
        count += reference_dequant.numel()

    mean_abs = total_abs / count
    mean_relative = total_abs / max(total_reference_abs, 1e-12)
    cosine = dot / max(math.sqrt(reference_norm * candidate_norm), 1e-12)
    passed = (
        finite
        and ratio_min >= 0.5
        and ratio_max <= 2.0
        and scale_log2_residual < 1e-5
        and (
            tolerances.max_mean_relative is None
            or mean_relative <= tolerances.max_mean_relative
        )
        and (tolerances.min_cosine is None or cosine >= tolerances.min_cosine)
    )
    return {
        "passed": passed,
        "finite": finite,
        "scale_ratio_min": ratio_min,
        "scale_ratio_max": ratio_max,
        "scale_log2_residual": scale_log2_residual,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": mean_abs,
        "mean_relative": mean_relative,
        "cosine": cosine,
        "max_mean_relative": tolerances.max_mean_relative,
        "min_cosine": tolerances.min_cosine,
    }


def build_b02_inv_rope_fp8_quant_case(args: Mapping[str, Any]) -> ChainCase:
    num_tokens = int(args.get("num_tokens", MIN_CUTE_TOKENS))
    num_groups = int(args.get("num_groups", 8))
    seed = int(args.get("seed", 0))
    if num_tokens <= 0:
        raise ValueError("B02 num_tokens must be positive")
    if num_groups not in (2, 8):
        raise ValueError("B02 num_groups must be 2 or 8")

    set_random_seed(seed)
    num_heads = num_groups * HEADS_PER_GROUP
    hidden_size = HEADS_PER_GROUP * HEAD_DIM
    max_position = max(num_tokens, 4096)
    o = torch.randn(
        (num_tokens, num_heads, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(max_position)
    baseline_fp8 = torch.empty(
        (num_groups, num_tokens, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    candidate_fp8 = torch.empty_like(baseline_fp8)
    baseline_scale = torch.empty(
        (num_groups, SCALE_BLOCKS_PER_GROUP, num_tokens),
        device="cuda",
        dtype=torch.float32,
    )
    candidate_scale = torch.empty_like(baseline_scale)

    def run_baseline() -> torch.Tensor:
        _launch_triton_reference(
            o,
            positions,
            cos_sin_cache,
            baseline_fp8,
            baseline_scale,
        )
        return baseline_fp8

    def run_candidate() -> torch.Tensor:
        if num_tokens < MIN_CUTE_TOKENS:
            _launch_triton_reference(
                o,
                positions,
                cos_sin_cache,
                candidate_fp8,
                candidate_scale,
            )
        else:
            from flashinfer.cute_dsl import inverse_rope_fp8_quant_cute

            inverse_rope_fp8_quant_cute(
                o,
                positions,
                cos_sin_cache,
                candidate_fp8,
                candidate_scale,
                enable_pdl=current_platform.is_arch_support_pdl(),
            )
        return candidate_fp8

    def compare_candidate(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        return _compare_quantized_outputs(
            reference,
            candidate,
            baseline_scale,
            candidate_scale,
            tolerances,
        )

    return ChainCase(
        baseline=Provider(
            "triton-inverse-rope-fp8-quant",
            run_baseline,
            {
                "symbol": "_fused_inv_rope_fp8_quant_per_head",
                "launches": 1,
                "preallocated_output": True,
                "compact_scale_layout": "GST",
            },
        ),
        candidate=Provider(
            "sm120-hybrid-inverse-rope-fp8-quant-cute",
            run_candidate,
            {
                "symbol": "flashinfer.cute_dsl.inverse_rope_fp8_quant_cute",
                "launches": 1,
                "preallocated_output": True,
                "dispatch_min_tokens": MIN_CUTE_TOKENS,
                "compact_scale_layout": "GST",
            },
            correctness_comparator=compare_candidate,
        ),
        shape={
            "name": f"t{num_tokens}-g{num_groups}-hpg{HEADS_PER_GROUP}",
            "T": num_tokens,
            "G": num_groups,
            "heads_per_group": HEADS_PER_GROUP,
            "head_dim": HEAD_DIM,
            "scale_blocks": SCALE_BLOCKS_PER_GROUP,
            "dtype": str(o.dtype),
            "chain": "inverse-rope-fp8-quant",
        },
        tolerances=CorrectnessTolerances(
            atol=0.0,
            rtol=0.0,
            max_mean_relative=1e-3,
            min_cosine=0.9999,
            require_allclose=False,
        ),
    )


__all__ = ["build_b02_inv_rope_fp8_quant_case"]
