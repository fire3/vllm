# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_fp8_einsum_config,
    use_deepseek_v4_sm120_cutlass_fp8_einsum,
)
from vllm.platforms import current_platform


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.
    SM89/SM12x: use FP32 block scales with the Triton fallback.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    return deepseek_v4_fp8_einsum_config(cap.major)


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    wo_a_scale = getattr(wo_a, "weight_scale_inv", None)
    if wo_a_scale is None:
        wo_a_scale = wo_a.weight_scale
    use_sm120_cutlass = (
        wo_a_scale.dtype == torch.float32
        and use_deepseek_v4_sm120_cutlass_fp8_einsum(
            o.shape[0],
            n_groups,
            o_lora_rank,
            heads_per_group * (nope_dim + rope_dim),
        )
    )
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
        compact_scales=use_sm120_cutlass,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    deepseek_v4_fp8_einsum(
        o_fp8,
        o_scale,
        wo_a.weight,
        wo_a_scale,
        z,
        "bhr,hdr->bhd",
        list(einsum_recipe),
    )
    return wo_b(z.flatten(1))
