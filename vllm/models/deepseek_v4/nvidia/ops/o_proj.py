# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_einsum, is_deep_gemm_supported


@triton.jit
def _grouped_bf16_matmul_kernel(
    x_ptr,  # [T, G, D]
    w_ptr,  # [G, D, R]
    out_ptr,  # [T, G, R]
    num_tokens,
    hidden_dim: tl.constexpr,
    o_lora_rank: tl.constexpr,
    stride_x_t: tl.int64,
    stride_x_g: tl.int64,
    stride_x_d: tl.constexpr,
    stride_w_g: tl.int64,
    stride_w_d: tl.int64,
    stride_w_r: tl.constexpr,
    stride_out_t: tl.int64,
    stride_out_g: tl.int64,
    stride_out_r: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_gr = tl.program_id(1)

    group_id = pid_gr // tl.cdiv(o_lora_rank, BLOCK_R)
    rank_block = pid_gr % tl.cdiv(o_lora_rank, BLOCK_R)

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    r_offsets = rank_block * BLOCK_R + tl.arange(0, BLOCK_R)
    d_offsets = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_T, BLOCK_R), dtype=tl.float32)
    for d_start in range(0, hidden_dim, BLOCK_D):
        d = d_start + d_offsets
        x_ptrs = (
            x_ptr
            + t_offsets[:, None] * stride_x_t
            + group_id * stride_x_g
            + d[None, :] * stride_x_d
        )
        w_ptrs = (
            w_ptr
            + group_id * stride_w_g
            + d[:, None] * stride_w_d
            + r_offsets[None, :] * stride_w_r
        )
        x = tl.load(
            x_ptrs,
            mask=(t_offsets[:, None] < num_tokens) & (d[None, :] < hidden_dim),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(d[:, None] < hidden_dim) & (r_offsets[None, :] < o_lora_rank),
            other=0.0,
        )
        acc = tl.dot(x, w, acc=acc, input_precision="ieee")

    out_ptrs = (
        out_ptr
        + t_offsets[:, None] * stride_out_t
        + group_id * stride_out_g
        + r_offsets[None, :] * stride_out_r
    )
    tl.store(
        out_ptrs,
        acc.to(tl.bfloat16),
        mask=(t_offsets[:, None] < num_tokens) & (r_offsets[None, :] < o_lora_rank),
    )


def _get_cached_wo_a_bf16_t(
    wo_a: nn.Module,
    n_groups: int,
    o_lora_rank: int,
    hidden_dim: int,
) -> torch.Tensor:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _get_cached_wo_a_bf16

    cached_t = getattr(wo_a, "_dsv4_wo_a_bf16_t", None)
    if cached_t is not None:
        return cached_t

    cached = _get_cached_wo_a_bf16(wo_a, n_groups, o_lora_rank, hidden_dim)
    cached_t = cached.transpose(-1, -2).contiguous()
    wo_a._dsv4_wo_a_bf16_t = cached_t
    return cached_t


def _triton_o_proj_fallback(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fused_inverse_rope_gptj,
    )

    o_ref = _fused_inverse_rope_gptj(o, positions, cos_sin_cache, rope_dim)
    o_ref = o_ref.view(o.shape[0], n_groups, -1)
    wo_a_weight_t = _get_cached_wo_a_bf16_t(
        wo_a, n_groups, o_lora_rank, o_ref.shape[-1]
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    block_t = 32
    block_r = 32
    block_d = 64
    grid = (
        triton.cdiv(o.shape[0], block_t),
        n_groups * triton.cdiv(o_lora_rank, block_r),
    )
    _grouped_bf16_matmul_kernel[grid](
        x_ptr=o_ref,
        w_ptr=wo_a_weight_t,
        out_ptr=z,
        num_tokens=o.shape[0],
        hidden_dim=o_ref.shape[-1],
        o_lora_rank=o_lora_rank,
        stride_x_t=o_ref.stride(0),
        stride_x_g=o_ref.stride(1),
        stride_x_d=o_ref.stride(2),
        stride_w_g=wo_a_weight_t.stride(0),
        stride_w_d=wo_a_weight_t.stride(1),
        stride_w_r=wo_a_weight_t.stride(2),
        stride_out_t=z.stride(0),
        stride_out_g=z.stride(1),
        stride_out_r=z.stride(2),
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return wo_b(z.flatten(1))


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


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
    if current_platform.is_cuda() and not is_deep_gemm_supported():
        return _triton_o_proj_fallback(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            wo_b,
            n_groups=n_groups,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
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
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
