# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 Triton top-k sparse MLA kernel for GLM-5.3-Flash.

The DSA (indexer top-k) layers are plain MQA over the indexer-selected KV:
each query token attends its own ``topk + trailing-pool`` rows (already
compacted by ``triton_convert_req_index_to_global_index`` into a contiguous
prefix of valid entries). No SWA, no compression, no RoPE
(``qk_rope_head_dim == 0``); KV cache is plain contiguous
``[num_blocks, block_size, 512]`` bf16 or fp8 e4m3 (per-tensor ``k_scale``).

The kernel is query-major and head-blocked (BLOCK_H=8 heads per program),
computing QK scores and K=V accumulation per 64-dim group with ``tl.dot``
(fp8 -> bf16 upcast, fp32 accumulate, the SM89 pattern), sharing one
base-2 online-softmax state. It is shaped like the DSv4 SM89 Triton sparse
kernel but with the DSv4 SWA/compressed/packed-layout machinery removed.
"""

import torch

from vllm.triton_utils import tl, triton

LOG2E = tl.constexpr(1.4426950408889634)

_GROUP_DIM = 64
_DEFAULT_BLOCK_H = 8
_DEFAULT_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 8


@triton.jit
def _triton_topk_mla_kernel(
    Q_ptr,  # [T, H, D] bf16 (absorbed q_nope, D = kv_lora_rank)
    KV_ptr,  # [num_rows, D] bf16 or fp8 e4m3 flat row view
    IDX_ptr,  # [T, NUM_TOPK] int32 flat row indices (-1 masked tail)
    LENS_ptr,  # [T] int32 valid count per row (compact prefix length)
    OUT_ptr,  # [T, H, D] bf16
    sm_scale,  # fp32
    kv_scale,  # fp32 (1.0 for bf16 caches)
    H,  # runtime head count (may be smaller than BLOCK_H on TP8)
    num_rows,  # runtime row count of the flat KV view
    stride_qb: tl.int64,
    stride_qh: tl.int64,
    stride_ob: tl.int64,
    stride_oh: tl.int64,
    NUM_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    NGROUPS: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """One program per (query token, head block)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_offs = tl.arange(0, GROUP_DIM)

    # Q per 64-dim group, loaded once and reused by every KV tile.
    q_base = t * stride_qb + h_offs[:, None] * stride_qh
    q0 = tl.load(
        Q_ptr + q_base + (0 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q1 = tl.load(
        Q_ptr + q_base + (1 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q2 = tl.load(
        Q_ptr + q_base + (2 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q3 = tl.load(
        Q_ptr + q_base + (3 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q4 = tl.load(
        Q_ptr + q_base + (4 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q5 = tl.load(
        Q_ptr + q_base + (5 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q6 = tl.load(
        Q_ptr + q_base + (6 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )
    q7 = tl.load(
        Q_ptr + q_base + (7 * GROUP_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    # Online-softmax state, base-2 like the DSv4 SM89 decode kernel.
    m_i = tl.full([BLOCK_H], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc7 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)

    valid_len = tl.load(LENS_ptr + t)
    k_offs = tl.arange(0, BLOCK_K)
    # Loop only over the tiles that can contain valid entries: the convert
    # kernel compacts valid slots to [0, valid_len), and the tail is -1.
    bound = tl.minimum(NUM_TOPK, tl.cdiv(valid_len, BLOCK_K) * BLOCK_K)

    for k_start in tl.range(0, bound, BLOCK_K):
        pos = k_start + k_offs
        kmask = pos < valid_len
        idx = tl.load(IDX_ptr + t * NUM_TOPK + pos, mask=kmask, other=-1)
        valid = kmask & (idx >= 0)
        # Clamp masked tails and any out-of-range slot to a legal address;
        # scores/values are masked off for invalid lanes, so the clamped
        # index is never dereferenced for them.
        safe_idx = tl.maximum(
            tl.minimum(tl.where(valid, idx, 0), num_rows - 1), 0
        )
        kbase = KV_ptr + safe_idx[:, None] * (NGROUPS * GROUP_DIM)

        kv0 = tl.load(
            kbase + (0 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv1 = tl.load(
            kbase + (1 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv2 = tl.load(
            kbase + (2 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv3 = tl.load(
            kbase + (3 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv4 = tl.load(
            kbase + (4 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv5 = tl.load(
            kbase + (5 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv6 = tl.load(
            kbase + (6 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv7 = tl.load(
            kbase + (7 * GROUP_DIM + d_offs)[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        if IS_FP8:
            # Per-tensor k_scale (fp8 e4m3 plain cache). Apply once so both
            # the score phase (K) and value phase (V) see the dequantized KV.
            kv0 = (kv0.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv1 = (kv1.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv2 = (kv2.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv3 = (kv3.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv4 = (kv4.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv5 = (kv5.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv6 = (kv6.to(tl.float32) * kv_scale).to(tl.bfloat16)
            kv7 = (kv7.to(tl.float32) * kv_scale).to(tl.bfloat16)

        # Score phase: per-group dot (fp8 losslessly upcast to bf16).
        scores = tl.dot(q0, tl.trans(kv0))
        scores += tl.dot(q1, tl.trans(kv1))
        scores += tl.dot(q2, tl.trans(kv2))
        scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q4, tl.trans(kv4))
        scores += tl.dot(q5, tl.trans(kv5))
        scores += tl.dot(q6, tl.trans(kv6))
        scores += tl.dot(q7, tl.trans(kv7))
        scores = scores * sm_scale
        scores = tl.where(valid[None, :] & h_mask[:, None], scores, -1e30)

        # Online softmax update (base-2, tile-level), like the DSv4 kernel:
        # the accumulator runs in log2 space.
        scores_log2 = scores * LOG2E
        tile_max = tl.max(scores_log2, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores_log2 - m_new[:, None])
        p = tl.where(valid[None, :] & h_mask[:, None], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

        # Value phase: p x kv per group, fp32 accumulate.
        acc0 = acc0 * alpha[:, None] + tl.dot(p_bf16, kv0)
        acc1 = acc1 * alpha[:, None] + tl.dot(p_bf16, kv1)
        acc2 = acc2 * alpha[:, None] + tl.dot(p_bf16, kv2)
        acc3 = acc3 * alpha[:, None] + tl.dot(p_bf16, kv3)
        acc4 = acc4 * alpha[:, None] + tl.dot(p_bf16, kv4)
        acc5 = acc5 * alpha[:, None] + tl.dot(p_bf16, kv5)
        acc6 = acc6 * alpha[:, None] + tl.dot(p_bf16, kv6)
        acc7 = acc7 * alpha[:, None] + tl.dot(p_bf16, kv7)
        m_i = m_new

    # Finalize: normalize and write the output.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = t * stride_ob + h_offs[:, None] * stride_oh
    tl.store(
        OUT_ptr + o_base + (0 * GROUP_DIM + d_offs)[None, :],
        (acc0 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (1 * GROUP_DIM + d_offs)[None, :],
        (acc1 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (2 * GROUP_DIM + d_offs)[None, :],
        (acc2 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (3 * GROUP_DIM + d_offs)[None, :],
        (acc3 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (4 * GROUP_DIM + d_offs)[None, :],
        (acc4 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (5 * GROUP_DIM + d_offs)[None, :],
        (acc5 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (6 * GROUP_DIM + d_offs)[None, :],
        (acc6 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        OUT_ptr + o_base + (7 * GROUP_DIM + d_offs)[None, :],
        (acc7 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )


def triton_topk_mla_forward(
    q: torch.Tensor,  # [T, H, D] bf16, contiguous
    kv_rows: torch.Tensor,  # [num_rows, D] bf16 or fp8 e4m3 flat view
    kv_indices: torch.Tensor,  # [T, NUM_TOPK] int32 global row indices
    kv_lens: torch.Tensor,  # [T] int32 valid counts
    out: torch.Tensor,  # [T, H, D] bf16
    sm_scale: float,
    kv_scale: float = 1.0,
    num_heads: int | None = None,
) -> None:
    """Launch the GLM-5.3 top-k sparse MLA kernel (prefill and decode).

    Both phases run one query row per program; decode rows are simply
    one-token queries, so there is no separate decode kernel.
    """
    assert q.dtype == torch.bfloat16
    assert kv_indices.dtype == torch.int32
    assert kv_lens.dtype == torch.int32
    assert q.shape[0] == kv_indices.shape[0] == kv_lens.shape[0]
    T, H, D = q.shape
    num_rows, kv_dim = kv_rows.shape
    assert kv_dim == D == 512, (kv_dim, D)
    num_topk = kv_indices.shape[1]
    assert num_topk % _DEFAULT_BLOCK_K == 0, (num_topk, _DEFAULT_BLOCK_K)
    is_fp8 = kv_rows.dtype == torch.float8_e4m3fn
    if not is_fp8:
        assert kv_rows.dtype == torch.bfloat16

    grid = (T, triton.cdiv(H, _DEFAULT_BLOCK_H))
    _triton_topk_mla_kernel[grid](
        Q_ptr=q,
        KV_ptr=kv_rows,
        IDX_ptr=kv_indices,
        LENS_ptr=kv_lens,
        OUT_ptr=out,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        H=H,
        num_rows=num_rows,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        NUM_TOPK=num_topk,
        BLOCK_H=_DEFAULT_BLOCK_H,
        BLOCK_K=_DEFAULT_BLOCK_K,
        GROUP_DIM=_GROUP_DIM,
        NGROUPS=D // _GROUP_DIM,
        IS_FP8=is_fp8,
        num_warps=_DEFAULT_NUM_WARPS,
        num_stages=2,
    )
