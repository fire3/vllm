# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA prefill kernel (phase 2A).

Phase 1 expands every prefill token into a decode-style row and runs the
decode kernel twice (SWA cache and compressed cache), then merges the partial
attentions in Python via LSE. That is correct but pays two full kernel passes
plus a ``[T, H, D]`` merge round-trip per layer.

Phase 2A replaces that path with a single tiled, query-major, head-blocked
kernel that:

  * consumes per-query CSR metadata (flat physical-slot lists + indptr)
    instead of fixed-width padded rows, so short prefixes stop paying for the
    full window scan;
  * fuses the SWA and compressed (c4/c128) sources into one launch that
    shares the online-softmax state (``m_i``/``l_i``/``acc``) across both
    sources, dropping the intermediate out/LSE tensors and the Python merge;
  * computes QK scores and K=V accumulation per 64-dim group with ``tl.dot``
    (fp8 -> bf16 upcast, fp32 accumulate -- the SM89 pattern established by
    the MQA-logits fix), applying the per-(token, group) UE8M0 scale after
    the score dot and pre-scaling the value operand;
  * keeps attn-sink handling in Python (same semantics as the decode
    wrapper) so the kernel numerics stay identical to the decode path.

The kernel still runs on the padded head count like phase 1; removing padded
heads is explicitly out of scope for 2A (see notes 2026-08-23, sec. 2.1-3).
"""

from typing import Optional

import torch

from vllm.envs import VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE
from vllm.triton_utils import tl, triton

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants (shared with the decode kernel).
_NOPE_DIM = 448
_ROPE_DIM = 64
_GROUP_DIM = 64
_NOPE_GROUPS = _NOPE_DIM // _GROUP_DIM  # 7
_TOKEN_DATA_STRIDE = 576  # bytes per token in the data section
_SCALE_STRIDE = 8  # bytes per token in the scale section

_PACK_BLOCK = 128
_DEFAULT_BLOCK_H = 16
_DEFAULT_BLOCK_K = 16
_DEFAULT_NUM_WARPS = 8


@triton.jit
def _pack_sparse_rows_kernel(
    dense_ptr,  # [T, W] int32
    lens_ptr,  # [T] int32
    flat_ptr,  # [T*W] int32 scratch, only the valid prefix is written
    indptr_ptr,  # [T+1] int32
    stride_d: tl.int32,
    BLOCK: tl.constexpr,
):
    """Copy each row's ``lens[t]``-long prefix into a flat CSR list."""
    t = tl.program_id(0)
    blk = tl.program_id(1)
    length = tl.load(lens_ptr + t)
    start = tl.load(indptr_ptr + t)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < length
    vals = tl.load(dense_ptr + t * stride_d + offs, mask=mask, other=-1)
    tl.store(flat_ptr + start + offs, vals, mask=mask)


@triton.jit
def _tiled_sparse_prefill_kernel(
    Q_ptr,  # [T, H, D] bf16
    O_ptr,  # [T, H, D] bf16, pre-sink output
    LSE_ptr,  # [T, H] f32, pre-sink LSE (for the Python-side sink)
    swa_cache_fp8_ptr,  # SWA NoPE fp8 flat view
    swa_cache_uint8_ptr,  # SWA scale uint8 flat view
    swa_cache_bf16_ptr,  # SWA RoPE bf16 flat view
    swa_idx_ptr,  # [nnz_swa] int32 physical slots
    swa_indptr_ptr,  # [T+1] int32
    extra_cache_fp8_ptr,  # compressed NoPE fp8 flat view
    extra_cache_uint8_ptr,  # compressed scale uint8 flat view
    extra_cache_bf16_ptr,  # compressed RoPE bf16 flat view
    extra_idx_ptr,  # [nnz_extra] int32 physical slots
    extra_indptr_ptr,  # [T+1] int32
    sm_scale: tl.float32,
    swa_page_size: tl.int32,
    swa_page_bytes: tl.int64,
    swa_layer_off: tl.int64,
    swa_scale_off: tl.int64,
    extra_page_size: tl.int32,
    extra_page_bytes: tl.int64,
    extra_layer_off: tl.int64,
    extra_scale_off: tl.int64,
    H: tl.int32,
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    HAS_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    TOKEN_DATA_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
):
    """Query-major, head-blocked, dual-source fused sparse prefill.

    Grid: ``(T, ceil(H / BLOCK_H))``. One program handles one query token and
    ``BLOCK_H`` query heads, scanning the SWA slot list and (optionally) the
    compressed slot list in ``BLOCK_K`` tiles while sharing one online-softmax
    accumulator. All heads share the same KV (MQA), so the paged gather is
    done once per tile and reused across heads through ``tl.dot``.
    """
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
    q_rope = tl.load(
        Q_ptr + q_base + (NOPE_DIM + d_offs)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    )

    # Online-softmax state, base-2 like the decode kernel.
    m_i = tl.full([BLOCK_H], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc4 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc5 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc6 = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)
    acc_rope = tl.zeros([BLOCK_H, GROUP_DIM], dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)

    for src in range(2):
        if src == 0:
            fp8_p = swa_cache_fp8_ptr
            uint8_p = swa_cache_uint8_ptr
            bf16_p = swa_cache_bf16_ptr
            idx_p = swa_idx_ptr
            indptr_p = swa_indptr_ptr
            page_size = swa_page_size
            page_bytes = swa_page_bytes
            layer_off = swa_layer_off
            scale_off = swa_scale_off
        else:
            fp8_p = extra_cache_fp8_ptr
            uint8_p = extra_cache_uint8_ptr
            bf16_p = extra_cache_bf16_ptr
            idx_p = extra_idx_ptr
            indptr_p = extra_indptr_ptr
            page_size = extra_page_size
            page_bytes = extra_page_bytes
            layer_off = extra_layer_off
            scale_off = extra_scale_off

        if src == 0 or HAS_EXTRA:
            start = tl.load(indptr_p + t)
            end = tl.load(indptr_p + t + 1)
            length = end - start

            for k_start in tl.range(0, length, BLOCK_K):
                pos = k_start + k_offs
                in_range = pos < length
                slot = tl.load(idx_p + start + pos, mask=in_range, other=-1)
                valid = in_range & (slot >= 0)
                safe_slot = tl.maximum(slot, 0)
                page_ids = (safe_slot // page_size).to(tl.int64)
                page_offs = (safe_slot % page_size).to(tl.int64)
                data_base = (
                    page_ids * page_bytes
                    + layer_off
                    + page_offs * TOKEN_DATA_STRIDE
                )
                scale_base = (
                    page_ids * page_bytes
                    + layer_off
                    + scale_off
                    + page_offs * SCALE_STRIDE
                )

                # Gather all 8 group tiles once per KV tile; they feed both
                # the score phase and the value phase of this tile.
                addrs0 = (
                    data_base[:, None]
                    + (0 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv0 = tl.load(fp8_p + addrs0, mask=valid[:, None], other=0.0)
                addrs1 = (
                    data_base[:, None]
                    + (1 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv1 = tl.load(fp8_p + addrs1, mask=valid[:, None], other=0.0)
                addrs2 = (
                    data_base[:, None]
                    + (2 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv2 = tl.load(fp8_p + addrs2, mask=valid[:, None], other=0.0)
                addrs3 = (
                    data_base[:, None]
                    + (3 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv3 = tl.load(fp8_p + addrs3, mask=valid[:, None], other=0.0)
                addrs4 = (
                    data_base[:, None]
                    + (4 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv4 = tl.load(fp8_p + addrs4, mask=valid[:, None], other=0.0)
                addrs5 = (
                    data_base[:, None]
                    + (5 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv5 = tl.load(fp8_p + addrs5, mask=valid[:, None], other=0.0)
                addrs6 = (
                    data_base[:, None]
                    + (6 * GROUP_DIM + d_offs)[None, :].to(tl.int64)
                )
                kv6 = tl.load(fp8_p + addrs6, mask=valid[:, None], other=0.0)
                rope_elem_base = ((data_base + NOPE_DIM) // 2)[:, None]
                kv_rope = tl.load(
                    bf16_p + rope_elem_base + d_offs[None, :].to(tl.int64),
                    mask=valid[:, None],
                    other=0.0,
                )
                sc0 = tl.load(
                    uint8_p + scale_base + 0, mask=valid, other=127
                )
                sc1 = tl.load(
                    uint8_p + scale_base + 1, mask=valid, other=127
                )
                sc2 = tl.load(
                    uint8_p + scale_base + 2, mask=valid, other=127
                )
                sc3 = tl.load(
                    uint8_p + scale_base + 3, mask=valid, other=127
                )
                sc4 = tl.load(
                    uint8_p + scale_base + 4, mask=valid, other=127
                )
                sc5 = tl.load(
                    uint8_p + scale_base + 5, mask=valid, other=127
                )
                sc6 = tl.load(
                    uint8_p + scale_base + 6, mask=valid, other=127
                )

                # Score phase: per-group dot (fp8 losslessly upcast to bf16),
                # then the group scale is applied to the finished score row
                # (constant per token and group, so it commutes with the dot).
                s0 = tl.math.exp2(sc0.to(tl.float32) - 127.0)
                s1 = tl.math.exp2(sc1.to(tl.float32) - 127.0)
                s2 = tl.math.exp2(sc2.to(tl.float32) - 127.0)
                s3 = tl.math.exp2(sc3.to(tl.float32) - 127.0)
                s4 = tl.math.exp2(sc4.to(tl.float32) - 127.0)
                s5 = tl.math.exp2(sc5.to(tl.float32) - 127.0)
                s6 = tl.math.exp2(sc6.to(tl.float32) - 127.0)
                scores = tl.dot(q0, tl.trans(kv0.to(tl.bfloat16)))
                scores = scores * s0[None, :]
                scores += tl.dot(q1, tl.trans(kv1.to(tl.bfloat16))) * s1[None, :]
                scores += tl.dot(q2, tl.trans(kv2.to(tl.bfloat16))) * s2[None, :]
                scores += tl.dot(q3, tl.trans(kv3.to(tl.bfloat16))) * s3[None, :]
                scores += tl.dot(q4, tl.trans(kv4.to(tl.bfloat16))) * s4[None, :]
                scores += tl.dot(q5, tl.trans(kv5.to(tl.bfloat16))) * s5[None, :]
                scores += tl.dot(q6, tl.trans(kv6.to(tl.bfloat16))) * s6[None, :]
                scores += tl.dot(q_rope, tl.trans(kv_rope))
                scores = scores * sm_scale
                scores = tl.where(valid[None, :] & h_mask[:, None], scores, -1e30)

                # Online softmax update (base-2, tile-level).
                # The accumulator runs in log2 space (like the decode kernel:
                # ``scores_log2 = scores * LOG2E``); without the conversion the
                # exp2 weights are flattened by 2^(s) instead of e^(s).
                scores_log2 = scores * LOG2E
                tile_max = tl.max(scores_log2, axis=1)
                m_new = tl.maximum(m_i, tile_max)
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(scores_log2 - m_new[:, None])
                p = tl.where(valid[None, :] & h_mask[:, None], p, 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                p_bf16 = p.to(tl.bfloat16)

                # Value phase: pre-scale each group by its power-of-2 scale
                # (exact in bf16), then dot with the bf16 weights.
                kv0s = (kv0.to(tl.float32) * s0[:, None]).to(tl.bfloat16)
                kv1s = (kv1.to(tl.float32) * s1[:, None]).to(tl.bfloat16)
                kv2s = (kv2.to(tl.float32) * s2[:, None]).to(tl.bfloat16)
                kv3s = (kv3.to(tl.float32) * s3[:, None]).to(tl.bfloat16)
                kv4s = (kv4.to(tl.float32) * s4[:, None]).to(tl.bfloat16)
                kv5s = (kv5.to(tl.float32) * s5[:, None]).to(tl.bfloat16)
                kv6s = (kv6.to(tl.float32) * s6[:, None]).to(tl.bfloat16)
                acc0 = acc0 * alpha[:, None] + tl.dot(p_bf16, kv0s)
                acc1 = acc1 * alpha[:, None] + tl.dot(p_bf16, kv1s)
                acc2 = acc2 * alpha[:, None] + tl.dot(p_bf16, kv2s)
                acc3 = acc3 * alpha[:, None] + tl.dot(p_bf16, kv3s)
                acc4 = acc4 * alpha[:, None] + tl.dot(p_bf16, kv4s)
                acc5 = acc5 * alpha[:, None] + tl.dot(p_bf16, kv5s)
                acc6 = acc6 * alpha[:, None] + tl.dot(p_bf16, kv6s)
                acc_rope = acc_rope * alpha[:, None] + tl.dot(p_bf16, kv_rope)
                m_i = m_new

    # Finalize: normalize and write the pre-sink output and LSE.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    o_base = t * stride_ob + h_offs[:, None] * stride_oh
    tl.store(
        O_ptr + o_base + (0 * GROUP_DIM + d_offs)[None, :],
        (acc0 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (1 * GROUP_DIM + d_offs)[None, :],
        (acc1 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (2 * GROUP_DIM + d_offs)[None, :],
        (acc2 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (3 * GROUP_DIM + d_offs)[None, :],
        (acc3 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (4 * GROUP_DIM + d_offs)[None, :],
        (acc4 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (5 * GROUP_DIM + d_offs)[None, :],
        (acc5 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (6 * GROUP_DIM + d_offs)[None, :],
        (acc6 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    tl.store(
        O_ptr + o_base + (NOPE_DIM + d_offs)[None, :],
        (acc_rope / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    lse = tl.where(
        l_i > 0.0,
        m_i / LOG2E + tl.math.log(safe_l),
        float("-inf"),
    )
    tl.store(LSE_ptr + t * H + h_offs, lse, mask=h_mask)


_TILED_SPARSE_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_H": 8, "BLOCK_K": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_H": 16, "BLOCK_K": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_H": 16, "BLOCK_K": 32}, num_warps=8, num_stages=2),
]

# Autotune is opt-in (same policy as the decode kernel): the default fixed
# config keeps first-call latency and results deterministic. The kernel is
# gather-bound, so num_stages is left out of the search.
_TILED_SPARSE_PREFILL_AUTOTUNED = triton.autotune(
    configs=_TILED_SPARSE_PREFILL_CONFIGS,
    key=["H"],
)(_tiled_sparse_prefill_kernel)


def _paged_cache_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Flat fp8/uint8/bf16 views over a packed fp8_ds_mla cache slab view."""
    storage_nbytes = kv_cache.untyped_storage().nbytes()
    raw_flat = kv_cache.as_strided((storage_nbytes,), (1,), storage_offset=0)
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)
    return (
        raw_fp8,
        raw_uint8,
        raw_bf16,
        kv_cache.storage_offset(),
        kv_cache.shape[1],
        kv_cache.stride(0),
    )


_CSR_FLAT_BUFFERS: dict[tuple[int, int, int, int], torch.Tensor] = {}


def _get_csr_flat_buffer(
    capacity: int, device: torch.device, slot: int = 0
) -> torch.Tensor:
    """Reusable max-capacity flat index buffer (no per-call allocation).

    ``slot`` keeps the swa and extra CSR packs from aliasing the same buffer
    when both sources have the same capacity; otherwise the second pack would
    overwrite the first and both kernel pointers would see the extra rows.
    The key also includes the current CUDA stream so concurrent calls on
    different streams cannot race on one buffer (a pack on stream A would
    otherwise be overwritten by stream B before A's fused kernel reads it).
    """
    stream = torch.cuda.current_stream(device)
    key = (
        device.index if device.index is not None else 0,
        capacity,
        slot,
        stream.cuda_stream,
    )
    buf = _CSR_FLAT_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(capacity, dtype=torch.int32, device=device)
        _CSR_FLAT_BUFFERS[key] = buf
    return buf


def _pack_sparse_rows(
    indices: torch.Tensor,  # [T, W] or [T, 1, W] int32 physical slots
    lens: torch.Tensor,  # [T] int32 valid prefix length per row
    slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack dense padded rows into flat CSR lists (flat, indptr).

    ``flat`` is a max-capacity cached buffer; only the valid prefix of each
    row is written, so no CPU/GPU sync is needed to learn the true nnz.
    """
    T = lens.shape[0]
    dense = indices.reshape(T, -1)
    if not dense.is_contiguous():
        dense = dense.contiguous()
    width = dense.shape[1]
    lens32 = lens.reshape(T).to(torch.int32)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=lens.device)
    torch.cumsum(lens32, dim=0, out=indptr[1:])
    flat = _get_csr_flat_buffer(T * width, lens.device, slot)
    _pack_sparse_rows_kernel[(T, triton.cdiv(width, _PACK_BLOCK))](
        dense,
        lens32,
        flat,
        indptr,
        dense.stride(0),
        BLOCK=_PACK_BLOCK,
    )
    return flat, indptr


def triton_sparse_mla_prefill_vllm(
    q: torch.Tensor,  # [T, H, D] bf16 (padded to backend head count)
    swa_kv_cache: torch.Tensor,  # paged fp8_ds_mla cache
    swa_indices: torch.Tensor,  # [T, 1, W] or [T, W] int32 slots
    swa_lens: torch.Tensor,  # [T] int32
    extra_kv_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_lens: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],  # [H] f32, padded
    softmax_scale: float,
    out: torch.Tensor,  # [T, H, D] bf16, written in place
) -> None:
    """vLLM entry point for the phase-2A tiled Triton sparse-MLA prefill."""
    if not q.is_contiguous():
        q = q.contiguous()
    T, H, _ = q.shape

    (
        swa_fp8,
        swa_uint8,
        swa_bf16,
        swa_layer_off,
        swa_page_size,
        swa_page_bytes,
    ) = _paged_cache_views(swa_kv_cache)
    swa_flat, swa_indptr = _pack_sparse_rows(swa_indices, swa_lens, slot=0)

    has_extra = extra_kv_cache is not None
    if has_extra:
        assert extra_indices is not None and extra_lens is not None
        (
            extra_fp8,
            extra_uint8,
            extra_bf16,
            extra_layer_off,
            extra_page_size,
            extra_page_bytes,
        ) = _paged_cache_views(extra_kv_cache)
        extra_flat, extra_indptr = _pack_sparse_rows(
            extra_indices, extra_lens, slot=1
        )
    else:
        empty_u8 = torch.empty(0, dtype=torch.uint8, device=q.device)
        empty_fp8 = torch.empty(0, dtype=torch.float8_e4m3fn, device=q.device)
        empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=q.device)
        empty_i32 = torch.empty(0, dtype=torch.int32, device=q.device)
        extra_fp8 = empty_fp8
        extra_uint8 = empty_u8
        extra_bf16 = empty_bf16
        extra_layer_off = 0
        extra_page_size = 1
        extra_page_bytes = 0
        extra_flat = empty_i32
        extra_indptr = empty_i32

    swa_scale_off = swa_page_size * _TOKEN_DATA_STRIDE
    extra_scale_off = extra_page_size * _TOKEN_DATA_STRIDE
    lse = torch.empty(T, H, dtype=torch.float32, device=q.device)

    grid = (T, triton.cdiv(H, _DEFAULT_BLOCK_H))
    kernel = (
        _TILED_SPARSE_PREFILL_AUTOTUNED
        if VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE
        else _tiled_sparse_prefill_kernel
    )
    launch_args = dict(
        Q_ptr=q,
        O_ptr=out,
        LSE_ptr=lse,
        swa_cache_fp8_ptr=swa_fp8,
        swa_cache_uint8_ptr=swa_uint8,
        swa_cache_bf16_ptr=swa_bf16,
        swa_idx_ptr=swa_flat,
        swa_indptr_ptr=swa_indptr,
        extra_cache_fp8_ptr=extra_fp8,
        extra_cache_uint8_ptr=extra_uint8,
        extra_cache_bf16_ptr=extra_bf16,
        extra_idx_ptr=extra_flat,
        extra_indptr_ptr=extra_indptr,
        sm_scale=softmax_scale,
        swa_page_size=swa_page_size,
        swa_page_bytes=int(swa_page_bytes),
        swa_layer_off=int(swa_layer_off),
        swa_scale_off=int(swa_scale_off),
        extra_page_size=extra_page_size,
        extra_page_bytes=int(extra_page_bytes),
        extra_layer_off=int(extra_layer_off),
        extra_scale_off=int(extra_scale_off),
        H=H,
        stride_qb=q.stride(0),
        stride_qh=q.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        HAS_EXTRA=has_extra,
        GROUP_DIM=_GROUP_DIM,
        NOPE_DIM=_NOPE_DIM,
        TOKEN_DATA_STRIDE=_TOKEN_DATA_STRIDE,
        SCALE_STRIDE=_SCALE_STRIDE,
    )
    if VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE:
        kernel[grid](**launch_args)
    else:
        kernel[grid](
            **launch_args,
            BLOCK_H=_DEFAULT_BLOCK_H,
            BLOCK_K=_DEFAULT_BLOCK_K,
            num_warps=_DEFAULT_NUM_WARPS,
            num_stages=2,
        )

    if attn_sink is not None:
        combined = torch.logaddexp(lse, attn_sink.view(1, -1).expand_as(lse))
        w = torch.where(
            lse > -1e20,
            torch.exp(lse - combined),
            torch.zeros_like(lse),
        )
        out.copy_((out.float() * w.unsqueeze(-1)).to(torch.bfloat16))
