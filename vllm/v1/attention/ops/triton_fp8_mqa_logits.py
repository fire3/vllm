# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits fallback for environments without DeepGEMM.

Ported from the v0.28.0-sm89 branch (where it is the SM89 DeepSeek V4
fallback): a dense per-row kernel for prefill
(``fp8_mqa_logits_triton``) and a paged kernel for decode
(``fp8_paged_mqa_logits_triton``). Both preserve the DeepGEMM reference
semantics:

    logits[m, k] = kv_scale[k] * sum_h relu(q[m, h] . k[k]) * w[m, h]

with FP8 operands losslessly upcast to fp16 for the tensor-core dot and
fp32 accumulation.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_INDEXER_BLOCK_COL_ENV = "VLLM_SM89_INDEXER_BLOCK_COL"
_INDEXER_BLOCK_COL_DEFAULT = 64
_INDEXER_BLOCK_COL_ALLOWED = (64, 128, 256)


@triton.jit
def _fp8_mqa_logits_kernel(
    Q_ptr,  # fp8e4m3 [seq_len, H, D]
    KV_ptr,  # fp8e4m3 [seq_len_kv, D]
    kv_scales_ptr,  # fp32 [seq_len_kv]
    weights_ptr,  # fp32 [seq_len, H]
    cu_start_ptr,  # int32 [seq_len]
    cu_end_ptr,  # int32 [seq_len]
    logits_ptr,  # fp32 [seq_len, seq_len_kv]
    seq_len,
    seq_len_kv,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    # strides
    stride_q_s: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_s: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_w_s: tl.int64,
    stride_w_h: tl.constexpr,
    stride_logits_s: tl.int64,
    stride_logits_k: tl.int64,
    # block sizes
    BLOCK_KV: tl.constexpr,
    # Scale-application order. DeepGEMM's sm90/sm100 fp8_fp4_mqa_logits
    # kernels apply it AFTER the weighted head sum
    # (relu -> weights -> sum -> * kv_scales); AITER's gfx942 reference
    # multiplies the per-token KV scale into the scores BEFORE ReLU. The two
    # orders are mathematically equivalent (scale > 0) and with the ue8m0
    # power-of-two scales used by the indexer cache they are bit-identical.
    # SCALE_AFTER_REDUCE=True keeps this fallback on the DeepGEMM reference
    # order used by SM90/SM100/SM120.
    SCALE_AFTER_REDUCE: tl.constexpr,
):
    row_id = tl.program_id(0)
    # go from larger to smaller in terms of work
    # to reduce the tail effect
    row_id = tl.num_programs(0) - row_id - 1
    tl.assume(row_id >= 0)
    tl.assume(stride_q_s > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_q_d > 0)
    tl.assume(stride_kv_s > 0)
    tl.assume(stride_kv_d > 0)
    tl.assume(stride_w_s > 0)
    tl.assume(stride_w_h > 0)

    logits_row_ptrs = logits_ptr + row_id * stride_logits_s

    h_inds = tl.arange(0, NUM_HEADS)[:, None]
    d_inds = tl.arange(0, HEAD_SIZE)

    # load Q[BLOCK_Q, NUM_HEADS, HEAD_SIZE]
    q_ptrs = (
        Q_ptr + row_id * stride_q_s + h_inds * stride_q_h + d_inds[None, :] * stride_q_d
    )

    q_block = tl.load(q_ptrs, cache_modifier=".cg")
    w_ptrs = weights_ptr + row_id * stride_w_s + h_inds * stride_w_h
    w_block = tl.load(w_ptrs, cache_modifier=".cg").to(tl.float32)

    # Load start/end for each row in this block
    start_ind = tl.load(cu_start_ptr + row_id)
    end_ind = tl.load(cu_end_ptr + row_id)

    start_ind = tl.maximum(start_ind, 0)
    end_ind = tl.minimum(end_ind, seq_len_kv)
    shifted_end = end_ind - start_ind
    shifted_unmasked_end = shifted_end // BLOCK_KV * BLOCK_KV

    kv_col_offsets = tl.arange(0, BLOCK_KV) + start_ind
    kv_ptrs = (
        KV_ptr + kv_col_offsets[None, :] * stride_kv_s + d_inds[:, None] * stride_kv_d
    )

    kv_scales_ptrs = kv_scales_ptr + kv_col_offsets

    logits_ptrs = logits_row_ptrs + kv_col_offsets * stride_logits_k

    # Loop over KV tiles
    for _ in tl.range(0, shifted_unmasked_end, BLOCK_KV):
        kv_block = tl.load(kv_ptrs)
        kv_scales = tl.load(kv_scales_ptrs)

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV].
        # Upcast the FP8 operands to fp16 (lossless: fp16 mantissa >= fp8) so
        # the MMA accumulates in fp32 through fp16 tensor cores.
        scores = tl.dot(
            q_block.to(tl.float16), kv_block.to(tl.float16), out_dtype=tl.float32
        )
        if SCALE_AFTER_REDUCE:
            # DeepGEMM order: logit = (sum_h relu(q·k_h) * w_h) * s_kv.
            scores = tl.maximum(scores, 0.0)
            scores = scores * w_block
            # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
            scores = tl.sum(scores, axis=0)
            scores = scores * kv_scales
        else:
            # AITER/gfx942 order: sum_h relu((q·k_h) * s_kv) * w_h.
            scores = scores * kv_scales[None, :]
            # ReLU
            scores = tl.maximum(scores, 0.0)
            scores = scores * w_block
            # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
            scores = tl.sum(scores, axis=0)
        tl.store(logits_ptrs, scores)

        kv_ptrs += BLOCK_KV * stride_kv_s
        kv_scales_ptrs += BLOCK_KV
        logits_ptrs += BLOCK_KV * stride_logits_k
        kv_col_offsets += BLOCK_KV

    # masked load
    kv_col_mask = kv_col_offsets < end_ind
    kv_block = tl.load(kv_ptrs, mask=kv_col_mask[None, :], other=0.0)
    kv_scales = tl.load(kv_scales_ptrs, mask=kv_col_mask, other=0.0)

    scores = tl.dot(
        q_block.to(tl.float16), kv_block.to(tl.float16), out_dtype=tl.float32
    )
    if SCALE_AFTER_REDUCE:
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
        scores = scores * kv_scales
    else:
        scores = scores * kv_scales[None, :]
        scores = tl.maximum(scores, 0.0)
        scores = scores * w_block
        scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits with Triton on CUDA/ROCm.

    Drop-in fallback for DeepGEMM's ``fp8_fp4_mqa_logits`` (prefill path).
    The launcher uses ``torch.empty`` rather than an ``-inf`` pre-fill: the
    consumer (``top_k_per_row_prefill``) selects candidates from
    ``[cu_starts[i], cu_ends[i])`` only and the kernel writes every position
    in that range, so out-of-range memory never affects the result.

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32 (the per-token
            Q scale and attention scaling are folded in by the caller).
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32; positions outside each row's
        ``[cu_starts[i], cu_ends[i])`` window are left unwritten.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    kv_scales_1d = kv_scales.reshape(-1)
    logits = torch.empty(
        (seq_len, seq_len_kv),
        dtype=torch.float32,
        device=q.device,
    )

    block_kv = min(128, max(16, triton.next_power_of_2(seq_len_kv)))
    stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_kv_s, stride_kv_d = k_fp8.stride()
    stride_w_s, stride_w_h = weights.stride()
    stride_logits_s, stride_logits_k = logits.stride()

    _fp8_mqa_logits_kernel[(seq_len,)](
        Q_ptr=q,
        KV_ptr=k_fp8,
        kv_scales_ptr=kv_scales_1d,
        weights_ptr=weights,
        cu_start_ptr=cu_starts,
        cu_end_ptr=cu_ends,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        stride_q_s=stride_q_s,
        stride_q_h=stride_q_h,
        stride_q_d=stride_q_d,
        stride_kv_s=stride_kv_s,
        stride_kv_d=stride_kv_d,
        stride_w_s=stride_w_s,
        stride_w_h=stride_w_h,
        stride_logits_s=stride_logits_s,
        stride_logits_k=stride_logits_k,
        BLOCK_KV=block_kv,
        SCALE_AFTER_REDUCE=True,
        num_warps=4,
        num_stages=2,
    )
    return logits


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,  # [rows, H, D] fp8
    kv_values_ptr,  # [num_blocks, block_size, D] fp8
    kv_scales_ptr,  # [num_blocks, block_size] fp32
    weights_ptr,  # [rows, H] fp32
    context_limits_ptr,  # [rows] int32
    q_offsets_ptr,  # [rows] int32
    block_tables_ptr,  # [B, max_blocks] int32
    logits_ptr,  # [rows, max_model_len] fp32
    next_n,
    max_blocks,
    max_model_len,
    NUM_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    stride_q_row: tl.int64,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_kv_block: tl.int64,
    stride_kv_token: tl.int64,
    stride_kv_d: tl.constexpr,
    stride_kv_scale_block: tl.int64,
    stride_kv_scale_token: tl.int64,
    stride_w_row: tl.int64,
    stride_w_h: tl.constexpr,
    stride_bt_batch: tl.int64,
    stride_bt_block: tl.int64,
    stride_logits_row: tl.int64,
    stride_logits_col: tl.int64,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(0)
    col_block_id = tl.program_id(1)

    tl.assume(stride_q_row > 0)
    tl.assume(stride_kv_block > 0)
    tl.assume(stride_kv_token > 0)
    tl.assume(stride_w_row > 0)
    tl.assume(stride_bt_batch > 0)
    tl.assume(stride_bt_block > 0)
    tl.assume(stride_logits_row > 0)
    tl.assume(stride_logits_col > 0)

    context_limit = tl.load(context_limits_ptr + row_id)
    q_offset = tl.load(q_offsets_ptr + row_id)
    batch_idx = row_id // next_n

    col_offsets = col_block_id * BLOCK_COL + tl.arange(0, BLOCK_COL)
    valid_cols = (
        (col_offsets < max_model_len)
        & (col_offsets < context_limit)
        & (col_offsets <= q_offset)
    )

    logical_blocks = col_offsets // block_size
    in_bt_range = logical_blocks < max_blocks
    safe_blocks = tl.where(in_bt_range, logical_blocks, 0)
    physical_blocks = tl.load(
        block_tables_ptr + batch_idx * stride_bt_batch + safe_blocks * stride_bt_block,
        mask=valid_cols & in_bt_range,
        other=0,
    )
    block_offsets = col_offsets % block_size

    h_offsets = tl.arange(0, NUM_HEADS)[:, None]
    d_offsets = tl.arange(0, HEAD_SIZE)
    q_ptrs = (
        q_ptr
        + row_id * stride_q_row
        + h_offsets * stride_q_h
        + d_offsets[None, :] * stride_q_d
    )
    q_block = tl.load(q_ptrs)
    w_ptrs = weights_ptr + row_id * stride_w_row + h_offsets * stride_w_h
    w_block = tl.load(w_ptrs).to(tl.float32)

    kv_ptrs = (
        kv_values_ptr
        + physical_blocks[None, :] * stride_kv_block
        + block_offsets[None, :] * stride_kv_token
        + d_offsets[:, None] * stride_kv_d
    )
    kv_block = tl.load(kv_ptrs, mask=valid_cols[None, :], other=0.0)
    kv_scales = tl.load(
        kv_scales_ptr
        + physical_blocks * stride_kv_scale_block
        + block_offsets * stride_kv_scale_token,
        mask=valid_cols,
        other=0.0,
    )

    # fp16 MMA (lossless upcast from fp8) with fp32 accumulation, matching the
    # dense fallback and the DeepGEMM reference op order.
    scores = tl.dot(
        q_block.to(tl.float16), kv_block.to(tl.float16), out_dtype=tl.float32
    )
    # DeepGEMM paged order: logit = (sum_h relu(q·k_h) * w_h) * s_kv.
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)
    scores = scores * kv_scales

    logits_ptrs = logits_ptr + row_id * stride_logits_row + col_offsets * stride_logits_col
    tl.store(logits_ptrs, scores, mask=valid_cols)


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    *,
    max_context_len: int | None = None,
) -> torch.Tensor:
    """Compute paged FP8 MQA logits with Triton.

    Drop-in fallback for DeepGEMM's ``fp8_fp4_paged_mqa_logits`` (decode
    path). The paged KV cache uses the packed FP8 layout shared by the C++
    ``indexer_k_quant_and_cache`` writer and the Triton compressor: per
    block, ``[block_size * head_dim]`` FP8 data bytes followed by
    ``[block_size * 4]`` fp32 scale bytes. Torch exposes the cache as
    ``[num_blocks, block_size, head_dim + 4]`` (optionally with an extra
    size-1 page dim), but the physical byte offsets are ``token * head_dim``
    for data and ``block_size * head_dim + token * 4`` for scales, so the
    launcher reads them with ``as_strided`` over the raw bytes.

    ``max_context_len`` optionally clips the dense-logits width (and the
    launch grid) to the maximum context length present in this batch.

    Args:
        q: Query tensor of shape ``[B, next_n, H, D]``, FP8 dtype.
        kv_cache: Paged KV cache. FP8 layout is
            ``[num_blocks, block_size, (1,) head_dim + 4]`` uint8 with the
            last 4 bytes per (block, pos) storing the float dequant scale.
        weights: Per-head weights of shape ``[B * next_n, H]``, float32.
        context_lens: Effective context lengths, int32, either 1D ``[B]``
            (``q_offsets`` derived as ``ctx - next_n + step``) or 2D
            ``[B, next_n]`` (``q_offsets = ctx - 1``).
        block_tables: ``[B, max_blocks]`` int32 logical->physical mapping.
        max_model_len: Maximum sequence length used to size the logits.
    """
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, num_heads, head_dim = q.shape
    rows = batch_size * next_n
    q_rows = q.reshape(rows, num_heads, head_dim).contiguous()
    weight_rows = weights.view(rows, num_heads).contiguous()

    if kv_cache.shape[-2] == 1:
        kv_cache = kv_cache.squeeze(-2)
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    block_bytes = kv_cache.stride(0)
    kv_values = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, head_dim),
        (block_bytes, head_dim, 1),
    ).view(fp8_dtype)
    kv_scales = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, 4),
        (block_bytes, 4, 1),
        storage_offset=block_size * head_dim,
    ).view(torch.float32).squeeze(-1)

    if context_lens.ndim == 1:
        steps = torch.arange(next_n, device=q.device, dtype=torch.int32)
        context_limits = context_lens.to(device=q.device, dtype=torch.int32)[:, None]
        context_limits = context_limits.expand(batch_size, next_n).reshape(-1).contiguous()
        q_offsets = (
            context_lens.to(device=q.device, dtype=torch.int32)[:, None]
            - next_n
            + steps[None, :]
        ).reshape(-1).contiguous()
    else:
        context_limits = context_lens.to(device=q.device, dtype=torch.int32).reshape(-1)
        q_offsets = (context_limits - 1).contiguous()

    try:
        block_col = int(os.environ.get(_INDEXER_BLOCK_COL_ENV, ""))
    except ValueError:
        block_col = _INDEXER_BLOCK_COL_DEFAULT
    if block_col not in _INDEXER_BLOCK_COL_ALLOWED:
        if _INDEXER_BLOCK_COL_ENV in os.environ:
            logger.warning(
                "fp8_paged_mqa_logits: ignoring invalid %s=%r, "
                "falling back to BLOCK_COL=%s",
                _INDEXER_BLOCK_COL_ENV,
                os.environ.get(_INDEXER_BLOCK_COL_ENV),
                _INDEXER_BLOCK_COL_DEFAULT,
            )
        block_col = _INDEXER_BLOCK_COL_DEFAULT
    if context_limits.numel() == 0:
        max_ctx = max_model_len
    elif max_context_len is None or max_context_len <= 0:
        max_ctx = int(context_limits.max().item())
    else:
        max_ctx = max_context_len
    max_ctx = min(max(max_ctx, 1), max_model_len)
    logits_cols = min(max_model_len, triton.cdiv(max_ctx, block_col) * block_col)

    logits = torch.full(
        (rows, logits_cols),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    grid = (rows, logits_cols // block_col)
    _fp8_paged_mqa_logits_kernel[grid](
        q_ptr=q_rows,
        kv_values_ptr=kv_values,
        kv_scales_ptr=kv_scales,
        weights_ptr=weight_rows,
        context_limits_ptr=context_limits,
        q_offsets_ptr=q_offsets,
        block_tables_ptr=block_tables,
        logits_ptr=logits,
        next_n=next_n,
        max_blocks=block_tables.shape[1],
        max_model_len=logits_cols,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_dim,
        BLOCK_COL=block_col,
        stride_q_row=q_rows.stride(0),
        stride_q_h=q_rows.stride(1),
        stride_q_d=q_rows.stride(2),
        stride_kv_block=kv_values.stride(0),
        stride_kv_token=kv_values.stride(1),
        stride_kv_d=kv_values.stride(2),
        stride_kv_scale_block=kv_scales.stride(0),
        stride_kv_scale_token=kv_scales.stride(1),
        stride_w_row=weight_rows.stride(0),
        stride_w_h=weight_rows.stride(1),
        stride_bt_batch=block_tables.stride(0),
        stride_bt_block=block_tables.stride(1),
        stride_logits_row=logits.stride(0),
        stride_logits_col=logits.stride(1),
        block_size=kv_values.shape[1],
        num_warps=4,
        num_stages=2,
    )
    return logits
