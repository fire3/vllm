# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton FP8 MQA logits helpers.

This module started as a vendored gfx942 fallback for AITER's Triton
``fp8_mqa_logits`` kernel. It now also provides a CUDA/Triton fallback used by
SM89 DeepSeek V4 sparse attention when DeepGEMM is unavailable.
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

# gfx942 (MI300X) has 64 KiB of LDS per CU. We accept the default
# (BLOCK_KV=128, num_stages=2) tile only when *both* of these hold:
#
# 1. Occupancy gate. With waves_per_eu=2 and num_warps=4 we target two
#    workgroups co-resident on a CU -> per-WG LDS budget = 32 KiB. Triton
#    keeps Q in registers (loop-invariant) and the fp32 scores accumulator
#    in VGPRs (heavy VALU), so only the double-buffered KV tile is
#    expected to live in LDS. A 0.9 safety factor leaves headroom for any
#    LDS overhead the compiler may add.
#
# 2. Hardware ceiling. Defensive upper bound that also counts Q and
#    scores against the 64 KiB CU limit, in case a Triton version (older
#    or future) decides to spill them to LDS. False positives here only
#    shrink the tile; false negatives are JIT-aborts, so we lean
#    conservative.
_GFX942_CU_LDS_BYTES = 64 * 1024
_GFX942_PER_WG_LDS_BUDGET_BYTES = _GFX942_CU_LDS_BYTES * 9 // 20  # ~28.8 KiB


def _gfx942_default_tile_fits_lds(num_heads: int, head_size: int) -> bool:
    """Return True iff (BLOCK_KV=128, num_stages=2) fits in MI300X LDS."""
    BLOCK_KV = 128
    NUM_STAGES = 2
    kv_bytes = head_size * BLOCK_KV * NUM_STAGES
    scores_bytes = num_heads * BLOCK_KV * 4
    q_bytes = num_heads * head_size
    fits_occupancy = kv_bytes < _GFX942_PER_WG_LDS_BUDGET_BYTES
    fits_hardware = q_bytes + kv_bytes + scores_bytes <= _GFX942_CU_LDS_BYTES
    return fits_occupancy and fits_hardware


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

        # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
        scores = tl.dot(q_block, kv_block, input_precision="ieee")
        # Multiply by kv_scales (broadcast along rows)
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

    # [NUM_HEADS, BLOCK_KV] = [NUM_HEADS, HEAD_SIZE] x [HEAD_SIZE, BLOCK_KV]
    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    # Multiply by kv_scales (broadcast along rows)
    scores = scores * kv_scales[None, :]
    # ReLU
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_block
    # [NUM_HEADS, BLOCK_KV] -> [BLOCK_KV, ]
    scores = tl.sum(scores, axis=0)
    # masked store
    in_window = (kv_col_offsets >= start_ind) & (kv_col_offsets < end_ind)
    tl.store(logits_ptrs, scores, mask=in_window)


def fp8_mqa_logits_gfx942(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits on MI300X (gfx942) using the vendored kernel.

    Drop-in replacement for ``aiter.ops.triton.attention.fp8_mqa_logits.
    fp8_mqa_logits`` on MI300X. Selects ``(BLOCK_KV, num_stages)`` based on
    whether the default tile fits within the 64 KiB LDS budget of a gfx942
    CU (see module docstring).

    Args:
        q: Query tensor of shape ``[M, H, D]``, FP8 dtype.
        k_fp8: Key tensor of shape ``[N, D]``, FP8 dtype.
        kv_scales: K scales of shape ``[N]`` (or ``[N, 1]`` -- viewed as
            ``[N]``), float32.
        weights: Per-head weights of shape ``[M, H]``, float32.
        cu_starts: Start indices (inclusive) of shape ``[M]``, int32.
        cu_ends: End indices (exclusive) of shape ``[M]``, int32.

    Returns:
        Logits of shape ``[M, N]``, float32 -- positions outside
        ``[cu_starts[i], cu_ends[i])`` for row ``i`` are pre-filled with
        ``-inf`` so the caller can run a top-k without masking.
    """
    seq_len, num_heads, head_size = q.shape
    seq_len_kv = k_fp8.shape[0]
    assert num_heads & (num_heads - 1) == 0, (
        f"num_heads must be a power of two (got {num_heads})"
    )
    assert head_size & (head_size - 1) == 0, (
        f"head_size must be a power of two (got {head_size})"
    )

    # The kernel walks ``kv_scales`` as a 1-D contiguous array of size N
    # (it indexes by ``kv_scales_ptr + kv_col_offsets``). The vLLM caller
    # passes a ``[N, 4]`` uint8 view-cast-to-float32 which lands as
    # ``[N, 1]`` contiguous -- byte-identical to ``[N]`` -- but flatten
    # explicitly to keep the kernel's pointer arithmetic intent clear.
    kv_scales_1d = kv_scales.reshape(-1)

    # Initialise with -inf so positions outside [cu_starts, cu_ends) read
    # as ``-inf`` after the masked store path -- this matches AITER's
    # ``fp8_mqa_logits`` semantics and is what the top-k consumer expects.
    logits = torch.full(
        (seq_len, seq_len_kv),
        fill_value=-float("inf"),
        dtype=torch.float32,
        device=q.device,
    )

    if _gfx942_default_tile_fits_lds(num_heads, head_size):
        block_kv = 128
        num_stages = 2
    else:
        # DSv4 sparse indexer (NUM_HEADS=64, HEAD_SIZE=128) lands here:
        # default tile spills past gfx942's 64 KiB LDS budget. (64, 1)
        # needs ~33 KiB and clears the per-WG budget with margin.
        block_kv = 64
        num_stages = 1

    # heuristic for MFMA instruction shape, identical to AITER's choice
    matrix_instr_nonkdim = 32
    if seq_len <= 1024:
        matrix_instr_nonkdim = 16

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
        num_warps=4,
        num_stages=num_stages,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )

    return logits


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    k_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_starts: torch.Tensor,
    cu_ends: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits with Triton on CUDA/ROCm.

    This is a lightweight local fallback for environments that do not provide
    DeepGEMM's ``fp8_fp4_mqa_logits`` entry points. It preserves the same output
    contract as the ROCm reference path: a dense ``[M, N]`` fp32 logits matrix.
    The launcher uses ``torch.empty`` rather than an ``-inf`` pre-fill: the
    only consumer (``top_k_per_row_prefill``) reads positions inside
    ``[cu_starts[i], cu_ends[i])`` only, and the kernel writes every position
    in that range, so positions outside the range are never read. At
    prefill-sized batches the ``-inf`` fill was O(M*N) fp32 writes per layer
    (e.g. ~2.5 GB at 25k tokens), pure fixed overhead.
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

    launch_kwargs = dict(num_warps=4, num_stages=2)
    if current_platform.is_rocm():
        matrix_instr_nonkdim = 32 if seq_len > 1024 else 16
        launch_kwargs.update(
            waves_per_eu=2, matrix_instr_nonkdim=matrix_instr_nonkdim
        )

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
        **launch_kwargs,
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

    scores = tl.dot(q_block, kv_block, input_precision="ieee")
    scores = tl.maximum(scores * kv_scales[None, :], 0.0)
    scores = scores * w_block
    scores = tl.sum(scores, axis=0)

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

    The paged KV cache uses the packed FP8 layout shared by the C++
    ``indexer_k_quant_and_cache`` writer and the Triton compressor:
    per block, ``[block_size * head_dim]`` FP8 data bytes followed by
    ``[block_size * 4]`` fp32 scale bytes. Torch exposes the cache as
    ``[num_blocks, block_size, head_dim + 4]``, but the physical byte offsets
    are ``token * head_dim`` for data and ``block_size * head_dim + token * 4``
    for scales, so the launcher reads them with ``as_strided`` over the raw
    bytes. This helper keeps the existing dense-logits interface but replaces
    the Python loop fallback with a single Triton kernel.

    ``max_context_len`` optionally clips the dense-logits width (and the launch
    grid) to the maximum context length present in this batch. Columns beyond
    that bound are never stored by the kernel (``context_limit`` masking) and
    never read by the downstream top-k (``seq_lens`` masking), so skipping them
    is semantics-preserving and removes the fixed O(max_model_len) fill +
    launch cost that would otherwise dominate short-context decode.

    The as-strided byte views are passed straight to the Triton kernel (which
    already takes explicit ``stride_kv_*`` arguments), so no ``.contiguous()``
    copy of the per-layer indexer KV cache is materialized. The full pool can
    be tens of MB per layer on 262k-model-len deployments; copying it every
    decode step (× the number of indexer layers) was pure fixed overhead.
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

    # Clip the logits width to the maximum context length actually present.
    # If the caller does not provide a trusted CPU-side bound (e.g. the
    # scheduler's max_seq_len), fall back to a device max (one small sync).
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
