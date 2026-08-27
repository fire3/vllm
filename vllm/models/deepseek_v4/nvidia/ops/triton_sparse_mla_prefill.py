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

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.envs import (
    VLLM_TRITON_SPARSE_MLA_KSPLIT,
    VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

LOG2E = tl.constexpr(1.4426950408889634)

logger = init_logger(__name__)

# DSv4 KV cache layout constants (shared with the decode kernel).
_NOPE_DIM = 448
_ROPE_DIM = 64
_GROUP_DIM = 64
_NOPE_GROUPS = _NOPE_DIM // _GROUP_DIM  # 7
_TOKEN_DATA_STRIDE = 576  # bytes per token in the data section
_SCALE_STRIDE = 8  # bytes per token in the scale section

_PACK_BLOCK = 128
_DEFAULT_BLOCK_H = 8
_DEFAULT_BLOCK_K = 32
_DEFAULT_NUM_WARPS = 8
_KSPLIT_CHUNK = 32
# K-split is only useful when the batch is too small to fill the GPU on its
# own (decode); prefill rows already give (T, 1) parallelism. The cutoff also
# bounds the cached [T, S, H, D] partial buffers: at T=2048/S=68/H=8 the
# bf16 partial out alone would be 1.1 GiB per (T, S) combo, accumulating
# across chunked-prefill sizes and OOMing at high GPU-memory utilization.
_KSPLIT_MAX_T = 64

# A/B toggle: keep K-split inside FULL graph captures. On by default: the
# K-split grid is now width-independent (fixed S_MAX derived from the max
# buffer row capacity; per-CTA source/chunk selection reads the runtime CSR
# lengths), so a FULL graph is correct for any c128 width up to max-model-len.
# Set VLLM_TRITON_SPARSE_MLA_KSPLIT_FULL=0 to force the fused kernel inside
# FULL captures.
_KSPLIT_FULL = os.environ.get("VLLM_TRITON_SPARSE_MLA_KSPLIT_FULL", "1") != "0"
_DEBUG = os.environ.get("VLLM_TRITON_SPARSE_MLA_DEBUG", "0") == "1"
_DEBUG_SEEN: set[tuple] = set()
_DEBUG_DIFF = os.environ.get("VLLM_TRITON_SPARSE_MLA_DEBUG_DIFF", "0") == "1"
_DEBUG_DIFF_STATES: dict[tuple, dict[int, torch.Tensor]] = {}
_DEBUG_DIFF_FIRED: set[int] = set()
_DEBUG_DIFF_READER_STARTED = [False]
_DEBUG_DIFF_THRESHOLD = 0.15
# Bound the eager scratch caches so they cannot silently grow to gigabytes and
# erode the GPU-memory headroom needed for transient peak allocations.
# Captured-graph buffers (capture token != 0) are never evicted: their
# addresses are baked into the CUDA graphs. Eager entries are recreated on
# demand, so evicting them is safe.
_MAX_SCRATCH_ENTRIES = 64

# Per-process capture-session counter: every CUDA graph capture gets a unique
# token so scratch buffers are keyed per graph instance. Eager execution
# (token 0) never aliases any captured graph's scratch.
_CAPTURE_SESSION = [0]
_PREV_CAPTURING = [False]


def _capture_token() -> int:
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and not _PREV_CAPTURING[0]:
        _CAPTURE_SESSION[0] += 1
    _PREV_CAPTURING[0] = capturing
    return _CAPTURE_SESSION[0] if capturing else 0


def _is_full_graph_capture() -> bool:
    """True when the current call is being captured into a FULL CUDA graph.

    K-split is width-independent (fixed grid/scratch sized to the max buffer
    row capacity, runtime source selection from the CSR indptr), so it is safe
    inside FULL captures. This helper is kept for the opt-out escape hatch
    (``VLLM_TRITON_SPARSE_MLA_KSPLIT_FULL=0``) and for A/B experiments.
    """
    if not torch.cuda.is_current_stream_capturing():
        return False
    try:
        ctx = get_forward_context()
    except Exception:
        return False
    return ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL


def _debug_launch(tag: str, key: tuple) -> None:
    """One-time per-shape launch logging for K-split A/B diagnostics."""
    if not _DEBUG or key in _DEBUG_SEEN:
        return
    _DEBUG_SEEN.add(key)
    logger.info("triton sparse MLA [%s] %s", tag, key)


@triton.jit
def _max_abs_diff_kernel(
    a_ptr,
    b_ptr,
    max_diff_ptr,
    nan_ptr,
    layer_off_ptr,
    layer_off: tl.int64,
    N: tl.int32,
    BLOCK: tl.constexpr,
):
    """Max |a - b| + NaN/Inf count over one [T, H, D] output pair."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    d = tl.abs(a - b)
    m = tl.max(d, axis=0)
    tl.atomic_max(max_diff_ptr, m)
    nan = tl.sum(((a != a) & (b == b)).to(tl.int32), axis=0) + tl.sum(
        ((b != b) & (a == a)).to(tl.int32), axis=0
    )
    tl.atomic_add(nan_ptr, nan)
    tl.store(layer_off_ptr, layer_off)


def _debug_diff_buffers(
    key: tuple,
    layer_off: int,
    T: int,
    H: int,
    D: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(call, layer) fused reference + diff state for K-split A/B."""
    state = _DEBUG_DIFF_STATES.setdefault(key, {})
    if layer_off not in state:
        out_ref = torch.empty(T, H, D, dtype=torch.bfloat16, device=device)
        lse_ref = torch.empty(T, H, dtype=torch.float32, device=device)
        max_diff = torch.zeros(1, dtype=torch.float32, device=device)
        nan_cnt = torch.zeros(1, dtype=torch.int32, device=device)
        layer_off_t = torch.zeros(1, dtype=torch.int64, device=device)
        # Pinned host mirrors updated by a capturable D2H copy inside the
        # graph; the reader samples them without any CUDA interaction, so it
        # can never invalidate an in-progress capture.
        host_state = torch.zeros(3, dtype=torch.float32, pin_memory=True)
        state[layer_off] = (
            out_ref,
            lse_ref,
            max_diff,
            nan_cnt,
            layer_off_t,
            host_state,
        )
    return state[layer_off]


def _capture_host_state_copy(
    max_diff: torch.Tensor,
    nan_cnt: torch.Tensor,
    layer_off_t: torch.Tensor,
    host_state: torch.Tensor,
) -> None:
    """Capturable device->host copy of the diff state on the current stream."""
    import ctypes

    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cudart = ctypes.CDLL("libcudart.so.12")
    cudart.cudaMemcpyAsync.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    for src, slot in (
        (max_diff, 0),
        (nan_cnt.to(torch.float32), 1),
        (layer_off_t.to(torch.float32), 2),
    ):
        ret = cudart.cudaMemcpyAsync(
            ctypes.c_void_p(host_state.data_ptr() + slot * 4),
            ctypes.c_void_p(src.data_ptr()),
            ctypes.c_size_t(4),
            ctypes.c_int(2),  # cudaMemcpyDeviceToHost
            ctypes.c_void_p(stream),
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed with code {ret}")


def _start_debug_diff_reader() -> None:
    if not _DEBUG_DIFF or _DEBUG_DIFF_READER_STARTED[0]:
        return
    _DEBUG_DIFF_READER_STARTED[0] = True

    import threading
    import time

    def _run() -> None:
        while True:
            time.sleep(1.0)
            for key, layers in list(_DEBUG_DIFF_STATES.items()):
                for layer_off, bufs in list(layers.items()):
                    host_state = bufs[5]
                    md = float(host_state[0].item())
                    nn = int(round(host_state[1].item()))
                    lo = int(round(host_state[2].item()))
                    if md > _DEBUG_DIFF_THRESHOLD or nn > 0:
                        if layer_off not in _DEBUG_DIFF_FIRED:
                            _DEBUG_DIFF_FIRED.add(layer_off)
                            logger.warning(
                                "MLA K-split vs fused DIVERGENCE: layer_off=%s "
                                "max_abs_diff=%.4f nan=%d key=%s",
                                lo,
                                md,
                                nn,
                                key,
                            )

    threading.Thread(target=_run, daemon=True).start()


@triton.jit
def _tiled_sparse_prefill_ksplit_kernel(
    Q_ptr,  # [T, H, D] bf16
    partial_out_ptr,  # [T, S, H, D] bf16, per-chunk normalized output
    partial_lse_ptr,  # [T, S, H] f32, per-chunk natural-log LSE
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
    stride_pt: tl.int32,  # S*H*D for partial out
    stride_ps: tl.int32,  # H*D
    stride_ph: tl.int32,  # D
    stride_lt: tl.int32,  # S*H for partial lse
    stride_ls: tl.int32,  # H
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    TOKEN_DATA_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    """K-split variant: one (SWA|extra) chunk of ``CHUNK_SIZE`` tokens per CTA.

    Grid: ``(T, S_MAX, ceil(H / BLOCK_H))`` where ``S_MAX`` is the fixed chunk
    count for the maximum possible row width (buffer capacity), independent of
    the runtime width. Each CTA reads the runtime CSR row lengths and picks its
    source (SWA first, then extra) and chunk index from them, so a FULL CUDA
    graph replayed against any runtime width is correct: chunks beyond the
    runtime length are no-ops and the merge masks them by runtime lens.
    """
    t = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_h = tl.program_id(2)

    swa_start = tl.load(swa_indptr_ptr + t)
    swa_end = tl.load(swa_indptr_ptr + t + 1)
    swa_len = swa_end - swa_start
    swa_chunks = tl.cdiv(swa_len, CHUNK_SIZE)
    if HAS_EXTRA:
        extra_start = tl.load(extra_indptr_ptr + t)
        extra_end = tl.load(extra_indptr_ptr + t + 1)
        extra_len = extra_end - extra_start
        extra_chunks = tl.cdiv(extra_len, CHUNK_SIZE)
    else:
        extra_chunks = 0
        extra_start = 0
        extra_end = 0
    total_chunks = swa_chunks + extra_chunks

    if pid_chunk < total_chunks:
        _ksplit_chunk_body(
            Q_ptr,
            partial_out_ptr,
            partial_lse_ptr,
            t,
            pid_h,
            swa_cache_fp8_ptr,
            swa_cache_uint8_ptr,
            swa_cache_bf16_ptr,
            swa_idx_ptr,
            swa_start,
            swa_end,
            extra_cache_fp8_ptr,
            extra_cache_uint8_ptr,
            extra_cache_bf16_ptr,
            extra_idx_ptr,
            extra_start,
            extra_end,
            sm_scale,
            swa_page_size,
            swa_page_bytes,
            swa_layer_off,
            swa_scale_off,
            extra_page_size,
            extra_page_bytes,
            extra_layer_off,
            extra_scale_off,
            H,
            stride_qb,
            stride_qh,
            swa_chunks,
            pid_chunk,
            stride_pt,
            stride_ps,
            stride_ph,
            stride_lt,
            stride_ls,
            CHUNK_SIZE,
            BLOCK_H,
            BLOCK_K,
            GROUP_DIM,
            NOPE_DIM,
            TOKEN_DATA_STRIDE,
            SCALE_STRIDE,
            HAS_EXTRA,
        )


@triton.jit
def _ksplit_chunk_body(
    Q_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    t: tl.int32,
    pid_h: tl.int32,
    swa_cache_fp8_ptr,
    swa_cache_uint8_ptr,
    swa_cache_bf16_ptr,
    swa_idx_ptr,
    swa_start: tl.int32,
    swa_end: tl.int32,
    extra_cache_fp8_ptr,
    extra_cache_uint8_ptr,
    extra_cache_bf16_ptr,
    extra_idx_ptr,
    extra_start: tl.int32,
    extra_end: tl.int32,
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
    swa_chunks: tl.int32,
    pid_chunk: tl.int32,
    stride_pt: tl.int32,
    stride_ps: tl.int32,
    stride_ph: tl.int32,
    stride_lt: tl.int32,
    stride_ls: tl.int32,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    TOKEN_DATA_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
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

    # Select this CTA's source (SWA chunks first, then extra) from the runtime
    # CSR row lengths and its range within the row.
    is_extra = pid_chunk >= swa_chunks
    fp8_p = tl.where(is_extra, extra_cache_fp8_ptr, swa_cache_fp8_ptr)
    uint8_p = tl.where(is_extra, extra_cache_uint8_ptr, swa_cache_uint8_ptr)
    bf16_p = tl.where(is_extra, extra_cache_bf16_ptr, swa_cache_bf16_ptr)
    idx_p = tl.where(is_extra, extra_idx_ptr, swa_idx_ptr)
    page_size = tl.where(is_extra, extra_page_size, swa_page_size)
    page_bytes = tl.where(is_extra, extra_page_bytes, swa_page_bytes)
    layer_off = tl.where(is_extra, extra_layer_off, swa_layer_off)
    scale_off = tl.where(is_extra, extra_scale_off, swa_scale_off)
    start = tl.where(is_extra, extra_start, swa_start)
    end = tl.where(is_extra, extra_end, swa_end)
    chunk_idx = pid_chunk - tl.where(is_extra, swa_chunks, 0)
    length = end - start
    k_begin = chunk_idx * CHUNK_SIZE
    k_end = tl.minimum(k_begin + CHUNK_SIZE, length)

    k_offs = tl.arange(0, BLOCK_K)
    for k_start in tl.range(k_begin, k_end, BLOCK_K):
        pos = k_start + k_offs
        in_range = pos < k_end
        slot = tl.load(idx_p + start + pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0)
        safe_slot = tl.maximum(slot, 0)
        page_ids = (safe_slot // page_size).to(tl.int64)
        page_offs = (safe_slot % page_size).to(tl.int64)
        data_base = (
            page_ids * page_bytes + layer_off + page_offs * TOKEN_DATA_STRIDE
        )
        scale_base = (
            page_ids * page_bytes
            + layer_off
            + scale_off
            + page_offs * SCALE_STRIDE
        )

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
        sc0 = tl.load(uint8_p + scale_base + 0, mask=valid, other=127)
        sc1 = tl.load(uint8_p + scale_base + 1, mask=valid, other=127)
        sc2 = tl.load(uint8_p + scale_base + 2, mask=valid, other=127)
        sc3 = tl.load(uint8_p + scale_base + 3, mask=valid, other=127)
        sc4 = tl.load(uint8_p + scale_base + 4, mask=valid, other=127)
        sc5 = tl.load(uint8_p + scale_base + 5, mask=valid, other=127)
        sc6 = tl.load(uint8_p + scale_base + 6, mask=valid, other=127)

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
        scores_log2 = scores * LOG2E
        tile_max = tl.max(scores_log2, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores_log2 - m_new[:, None])
        p = tl.where(valid[None, :] & h_mask[:, None], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_bf16 = p.to(tl.bfloat16)

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

    # Finalize: normalize and write this chunk's partial output + LSE.
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    has_data = l_i > 0.0
    p_base = t * stride_pt + pid_chunk * stride_ps
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (0 * GROUP_DIM + d_offs)[None, :],
        (acc0 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (1 * GROUP_DIM + d_offs)[None, :],
        (acc1 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (2 * GROUP_DIM + d_offs)[None, :],
        (acc2 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (3 * GROUP_DIM + d_offs)[None, :],
        (acc3 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (4 * GROUP_DIM + d_offs)[None, :],
        (acc4 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (5 * GROUP_DIM + d_offs)[None, :],
        (acc5 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (6 * GROUP_DIM + d_offs)[None, :],
        (acc6 / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    tl.store(
        partial_out_ptr + p_base + h_offs[:, None] * stride_ph
        + (NOPE_DIM + d_offs)[None, :],
        (acc_rope / safe_l[:, None]).to(tl.bfloat16),
        mask=h_mask[:, None] & has_data[:, None],
    )
    lse = tl.where(
        l_i > 0.0,
        m_i / LOG2E + tl.math.log(safe_l),
        float("-inf"),
    )
    tl.store(
        partial_lse_ptr + t * stride_lt + pid_chunk * stride_ls + h_offs,
        lse,
        mask=h_mask & has_data,
    )


@triton.jit
def _merge_ksplit_kernel(
    partial_out_ptr,  # [T, S, H, D] bf16
    partial_lse_ptr,  # [T, S, H] f32
    O_ptr,  # [T, H, D] bf16
    LSE_ptr,  # [T, H] f32
    swa_indptr_ptr,  # [T+1] int32, runtime SWA CSR indptr
    extra_indptr_ptr,  # [T+1] int32, runtime extra CSR indptr
    H: tl.int32,
    stride_pt: tl.int32,
    stride_ps: tl.int32,
    stride_ph: tl.int32,
    stride_lt: tl.int32,
    stride_ls: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    """Flash-decoding merge: combine per-chunk partials via LSE weighting.

    The chunk count and source boundary are derived from the runtime row
    CSR indptr (swa chunks first, then extra), so the merge is independent of
    the captured width and -- critically -- reads the exact same row lengths
    that the ksplit kernel used to write the partials. A stale/oversized lens
    row cannot make the merge trust a chunk the ksplit kernel did not write.
    """
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_d = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Per-row valid chunk range: chunks beyond the runtime row length were not
    # (re)written by this step's ksplit kernel, so they must be masked instead
    # of trusting their (potentially stale) partial LSE.
    swa_len = tl.load(swa_indptr_ptr + t + 1) - tl.load(swa_indptr_ptr + t)
    swa_chunks = tl.cdiv(swa_len, CHUNK_SIZE)
    if HAS_EXTRA:
        extra_len = (
            tl.load(extra_indptr_ptr + t + 1) - tl.load(extra_indptr_ptr + t)
        )
        extra_chunks = tl.cdiv(extra_len, CHUNK_SIZE)
    else:
        extra_chunks = 0
    total_chunks = swa_chunks + extra_chunks

    # Pass 1: max LSE over chunks (per head).
    m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    for s0 in tl.range(0, total_chunks, BLOCK_S):
        s_offs = s0 + tl.arange(0, BLOCK_S)
        smask = s_offs < total_chunks
        is_extra = s_offs >= swa_chunks
        chunk_in_source = tl.where(
            is_extra, s_offs - swa_chunks, s_offs
        )
        valid_chunk = tl.where(
            is_extra,
            chunk_in_source < extra_chunks,
            chunk_in_source < swa_chunks,
        )
        smask = smask & valid_chunk
        lse = tl.load(
            partial_lse_ptr + t * stride_lt + s_offs[:, None] * stride_ls
            + h_offs[None, :],
            mask=smask[:, None] & h_mask[None, :],
            other=float("-inf"),
        )
        m = tl.maximum(m, tl.max(lse, axis=0))
    m = tl.where(h_mask, m, 0.0)

    # Pass 2: weighted sum over chunks.
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    denom = tl.zeros([BLOCK_H], dtype=tl.float32)
    for s0 in tl.range(0, total_chunks, BLOCK_S):
        s_offs = s0 + tl.arange(0, BLOCK_S)
        smask = s_offs < total_chunks
        is_extra = s_offs >= swa_chunks
        chunk_in_source = tl.where(
            is_extra, s_offs - swa_chunks, s_offs
        )
        valid_chunk = tl.where(
            is_extra,
            chunk_in_source < extra_chunks,
            chunk_in_source < swa_chunks,
        )
        smask = smask & valid_chunk
        lse = tl.load(
            partial_lse_ptr + t * stride_lt + s_offs[:, None] * stride_ls
            + h_offs[None, :],
            mask=smask[:, None] & h_mask[None, :],
            other=float("-inf"),
        )
        w = tl.math.exp(lse - m[None, :])
        w = tl.where(smask[:, None] & h_mask[None, :] & (lse > -1e20), w, 0.0)
        po = tl.load(
            partial_out_ptr + t * stride_pt + s_offs[:, None, None] * stride_ps
            + h_offs[None, :, None] * stride_ph + d_offs[None, None, :],
            mask=smask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :],
            other=0.0,
        )
        acc += tl.sum(w[:, :, None] * po.to(tl.float32), axis=0)
        denom += tl.sum(w, axis=0)

    denom_s = tl.where(denom > 0.0, denom, 1.0)
    res = acc / denom_s[:, None]
    o_base = t * stride_ob + h_offs[:, None] * stride_oh
    tl.store(
        O_ptr + o_base + d_offs[None, :],
        res.to(tl.bfloat16),
        mask=h_mask[:, None] & d_mask[None, :],
    )
    out_lse = m + tl.math.log(denom_s)
    out_lse = tl.where(denom > 0.0, out_lse, float("-inf"))
    tl.store(LSE_ptr + t * H + h_offs, out_lse, mask=h_mask)


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
    triton.Config({"BLOCK_H": 8, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_H": 8, "BLOCK_K": 16}, num_warps=8, num_stages=2),
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


_CSR_FLAT_BUFFERS: dict[tuple[int, int, int, int, int], torch.Tensor] = {}


def _evict_eager_scratch(cache: dict) -> None:
    if len(cache) <= _MAX_SCRATCH_ENTRIES:
        return
    for key in list(cache):
        if key[-1] == 0:  # eager-only entries (capture token 0)
            cache.pop(key, None)
            if len(cache) <= _MAX_SCRATCH_ENTRIES:
                return


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
        _capture_token(),
    )
    buf = _CSR_FLAT_BUFFERS.get(key)
    if buf is None:
        buf = torch.empty(capacity, dtype=torch.int32, device=device)
        _CSR_FLAT_BUFFERS[key] = buf
        _evict_eager_scratch(_CSR_FLAT_BUFFERS)
    return buf


def _pack_sparse_rows(
    indices: torch.Tensor,  # [T, W] or [T, 1, W] int32 physical slots
    lens: torch.Tensor,  # [T] int32 valid prefix length per row
    slot: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack dense padded rows into flat CSR lists (flat, indptr).

    ``flat`` is a max-capacity cached buffer (T * buffer-row-width); only the
    valid prefix of each row is written, so no CPU/GPU sync is needed to learn
    the true nnz. The grid covers the buffer row capacity (outer stride of the
    sliced max-width buffer view), not the active width, so a FULL CUDA graph
    packs every runtime row length up to the capacity.
    """
    T = lens.shape[0]
    dense = indices.reshape(T, -1)
    max_width = max(dense.stride(0), dense.shape[1])
    lens32 = lens.reshape(T).to(torch.int32)
    # Defensive clamp: a stale/negative lens row (e.g. a padding row that was
    # not rewritten this step, or a corrupted length from a batch transition)
    # must never make indptr non-monotonic or drive the pack past the flat
    # buffer. Valid lengths are always within [0, max_width] by construction,
    # so this is a no-op for healthy metadata and only sanitizes garbage.
    lens32 = torch.clamp(lens32, min=0, max=max_width)
    indptr = torch.zeros(T + 1, dtype=torch.int32, device=lens.device)
    torch.cumsum(lens32, dim=0, out=indptr[1:])
    flat = _get_csr_flat_buffer(T * max_width, lens.device, slot)
    _pack_sparse_rows_kernel[(T, triton.cdiv(max_width, _PACK_BLOCK))](
        dense,
        lens32,
        flat,
        indptr,
        dense.stride(0),
        BLOCK=_PACK_BLOCK,
    )
    return flat, indptr


_KSPLIT_PARTIAL_BUFFERS: dict[tuple[int, int, int, int, int, int], tuple] = {}


def _get_ksplit_partial_buffers(
    T: int,
    S: int,
    H: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static [T, S, H, D] partial buffers for the K-split path."""
    stream = torch.cuda.current_stream(device)
    key = (
        device.index if device.index is not None else 0,
        T,
        S,
        H,
        stream.cuda_stream,
        _capture_token(),
    )
    buf = _KSPLIT_PARTIAL_BUFFERS.get(key)
    if buf is None:
        d = _NOPE_DIM + _ROPE_DIM
        partial_out = torch.zeros(
            T, S, H, d, dtype=torch.bfloat16, device=device
        )
        # Initialize the LSE slab to -inf so a chunk that is ever (mistakenly)
        # read outside the runtime write range contributes zero weight instead
        # of stale finite logits. The merge masks by runtime CSR lengths, so
        # this is belt-and-suspenders, but it makes a logic hole benign.
        partial_lse = torch.full(
            (T, S, H), float("-inf"), dtype=torch.float32, device=device
        )
        _KSPLIT_PARTIAL_BUFFERS[key] = (partial_out, partial_lse)
        _evict_eager_scratch(_KSPLIT_PARTIAL_BUFFERS)
        buf = (partial_out, partial_lse)
    return buf


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

    use_ksplit = VLLM_TRITON_SPARSE_MLA_KSPLIT and T <= _KSPLIT_MAX_T
    if use_ksplit and _is_full_graph_capture() and not _KSPLIT_FULL:
        use_ksplit = False
    _debug_launch(
        "ksplit" if use_ksplit else "fused",
        (
            T,
            H,
            has_extra,
            _is_full_graph_capture(),
            _KSPLIT_FULL,
        ),
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
    if use_ksplit:
        # K-split path: (T, S_MAX, H-blocks) CTAs each scan one 32-token chunk
        # of one source, then a small merge kernel combines the partials with
        # flash-decoding LSE weighting. S_MAX is the fixed chunk count for the
        # max buffer row capacity (width-independent): every CTA selects its
        # source/chunk from the runtime CSR lengths, so FULL-graph replay is
        # correct for any runtime c128 width. Default on for decode-sized
        # batches; VLLM_TRITON_SPARSE_MLA_KSPLIT=0 (or T > _KSPLIT_MAX_T)
        # falls back to the single-CTA fused kernel.
        swa_max_width = max(
            swa_indices.reshape(T, -1).stride(0),
            swa_indices.reshape(T, -1).shape[1],
        )
        extra_max_width = (
            max(
                extra_indices.reshape(T, -1).stride(0),
                extra_indices.reshape(T, -1).shape[1],
            )
            if has_extra
            else 0
        )
        S = (
            triton.cdiv(swa_max_width, _KSPLIT_CHUNK)
            + triton.cdiv(extra_max_width, _KSPLIT_CHUNK)
        )
        D = _NOPE_DIM + _ROPE_DIM
        partial_out, partial_lse = _get_ksplit_partial_buffers(T, S, H, q.device)
        grid = (T, S, triton.cdiv(H, _DEFAULT_BLOCK_H))
        _tiled_sparse_prefill_ksplit_kernel[grid](
            Q_ptr=q,
            partial_out_ptr=partial_out,
            partial_lse_ptr=partial_lse,
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
            stride_pt=S * H * D,
            stride_ps=H * D,
            stride_ph=D,
            stride_lt=S * H,
            stride_ls=H,
            CHUNK_SIZE=_KSPLIT_CHUNK,
            BLOCK_H=_DEFAULT_BLOCK_H,
            BLOCK_K=_DEFAULT_BLOCK_K,
            GROUP_DIM=_GROUP_DIM,
            NOPE_DIM=_NOPE_DIM,
            TOKEN_DATA_STRIDE=_TOKEN_DATA_STRIDE,
            SCALE_STRIDE=_SCALE_STRIDE,
            HAS_EXTRA=has_extra,
            num_warps=_DEFAULT_NUM_WARPS,
            num_stages=2,
        )
        _merge_ksplit_kernel[(T, triton.cdiv(H, 8), triton.cdiv(D, 128))](
            partial_out,
            partial_lse,
            out,
            lse,
            swa_indptr,
            extra_indptr if has_extra else swa_indptr,
            H=H,
            stride_pt=S * H * D,
            stride_ps=H * D,
            stride_ph=D,
            stride_lt=S * H,
            stride_ls=H,
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            D=D,
            BLOCK_H=8,
            BLOCK_S=8,
            BLOCK_D=128,
            CHUNK_SIZE=_KSPLIT_CHUNK,
            HAS_EXTRA=has_extra,
            num_warps=8,
        )
        if _DEBUG_DIFF:
            _start_debug_diff_reader()
            stream = torch.cuda.current_stream(q.device)
            diff_key = (
                q.device.index if q.device.index is not None else 0,
                T,
                H,
                stream.cuda_stream,
                _capture_token(),
            )
            (
                out_ref,
                lse_ref,
                max_diff,
                nan_cnt,
                layer_off_t,
                host_state,
            ) = (
                _debug_diff_buffers(
                    diff_key,
                    int(swa_layer_off),
                    T,
                    H,
                    D,
                    q.device,
                )
            )
            fused_grid = (T, triton.cdiv(H, _DEFAULT_BLOCK_H))
            ref_args = dict(launch_args)
            ref_args["O_ptr"] = out_ref
            ref_args["LSE_ptr"] = lse_ref
            _tiled_sparse_prefill_kernel[fused_grid](
                **ref_args,
                BLOCK_H=_DEFAULT_BLOCK_H,
                BLOCK_K=_DEFAULT_BLOCK_K,
                num_warps=_DEFAULT_NUM_WARPS,
                num_stages=2,
            )
            _max_abs_diff_kernel[
                (triton.cdiv(T * H * D, 1024),)
            ](
                out,
                out_ref,
                max_diff,
                nan_cnt,
                layer_off_t,
                layer_off=int(swa_layer_off),
                N=T * H * D,
                BLOCK=1024,
            )
            _capture_host_state_copy(max_diff, nan_cnt, layer_off_t, host_state)
    else:
        grid = (T, triton.cdiv(H, _DEFAULT_BLOCK_H))
        kernel = (
            _TILED_SPARSE_PREFILL_AUTOTUNED
            if VLLM_TRITON_SPARSE_MLA_PREFILL_AUTOTUNE
            else _tiled_sparse_prefill_kernel
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
