# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA decode kernel ported from SGLang's SM120 FlashMLA
implementation (``sglang/kernels/ops/attention/flash_mla_sm120_triton.py``).

The kernel implements the DeepSeek V4 (Flash) sparse-MLA decode path entirely
in Triton: for each (batch, head) pair it gathers the top-k KV tokens pointed
to by the sparse index matrix, computes QK scores in tiles with an online
softmax, and accumulates K=V directly -- no FlashInfer/CuTe-DSL dependency.

It consumes the DSv4 ``fp8_ds_mla`` packed page layout (which vLLM and SGLang
share):

  Per 64-token page (576B alignment):
    [0, 64*576)              token data; each token has 448 FP8 NoPE + 128B
                             BF16 RoPE (64 values)
    [64*576, 64*576+64*8)    scales; each token has 8 UE8M0 uint8 exponents
                             (7 real groups of 64 + 1 padding)

Only the decode path (one query token per batch row) is ported; DSv4 prefill
continues to use the FlashInfer launcher.
"""

from typing import Optional, Tuple

import os
import threading
import time

import torch

from vllm.envs import (
    VLLM_TRITON_SPARSE_MLA_DECODE_AUTOTUNE,
    VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY,
)
from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
    triton_sparse_mla_prefill_vllm,
)
from vllm.triton_utils import tl, triton

logger = None  # bound lazily to avoid import cycles

LOG2E = tl.constexpr(1.4426950408889634)

# DSv4 KV cache layout constants.
_NOPE_DIM = 448
_ROPE_DIM = 64
_TOKEN_DATA_STRIDE = 576  # bytes per token in data section
_SCALE_STRIDE = 8  # bytes per token in scale section

# ---------------------------------------------------------------------------
# Decode-time KV/input watchdog (VLLM_TRITON_SPARSE_MLA_WATCHDOG=1).
#
# Every decode step, for each batch row, a tiny Triton kernel checksums the
# compressed (c128a) KV tokens that the row attends to (excluding the newest
# compressed slot, which legitimately changes as the sequence grows) and
# validates the sparse index rows for bounds. Results land in a persistent
# device buffer that a background poller reads through a pinned host mirror
# updated by a capturable D2H copy -- the poller never touches CUDA, so it
# cannot perturb an in-progress capture or replay.
#
# The poller flags:
#   * KV fingerprint change while the row's compressed length (L) is
#     unchanged -- i.e. previously written KV content mutated under the row
#     (block reuse / COW / concurrent-write corruption), and
#   * out-of-range sparse indices (metadata corruption).
# ---------------------------------------------------------------------------
_WATCHDOG = os.environ.get("VLLM_TRITON_SPARSE_MLA_WATCHDOG", "0") == "1"
_WATCHDOG_MAX_ROWS = 64  # >= max_num_seqs
_WATCHDOG_DEV: dict[int, dict] = {}
_WATCHDOG_HOST: dict[int, dict] = {}
_WATCHDOG_READER_STARTED = [False]
_WATCHDOG_PREV: dict[int, dict] = {}
_WATCHDOG_LAST_VIOL: dict[int, int] = {}
_WATCHDOG_CORRUPT_LOG = 0
# Must be a Triton-visible constexpr: the watchdog kernel indexes records with
# it. (fp0, fp1, L, layer_off, row, computed, viol, rsv, 8 sample slots)
_WATCHDOG_REC_W = tl.constexpr(16)


@triton.jit
def _tiled_sparse_decode_kernel(
    # Q: [B, H, D] bf16
    Q_ptr,
    # Paged KV cache -- three typed views of same underlying memory
    cache_fp8_ptr,  # float8_e4m3fn flat (1 byte/elem) -- for nope
    cache_uint8_ptr,  # uint8 flat (1 byte/elem) -- for scales
    cache_bf16_ptr,  # bfloat16 flat (2 bytes/elem) -- for rope
    # Indices: [B, topk] int32
    indices_ptr,
    # Valid lengths: [B] int32
    topk_len_ptr,
    # Output: [B, H, D] bf16 and LSE: [B, H] float32
    O_ptr,
    LSE_ptr,
    # Scalars
    sm_scale: tl.float32,
    page_size: tl.int32,
    page_bytes: tl.int64,
    layer_offset: tl.int64,  # bytes from the flat storage base to this layer's region
    scale_section_off: tl.int64,  # page_size * 576
    H: tl.int32,
    topk: tl.int32,
    topk_rounded: tl.int32,  # for autotune key
    has_topk_len: tl.constexpr,
    # Strides
    stride_qb: tl.int32,
    stride_qh: tl.int32,
    stride_ob: tl.int32,
    stride_oh: tl.int32,
    stride_ib: tl.int32,  # indices batch stride
    # Constexprs
    NOPE_PAD: tl.constexpr,  # 512 (padded from 448)
    ROPE_DIM: tl.constexpr,  # 64
    NOPE_DIM_RT: tl.int32,  # 448 (runtime, for masking)
    BLOCK_T: tl.constexpr,  # tokens per tile (16 or 32)
):
    """Tiled sparse decode: vectorized gather + QK + softmax + V accumulation.

    Grid: (B, H) -- one block per (batch, head) pair.
    Each block processes all topk tokens in tiles of BLOCK_T.
    """
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    # ---- Load Q for this (batch, head) ----
    q_base = bid * stride_qb + hid * stride_qh
    nope_offs = tl.arange(0, NOPE_PAD)  # [512]
    nope_mask = nope_offs < NOPE_DIM_RT  # [512], True for [0:448]
    rope_offs = tl.arange(0, ROPE_DIM)  # [64]

    q_nope = tl.load(Q_ptr + q_base + nope_offs, mask=nope_mask, other=0.0)
    q_nope = q_nope.to(tl.float32) * sm_scale
    q_rope = tl.load(Q_ptr + q_base + NOPE_DIM_RT + rope_offs)
    q_rope = q_rope.to(tl.float32) * sm_scale

    # ---- Valid token count ----
    valid_topk = topk
    if has_topk_len:
        valid_topk = tl.load(topk_len_ptr + bid).to(tl.int32)
        valid_topk = tl.minimum(valid_topk, topk)

    # ---- Online softmax state (base-2 math for SM120 efficiency) ----
    m_i: tl.float32 = -1e30
    l_i: tl.float32 = 0.0
    acc_nope = tl.zeros([NOPE_PAD], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    # ---- Precompute constant index vectors ----
    group_ids = (nope_offs // 64).to(tl.int64)  # [NOPE_PAD], scale group for each dim
    t_offs = tl.arange(0, BLOCK_T)  # [BLOCK_T], token offsets within tile

    # ---- Process tokens in tiles of BLOCK_T ----
    for tile_start in range(0, topk, BLOCK_T):
        t_idx = tile_start + t_offs  # [BLOCK_T], global token indices
        t_in_bounds = t_idx < topk  # bounds for index load
        t_valid = t_idx < valid_topk  # bounds for actual processing

        # Load indices for this tile: [BLOCK_T]
        raw_indices = tl.load(
            indices_ptr + bid * stride_ib + t_idx,
            mask=t_in_bounds,
            other=-1,
        )
        idx_valid = t_valid & (raw_indices >= 0)  # [BLOCK_T] mask

        # Page addressing: [BLOCK_T] (clamp for safe addressing of invalid tokens)
        safe_indices = tl.where(idx_valid, raw_indices, tl.zeros_like(raw_indices))
        page_ids = (safe_indices // page_size).to(tl.int64)
        page_offs_t = (safe_indices % page_size).to(tl.int64)
        token_data_bases = (
            page_ids * page_bytes + layer_offset + page_offs_t * 576
        )  # [BLOCK_T] int64

        # ---- Vectorized NOPE FP8 gather: [BLOCK_T, NOPE_PAD] ----
        nope_addrs = token_data_bases[:, None] + nope_offs[None, :].to(tl.int64)
        nope_2d_mask = idx_valid[:, None] & nope_mask[None, :]
        kv_nope_fp8 = tl.load(
            cache_fp8_ptr + nope_addrs,
            mask=nope_2d_mask,
            other=0.0,
        )

        # ---- Vectorized scale gather + dequant: [BLOCK_T, NOPE_PAD] ----
        scale_bases = (
            page_ids * page_bytes + layer_offset + scale_section_off + page_offs_t * 8
        )
        scale_addrs = scale_bases[:, None] + group_ids[None, :]
        scale_raw = tl.load(
            cache_uint8_ptr + scale_addrs,
            mask=nope_2d_mask,
            other=127,
        )
        scale_f32 = tl.math.exp2(scale_raw.to(tl.float32) - 127.0)
        kv_nope = tl.where(nope_2d_mask, kv_nope_fp8.to(tl.float32) * scale_f32, 0.0)

        # ---- Vectorized ROPE BF16 gather: [BLOCK_T, ROPE_DIM] ----
        rope_byte_bases = token_data_bases + 448
        rope_elem_bases = (rope_byte_bases // 2).to(tl.int64)
        rope_addrs = rope_elem_bases[:, None] + rope_offs[None, :].to(tl.int64)
        kv_rope = tl.load(
            cache_bf16_ptr + rope_addrs,
            mask=idx_valid[:, None],
            other=0.0,
        ).to(tl.float32)

        # ---- Vectorized QK scores: [BLOCK_T] ----
        # scores[t] = dot(q_nope, kv_nope[t]) + dot(q_rope, kv_rope[t])
        scores = tl.sum(q_nope[None, :] * kv_nope, axis=1) + tl.sum(
            q_rope[None, :] * kv_rope, axis=1
        )
        scores = tl.where(idx_valid, scores, -1e30)

        # ---- Online softmax update (base-2, tile-level) ----
        scores_log2 = scores * LOG2E  # [BLOCK_T]
        tile_max = tl.max(scores_log2)  # scalar
        m_new = tl.maximum(m_i, tile_max)

        alpha = tl.math.exp2(m_i - m_new)  # rescale factor
        p = tl.math.exp2(scores_log2 - m_new)  # [BLOCK_T] attention weights
        p = tl.where(idx_valid, p, 0.0)  # zero out invalid

        l_i = l_i * alpha + tl.sum(p)

        # ---- Vectorized V accumulation (K=V in MLA) ----
        # acc += sum_t(p[t] * kv[t, :]) for both nope and rope
        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * kv_nope, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * kv_rope, axis=0)
        m_i = m_new

    # ---- Normalize output ----
    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    acc_nope = acc_nope / safe_l
    acc_rope = acc_rope / safe_l

    # LSE: convert from log2 back to natural log
    lse = tl.where(l_i > 0.0, m_i / LOG2E + tl.math.log(safe_l), float("-inf"))

    # ---- Store output ----
    o_base = bid * stride_ob + hid * stride_oh
    tl.store(O_ptr + o_base + nope_offs, acc_nope.to(tl.bfloat16), mask=nope_mask)
    tl.store(O_ptr + o_base + NOPE_DIM_RT + rope_offs, acc_rope.to(tl.bfloat16))
    tl.store(LSE_ptr + bid * H + hid, lse)


_TILED_SPARSE_DECODE_CONFIGS = [
    triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 16}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_T": 32}, num_warps=8, num_stages=2),
]

# Autotuned variant mirrors the upstream SGLang operator. vLLM's DSv4 attention
# runs in an eager break around CUDA graph capture, so autotuning would be safe,
# but it adds first-call latency and a runtime benchmark. It is therefore
# opt-in via ``VLLM_TRITON_SPARSE_MLA_DECODE_AUTOTUNE=1``; the default path
# uses the fixed mid-range config below.
_TILED_SPARSE_DECODE_AUTOTUNED = triton.autotune(
    configs=_TILED_SPARSE_DECODE_CONFIGS,
    key=["topk_rounded"],
)(_tiled_sparse_decode_kernel)


@triton.jit
def _kv_watchdog_kernel(
    cache_ptr,  # flat uint8 view of the layer's cache backing storage
    indices_ptr,  # [B, topk] int32 physical slot ids (global)
    lens_ptr,  # [B] int32 valid lengths
    rec_ptr,  # [B, 7] f32: (fp0, fp1, L, layer_off, row, computed, viol_code)
    viol_ptr,  # [3] f32: (count, row, code)
    layer_offset,  # int64 byte offset of this layer's region in cache_ptr
    page_size,  # tokens per page
    page_bytes,  # bytes per page (uint8 elements == bytes)
    scale_section_off,  # page_size * 576
    num_slots,  # total physical slots (num_pages * page_size)
    topk,  # row width
    BLOCK: tl.constexpr,
    N_SAMPLES: tl.constexpr,
):
    """Per-row c128a KV fingerprint + index-bounds validation."""
    pid = tl.program_id(0)

    L = tl.load(lens_ptr + pid).to(tl.int32)
    L = tl.minimum(L, topk)
    n = L - 1  # exclude the newest compressed slot (legitimately written now)

    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 2, L.to(tl.float32))
    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 3, layer_offset.to(tl.float32))
    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 4, pid.to(tl.float32))
    if n <= 0:
        tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 5, 0.0)
        return

    acc0 = tl.zeros((), dtype=tl.int64)
    acc1 = tl.zeros((), dtype=tl.int64)
    bad = tl.zeros((), dtype=tl.int32)
    for i in range(0, n, BLOCK):
        off = i + tl.arange(0, BLOCK)
        mask = off < n
        idx = tl.load(indices_ptr + pid * topk + off, mask=mask, other=-1)
        valid = mask & (idx >= 0) & (idx < num_slots)
        safe = tl.where(valid, idx, 0)
        page_ids = (safe // page_size).to(tl.int64)
        toffs = (safe % page_size).to(tl.int64)
        base = page_ids * page_bytes + layer_offset + toffs * 576
        b0 = tl.load(cache_ptr + base, mask=valid, other=0).to(tl.int64)
        b1 = tl.load(
            cache_ptr + base + scale_section_off, mask=valid, other=0
        ).to(tl.int64)
        acc0 += tl.sum(b0, axis=0)
        acc1 += tl.sum(b1, axis=0)
        bad += tl.sum((1 - valid.to(tl.int32)) * mask.to(tl.int32), axis=0)

    # Sample a few individual slots (early system-prompt region, mid-history,
    # and the tail of the stable range) so a fingerprint change can be
    # pinpointed to a region. Each sample packs (data_byte, scale_byte).
    sample_slots = tl.zeros([N_SAMPLES], dtype=tl.int32)
    if N_SAMPLES >= 4:
        sample_slots = tl.where(
            tl.arange(0, N_SAMPLES) < 4,
            tl.arange(0, N_SAMPLES),
            sample_slots,
        )
    if N_SAMPLES >= 6:
        sample_slots = tl.where(
            tl.arange(0, N_SAMPLES) == 4,
            (n // 2).to(tl.int32),
            sample_slots,
        )
        sample_slots = tl.where(
            tl.arange(0, N_SAMPLES) == 5,
            (n // 2 + 1).to(tl.int32),
            sample_slots,
        )
    if N_SAMPLES >= 8:
        sample_slots = tl.where(
            tl.arange(0, N_SAMPLES) == 6,
            n - 1,
            sample_slots,
        )
        sample_slots = tl.where(
            tl.arange(0, N_SAMPLES) == 7,
            n - 2,
            sample_slots,
        )
    sample_slots = tl.maximum(sample_slots, 0)
    s_idx = tl.load(
        indices_ptr + pid * topk + sample_slots,
        mask=sample_slots < n,
        other=-1,
    )
    s_valid = (s_idx >= 0) & (s_idx < num_slots)
    s_safe = tl.where(s_valid, s_idx, 0)
    s_pages = (s_safe // page_size).to(tl.int64)
    s_toffs = (s_safe % page_size).to(tl.int64)
    s_base = s_pages * page_bytes + layer_offset + s_toffs * 576
    sb0 = tl.load(cache_ptr + s_base, mask=s_valid, other=0).to(tl.int32)
    sb1 = tl.load(
        cache_ptr + s_base + scale_section_off, mask=s_valid, other=0
    ).to(tl.int32)
    packed = sb0 * 256 + sb1
    tl.store(
        rec_ptr + pid * _WATCHDOG_REC_W + 8 + tl.arange(0, N_SAMPLES),
        packed.to(tl.float32),
        mask=tl.arange(0, N_SAMPLES) < N_SAMPLES,
    )

    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 0, acc0.to(tl.float32))
    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 1, acc1.to(tl.float32))
    tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 5, 1.0)
    if bad > 0:
        tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 6, 2.0)
        tl.atomic_add(viol_ptr + 0, 1.0)
        tl.store(viol_ptr + 1, pid.to(tl.float32))
        tl.store(viol_ptr + 2, 2.0)
    else:
        tl.store(rec_ptr + pid * _WATCHDOG_REC_W + 6, 0.0)


def _watchdog_buffers(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-device persistent watchdog buffers + pinned host mirrors."""
    if device.index is None:
        key = 0
    else:
        key = device.index
    if key in _WATCHDOG_DEV:
        return _WATCHDOG_DEV[key]
    rec_dev = torch.zeros(
        (_WATCHDOG_MAX_ROWS, _WATCHDOG_REC_W), dtype=torch.float32, device=device
    )
    viol_dev = torch.zeros(3, dtype=torch.float32, device=device)
    rec_host = torch.zeros(
        (_WATCHDOG_MAX_ROWS, _WATCHDOG_REC_W), dtype=torch.float32, pin_memory=True
    )
    viol_host = torch.zeros(3, dtype=torch.float32, pin_memory=True)
    bufs = (rec_dev, viol_dev, rec_host, viol_host)
    _WATCHDOG_DEV[key] = bufs
    _WATCHDOG_HOST[key] = {"rec": rec_host, "viol": viol_host}
    return bufs


def _watchdog_capture_host_copy(
    rec_dev: torch.Tensor, viol_dev: torch.Tensor,
    rec_host: torch.Tensor, viol_host: torch.Tensor,
) -> None:
    """Capturable device->host copy of the watchdog state (stream-ordered)."""
    import ctypes

    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError:
        cudart = ctypes.CDLL("libcudart.so.12")
    cudart.cudaMemcpyAsync.restype = ctypes.c_int
    stream = torch.cuda.current_stream().cuda_stream
    nbytes = rec_dev.numel() * rec_dev.element_size()
    ret = cudart.cudaMemcpyAsync(
        ctypes.c_void_p(rec_host.data_ptr()),
        ctypes.c_void_p(rec_dev.data_ptr()),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(2),  # cudaMemcpyDeviceToHost
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        raise RuntimeError(f"watchdog rec copy failed: {ret}")
    nbytes = viol_dev.numel() * viol_dev.element_size()
    ret = cudart.cudaMemcpyAsync(
        ctypes.c_void_p(viol_host.data_ptr()),
        ctypes.c_void_p(viol_dev.data_ptr()),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(2),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        raise RuntimeError(f"watchdog viol copy failed: {ret}")


def _start_watchdog_reader() -> None:
    if not _WATCHDOG or _WATCHDOG_READER_STARTED[0]:
        return
    _WATCHDOG_READER_STARTED[0] = True

    def _run() -> None:
        global _WATCHDOG_CORRUPT_LOG
        time.sleep(1.0)
        while True:
            time.sleep(0.5)
            for key, host in list(_WATCHDOG_HOST.items()):
                rec = host["rec"]
                viol = host["viol"]
                vcnt = int(round(viol[0].item()))
                if vcnt != _WATCHDOG_LAST_VIOL.get(key, 0):
                    _WATCHDOG_LAST_VIOL[key] = vcnt
                    from vllm.logger import init_logger
                    wlog = init_logger(__name__)
                    wlog.warning(
                        "MLA watchdog SPARSE-INDEX VIOLATION count=%d row=%d code=%d",
                        vcnt,
                        int(round(viol[1].item())),
                        int(round(viol[2].item())),
                    )
                prev = _WATCHDOG_PREV.setdefault(key, {})
                for r in range(_WATCHDOG_MAX_ROWS):
                    computed = rec[r, 5].item()
                    if computed <= 0:
                        continue
                    L = int(round(rec[r, 2].item()))
                    layer_off = int(round(rec[r, 3].item()))
                    fp0 = rec[r, 0].item()
                    fp1 = rec[r, 1].item()
                    vcode = int(round(rec[r, 6].item()))
                    # Key on the physical row index too: two different requests
                    # can share the same compressed length in one batch, and
                    # their KV fingerprints legitimately differ. Without the
                    # row in the key, alternating rows masquerade as a
                    # KV-content change.
                    ident = (r, L, layer_off)
                    prev_fp = prev.get(ident)
                    if prev_fp is not None and (
                        prev_fp[0] != fp0 or prev_fp[1] != fp1
                    ):
                        if _WATCHDOG_CORRUPT_LOG < 5000:
                            _WATCHDOG_CORRUPT_LOG += 1
                            from vllm.logger import init_logger
                            wlog = init_logger(__name__)
                            samples = []
                            for k in range(8):
                                ov = rec[r, 8 + k].item()
                                nv = prev_fp[2 + k] if len(prev_fp) > 2 + k else None
                                if nv is not None and ov != nv:
                                    samples.append(
                                        f"s{k}={int(nv)}->{int(ov)}"
                                    )
                            wlog.warning(
                                "MLA watchdog KV-CONTENT CHANGE at same L: "
                                "row=%d L=%d layer_off=%d "
                                "fp=(%.0f,%.0f)->(%.0f,%.0f) %s "
                                "viol_code=%d (count=%d)",
                                r,
                                L,
                                layer_off,
                                prev_fp[0],
                                prev_fp[1],
                                fp0,
                                fp1,
                                " ".join(samples),
                                vcode,
                                _WATCHDOG_CORRUPT_LOG,
                            )
                    prev[ident] = (
                        fp0,
                        fp1,
                        rec[r, 8].item(),
                        rec[r, 9].item(),
                        rec[r, 10].item(),
                        rec[r, 11].item(),
                        rec[r, 12].item(),
                        rec[r, 13].item(),
                        rec[r, 14].item(),
                        rec[r, 15].item(),
                    )

    threading.Thread(target=_run, daemon=True).start()


def _launch_tiled_sparse_decode_kernel(
    q3: torch.Tensor,  # [B, H, D] bf16, contiguous
    cache_fp8: torch.Tensor,  # flat float8_e4m3fn view (NoPE bytes)
    cache_uint8: torch.Tensor,  # flat uint8 view (scale bytes)
    cache_bf16: torch.Tensor,  # flat bfloat16 view (RoPE bytes)
    flat_indices: torch.Tensor,  # [B, topk] int32, contiguous
    topk_length: Optional[torch.Tensor],
    out: torch.Tensor,  # [B, H, D] bf16
    lse: torch.Tensor,  # [B, H] float32
    softmax_scale: float,
    page_size: int,
    page_bytes: int,
    layer_offset: int,
    H: int,
    topk: int,
    topk_rounded: int,
) -> None:
    B = q3.shape[0]
    if VLLM_TRITON_SPARSE_MLA_DECODE_AUTOTUNE:
        kernel = _TILED_SPARSE_DECODE_AUTOTUNED
    else:
        kernel = _tiled_sparse_decode_kernel

    grid = (B, H)
    launch_args = dict(
        Q_ptr=q3,
        cache_fp8_ptr=cache_fp8,
        cache_uint8_ptr=cache_uint8,
        cache_bf16_ptr=cache_bf16,
        indices_ptr=flat_indices,
        topk_len_ptr=(
            topk_length
            if topk_length is not None
            else torch.empty(0, device=q3.device, dtype=torch.int32)
        ),
        O_ptr=out,
        LSE_ptr=lse,
        sm_scale=softmax_scale,
        page_size=page_size,
        page_bytes=int(page_bytes),
        layer_offset=int(layer_offset),
        scale_section_off=int(page_size * _TOKEN_DATA_STRIDE),
        H=H,
        topk=topk,
        topk_rounded=topk_rounded,
        has_topk_len=topk_length is not None,
        stride_qb=q3.stride(0),
        stride_qh=q3.stride(1),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ib=flat_indices.stride(0),
        NOPE_PAD=512,
        ROPE_DIM=_ROPE_DIM,
        NOPE_DIM_RT=_NOPE_DIM,
    )
    if VLLM_TRITON_SPARSE_MLA_DECODE_AUTOTUNE:
        kernel[grid](**launch_args)
    else:
        kernel[grid](
            **launch_args,
            BLOCK_T=16,
            num_warps=8,
            num_stages=2,
        )


def _run_triton_sparse_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    k_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] uint8/fp8
    indices: torch.Tensor,  # [B, ...] int32 global physical token ids
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the tiled Triton sparse decode kernel on one paged KV cache."""
    B, _, H, D = q.shape
    num_pages = k_cache.shape[0]
    page_size = k_cache.shape[1]
    page_bytes = k_cache.stride(0)  # elements = bytes for uint8/fp8
    # The packed fp8_ds_mla cache aliases one shared block slab: each attention
    # layer is a strided view starting at ``storage_offset`` with per-page
    # stride ``stride(0)`` (the slab stride, 576B-padded). Flatten the *whole
    # backing storage* (always in bounds) and carry the layer's byte offset
    # into the kernel so page addressing works for packed/unpacked and
    # padded/unpadded layouts alike.
    layer_offset = k_cache.storage_offset()
    storage_nbytes = k_cache.untyped_storage().nbytes()

    # Flatten indices to [B, topk]
    flat_indices = indices.reshape(B, -1).contiguous()
    topk = flat_indices.shape[1]

    # Create three typed views of the flat cache memory (uint8, fp8, bf16).
    raw_flat = k_cache.as_strided((storage_nbytes,), (1,), storage_offset=0)
    raw_uint8 = raw_flat.view(torch.uint8)
    raw_fp8 = raw_uint8.view(torch.float8_e4m3fn)
    raw_bf16 = raw_uint8.view(torch.bfloat16)

    # Squeeze Q: [B, H, D]
    q3 = q.squeeze(1)
    if not q3.is_contiguous():
        q3 = q3.contiguous()

    out = torch.zeros(B, H, D, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((B, H), float("-inf"), dtype=torch.float32, device=q.device)

    # Round topk for autotune key stability
    topk_rounded = triton.next_power_of_2(topk)

    _launch_tiled_sparse_decode_kernel(
        q3=q3,
        cache_fp8=raw_fp8,
        cache_uint8=raw_uint8,
        cache_bf16=raw_bf16,
        flat_indices=flat_indices,
        topk_length=topk_length,
        out=out,
        lse=lse,
        softmax_scale=softmax_scale,
        page_size=page_size,
        page_bytes=page_bytes,
        layer_offset=layer_offset,
        H=H,
        topk=topk,
        topk_rounded=topk_rounded,
    )

    # Return [B, 1, H, D] and [B, 1, H]
    return out.unsqueeze(1), lse.unsqueeze(1)


def _merge_partial_attn(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention outputs using LSE-weighted combination.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] float32
    """
    max_lse = torch.maximum(lse1, lse2)
    w1 = torch.where(lse1 > -1e20, torch.exp(lse1 - max_lse), torch.zeros_like(lse1))
    w2 = torch.where(lse2 > -1e20, torch.exp(lse2 - max_lse), torch.zeros_like(lse2))
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * out1.float() + w2.unsqueeze(-1) * out2.float()
    ) / total.unsqueeze(-1)
    merged_lse = max_lse + torch.log(total)
    return merged.to(torch.bfloat16), merged_lse


def _apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply attention sink normalization.

    The sink adds to the softmax denominator without contributing output,
    effectively down-weighting all attention scores.

    out: [B, 1, H, D] bf16,  lse: [B, 1, H] f32,  attn_sink: [H] f32
    """
    sink_lse = attn_sink.view(1, 1, -1).expand_as(lse)
    combined_lse = torch.logaddexp(lse, sink_lse)
    w = torch.where(
        lse > -1e20,
        torch.exp(lse - combined_lse),
        torch.zeros_like(lse),
    )
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16), combined_lse


def flash_mla_sparse_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    head_dim_v: int,
    softmax_scale: float,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton sparse MLA decode with optional extra (c4/c128) cache and sink.

    Processes the main (SWA) cache and the extra (compressed top-k) cache
    separately through the same kernel, then merges via LSE-weighted
    combination.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    # Process main cache (SWA)
    out, lse = _run_triton_sparse_decode(
        q,
        k_cache,
        indices,
        topk_length,
        softmax_scale,
    )

    # Process extra cache (c4 / c128) if present
    if extra_k_cache is not None and extra_indices is not None:
        out_extra, lse_extra = _run_triton_sparse_decode(
            q,
            extra_k_cache,
            extra_indices,
            extra_topk_length,
            softmax_scale,
        )
        out, lse = _merge_partial_attn(out, lse, out_extra, lse_extra)

    # Apply attention sink
    if attn_sink is not None:
        out, lse = _apply_attn_sink(out, lse, attn_sink)

    return out, lse.permute(0, 2, 1)


def triton_sparse_mla_decode_vllm(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_kv_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] uint8
    swa_indices: torch.Tensor,  # [B, window_size] int32 physical slots
    swa_lens: Optional[torch.Tensor],  # [B] int32
    extra_kv_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_lens: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],  # [H] f32
    softmax_scale: float,
    out: torch.Tensor,  # [B, H, D] bf16, written in place
    watchdog_c128a: bool = False,
) -> None:
    """vLLM entry point for the DSv4 triton sparse-MLA decode path.

    Defaults to the tiled dual-source fused kernel shared with prefill
    (head-blocked, per-64-dim-group ``tl.dot``, single launch replacing the
    two-pass elementwise kernel + Python LSE merge). The phase-1 path stays
    available via ``VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY=1`` for A/B.
    """
    if _WATCHDOG and watchdog_c128a and extra_kv_cache is not None:
        _start_watchdog_reader()
        B = q.shape[0]
        if B > 0 and extra_indices is not None and extra_lens is not None:
            rec_dev, viol_dev, rec_host, viol_host = _watchdog_buffers(q.device)
            if B < _WATCHDOG_MAX_ROWS:
                # Zero stale rows so the poller never compares dead rows.
                rec_dev[B:] = 0
            flat_indices = extra_indices.reshape(B, -1).contiguous()
            page_size = extra_kv_cache.shape[1]
            num_slots = extra_kv_cache.shape[0] * page_size
            storage_nbytes = extra_kv_cache.untyped_storage().nbytes()
            raw_flat = extra_kv_cache.as_strided(
                (storage_nbytes,), (1,), storage_offset=0
            )
            raw_uint8 = raw_flat.view(torch.uint8)
            _kv_watchdog_kernel[(B,)](
                raw_uint8,
                flat_indices,
                extra_lens,
                rec_dev,
                viol_dev,
                extra_kv_cache.storage_offset(),
                page_size,
                extra_kv_cache.stride(0),
                page_size * _TOKEN_DATA_STRIDE,
                num_slots,
                flat_indices.shape[1],
                BLOCK=64,
                N_SAMPLES=8,
            )
            _watchdog_capture_host_copy(
                rec_dev, viol_dev, rec_host, viol_host
            )

    if VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY:
        out4, _ = flash_mla_sparse_decode_triton(
            q=q,
            k_cache=swa_kv_cache,
            indices=swa_indices,
            topk_length=swa_lens,
            attn_sink=attn_sink,
            head_dim_v=q.shape[-1],
            softmax_scale=softmax_scale,
            extra_k_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        out.copy_(out4.squeeze(1))
        return
    triton_sparse_mla_prefill_vllm(
        q=q.squeeze(1),
        swa_kv_cache=swa_kv_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        out=out,
    )
