"""Config + topk sweep for the fused sparse-MLA decode kernel.

Times the kernel via CUDA-graph replay (pure GPU time, no launch overhead)
across BLOCK_H/BLOCK_K/num_warps and SWA/extra topk combinations, to decide
whether config tuning suffices or a K-split rewrite is warranted.

Run on gserver (small footprint):

    python benchmarks/kernels/sweep_sparse_mla_decode_configs.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sparse_mla_decode_concurrency import Case

from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
    _GROUP_DIM,
    _NOPE_DIM,
    _SCALE_STRIDE,
    _TOKEN_DATA_STRIDE,
    _paged_cache_views,
    _tiled_sparse_prefill_kernel,
)


def _launch_args(c: Case, BH: int, BK: int):
    swa_fp8, swa_u8, swa_bf16, swa_lo, _, swa_pb = _paged_cache_views(c.swa_cache)
    extra_fp8, extra_u8, extra_bf16, extra_lo, _, extra_pb = _paged_cache_views(
        c.extra_cache
    )
    swa_page = c.swa_cache.shape[1]
    extra_page = c.extra_cache.shape[1]
    swa_csr = c.swa_idx.reshape(-1).contiguous()
    extra_csr = c.extra_idx.reshape(-1).contiguous()
    swa_indptr = torch.arange(c.B + 1, dtype=torch.int32, device=c.dev) * c.swa_topk
    extra_indptr = torch.arange(c.B + 1, dtype=torch.int32, device=c.dev) * c.extra_topk
    lse = torch.empty(c.B, c.H, dtype=torch.float32, device=c.dev)
    return dict(
        Q_ptr=c.q.squeeze(1),
        O_ptr=c.out,
        LSE_ptr=lse,
        swa_cache_fp8_ptr=swa_fp8,
        swa_cache_uint8_ptr=swa_u8,
        swa_cache_bf16_ptr=swa_bf16,
        swa_idx_ptr=swa_csr,
        swa_indptr_ptr=swa_indptr,
        extra_cache_fp8_ptr=extra_fp8,
        extra_cache_uint8_ptr=extra_u8,
        extra_cache_bf16_ptr=extra_bf16,
        extra_idx_ptr=extra_csr,
        extra_indptr_ptr=extra_indptr,
        sm_scale=c.scale,
        swa_page_size=swa_page,
        swa_page_bytes=c.swa_cache.stride(0),
        swa_layer_off=swa_lo,
        swa_scale_off=swa_page * _TOKEN_DATA_STRIDE,
        extra_page_size=extra_page,
        extra_page_bytes=c.extra_cache.stride(0),
        extra_layer_off=extra_lo,
        extra_scale_off=extra_page * _TOKEN_DATA_STRIDE,
        H=c.H,
        stride_qb=c.q.stride(0),
        stride_qh=c.q.stride(2),
        stride_ob=c.out.stride(0),
        stride_oh=c.out.stride(1),
        HAS_EXTRA=True,
        GROUP_DIM=_GROUP_DIM,
        NOPE_DIM=_NOPE_DIM,
        TOKEN_DATA_STRIDE=_TOKEN_DATA_STRIDE,
        SCALE_STRIDE=_SCALE_STRIDE,
    )


def time_config(c: Case, BH: int, BK: int, nw: int, ns: int, iters: int = 200) -> float:
    import triton

    grid = (c.B, triton.cdiv(c.H, BH))
    args = _launch_args(c, BH, BK)
    # Compile + warmup on the current stream.
    _tiled_sparse_prefill_kernel[grid](
        **args, BLOCK_H=BH, BLOCK_K=BK, num_warps=nw, num_stages=ns
    )
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _tiled_sparse_prefill_kernel[grid](
            **args, BLOCK_H=BH, BLOCK_K=BK, num_warps=nw, num_stages=ns
        )
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters


def main() -> None:
    configs = [
        (16, 16, 8, 2),  # baseline
        (8, 16, 8, 2),
        (8, 16, 4, 2),
        (16, 32, 8, 2),
        (8, 32, 8, 2),
        (8, 32, 4, 2),
        (8, 32, 8, 3),
    ]
    shapes = [(128, 128), (128, 512), (128, 1024), (128, 2048)]
    print("config sweep (us/call, B=8 H=8):")
    print(f"{'BH,BK,warps,stages':>20} | " + " | ".join(f"swa+{e}" for _, e in shapes))
    for BH, BK, nw, ns in configs:
        row = []
        for swa_t, extra_t in shapes:
            c = Case(
                6000 + swa_t + extra_t,
                B=8,
                H=8,
                swa_topk=swa_t,
                extra_topk=extra_t,
                swa_page=64,
                extra_page=64,
                lens_mode="full",
                score_scale=500.0,
                exp_range=0,
            )
            row.append(f"{time_config(c, BH, BK, nw, ns):8.1f}")
        print(f"({BH:>2},{BK:>2},{nw:>1},{ns}) | " + " | ".join(row))

    print("topk scaling (baseline 16,16,8,2):")
    for swa_t, extra_t in shapes:
        c = Case(
            7000 + swa_t + extra_t,
            B=8,
            H=8,
            swa_topk=swa_t,
            extra_topk=extra_t,
            swa_page=64,
            extra_page=64,
            lens_mode="full",
            score_scale=500.0,
            exp_range=0,
        )
        tiles = (swa_t + extra_t) // 16
        t = time_config(c, 16, 16, 8, 2)
        print(
            f"  topk {swa_t}+{extra_t}: {t:8.1f} us  ({tiles:3d} tiles, "
            f"{t / tiles:5.2f} us/tile)"
        )


if __name__ == "__main__":
    main()
