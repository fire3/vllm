"""Microbenchmark + profiling harness for the DSv4 Triton sparse-MLA decode.

Reproduces the packed fp8_ds_mla page layout and the decode shapes used by
``TRITON_MLA_SPARSE_DSV4`` on the SM89 (L40S) branch, then times and (under
ncu/nsys) profiles the tiled dual-source fused kernel that the decode entry
point routes to.

Modes
-----
``--mode fused`` (default): public ``triton_sparse_mla_decode_vllm`` path
    (CSR pack + one fused kernel launch), what runs in production decode.
``--mode fused-direct``: the fused ``_tiled_sparse_prefill_kernel`` only,
    with prebuilt CSR metadata; gives ncu a single-kernel launch stream.
``--mode legacy``: phase-1 elementwise two-kernel path
    (``VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY=1`` equivalent).
``--mode both``: fused then legacy in one process.

Examples
--------
python benchmarks/kernels/profile_sparse_mla_decode.py --mode both
python benchmarks/kernels/profile_sparse_mla_decode.py --mode fused-direct
"""

import argparse
import os

import torch

_PAGE_SIZE = 64
_DATA_STRIDE = 576
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM
_ROPE_BYTES = 2 * _ROPE_DIM
_BLOCK_BYTES = _PAGE_SIZE * 584


def _make_packed_cache(
    num_blocks: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Vectorized packed fp8_ds_mla cache, layout identical to the tests."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_tokens = num_blocks * _PAGE_SIZE
    kv_nope = torch.randn(num_tokens, _NOPE_DIM, generator=gen) * 0.05
    kv_rope = torch.randn(num_tokens, _ROPE_DIM, generator=gen) * 0.05
    exp = torch.randint(-8, 9, (num_tokens, 7), generator=gen)
    scale = torch.exp2(exp.to(torch.float32))

    scale_448 = scale[:, :, None].expand(-1, -1, 64).reshape(-1, _NOPE_DIM)
    fp8 = (kv_nope / scale_448).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    data = torch.zeros(num_tokens, _DATA_STRIDE, dtype=torch.uint8)
    data[:, :_NOPE_DIM] = fp8.view(torch.uint8)
    data[:, _NOPE_DIM:] = kv_rope.to(torch.bfloat16).view(torch.uint8)

    cache_flat = torch.zeros(num_blocks * _BLOCK_BYTES, dtype=torch.uint8)
    cache_flat.view(num_blocks, _BLOCK_BYTES)[:, : _PAGE_SIZE * _DATA_STRIDE] = (
        data.view(num_blocks, _PAGE_SIZE * _DATA_STRIDE)
    )
    scales = (exp + 127).to(torch.uint8)
    scale_view = cache_flat.view(num_blocks, _BLOCK_BYTES)[
        :, _PAGE_SIZE * _DATA_STRIDE : _PAGE_SIZE * (_DATA_STRIDE + 8)
    ].view(num_blocks, _PAGE_SIZE, 8)
    scale_view[:, :, :7] = scales.view(num_blocks, _PAGE_SIZE, 7)
    return cache_flat.to(device)


def _cache_view(cache_flat: torch.Tensor, num_blocks: int) -> torch.Tensor:
    return cache_flat.view(num_blocks, _PAGE_SIZE, 584)


def _make_indices(
    batch: int,
    topk: int,
    num_tokens: int,
    device: torch.device,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Physical slot indices: ``window`` = recent contiguous, ``scatter`` = random."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "window":
        ends = torch.randint(topk, num_tokens - topk + 1, (batch,), generator=gen)
        idx = ends[:, None] + torch.arange(topk)[None, :]
    else:
        idx = torch.randint(0, num_tokens, (batch, topk), generator=gen)
    return idx.to(torch.int32).to(device)


def _bench(
    fn,
    iters: int,
    warmup: int,
    stream: torch.cuda.Stream,
) -> tuple[float, float]:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1e3 / iters
    return us, us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["fused", "fused-direct", "legacy", "both"], default="fused"
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--heads", type=int, default=8, help="padded per-rank head count (TP=8 -> 8)"
    )
    ap.add_argument("--swa-topk", type=int, default=128)
    ap.add_argument("--extra-topk", type=int, default=128)
    ap.add_argument("--pages", type=int, default=1024)
    ap.add_argument("--index-mode", choices=["window", "scatter"], default="window")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--check",
        action="store_true",
        help="assert public vs direct fused outputs match",
    )
    args = ap.parse_args()

    if args.mode == "legacy":
        os.environ["VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY"] = "1"

    import triton  # noqa: E402

    from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_vllm,
    )
    from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
        _DEFAULT_BLOCK_H,
        _DEFAULT_BLOCK_K,
        _DEFAULT_NUM_WARPS,
        _GROUP_DIM,
        _NOPE_DIM,
        _SCALE_STRIDE,
        _TOKEN_DATA_STRIDE,
        _paged_cache_views,
        _tiled_sparse_prefill_kernel,
    )

    device = torch.device(f"cuda:{args.device}")
    assert torch.cuda.is_available()
    torch.manual_seed(args.seed)

    B, H, T = args.batch, args.heads, args.swa_topk
    ET = args.extra_topk
    num_tokens = args.pages * _PAGE_SIZE

    swa_flat = _make_packed_cache(args.pages, device, args.seed)
    extra_flat = _make_packed_cache(args.pages, device, args.seed + 1)
    swa_cache = _cache_view(swa_flat, args.pages)
    extra_cache = _cache_view(extra_flat, args.pages)

    swa_idx = _make_indices(B, T, num_tokens, device, args.index_mode, args.seed)
    extra_idx = _make_indices(B, ET, num_tokens, device, args.index_mode, args.seed + 1)
    swa_lens = torch.full((B,), T, dtype=torch.int32, device=device)
    extra_lens = torch.full((B,), ET, dtype=torch.int32, device=device)
    attn_sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    q = torch.randn(B, 1, H, _D, dtype=torch.bfloat16, device=device) * 0.05
    out = torch.zeros(B, H, _D, dtype=torch.bfloat16, device=device)
    scale = _D**-0.5

    def run_public() -> None:
        triton_sparse_mla_decode_vllm(
            q=q,
            swa_kv_cache=swa_cache,
            swa_indices=swa_idx[:, None, :],
            swa_lens=swa_lens,
            extra_kv_cache=extra_cache,
            extra_indices=extra_idx[:, None, :],
            extra_lens=extra_lens,
            attn_sink=attn_sink,
            softmax_scale=scale,
            out=out,
        )

    # Prebuilt CSR for the direct kernel launch: rows are full so the flat
    # list is just the row-major flatten, and indptr is t*W.
    (
        swa_fp8,
        swa_u8,
        swa_bf16,
        swa_lo,
        swa_ps,
        swa_pb,
    ) = _paged_cache_views(swa_cache)
    (
        extra_fp8,
        extra_u8,
        extra_bf16,
        extra_lo,
        extra_ps,
        extra_pb,
    ) = _paged_cache_views(extra_cache)
    swa_csr = swa_idx.reshape(-1).contiguous()
    extra_csr = extra_idx.reshape(-1).contiguous()
    swa_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * T
    extra_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * ET
    lse = torch.empty(B, H, dtype=torch.float32, device=device)

    def run_direct() -> None:
        grid = (B, triton.cdiv(H, _DEFAULT_BLOCK_H))
        _tiled_sparse_prefill_kernel[grid](
            Q_ptr=q.squeeze(1),
            O_ptr=out,
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
            sm_scale=scale,
            swa_page_size=_PAGE_SIZE,
            swa_page_bytes=swa_cache.stride(0),
            swa_layer_off=swa_lo,
            swa_scale_off=_PAGE_SIZE * _TOKEN_DATA_STRIDE,
            extra_page_size=_PAGE_SIZE,
            extra_page_bytes=extra_cache.stride(0),
            extra_layer_off=extra_lo,
            extra_scale_off=_PAGE_SIZE * _TOKEN_DATA_STRIDE,
            H=H,
            stride_qb=q.stride(0),
            stride_qh=q.stride(2),
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            HAS_EXTRA=True,
            BLOCK_H=_DEFAULT_BLOCK_H,
            BLOCK_K=_DEFAULT_BLOCK_K,
            GROUP_DIM=_GROUP_DIM,
            NOPE_DIM=_NOPE_DIM,
            TOKEN_DATA_STRIDE=_TOKEN_DATA_STRIDE,
            SCALE_STRIDE=_SCALE_STRIDE,
            num_warps=_DEFAULT_NUM_WARPS,
            num_stages=2,
        )
        # The public entry applies the sink in Python after the kernel; mirror
        # it here so ``--check`` and per-call timings are comparable.
        combined = torch.logaddexp(lse, attn_sink.view(1, -1).expand_as(lse))
        w = torch.where(
            lse > -1e20,
            torch.exp(lse - combined),
            torch.zeros_like(lse),
        )
        out.copy_((out.float() * w.unsqueeze(-1)).to(torch.bfloat16))

    if args.check:
        out.zero_()
        run_public()
        ref = out.clone()
        out.zero_()
        run_direct()
        torch.cuda.synchronize()
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-3), (
            "direct kernel output diverges from public path"
        )
        print(
            f"[check] direct == public (max abs diff "
            f"{(out - ref).float().abs().max().item():.3e})"
        )

    stream = torch.cuda.Stream(device=device)
    print(
        f"torch={torch.__version__} triton={triton.__version__} "
        f"gpu={torch.cuda.get_device_name(device)}"
    )
    print(
        f"B={B} H={H} D={_D} swa_topk={T} extra_topk={ET} "
        f"pages={args.pages} index={args.index_mode}"
    )

    modes = {"fused": run_public, "fused-direct": run_direct, "legacy": run_public}
    labels = {
        "fused": "fused(public)",
        "fused-direct": "fused-direct(kernel)",
        "legacy": "legacy(2-kernel)",
    }
    if args.mode == "both":
        modes = {"fused": run_public, "fused-direct": run_direct}
    for mode, fn in modes.items():
        if args.mode not in (mode, "both"):
            continue
        us, _ = _bench(fn, args.iters, args.warmup, stream)
        per_token = us / B
        print(
            f"{labels[mode]:>22}: {us:8.2f} us/call  "
            f"{per_token:6.2f} us/token  ({B / us * 1e6:7.1f} calls/s)"
        )


if __name__ == "__main__":
    main()
