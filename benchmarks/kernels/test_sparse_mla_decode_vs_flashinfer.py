"""Differential check: Triton fused decode vs FlashInfer's sparse-MLA decode.

Same packed fp8_ds_mla cache, indices and lens fed to both operators; outputs
must match within the bf16 value-MMA rounding that the Triton path uses.
FlashInfer (``trtllm_batch_decode_sparse_mla_dsv4``) is the external oracle:
it validates layout interpretation (page addressing, UE8M0 scale section,
extra/sink semantics) that a clean-room torch reference could share mistakes
with.

Run on gserver (small footprint, works alongside a busy GPU):

    python benchmarks/kernels/test_sparse_mla_decode_vs_flashinfer.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sparse_mla_decode_concurrency import (
    Case,
)


def _run_flashinfer(
    c: Case,
    out: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4

    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=c.dev)
    trtllm_batch_decode_sparse_mla_dsv4(
        query=c.q.squeeze(1),
        swa_kv_cache=c.swa_cache.unsqueeze(-2),
        workspace_buffer=workspace,
        sparse_indices=c.swa_idx.unsqueeze(1),
        compressed_kv_cache=c.extra_cache.unsqueeze(-2),
        out=out,
        bmm1_scale=c.scale,
        sinks=c.sinks,
        kv_layout="NHD",
        swa_topk_lens=c.swa_lens,
        extra_sparse_indices=c.extra_idx.unsqueeze(1),
        extra_sparse_topk_lens=c.extra_lens,
    )
    torch.cuda.synchronize()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--tolerance", type=float, default=5e-2)
    args = ap.parse_args()
    assert torch.cuda.is_available()

    specs = [
        # Only shapes in FlashInfer's decode-dsv4 dispatch table: (8, 128/512/
        # 1024); the Triton path additionally supports the other shapes tested
        # in test_sparse_mla_decode_concurrency.py.
        dict(
            B=2,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=512,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=4,
            H=8,
            swa_topk=128,
            extra_topk=1024,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=16,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=16,
            score_scale=500.0,
            exp_range=0,
        ),
    ]
    nfail = 0
    for i in range(args.cases):
        spec = specs[i % len(specs)]
        c = Case(9000 + i * 7, lens_mode="random", **spec)
        out_t = c.run_public()
        out_f = torch.zeros_like(out_t)
        _run_flashinfer(c, out_f)
        d = (out_t.float() - out_f.float()).abs()
        scale = out_f.float().abs().max().item()
        ok = d.max().item() <= args.tolerance * max(scale, 1e-6)
        mx, rel = d.max().item(), d.max().item() / max(scale, 1e-6)
        print(
            f"  case {i} {spec}: max_abs={mx:.4e} max_rel={rel:.2f} "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            nfail += 1
    print(f"triton-vs-flashinfer: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


if __name__ == "__main__":
    main()
