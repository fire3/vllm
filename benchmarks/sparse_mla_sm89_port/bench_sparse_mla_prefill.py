# SPDX-License-Identifier: Apache-2.0
"""SM120 sparse MLA prefill benchmark: FlashInfer orchestrator vs Triton."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_sparse_mla_decode import run_flashinfer, run_triton  # noqa: E402
from common import (  # noqa: E402
    HEAD_DIM,
    SWA_WIDTH,
    bench_cuda,
    make_footer_kv_cache,
    make_queries,
    make_topk_indices,
)


def run_case(
    *,
    num_tokens: int,
    num_heads: int,
    num_slots: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    kv_cache = make_footer_kv_cache(num_slots)
    q = make_queries(num_tokens, num_heads)
    indices = make_topk_indices(num_tokens, num_slots, SWA_WIDTH)
    lens = torch.full((num_tokens,), SWA_WIDTH, device="cuda", dtype=torch.int32)
    workspace = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    scale = HEAD_DIM**-0.5

    out_flashinfer = run_flashinfer(q, kv_cache, indices, lens, workspace, scale)
    out_triton = run_triton(q, kv_cache, indices, lens, scale)
    torch.cuda.synchronize()
    rel_diff = (
        (out_flashinfer.float() - out_triton.float()).abs().max()
        / out_triton.float().abs().max().clamp_min(1e-6)
    ).item()

    flashinfer_ms = bench_cuda(
        lambda: run_flashinfer(q, kv_cache, indices, lens, workspace, scale),
        warmup=warmup,
        iters=iters,
    )
    triton_ms = bench_cuda(
        lambda: run_triton(q, kv_cache, indices, lens, scale),
        warmup=warmup,
        iters=iters,
    )
    return {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "topk": SWA_WIDTH,
        "rel_diff": rel_diff,
        "flashinfer_ms": flashinfer_ms,
        "triton_ms": triton_ms,
        "speedup_triton_over_flashinfer": triton_ms / flashinfer_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--num-slots", type=int, default=65536)
    parser.add_argument(
        "--out", default="benchmarks/sparse_mla_sm89_port/results_prefill.json"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    token_sweep = (256, 1024) if args.quick else (256, 1024, 4096, 8192)
    warmup = 2 if args.quick else 5
    iters = 3 if args.quick else 20
    results = [
        run_case(
            num_tokens=tokens,
            num_heads=args.num_heads,
            num_slots=max(args.num_slots, tokens + SWA_WIDTH),
            warmup=warmup,
            iters=iters,
        )
        for tokens in token_sweep
    ]
    for row in results:
        print(row)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
