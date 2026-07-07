# SPDX-License-Identifier: Apache-2.0
"""SM120 sparse MLA decode benchmark: FlashInfer DSV4 vs Triton reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    HEAD_DIM,
    PAGE_BLOCK_SIZE,
    SWA_WIDTH,
    bench_cuda,
    make_footer_kv_cache,
    make_queries,
    make_topk_indices,
)
import triton_sparse_mla_ref as triton_ref  # noqa: E402


def run_flashinfer(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    workspace: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4

    out = trtllm_batch_decode_sparse_mla_dsv4(
        query=q.unsqueeze(1),
        swa_kv_cache=kv_cache,
        workspace_buffer=workspace,
        sparse_indices=indices,
        swa_topk_lens=lens,
        bmm1_scale=scale,
        kv_layout="NHD",
    )
    return out[:, 0]


def run_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_tokens, num_heads, _ = q.shape
    max_score = torch.full(
        (num_tokens, num_heads), -float("inf"), device=q.device, dtype=torch.float32
    )
    denom = torch.zeros((num_tokens, num_heads), device=q.device, dtype=torch.float32)
    acc = torch.zeros(
        (num_tokens, num_heads, HEAD_DIM), device=q.device, dtype=torch.float32
    )
    head_block_size = triton_ref.sparse_mla_decode_head_block_size(num_tokens)
    triton_ref.accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q,
        kv_cache,
        indices,
        lens,
        PAGE_BLOCK_SIZE,
        scale,
        max_score,
        denom,
        acc,
        head_block_size=head_block_size,
    )
    return acc / denom.unsqueeze(-1)


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
    parser.add_argument("--out", default="benchmarks/sparse_mla_sm89_port/results_decode_h64.json")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    token_sweep = (1, 8, 32, 64) if args.quick else (1, 8, 32, 64, 128, 256)
    warmup = 3 if args.quick else 10
    iters = 5 if args.quick else 50
    results = [
        run_case(
            num_tokens=tokens,
            num_heads=args.num_heads,
            num_slots=args.num_slots,
            warmup=warmup,
            iters=iters,
        )
        for tokens in token_sweep
    ]
    for row in results:
        print(row)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
