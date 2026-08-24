"""Batch-size and index-locality sweep for the fused sparse-MLA decode.

Answers the "is there free parallelism" question without perf counters:

* if per-token time stays flat as B grows (CTAs are independent, SMs idle),
  the kernel is per-CTA latency-bound and K-split has headroom;
* window (L2-hot) vs scatter (DRAM-cold) indices show whether memory
  bandwidth limits the kernel.

Run on gserver:

    python benchmarks/kernels/sweep_sparse_mla_decode_batch.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_sparse_mla_decode_configs import time_config
from test_sparse_mla_decode_concurrency import Case


def main() -> None:
    print("batch sweep (topk 128+128, config 8,32,8,2, CUDA-graph replay):")
    print(f"{'B':>4} {'total_us':>9} {'per_token_us':>12}")
    for b in [1, 2, 4, 8, 16, 32, 64, 128]:
        c = Case(
            8000 + b,
            B=b,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            lens_mode="full",
            score_scale=500.0,
            exp_range=0,
        )
        t = time_config(c, 8, 32, 8, 2)
        print(f"{b:>4} {t:>9.1f} {t / b:>12.2f}")

    print("index locality A/B (B=8, topk 128+128, config 8,32,8,2):")
    for mode in ["window", "scatter"]:
        c = Case(
            9000 if mode == "window" else 9001,
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            lens_mode="full",
            score_scale=500.0,
            exp_range=0,
        )
        num_tokens = c.swa_nope.shape[0]
        for name in ["swa_idx", "extra_idx"]:
            idx = getattr(c, name)
            if mode == "scatter":
                idx.copy_(torch.randint(0, num_tokens, idx.shape, device=c.dev))
            else:
                ends = torch.randint(
                    128, num_tokens - 128, (c.B,), device=c.dev
                )
                idx.copy_(
                    (ends[:, None] + torch.arange(128, device=c.dev)[None, :]).to(
                        torch.int32
                    )
                )
        t = time_config(c, 8, 32, 8, 2)
        print(f"  {mode:>8}: {t:8.1f} us/call")


if __name__ == "__main__":
    main()
