# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.a04_factories import build_a04_logits_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--compress-ratio", type=int, choices=(4, 128), default=4)
    parser.add_argument("--request-batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = build_a04_logits_case(
        {
            "context_tokens": args.context_tokens,
            "compress_ratio": args.compress_ratio,
            "request_batch": args.request_batch,
            "draft_tokens": args.draft_tokens,
        }
    )
    provider = case.baseline if args.provider == "baseline" else case.candidate
    with torch.inference_mode():
        for _ in range(3):
            provider.fn()
        torch.accelerator.synchronize()
        for _ in range(args.iterations):
            provider.fn()
        torch.accelerator.synchronize()


if __name__ == "__main__":
    main()
