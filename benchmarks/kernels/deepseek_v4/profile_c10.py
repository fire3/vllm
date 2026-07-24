# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.c10_factories import (
    build_c10_combine_topk_swa_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--request-batch", type=int, choices=(1, 4), default=4)
    parser.add_argument("--compress-ratio", type=int, choices=(1, 4, 128), default=4)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--topk-width", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--context-jitter", type=int, default=0)
    parser.add_argument("--query-base", type=int, default=0)
    parser.add_argument("--offset-inputs", action="store_true")
    parser.add_argument(
        "--candidate-mode", choices=("mirror", "fused"), default="mirror"
    )
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = build_c10_combine_topk_swa_case(
        {
            "num_tokens": args.num_tokens,
            "context_tokens": args.context_tokens,
            "request_batch": args.request_batch,
            "compress_ratio": args.compress_ratio,
            "topk": args.topk,
            "topk_width": args.topk_width,
            "window_size": args.window_size,
            "context_jitter": args.context_jitter,
            "query_base": args.query_base,
            "offset_inputs": args.offset_inputs,
            "candidate_mode": args.candidate_mode,
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
