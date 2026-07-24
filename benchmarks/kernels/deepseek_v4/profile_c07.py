# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.c07_factories import (
    build_c07_uniform_decode_case,
    build_c07_uniform_decode_compression_chain_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--mode", choices=("standalone", "chain"), default="chain")
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--request-batch", type=int, choices=(1, 4), default=4)
    parser.add_argument("--decode-len", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--block-size", type=int, choices=(64, 256), default=256)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--context-jitter", type=int, default=0)
    parser.add_argument("--compress-ratio", type=int, choices=(1, 4, 128), default=4)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factory = (
        build_c07_uniform_decode_compression_chain_case
        if args.mode == "chain"
        else build_c07_uniform_decode_case
    )
    case = factory(
        {
            "context_tokens": args.context_tokens,
            "request_batch": args.request_batch,
            "decode_len": args.decode_len,
            "block_size": args.block_size,
            "max_model_len": args.max_model_len,
            "context_jitter": args.context_jitter,
            "compress_ratio": args.compress_ratio,
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
