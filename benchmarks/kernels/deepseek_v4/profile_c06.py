# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.c06_factories import (
    build_c06_prefill_chunk_metadata_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument("--num-query-tokens", type=int, default=8192)
    parser.add_argument("--request-batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--compress-ratio", type=int, choices=(4, 128), default=4)
    parser.add_argument("--context-jitter", type=int, default=0)
    parser.add_argument("--query-slice-start", type=int, default=0)
    parser.add_argument("--query-slice-tokens", type=int)
    parser.add_argument("--dcp-rank", type=int, default=0)
    parser.add_argument("--dcp-world", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--dcp-interleave", type=int, default=1)
    parser.add_argument(
        "--candidate-block-size", type=int, choices=(128, 256, 512, 1024), default=256
    )
    parser.add_argument("--candidate-num-warps", type=int, choices=(4, 8), default=4)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factory_args = {
        "context_tokens": args.context_tokens,
        "num_query_tokens": args.num_query_tokens,
        "request_batch": args.request_batch,
        "compress_ratio": args.compress_ratio,
        "context_jitter": args.context_jitter,
        "query_slice_start": args.query_slice_start,
        "dcp_rank": args.dcp_rank,
        "dcp_world": args.dcp_world,
        "dcp_interleave": args.dcp_interleave,
        "candidate_block_size": args.candidate_block_size,
        "candidate_num_warps": args.candidate_num_warps,
    }
    if args.query_slice_tokens is not None:
        factory_args["query_slice_tokens"] = args.query_slice_tokens
    case = build_c06_prefill_chunk_metadata_case(factory_args)
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
