# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.c04_factories import (
    build_c04_decode_case,
    build_c04_mixed_chain_case,
    build_c04_prefill_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--mode", choices=("decode", "prefill", "mixed"), required=True)
    parser.add_argument("--context-tokens", type=int, default=131072)
    parser.add_argument(
        "--decode-request-batch", type=int, choices=(0, 1, 4), default=0
    )
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--num-prefill-tokens", type=int, default=0)
    parser.add_argument(
        "--prefill-request-batch", type=int, choices=(0, 1, 4), default=0
    )
    parser.add_argument("--request-padding", type=int, default=0)
    parser.add_argument("--token-padding", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factories = {
        "decode": build_c04_decode_case,
        "prefill": build_c04_prefill_case,
        "mixed": build_c04_mixed_chain_case,
    }
    case = factories[args.mode](
        {
            "context_tokens": args.context_tokens,
            "decode_request_batch": args.decode_request_batch,
            "draft_tokens": args.draft_tokens,
            "num_prefill_tokens": args.num_prefill_tokens,
            "prefill_request_batch": args.prefill_request_batch,
            "request_padding": args.request_padding,
            "token_padding": args.token_padding,
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
