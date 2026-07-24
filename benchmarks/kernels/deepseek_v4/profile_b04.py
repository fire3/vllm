# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.b04_factories import (
    build_b04_indexer_fp8_case,
    build_b04_indexer_fp8_save_chain_case,
    build_b05_indexer_mxfp4_case,
    build_b05_indexer_mxfp4_save_chain_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--position-offset", type=int, default=0)
    parser.add_argument("--include-state-store", action="store_true")
    parser.add_argument("--cache-format", choices=("fp8", "mxfp4"), default="fp8")
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_format == "mxfp4":
        factory = (
            build_b05_indexer_mxfp4_save_chain_case
            if args.include_state_store
            else build_b05_indexer_mxfp4_case
        )
    else:
        factory = (
            build_b04_indexer_fp8_save_chain_case
            if args.include_state_store
            else build_b04_indexer_fp8_case
        )
    case = factory(
        {
            "num_tokens": args.num_tokens,
            "position_offset": args.position_offset,
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
