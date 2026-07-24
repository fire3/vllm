# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.a06_factories import (
    build_a06_kernel_case,
    build_a06_production_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--production-dispatch", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factory = (
        build_a06_production_case if args.production_dispatch else build_a06_kernel_case
    )
    case = factory(
        {
            "num_tokens": args.num_tokens,
            "hidden_size": args.hidden_size,
            "hc_mult": args.hc_mult,
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
