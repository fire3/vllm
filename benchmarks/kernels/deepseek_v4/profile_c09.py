# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.c09_factories import (
    build_c09_dequant_gather_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--gather-len", type=int)
    parser.add_argument("--request-batch", type=int, choices=(1, 4), default=1)
    parser.add_argument(
        "--block-size", type=int, choices=(2, 16, 64, 128, 256), default=64
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seq-jitter", type=int, default=0)
    parser.add_argument("--gather-jitter", type=int, default=0)
    parser.add_argument("--output-padding", type=int, default=0)
    parser.add_argument("--use-gather-lens", action="store_true")
    parser.add_argument("--regime", default="compressed-full")
    parser.add_argument(
        "--baseline-mode",
        choices=("triton", "cutedsl-fixed"),
        default="triton",
    )
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factory_args = {
        "seq_len": args.seq_len,
        "request_batch": args.request_batch,
        "block_size": args.block_size,
        "offset": args.offset,
        "seq_jitter": args.seq_jitter,
        "gather_jitter": args.gather_jitter,
        "output_padding": args.output_padding,
        "use_gather_lens": args.use_gather_lens,
        "regime": args.regime,
        "baseline_mode": args.baseline_mode,
    }
    if args.gather_len is not None:
        factory_args["gather_len"] = args.gather_len
    case = build_c09_dequant_gather_case(factory_args)
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
