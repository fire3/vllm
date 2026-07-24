# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json

import torch

from benchmarks.kernels.deepseek_v4.d02_factories import (
    build_d02_gumbel_sample_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--num-reqs", type=int, default=4)
    parser.add_argument("--valid-tokens", type=int)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--temperatures", type=str, default="[1.0]")
    parser.add_argument("--no-apply-temperature", action="store_true")
    parser.add_argument("--use-fp64", action="store_true")
    parser.add_argument(
        "--output-mode", choices=("none", "scalar", "per-token"), default="none"
    )
    parser.add_argument("--num-output-cols", type=int, default=7)
    parser.add_argument("--output-col", type=int, default=3)
    parser.add_argument("--logits-stride-extra", type=int, default=0)
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "mirror",
            "fused",
            "fused512",
            "fused2048",
            "torch-argmax",
            "dispatch",
        ),
        default="mirror",
    )
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = json.loads(args.temperatures)
    if not isinstance(temperatures, list):
        raise ValueError("temperatures must be a JSON array")
    case = build_d02_gumbel_sample_case(
        {
            "num_tokens": args.num_tokens,
            "num_reqs": args.num_reqs,
            "valid_tokens": (
                args.valid_tokens if args.valid_tokens is not None else args.num_tokens
            ),
            "vocab_size": args.vocab_size,
            "temperatures": temperatures,
            "apply_temperature": not args.no_apply_temperature,
            "use_fp64": args.use_fp64,
            "output_mode": args.output_mode,
            "num_output_cols": args.num_output_cols,
            "output_col": args.output_col,
            "logits_stride_extra": args.logits_stride_extra,
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
