# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json

import torch

from benchmarks.kernels.deepseek_v4.d01_factories import (
    build_d01_prepare_dflash_inputs_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-reqs", type=int, default=4)
    parser.add_argument("--total-target-tokens", type=int, default=32)
    parser.add_argument("--target-query-lens", type=str)
    parser.add_argument("--num-rejected", type=str)
    parser.add_argument("--num-sampled", type=str)
    parser.add_argument("--num-speculative-steps", type=int, default=7)
    parser.add_argument("--dflash-layout", action="store_true")
    parser.add_argument("--max-num-reqs", type=int, default=128)
    parser.add_argument("--max-num-tokens", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--context-jitter", type=int, default=0)
    parser.add_argument(
        "--candidate-mode",
        choices=("mirror", "fixed32", "fixed64", "fixed128", "fixed256", "dispatch"),
        default="mirror",
    )
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def _parse_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("list arguments must be JSON arrays")
    return [int(item) for item in parsed]


def main() -> None:
    args = parse_args()
    factory_args = {
        "num_reqs": args.num_reqs,
        "total_target_tokens": args.total_target_tokens,
        "num_speculative_steps": args.num_speculative_steps,
        "sample_from_anchor": not args.dflash_layout,
        "max_num_reqs": args.max_num_reqs,
        "max_num_tokens": args.max_num_tokens,
        "block_size": args.block_size,
        "max_model_len": args.max_model_len,
        "context_tokens": args.context_tokens,
        "context_jitter": args.context_jitter,
        "candidate_mode": args.candidate_mode,
    }
    for name, value in (
        ("target_query_lens", _parse_list(args.target_query_lens)),
        ("num_rejected", _parse_list(args.num_rejected)),
        ("num_sampled", _parse_list(args.num_sampled)),
    ):
        if value is not None:
            factory_args[name] = value
    case = build_d01_prepare_dflash_inputs_case(factory_args)
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
