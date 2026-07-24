# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from benchmarks.kernels.deepseek_v4.a03_factories import (
    build_a03_deepgemm_logits_case,
    build_a03_deepgemm_topk_case,
    build_a03_logits_case,
    build_a03_production_logits_case,
    build_a03_production_topk_case,
    build_a03_topk_case,
)

FACTORIES = {
    "native-fp8": build_a03_logits_case,
    "native-fp8-topk": build_a03_topk_case,
    "deepgemm": build_a03_deepgemm_logits_case,
    "deepgemm-topk": build_a03_deepgemm_topk_case,
    "production": build_a03_production_logits_case,
    "production-topk": build_a03_production_topk_case,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", choices=FACTORIES, default="native-fp8")
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--num-queries", type=int, default=8192)
    parser.add_argument("--num-keys", type=int, default=64)
    parser.add_argument("--compress-ratio", type=int, default=128)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    case = FACTORIES[args.factory](
        {
            "num_queries": args.num_queries,
            "num_keys": args.num_keys,
            "compress_ratio": args.compress_ratio,
            "query_offset": args.query_offset,
            "causal": args.factory.endswith("-topk"),
        }
    )
    provider = getattr(case, args.provider)

    with torch.inference_mode():
        for _ in range(args.warmup):
            provider.fn()
        torch.accelerator.synchronize()
        for _ in range(args.iterations):
            provider.fn()
        torch.accelerator.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
