# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.deepseek_v4.b01_factories import (
    build_b01_dual_rmsnorm_case,
    build_b01_hybrid_kernel_case,
    build_b01_production_case,
    build_b01_two_rmsnorm_case,
)

_CASE_BUILDERS = {
    "two-rmsnorm": build_b01_two_rmsnorm_case,
    "dual-rmsnorm": build_b01_dual_rmsnorm_case,
    "production": build_b01_production_case,
    "hybrid": build_b01_hybrid_kernel_case,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("baseline", "candidate"), required=True)
    parser.add_argument(
        "--candidate-mode",
        choices=tuple(_CASE_BUILDERS),
        default="hybrid",
    )
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--q-size", type=int, default=1536)
    parser.add_argument("--kv-size", type=int, default=512)
    parser.add_argument("--enable-pdl", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = _CASE_BUILDERS[args.candidate_mode](
        {
            "num_tokens": args.num_tokens,
            "q_size": args.q_size,
            "kv_size": args.kv_size,
            "enable_pdl": args.enable_pdl,
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
