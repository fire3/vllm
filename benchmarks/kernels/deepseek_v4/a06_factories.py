# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
    compare_outputs,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.model_executor.kernels.mhc.tilelang_kernels import compute_num_split
from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
    tf32_hc_prenorm_gemm_triton,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed


def _build_case(args: Mapping[str, Any], *, production_dispatch: bool) -> ChainCase:
    num_tokens = int(args.get("num_tokens", 1))
    hidden_size = int(args.get("hidden_size", 7168))
    hc_mult = int(args.get("hc_mult", 4))
    seed = int(args.get("seed", 0))
    if num_tokens <= 0 or hidden_size <= 0 or hc_mult <= 0:
        raise ValueError("A06 dimensions must be positive")

    k = hc_mult * hidden_size
    n = hc_mult * (2 + hc_mult)
    if k % 64 or n % 8:
        raise ValueError("A06 requires K divisible by 64 and N divisible by 8")

    set_random_seed(seed)
    x = torch.randn((num_tokens, k), device="cuda", dtype=torch.bfloat16)
    fn = torch.randn((n, k), device="cuda", dtype=torch.float32) * 1e-4
    num_split = compute_num_split(64, k, cdiv(num_tokens, 64))
    baseline_out = torch.empty(
        (num_split, num_tokens, n), device="cuda", dtype=torch.float32
    )
    baseline_sqrsum = torch.empty(
        (num_split, num_tokens), device="cuda", dtype=torch.float32
    )
    candidate_out = torch.empty_like(baseline_out)
    candidate_sqrsum = torch.empty_like(baseline_sqrsum)

    from vllm.third_party import deep_gemm as vendored_deep_gemm

    vendored_deep_gemm.set_pdl(True)

    def run_baseline() -> torch.Tensor:
        tf32_hc_prenorm_gemm_triton(
            x,
            fn,
            baseline_out,
            baseline_sqrsum,
            num_split,
        )
        return baseline_out

    if production_dispatch:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        candidate_kernel = "vllm.utils.deep_gemm.tf32_hc_prenorm_gemm"

        def run_candidate() -> torch.Tensor:
            tf32_hc_prenorm_gemm(
                x,
                fn,
                candidate_out,
                candidate_sqrsum,
                num_split,
            )
            return candidate_out

    else:
        candidate_kernel = "vllm.third_party.deep_gemm.tf32_hc_prenorm_gemm"

        def run_candidate() -> torch.Tensor:
            vendored_deep_gemm.tf32_hc_prenorm_gemm(
                x,
                fn,
                candidate_out,
                candidate_sqrsum,
                num_split,
            )
            return candidate_out

    def compare_split_outputs(
        _reference: torch.Tensor,
        _candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        output = compare_outputs(
            baseline_out.sum(0),
            candidate_out.sum(0),
            tolerances,
        )
        sqrsum = compare_outputs(
            baseline_sqrsum.sum(0),
            candidate_sqrsum.sum(0),
            CorrectnessTolerances(
                atol=0.125,
                rtol=1e-5,
                require_allclose=True,
            ),
        )
        return {
            "passed": output["passed"] and sqrsum["passed"],
            "output": output,
            "sqrsum": sqrsum,
        }

    shape = {
        "name": f"m{num_tokens}-n{n}-k{k}-s{num_split}",
        "M": num_tokens,
        "N": n,
        "K": k,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "num_split": num_split,
        "chain": "tf32-hc-prenorm-gemm",
    }
    return ChainCase(
        baseline=Provider(
            "triton-tf32-hc-prenorm",
            run_baseline,
            {
                "symbol": "tf32_hc_prenorm_gemm_triton",
                "preallocated_output": True,
            },
        ),
        candidate=Provider(
            "deepgemm-sm120-tf32-hc-prenorm",
            run_candidate,
            {
                "symbol": candidate_kernel,
                "preallocated_output": True,
                "pdl": True,
                "production_dispatch": production_dispatch,
                "minimum_deepgemm_tokens": 64,
            },
            correctness_comparator=compare_split_outputs,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=6e-5,
            rtol=5e-3,
            require_allclose=True,
        ),
    )


def build_a06_kernel_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, production_dispatch=False)


def build_a06_production_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, production_dispatch=True)
