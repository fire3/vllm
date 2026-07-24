# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from flashinfer.cute_dsl import dual_rmsnorm_cute, rmsnorm_cute

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
    compare_outputs,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (
    _fused_q_kv_rmsnorm_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed


def _build_b01_case(
    args: Mapping[str, Any],
    *,
    use_dual_kernel: bool,
    production_dispatch: bool = False,
    hybrid_dispatch: bool = False,
) -> ChainCase:
    num_tokens = int(args.get("num_tokens", 1))
    q_size = int(args.get("q_size", 1536))
    kv_size = int(args.get("kv_size", 512))
    eps = float(args.get("eps", 1e-6))
    seed = int(args.get("seed", 0))
    enable_pdl = bool(args.get("enable_pdl", False))
    parallel_tasks = bool(args.get("parallel_tasks", False))
    if hybrid_dispatch:
        parallel_tasks = num_tokens < 8192
    if num_tokens <= 0 or q_size <= 0 or kv_size <= 0:
        raise ValueError("B01 dimensions must be positive")

    set_random_seed(seed)
    dtype = torch.bfloat16
    qr_kv = torch.randn(
        (num_tokens, q_size + kv_size),
        device="cuda",
        dtype=dtype,
    )
    qr, kv = qr_kv.split([q_size, kv_size], dim=-1)
    q_weight = torch.randn(q_size, device="cuda", dtype=dtype)
    kv_weight = torch.randn(kv_size, device="cuda", dtype=dtype)

    baseline_q_out = torch.empty_like(qr)
    baseline_kv_out = torch.empty_like(kv)
    candidate_q_out = torch.empty_like(qr)
    candidate_kv_out = torch.empty_like(kv)
    block_size = triton.next_power_of_2(max(q_size, kv_size))

    def run_baseline() -> torch.Tensor:
        _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
            qr,
            baseline_q_out,
            q_weight,
            qr.stride(0),
            baseline_q_out.stride(0),
            kv,
            baseline_kv_out,
            kv_weight,
            kv.stride(0),
            baseline_kv_out.stride(0),
            eps,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=block_size,
        )
        return baseline_q_out

    candidate_outputs = {
        "q": candidate_q_out,
        "kv": candidate_kv_out,
    }
    if production_dispatch:
        from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (
            fused_q_kv_rmsnorm,
        )

        candidate_name = "vllm-production-fused-q-kv-rmsnorm"
        candidate_symbol = "fused_q_kv_rmsnorm"
        candidate_launches = 1

        def run_candidate() -> torch.Tensor:
            q_out, kv_out = fused_q_kv_rmsnorm(
                qr,
                kv,
                q_weight,
                kv_weight,
                eps,
            )
            candidate_outputs["q"] = q_out
            candidate_outputs["kv"] = kv_out
            return q_out

    elif hybrid_dispatch and num_tokens < 16:
        candidate_name = "triton-fused-q-kv-rmsnorm-hybrid-fallback"
        candidate_symbol = "_fused_q_kv_rmsnorm_kernel"
        candidate_launches = 1

        def run_candidate() -> torch.Tensor:
            _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
                qr,
                candidate_q_out,
                q_weight,
                qr.stride(0),
                candidate_q_out.stride(0),
                kv,
                candidate_kv_out,
                kv_weight,
                kv.stride(0),
                candidate_kv_out.stride(0),
                eps,
                Q_SIZE=q_size,
                KV_SIZE=kv_size,
                BLOCK_SIZE=block_size,
            )
            return candidate_q_out

    elif use_dual_kernel:
        candidate_name = "flashinfer-dual-rmsnorm-cute"
        candidate_symbol = "flashinfer.cute_dsl.dual_rmsnorm_cute"
        candidate_launches = 1

        def run_candidate() -> torch.Tensor:
            dual_rmsnorm_cute(
                qr,
                kv,
                q_weight,
                kv_weight,
                candidate_q_out,
                candidate_kv_out,
                eps,
                enable_pdl=enable_pdl,
                parallel_tasks=parallel_tasks,
            )
            return candidate_q_out

    else:
        candidate_name = "flashinfer-two-rmsnorm-cute"
        candidate_symbol = "flashinfer.cute_dsl.rmsnorm_cute"
        candidate_launches = 2

        def run_candidate() -> torch.Tensor:
            rmsnorm_cute(
                qr,
                q_weight,
                candidate_q_out,
                eps,
                enable_pdl=enable_pdl,
            )
            rmsnorm_cute(
                kv,
                kv_weight,
                candidate_kv_out,
                eps,
                enable_pdl=enable_pdl,
            )
            return candidate_q_out

    def compare_q_kv_outputs(
        _reference: torch.Tensor,
        _candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        q_result = compare_outputs(
            baseline_q_out,
            candidate_outputs["q"],
            tolerances,
        )
        kv_result = compare_outputs(
            baseline_kv_out,
            candidate_outputs["kv"],
            tolerances,
        )
        return {
            "passed": q_result["passed"] and kv_result["passed"],
            "q": q_result,
            "kv": kv_result,
        }

    return ChainCase(
        baseline=Provider(
            "triton-fused-q-kv-rmsnorm",
            run_baseline,
            {
                "symbol": "_fused_q_kv_rmsnorm_kernel",
                "launches": 1,
                "preallocated_output": True,
            },
        ),
        candidate=Provider(
            candidate_name,
            run_candidate,
            {
                "symbol": candidate_symbol,
                "launches": candidate_launches,
                "preallocated_output": True,
                "enable_pdl": enable_pdl,
                "parallel_tasks": (
                    num_tokens < 8192 if production_dispatch else parallel_tasks
                ),
                "production_dispatch": production_dispatch,
                "hybrid_dispatch": hybrid_dispatch,
            },
            correctness_comparator=compare_q_kv_outputs,
        ),
        shape={
            "name": f"m{num_tokens}-q{q_size}-kv{kv_size}",
            "M": num_tokens,
            "q_size": q_size,
            "kv_size": kv_size,
            "input_row_stride": qr.stride(0),
            "input_contiguous": qr.is_contiguous(),
            "dtype": str(dtype),
            "chain": "fused-q-kv-rmsnorm",
        },
        tolerances=CorrectnessTolerances(
            atol=1e-2,
            rtol=1e-2,
            require_allclose=True,
        ),
    )


def build_b01_two_rmsnorm_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b01_case(args, use_dual_kernel=False)


def build_b01_dual_rmsnorm_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b01_case(args, use_dual_kernel=True)


def build_b01_production_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b01_case(
        args,
        use_dual_kernel=True,
        production_dispatch=True,
    )


def build_b01_hybrid_kernel_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b01_case(
        args,
        use_dual_kernel=True,
        hybrid_dispatch=True,
    )
