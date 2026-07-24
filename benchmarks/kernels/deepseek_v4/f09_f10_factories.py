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
from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
    _fused_mtp_input_rmsnorm_kernel,
    _mtp_shared_head_rmsnorm_kernel,
    fused_mtp_input_rmsnorm,
    mtp_shared_head_rmsnorm,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

DEEPSEEK_V4_HIDDEN_SIZE = 4096
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_RMS_NORM_EPS = 1.0e-6
DEEPSEEK_V4_DTYPE = torch.bfloat16
DEEPSEEK_V4_DRAFT_TOKENS = 7
DEEPSEEK_V4_BATCH_SIZES = (1, 4, 32)
DEEPSEEK_V4_PREFILL_TOKENS = 256

_TRITON_WARPS = {
    "triton-w1": 1,
    "triton-w2": 2,
    "triton-w4": 4,
    "triton-w8": 8,
}


def _num_tokens(args: Mapping[str, Any]) -> int:
    if "num_tokens" in args:
        num_tokens = int(args["num_tokens"])
    else:
        shape_kind = str(args.get("shape_kind", "decode"))
        if shape_kind == "prefill":
            num_tokens = DEEPSEEK_V4_PREFILL_TOKENS
        elif shape_kind == "decode":
            batch_size = int(args.get("batch_size", 1))
            if batch_size not in DEEPSEEK_V4_BATCH_SIZES:
                raise ValueError("F09/F10 decode batch_size must be one of B1/B4/B32")
            num_tokens = batch_size
        else:
            raise ValueError("shape_kind must be 'decode' or 'prefill'")
    if num_tokens <= 0:
        raise ValueError("F09/F10 num_tokens must be positive")
    return num_tokens


def _positions(num_tokens: int, args: Mapping[str, Any]) -> torch.Tensor:
    if bool(args.get("include_position_zero", True)):
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    else:
        positions = torch.arange(1, num_tokens + 1, device="cuda", dtype=torch.long)
    return positions


def _shape_metadata(
    *,
    chain: str,
    num_tokens: int,
    candidate_mode: str,
    candidate_symbol: str,
) -> dict[str, Any]:
    batch_size = int(num_tokens if num_tokens in DEEPSEEK_V4_BATCH_SIZES else 0)
    shape_kind = "prefill" if num_tokens == DEEPSEEK_V4_PREFILL_TOKENS else "decode"
    name = (
        f"{chain}-T{num_tokens}-H{DEEPSEEK_V4_HIDDEN_SIZE}"
        f"-hc{DEEPSEEK_V4_HC_MULT}-draft{DEEPSEEK_V4_DRAFT_TOKENS}"
        f"-{candidate_mode}"
    )
    return {
        "name": name,
        "chain": chain,
        "shape_kind": shape_kind,
        "batch_size": batch_size or None,
        "num_tokens": num_tokens,
        "hidden_size": DEEPSEEK_V4_HIDDEN_SIZE,
        "hc_mult": DEEPSEEK_V4_HC_MULT,
        "draft_tokens": DEEPSEEK_V4_DRAFT_TOKENS,
        "dtype": str(DEEPSEEK_V4_DTYPE),
        "eps": DEEPSEEK_V4_RMS_NORM_EPS,
        "candidate_mode": candidate_mode,
        "candidate_symbol": candidate_symbol,
    }


def _tolerances() -> CorrectnessTolerances:
    return CorrectnessTolerances(atol=1e-2, rtol=1e-2, require_allclose=True)


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = DEEPSEEK_V4_RMS_NORM_EPS,
) -> torch.Tensor:
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    out = x_f32 * torch.rsqrt(variance + eps) * weight.float()
    return out.to(x.dtype)


def build_f09_triton_warps_case(args: Mapping[str, Any]) -> ChainCase:
    candidate_mode = str(args.get("candidate_mode", "triton-w4"))
    if candidate_mode not in _TRITON_WARPS:
        raise ValueError("F09 only benchmarks legal Triton warp candidates")

    num_tokens = _num_tokens(args)
    seed = int(args.get("seed", 0))
    set_random_seed(seed)
    inputs_embeds = torch.randn(
        num_tokens,
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    positions = _positions(num_tokens, args)
    previous_hidden_states = torch.randn(
        num_tokens,
        DEEPSEEK_V4_HC_MULT,
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    enorm_weight = torch.randn(
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    hnorm_weight = torch.randn(
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    candidate_enorm_out = torch.empty_like(inputs_embeds)
    candidate_hnorm_out = torch.empty_like(previous_hidden_states)
    block_size = triton.next_power_of_2(DEEPSEEK_V4_HIDDEN_SIZE)

    baseline_outputs: dict[str, torch.Tensor] = {}

    def run_baseline() -> torch.Tensor:
        enorm_out, hnorm_out = fused_mtp_input_rmsnorm(
            inputs_embeds,
            positions,
            previous_hidden_states,
            enorm_weight,
            hnorm_weight,
            DEEPSEEK_V4_RMS_NORM_EPS,
            DEEPSEEK_V4_HC_MULT,
        )
        baseline_outputs["enorm"] = enorm_out
        baseline_outputs["hnorm"] = hnorm_out
        return enorm_out

    def run_candidate() -> torch.Tensor:
        _fused_mtp_input_rmsnorm_kernel[(num_tokens, DEEPSEEK_V4_HC_MULT + 1)](
            inputs_embeds,
            positions,
            previous_hidden_states,
            enorm_weight,
            hnorm_weight,
            candidate_enorm_out,
            candidate_hnorm_out,
            DEEPSEEK_V4_RMS_NORM_EPS,
            HIDDEN=DEEPSEEK_V4_HIDDEN_SIZE,
            HC_MULT=DEEPSEEK_V4_HC_MULT,
            BLOCK_SIZE=block_size,
            num_warps=_TRITON_WARPS[candidate_mode],
        )
        return candidate_enorm_out

    def compare_f09_outputs(
        _reference: torch.Tensor,
        _candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        enorm_result = compare_outputs(
            baseline_outputs["enorm"],
            candidate_enorm_out,
            tolerances,
        )
        hnorm_result = compare_outputs(
            baseline_outputs["hnorm"],
            candidate_hnorm_out,
            tolerances,
        )
        masked_inputs = torch.where(positions[:, None] == 0, 0, inputs_embeds)
        reference_result = compare_outputs(
            _rmsnorm_reference(masked_inputs, enorm_weight),
            baseline_outputs["enorm"],
            tolerances,
        )
        return {
            "passed": (
                enorm_result["passed"]
                and hnorm_result["passed"]
                and reference_result["passed"]
            ),
            "enorm": enorm_result,
            "hnorm": hnorm_result,
            "torch_reference_enorm": reference_result,
        }

    return ChainCase(
        baseline=Provider(
            "vllm-production-fused-mtp-input-rmsnorm",
            run_baseline,
            {"symbol": "fused_mtp_input_rmsnorm", "launches": 1},
        ),
        candidate=Provider(
            f"f09-{candidate_mode}",
            run_candidate,
            {
                "symbol": "_fused_mtp_input_rmsnorm_kernel",
                "launches": 1,
                "num_warps": _TRITON_WARPS[candidate_mode],
                "benchmark_only": True,
            },
            correctness_comparator=compare_f09_outputs,
        ),
        shape=_shape_metadata(
            chain="f09-fused-mtp-input-rmsnorm",
            num_tokens=num_tokens,
            candidate_mode=candidate_mode,
            candidate_symbol="_fused_mtp_input_rmsnorm_kernel",
        ),
        tolerances=_tolerances(),
    )


def build_f10_rmsnorm_case(args: Mapping[str, Any]) -> ChainCase:
    candidate_mode = str(args.get("candidate_mode", "flashinfer-cute"))
    valid_modes = {
        *_TRITON_WARPS,
        "flashinfer-cute",
        "flashinfer-native",
        "vllm-native",
    }
    if candidate_mode not in valid_modes:
        raise ValueError("unsupported F10 RMSNorm candidate mode")

    num_tokens = _num_tokens(args)
    seed = int(args.get("seed", 0))
    set_random_seed(seed)
    hidden_states = torch.randn(
        num_tokens,
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    weight = torch.randn(
        DEEPSEEK_V4_HIDDEN_SIZE,
        device="cuda",
        dtype=DEEPSEEK_V4_DTYPE,
    )
    candidate_out = torch.empty_like(hidden_states)
    block_size = triton.next_power_of_2(DEEPSEEK_V4_HIDDEN_SIZE)

    def run_baseline() -> torch.Tensor:
        return mtp_shared_head_rmsnorm(
            hidden_states,
            weight,
            DEEPSEEK_V4_RMS_NORM_EPS,
        )

    if candidate_mode in _TRITON_WARPS:
        candidate_symbol = "_mtp_shared_head_rmsnorm_kernel"

        def run_candidate() -> torch.Tensor:
            _mtp_shared_head_rmsnorm_kernel[(num_tokens,)](
                hidden_states,
                weight,
                candidate_out,
                DEEPSEEK_V4_RMS_NORM_EPS,
                HIDDEN=DEEPSEEK_V4_HIDDEN_SIZE,
                BLOCK_SIZE=block_size,
                num_warps=_TRITON_WARPS[candidate_mode],
            )
            return candidate_out

    elif candidate_mode == "flashinfer-cute":
        candidate_symbol = "flashinfer.cute_dsl.rmsnorm_cute"

        def run_candidate() -> torch.Tensor:
            from flashinfer.cute_dsl import rmsnorm_cute

            rmsnorm_cute(
                hidden_states,
                weight,
                candidate_out,
                DEEPSEEK_V4_RMS_NORM_EPS,
            )
            return candidate_out

    elif candidate_mode == "flashinfer-native":
        candidate_symbol = "flashinfer.norm.rmsnorm"

        def run_candidate() -> torch.Tensor:
            from flashinfer.norm import rmsnorm

            return rmsnorm(
                hidden_states,
                weight,
                DEEPSEEK_V4_RMS_NORM_EPS,
                out=candidate_out,
            )

    else:
        candidate_symbol = "vllm._custom_ops.rms_norm"

        def run_candidate() -> torch.Tensor:
            from vllm import _custom_ops as ops

            ops.rms_norm(
                candidate_out,
                hidden_states,
                weight,
                DEEPSEEK_V4_RMS_NORM_EPS,
            )
            return candidate_out

    def compare_f10_outputs(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        candidate_result = compare_outputs(reference, candidate, tolerances)
        reference_result = compare_outputs(
            _rmsnorm_reference(hidden_states, weight),
            reference,
            tolerances,
        )
        return {
            "passed": candidate_result["passed"] and reference_result["passed"],
            "candidate": candidate_result,
            "torch_reference": reference_result,
        }

    return ChainCase(
        baseline=Provider(
            "vllm-production-mtp-shared-head-rmsnorm",
            run_baseline,
            {"symbol": "mtp_shared_head_rmsnorm", "launches": 1},
        ),
        candidate=Provider(
            f"f10-{candidate_mode}",
            run_candidate,
            {
                "symbol": candidate_symbol,
                "launches": 1,
                "benchmark_only": True,
                "num_warps": _TRITON_WARPS.get(candidate_mode),
            },
            correctness_comparator=compare_f10_outputs,
        ),
        shape=_shape_metadata(
            chain="f10-mtp-shared-head-rmsnorm",
            num_tokens=num_tokens,
            candidate_mode=candidate_mode,
            candidate_symbol=candidate_symbol,
        ),
        tolerances=_tolerances(),
    )
