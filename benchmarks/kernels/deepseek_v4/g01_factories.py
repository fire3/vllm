# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as HfDeepseekV4Config,
)

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
    compare_outputs,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _silu_mul_quant_fp8_packed_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

G01_HIDDEN_SIZE = int(HfDeepseekV4Config.moe_intermediate_size)
G01_GROUP_SIZE = 128
G01_GROUPS_PER_PACK = 4

# These rows are the contiguous DeepGEMM workspace upper bounds for
# DeepSeek-V4 top-k=6 with 256 experts and the SM120 alignment policy.
G01_DEFAULT_CASE_ARGS: tuple[dict[str, Any], ...] = (
    {"name": "decode-b1-m384", "rows": 384, "source_tokens": 1},
    {"name": "decode-b4-m1536", "rows": 1536, "source_tokens": 4},
    {"name": "decode-b16-m6144", "rows": 6144, "source_tokens": 16},
    {"name": "prefill-t64-m16512", "rows": 16512, "source_tokens": 64},
    {"name": "prefill-t8192-m81664", "rows": 81664, "source_tokens": 8192},
)


def _allocate_scale(rows: int, groups_per_row: int) -> torch.Tensor:
    packs_per_row = triton.cdiv(groups_per_row, G01_GROUPS_PER_PACK)
    return torch.empty_strided(
        (rows, packs_per_row),
        (1, rows),
        device="cuda",
        dtype=torch.int32,
    )


def _unpack_and_dequantize(
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    rows, hidden_size = output_q.shape
    groups_per_row = hidden_size // group_size
    scale_bytes = output_scale.contiguous().view(torch.uint8)
    scale_bytes = scale_bytes.reshape(rows, -1)[:, :groups_per_row]
    exponents = scale_bytes.to(torch.int32) - 127
    scales = torch.exp2(exponents.float()).unsqueeze(-1)
    return (output_q.float().reshape(rows, groups_per_row, group_size) * scales).view(
        rows, hidden_size
    )


def _launch_triton(
    input_tensor: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    rows, input_width = input_tensor.shape
    hidden_size = input_width // 2
    groups_per_row = hidden_size // group_size
    packs_per_row = triton.cdiv(groups_per_row, G01_GROUPS_PER_PACK)
    block_m = 1 if rows < 512 else 4
    packs_per_cta = 2 if rows < 512 else 1
    grid = (
        triton.cdiv(packs_per_row, packs_per_cta),
        min(triton.cdiv(rows, block_m), 4096),
    )
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    _silu_mul_quant_fp8_packed_kernel[grid](
        input_tensor,
        output_q,
        output_scale,
        rows,
        input_tensor.stride(0),
        output_q.stride(0),
        output_scale.stride(1),
        0.0,
        1.0,
        0.0,
        N=input_width,
        GROUPS_PER_ROW=groups_per_row,
        PACKS_PER_ROW=packs_per_row,
        fp8_min=fp8_info.min,
        fp8_max=fp8_info.max,
        GROUP_SIZE=group_size,
        PACKS_PER_CTA=packs_per_cta,
        BLOCK_M=block_m,
        HAS_CLAMP=False,
        num_warps=4,
        num_stages=2,
    )
    return output_q


def build_g01_silu_packed_case(args: Mapping[str, Any]) -> ChainCase:
    rows = int(args.get("rows", 384))
    hidden_size = int(args.get("hidden_size", G01_HIDDEN_SIZE))
    group_size = int(args.get("group_size", G01_GROUP_SIZE))
    seed = int(args.get("seed", 0))
    if rows <= 0 or rows % 4 != 0:
        raise ValueError("G01 requires a positive TMA-aligned row count")
    if hidden_size <= 0 or hidden_size % group_size != 0:
        raise ValueError("G01 hidden_size must be divisible by group_size")
    if group_size != G01_GROUP_SIZE:
        raise ValueError("the native G01 candidate supports group_size=128 only")

    set_random_seed(seed)
    input_tensor = (
        torch.randn(
            (rows, hidden_size * 2),
            device="cuda",
            dtype=torch.float32,
        )
        * 4.0
    ).to(torch.bfloat16)
    input_tensor[0, :group_size] = 0
    if rows > 1:
        input_tensor[1, group_size : 2 * group_size] = 1.0e-6

    reference_activation = torch.empty(
        (rows, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    torch.ops._C.silu_and_mul(reference_activation, input_tensor)

    baseline_q = torch.empty(
        (rows, hidden_size),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    candidate_q = torch.empty_like(baseline_q)
    groups_per_row = hidden_size // group_size
    baseline_scale = _allocate_scale(rows, groups_per_row)
    candidate_scale = _allocate_scale(rows, groups_per_row)
    tokens_per_expert = torch.tensor(
        [rows],
        device="cuda",
        dtype=torch.int32,
    )

    def run_baseline() -> torch.Tensor:
        return _launch_triton(
            input_tensor,
            baseline_q,
            baseline_scale,
            group_size,
        )

    def run_candidate() -> torch.Tensor:
        torch.ops._C.persistent_masked_m_silu_mul_quant(
            input_tensor.unsqueeze(0),
            tokens_per_expert,
            candidate_q.unsqueeze(0),
            candidate_scale.unsqueeze(0),
            True,
        )
        return candidate_q

    def compare_candidate(
        reference_q: torch.Tensor,
        candidate_q_result: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        baseline_dequant = _unpack_and_dequantize(
            reference_q,
            baseline_scale,
            group_size,
        )
        candidate_dequant = _unpack_and_dequantize(
            candidate_q_result,
            candidate_scale,
            group_size,
        )
        reference = reference_activation.float()
        baseline_metrics = compare_outputs(reference, baseline_dequant, tolerances)
        candidate_metrics = compare_outputs(reference, candidate_dequant, tolerances)
        pair_metrics = compare_outputs(
            baseline_dequant,
            candidate_dequant,
            CorrectnessTolerances(
                atol=1.0,
                rtol=0.25,
                max_mean_relative=0.005,
                min_cosine=0.999,
                require_allclose=False,
            ),
        )
        expected_scale_stride = (1, rows)
        layout_passed = (
            baseline_scale.dtype == candidate_scale.dtype == torch.int32
            and baseline_scale.shape == candidate_scale.shape
            and baseline_scale.stride() == expected_scale_stride
            and candidate_scale.stride() == expected_scale_stride
        )
        candidate_not_worse = (
            float(candidate_metrics["mean_relative"])
            <= float(baseline_metrics["mean_relative"]) * 1.01 + 1.0e-6
        )
        return {
            "passed": bool(
                baseline_metrics["passed"]
                and candidate_metrics["passed"]
                and pair_metrics["passed"]
                and layout_passed
                and candidate_not_worse
            ),
            "comparison": "dequantized_fp32_against_bf16_silu_reference",
            "baseline_reference": baseline_metrics,
            "candidate_reference": candidate_metrics,
            "candidate_vs_baseline": pair_metrics,
            "candidate_not_worse": candidate_not_worse,
            "q_mismatch_count": int((reference_q != candidate_q_result).sum().item()),
            "scale_mismatch_count": int(
                (baseline_scale != candidate_scale).sum().item()
            ),
            "scale_shape": list(candidate_scale.shape),
            "scale_stride": list(candidate_scale.stride()),
            "layout_passed": layout_passed,
        }

    shape = {
        "name": str(args.get("name", f"silu-packed-m{rows}-h{hidden_size}")),
        "rows": rows,
        "source_tokens": int(args.get("source_tokens", -1)),
        "hidden_size": hidden_size,
        "input_width": hidden_size * 2,
        "group_size": group_size,
        "groups_per_row": groups_per_row,
        "scale_shape": list(candidate_scale.shape),
        "scale_stride": list(candidate_scale.stride()),
        "activation": "silu",
        "clamp_limit": None,
        "alpha": 1.0,
        "beta": 0.0,
        "chain": "silu-mul-quantize-packed-ue8m0",
    }
    tolerances = CorrectnessTolerances(
        atol=1.0,
        rtol=0.25,
        max_mean_relative=0.03,
        min_cosine=0.999,
        require_allclose=False,
    )
    return ChainCase(
        baseline=Provider(
            "triton-packed-silu-quant",
            run_baseline,
            {
                "kernel": "_silu_mul_quant_fp8_packed_kernel",
                "scale_format": "packed_ue8m0_int32",
            },
        ),
        candidate=Provider(
            "native-persistent-masked-e1",
            run_candidate,
            {
                "op": "torch.ops._C.persistent_masked_m_silu_mul_quant",
                "experts": 1,
                "scale_format": "packed_ue8m0_int32",
                "semantic_guard": "plain_silu_no_clamp",
            },
            correctness_comparator=compare_candidate,
        ),
        shape=shape,
        tolerances=tolerances,
    )
