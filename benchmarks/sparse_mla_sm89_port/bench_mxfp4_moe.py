# SPDX-License-Identifier: Apache-2.0
"""SM120 MXFP4 MoE throughput probe: DeepGEMM FP4 vs Marlin MXFP4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.test_moe import (
    MarlinMoEWeightData,
    fused_marlin_moe,
    fused_topk,
    scalar_types,
)
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmFP4Experts,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    _pack_deepgemm_mxfp4_scales,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

from common import bench_cuda


def _init_runtime() -> None:
    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    DeepGemmQuantScaleFMT.init_oracle_cache()


def _make_topk(
    hidden_states: torch.Tensor, num_experts: int, topk: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=hidden_states.device)
    generator.manual_seed(seed)
    logits = torch.randn(
        hidden_states.shape[0],
        num_experts,
        device=hidden_states.device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(logits, k=topk, dim=-1)
    topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)
    return topk_weights, topk_ids


def _make_hidden(m: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    return torch.randn(
        m, k, device="cuda", dtype=torch.bfloat16, generator=generator
    ) * (k**-0.5)


class DeepGemmFP4Case:
    def __init__(self, *, n: int, k: int, num_experts: int, topk: int) -> None:
        self.n = n
        self.k = k
        self.num_experts = num_experts
        self.topk = topk
        self.moe_config = make_dummy_moe_config()

        # Synthetic finite-code packed FP4 tensors. The benchmark is a
        # throughput probe; correctness is not claimed because Marlin and
        # DeepGEMM use different packed weight layouts.
        self.w1 = torch.full(
            (num_experts, 2 * n, k // 2), 0x11, device="cuda", dtype=torch.uint8
        )
        self.w2 = torch.full(
            (num_experts, k, n // 2), 0x11, device="cuda", dtype=torch.uint8
        )
        w1_scale = torch.full(
            (num_experts, 2 * n, k // 32),
            1e-3,
            device="cuda",
            dtype=torch.float32,
        )
        w2_scale = torch.full(
            (num_experts, k, n // 32), 1e-3, device="cuda", dtype=torch.float32
        )
        self.w1_scale, self.w2_scale = _pack_deepgemm_mxfp4_scales(
            self.w1, self.w2, w1_scale, w2_scale
        )

    def make_kernel(self, hidden_states: torch.Tensor) -> mk.FusedMoEKernel:
        _, a1_scale = per_token_group_quant_fp8(hidden_states, 128)
        fp8_dtype = current_platform.fp8_dtype()
        quant_config = FusedMoEQuantConfig(
            _a1=FusedMoEQuantDesc(
                fp8_dtype, GroupShape(128, 128), a1_scale, None, None, None
            ),
            _a2=FusedMoEQuantDesc(
                fp8_dtype, GroupShape(128, 128), None, None, None, None
            ),
            _w1=FusedMoEQuantDesc(
                "mxfp4", None, self.w1_scale, None, None, None
            ),
            _w2=FusedMoEQuantDesc(
                "mxfp4", None, self.w2_scale, None, None, None
            ),
        )
        return mk.FusedMoEKernel(
            prepare_finalize=maybe_make_prepare_finalize(
                moe=self.moe_config,
                quant_config=quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            ),
            fused_experts=DeepGemmFP4Experts(
                moe_config=self.moe_config,
                quant_config=quant_config,
            ),
        )

    def run(
        self,
        kernel: mk.FusedMoEKernel,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return kernel.apply(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=self.num_experts,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )


class MarlinMXFP4Case:
    def __init__(self, *, n: int, k: int, num_experts: int, topk: int) -> None:
        self.num_experts = num_experts
        self.topk = topk
        dtype = torch.bfloat16
        generator = torch.Generator(device="cuda")
        generator.manual_seed(123)
        w1 = torch.randn(
            num_experts,
            2 * n,
            k,
            device="cuda",
            dtype=dtype,
            generator=generator,
        ) / 20
        w2 = torch.randn(
            num_experts,
            k,
            n,
            device="cuda",
            dtype=dtype,
            generator=generator,
        ) / 20
        self.w1_data = MarlinMoEWeightData.make(
            w=w1,
            quant_type=scalar_types.float4_e2m1f,
            group_size=32,
            act_order=None,
            input_type=None,
        )
        self.w2_data = MarlinMoEWeightData.make(
            w=w2,
            quant_type=scalar_types.float4_e2m1f,
            group_size=32,
            act_order=None,
            input_type=None,
        )

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return fused_marlin_moe(
            hidden_states,
            self.w1_data.qweight,
            self.w2_data.qweight,
            None,
            None,
            self.w1_data.scales,
            self.w2_data.scales,
            topk_weights,
            topk_ids,
            global_num_experts=self.num_experts,
            expert_map=None,
            global_scale1=self.w1_data.global_scale,
            global_scale2=self.w2_data.global_scale,
            g_idx1=self.w1_data.g_idx,
            g_idx2=self.w2_data.g_idx,
            input_global_scale1=self.w1_data.a_scales_factor,
            input_global_scale2=self.w2_data.a_scales_factor,
            sort_indices1=self.w1_data.sort_indices,
            sort_indices2=self.w2_data.sort_indices,
            w1_zeros=self.w1_data.zeros,
            w2_zeros=self.w2_data.zeros,
            input_dtype=None,
            quant_type_id=scalar_types.float4_e2m1f.id,
            is_k_full=True,
        )


def bench_case(
    *,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    deepgemm_case: DeepGemmFP4Case,
    marlin_case: MarlinMXFP4Case,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    hidden_states = _make_hidden(m, k, seed=1000 + m)
    topk_weights, topk_ids = _make_topk(hidden_states, num_experts, topk, seed=2000 + m)

    row: dict[str, Any] = {
        "tokens": m,
        "hidden_size": k,
        "intermediate_size": n,
        "num_experts": num_experts,
        "topk": topk,
    }

    try:
        deepgemm_kernel = deepgemm_case.make_kernel(hidden_states)
        out = deepgemm_case.run(deepgemm_kernel, hidden_states, topk_weights, topk_ids)
        torch.cuda.synchronize()
        row["deepgemm_output_all_finite"] = bool(torch.isfinite(out).all().item())
        row["deepgemm_ms"] = bench_cuda(
            lambda: deepgemm_case.run(
                deepgemm_kernel, hidden_states, topk_weights, topk_ids
            ),
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:  # noqa: BLE001
        row["deepgemm_error"] = f"{type(exc).__name__}: {exc}"

    try:
        out = marlin_case.run(hidden_states, topk_weights, topk_ids)
        torch.cuda.synchronize()
        row["marlin_output_all_finite"] = bool(torch.isfinite(out).all().item())
        row["marlin_ms"] = bench_cuda(
            lambda: marlin_case.run(hidden_states, topk_weights, topk_ids),
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:  # noqa: BLE001
        row["marlin_error"] = f"{type(exc).__name__}: {exc}"

    if "deepgemm_ms" in row and "marlin_ms" in row:
        row["marlin_over_deepgemm"] = row["marlin_ms"] / row["deepgemm_ms"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument(
        "--out", default="benchmarks/sparse_mla_sm89_port/results_mxfp4.json"
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    _init_runtime()
    tokens = (128, 256) if args.quick else (1, 16, 64, 256, 1024, 4096)
    warmup = 1 if args.quick else 5
    iters = 3 if args.quick else 20

    deepgemm_case = DeepGemmFP4Case(
        n=args.intermediate_size,
        k=args.hidden_size,
        num_experts=args.num_experts,
        topk=args.topk,
    )
    marlin_case = MarlinMXFP4Case(
        n=args.intermediate_size,
        k=args.hidden_size,
        num_experts=args.num_experts,
        topk=args.topk,
    )
    results = [
        bench_case(
            m=m,
            n=args.intermediate_size,
            k=args.hidden_size,
            num_experts=args.num_experts,
            topk=args.topk,
            deepgemm_case=deepgemm_case,
            marlin_case=marlin_case,
            warmup=warmup,
            iters=iters,
        )
        for m in tokens
    ]
    for row in results:
        print(row)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
