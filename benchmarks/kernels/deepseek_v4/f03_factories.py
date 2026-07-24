# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
    DeepseekV4Config as HfDeepseekV4Config,
)

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
    _prepare_megamoe_inputs_kernel,
    _prepare_megamoe_num_warps,
    prepare_megamoe_inputs,
)
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed

DEEPSEEK_V4_HIDDEN_SIZE = 4096
DEEPSEEK_V4_TOPK = int(HfDeepseekV4Config.num_experts_per_tok)
DEEPSEEK_V4_NUM_EXPERTS = int(HfDeepseekV4Config.n_routed_experts)
F03_BLOCK_K = 128
F03_GROUP_K = 32
F03_LEGACY_NUM_WARPS = 4
F03_CANDIDATE_NUM_WARPS = (1, 2, 4, 8)

F03_DEFAULT_CASE_ARGS: tuple[dict[str, Any], ...] = tuple(
    {
        "name": f"{phase.lower()}-{label}-{'padding' if is_padding else 'nopadding'}",
        "phase": phase,
        "num_tokens": num_tokens,
        "hidden_size": DEEPSEEK_V4_HIDDEN_SIZE,
        "top_k": DEEPSEEK_V4_TOPK,
        "num_experts": DEEPSEEK_V4_NUM_EXPERTS,
        "is_padding": is_padding,
    }
    for phase, sizes in (
        ("decode", (("b1", 1), ("b4", 4), ("b32", 32))),
        ("prefill", (("t256", 256), ("t8192", 8192))),
    )
    for label, num_tokens in sizes
    for is_padding in (False, True)
)


def deepseek_v4_f03_config() -> dict[str, int]:
    """Return the DeepSeek-V4 values used by the F03 benchmark shapes."""

    return {
        "hidden_size": DEEPSEEK_V4_HIDDEN_SIZE,
        "top_k": DEEPSEEK_V4_TOPK,
        "num_experts": DEEPSEEK_V4_NUM_EXPERTS,
    }


def _resolve_candidate_geometry(args: Mapping[str, Any]) -> tuple[int, int]:
    block_k = int(args.get("candidate_block_k", F03_BLOCK_K))
    if block_k != F03_BLOCK_K:
        raise ValueError(
            "F03 scale packing fixes BLOCK_K=128: one int32 stores four "
            "GROUP_K=32 E8M0 scale exponents, so benchmark candidates only "
            "sweep num_warps."
        )
    num_warps = int(args.get("candidate_num_warps", _prepare_megamoe_num_warps()))
    if num_warps not in F03_CANDIDATE_NUM_WARPS:
        raise ValueError(
            f"F03 candidate_num_warps must be one of {F03_CANDIDATE_NUM_WARPS}"
        )
    return block_k, num_warps


def _make_padding(
    num_tokens: int,
    enabled: bool,
    device: torch.device,
) -> torch.Tensor | None:
    if not enabled:
        return None
    if num_tokens == 1:
        return torch.ones((1,), device=device, dtype=torch.bool)
    return torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(4) == 1


def _make_inputs(args: Mapping[str, Any]):
    num_tokens = int(args["num_tokens"])
    hidden_size = int(args.get("hidden_size", DEEPSEEK_V4_HIDDEN_SIZE))
    top_k = int(args.get("top_k", DEEPSEEK_V4_TOPK))
    num_experts = int(args.get("num_experts", DEEPSEEK_V4_NUM_EXPERTS))
    seed = int(args.get("seed", 0))
    is_padding_enabled = bool(args.get("is_padding", False))
    phase = str(args.get("phase", "decode" if num_tokens <= 32 else "prefill"))
    if num_tokens <= 0:
        raise ValueError("F03 requires a positive num_tokens")
    if hidden_size <= 0 or hidden_size % F03_BLOCK_K != 0:
        raise ValueError("F03 requires hidden_size to be a positive multiple of 128")
    if top_k <= 0:
        raise ValueError("F03 requires a positive top_k")
    if num_experts <= 0:
        raise ValueError("F03 requires a positive num_experts")

    set_random_seed(seed)
    device = torch.device("cuda")
    hidden_states = (
        torch.randn(
            (num_tokens, hidden_size),
            device=device,
            dtype=torch.float32,
        )
        * 17.0
    ).to(torch.bfloat16)
    hidden_states[0, : min(F03_GROUP_K, hidden_size)] = 0
    if num_tokens > 1 and hidden_size >= 2 * F03_GROUP_K:
        hidden_states[1, F03_GROUP_K : 2 * F03_GROUP_K] = 1.0e-6

    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        device=device,
        dtype=torch.int32,
    )
    topk_weights = torch.randn(
        (num_tokens, top_k),
        device=device,
        dtype=torch.float32,
    )
    is_padding = _make_padding(num_tokens, is_padding_enabled, device)
    shape = {
        "name": str(
            args.get(
                "name",
                f"{phase.lower()}-t{num_tokens}-h{hidden_size}-topk{top_k}"
                f"-e{num_experts}-{'padding' if is_padding_enabled else 'nopadding'}",
            )
        ),
        "phase": phase,
        "T": num_tokens,
        "H": hidden_size,
        "top_k": top_k,
        "num_experts": num_experts,
        "is_padding": is_padding_enabled,
        "BLOCK_K": F03_BLOCK_K,
        "GROUP_K": F03_GROUP_K,
        "BLOCK_K_search": "fixed_by_packed_e8m0_scales",
        "chain": "prepare-megamoe-inputs",
    }
    return hidden_states, topk_weights, topk_ids, is_padding, shape


def _allocate_outputs(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    *,
    device: torch.device,
):
    x_fp8 = torch.empty(
        (num_tokens, hidden_size),
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    x_sf = torch.empty(
        (num_tokens, triton.cdiv(hidden_size, F03_BLOCK_K)),
        device=device,
        dtype=torch.int32,
    )
    topk_idx = torch.empty((num_tokens, top_k), device=device, dtype=torch.int64)
    topk_weights = torch.empty((num_tokens, top_k), device=device, dtype=torch.float32)
    return x_fp8, x_sf, topk_idx, topk_weights


def _pack_outputs(
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            x_fp8.view(torch.uint8).flatten(),
            x_sf.view(torch.uint8).flatten(),
            topk_idx.view(torch.uint8).flatten(),
            topk_weights.view(torch.uint8).flatten(),
        )
    )


def _compare_exact_packed_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerances: CorrectnessTolerances,
) -> dict[str, Any]:
    del tolerances
    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    matches = reference == candidate
    mismatch_count = int((~matches).sum().item())
    first_mismatch = -1
    if mismatch_count:
        first_mismatch = int((~matches).nonzero()[0].item())
    return {
        "passed": mismatch_count == 0,
        "comparison": "bitwise_exact_payload_scales_ids_weights",
        "mismatch_count": mismatch_count,
        "first_mismatch_byte": first_mismatch,
        "num_bytes": int(reference.numel()),
    }


def _run_candidate_kernel(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    is_padding: torch.Tensor | None,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    *,
    num_warps: int,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    grid = (num_tokens, triton.cdiv(hidden_size, F03_BLOCK_K))
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        padding_stride_m,
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        topk_ids.shape[1],
        BLOCK_K=F03_BLOCK_K,
        GROUP_K=F03_GROUP_K,
        BLOCK_TOPK=block_topk,
        num_warps=num_warps,
    )
    return _pack_outputs(x_fp8, x_sf, topk_idx_out, topk_weights_out)


def build_f03_prepare_megamoe_case(args: Mapping[str, Any]) -> ChainCase:
    block_k, candidate_num_warps = _resolve_candidate_geometry(args)
    candidate_provider = str(args.get("candidate_provider", "direct"))
    if candidate_provider not in {"direct", "production"}:
        raise ValueError("candidate_provider must be direct or production")
    if (
        candidate_provider == "production"
        and candidate_num_warps != _prepare_megamoe_num_warps()
    ):
        raise ValueError("production candidate warps must match the SM architecture")
    hidden_states, topk_weights, topk_ids, is_padding, shape = _make_inputs(args)
    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    baseline_outputs = _allocate_outputs(
        num_tokens,
        hidden_size,
        top_k,
        device=hidden_states.device,
    )
    candidate_outputs = _allocate_outputs(
        num_tokens,
        hidden_size,
        top_k,
        device=hidden_states.device,
    )

    def run_baseline() -> torch.Tensor:
        return _run_candidate_kernel(
            hidden_states,
            topk_weights,
            topk_ids,
            is_padding,
            *baseline_outputs,
            num_warps=F03_LEGACY_NUM_WARPS,
        )

    if candidate_provider == "production":

        def run_candidate() -> torch.Tensor:
            prepare_megamoe_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                *candidate_outputs,
                is_padding=is_padding,
            )
            return _pack_outputs(*candidate_outputs)

        candidate_symbol = "prepare_megamoe_inputs"
    else:

        def run_candidate() -> torch.Tensor:
            return _run_candidate_kernel(
                hidden_states,
                topk_weights,
                topk_ids,
                is_padding,
                *candidate_outputs,
                num_warps=candidate_num_warps,
            )

        candidate_symbol = "_prepare_megamoe_inputs_kernel"

    shape["candidate_BLOCK_K"] = block_k
    shape["candidate_num_warps"] = candidate_num_warps
    shape["candidate_provider"] = candidate_provider
    return ChainCase(
        baseline=Provider(
            "legacy-prepare-megamoe-warps4",
            run_baseline,
            {
                "kernel": "_prepare_megamoe_inputs_kernel",
                "BLOCK_K": F03_BLOCK_K,
                "GROUP_K": F03_GROUP_K,
                "num_warps": F03_LEGACY_NUM_WARPS,
            },
        ),
        candidate=Provider(
            f"{candidate_provider}-prepare-megamoe-warps{candidate_num_warps}",
            run_candidate,
            {
                "symbol": candidate_symbol,
                "candidate_provider": candidate_provider,
                "BLOCK_K": F03_BLOCK_K,
                "GROUP_K": F03_GROUP_K,
                "num_warps": candidate_num_warps,
                "BLOCK_K_search": "fixed_by_packed_e8m0_scales",
            },
            correctness_comparator=_compare_exact_packed_outputs,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
