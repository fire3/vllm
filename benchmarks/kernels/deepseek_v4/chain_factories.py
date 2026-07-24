# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops import fused_inv_rope_fp8_quant
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_sm12x_fp8_einsum,
)
from vllm.utils.torch_utils import set_random_seed

HEAD_DIM = 512
HEADS_PER_GROUP = 8
NOPE_DIM = 448
ROPE_DIM = 64
OUT_RANK = 1024


def build_a02_quant_einsum_chain(args: Mapping[str, Any]) -> ChainCase:
    num_tokens = int(args["num_tokens"])
    num_groups = int(args["num_groups"])
    seed = int(args.get("seed", 0))
    if num_groups not in (2, 8):
        raise ValueError("A02 chain requires num_groups in (2, 8)")

    set_random_seed(seed)
    device = torch.device("cuda")
    num_heads = num_groups * HEADS_PER_GROUP
    hidden_size = HEADS_PER_GROUP * HEAD_DIM
    o = torch.randn(
        (num_tokens, num_heads, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )
    positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
    frequencies = torch.randn(
        (num_tokens, ROPE_DIM // 2),
        device=device,
        dtype=torch.float32,
    )
    cos_sin_cache = torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)
    weight = torch.randn(
        (num_groups, OUT_RANK, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    weight_scale = torch.empty(
        (num_groups, OUT_RANK // 128, hidden_size // 128),
        device=device,
        dtype=torch.float32,
    ).uniform_(0.001, 0.01)
    baseline_output = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device=device,
        dtype=torch.bfloat16,
    )
    candidate_output = torch.empty_like(baseline_output)

    def run_baseline() -> torch.Tensor:
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            num_groups,
            HEADS_PER_GROUP,
            nope_dim=NOPE_DIM,
            rope_dim=ROPE_DIM,
        )
        deepseek_v4_sm12x_fp8_einsum(
            o_fp8,
            o_scale,
            weight,
            weight_scale,
            baseline_output,
        )
        return baseline_output

    def run_candidate() -> torch.Tensor:
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            num_groups,
            HEADS_PER_GROUP,
            nope_dim=NOPE_DIM,
            rope_dim=ROPE_DIM,
            compact_scales=True,
        )
        deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            weight.flatten(0, 1),
            weight_scale.flatten(0, 1),
            candidate_output,
            "bhr,hdr->bhd",
            [1, 128, 128],
        )
        return candidate_output

    shape = {
        "name": f"t{num_tokens}-g{num_groups}-n{OUT_RANK}-k{hidden_size}",
        "T": num_tokens,
        "G": num_groups,
        "N": OUT_RANK,
        "K": hidden_size,
        "chain": "inverse-rope-fp8-quant-einsum",
    }
    return ChainCase(
        baseline=Provider(
            "triton-chain",
            run_baseline,
            {
                "compact_scales": False,
                "einsum_backend": "triton",
            },
        ),
        candidate=Provider(
            "cutlass-chain",
            run_candidate,
            {
                "compact_scales": True,
                "einsum_backend": "cutlass-batched-direct",
            },
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=1.0,
            rtol=0.05,
            max_mean_relative=0.001,
            min_cosine=0.999,
            require_allclose=False,
        ),
    )
