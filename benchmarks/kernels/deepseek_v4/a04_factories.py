# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.a03_factories import _compare_topk_with_ties
from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
    _view_packed_fp8_paged_mqa_kv_cache,
    fp8_paged_mqa_logits_triton,
)
from vllm.utils.deep_gemm import (
    fp8_fp4_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.torch_utils import set_random_seed

NUM_HEADS = 64
HEAD_DIM = 128
BLOCK_SIZE = 64
TOPK_TOKENS = 512
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _make_inputs(args: Mapping[str, Any]):
    context_tokens = int(args["context_tokens"])
    compress_ratio = int(args["compress_ratio"])
    request_batch = int(args.get("request_batch", 1))
    draft_tokens = int(args.get("draft_tokens", 7))
    max_model_tokens = int(args.get("max_model_tokens", context_tokens))
    seed = int(args.get("seed", 0))
    if context_tokens <= 0 or max_model_tokens < context_tokens:
        raise ValueError("A04 requires 0 < context_tokens <= max_model_tokens")
    if compress_ratio not in (4, 128):
        raise ValueError("A04 compress_ratio must be 4 or 128")
    if request_batch not in (1, 4):
        raise ValueError("A04 request_batch must be 1 or 4")
    if draft_tokens < 0:
        raise ValueError("A04 draft_tokens must be non-negative")

    set_random_seed(seed)
    device = torch.device("cuda")
    decode_width = draft_tokens + 1
    num_rows = request_batch * decode_width
    max_model_len = max_model_tokens // compress_ratio
    token_offsets = torch.arange(decode_width, device=device, dtype=torch.int32)
    request_context_lens = (
        context_tokens - decode_width + 1 + token_offsets
    ) // compress_ratio
    request_context_lens.clamp_(min=0, max=max_model_len)
    context_lens = (
        request_context_lens.repeat(request_batch).view(num_rows, 1).contiguous()
    )

    q = torch.randn(
        (num_rows, 1, NUM_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (num_rows, NUM_HEADS),
        device=device,
        dtype=torch.float32,
    )

    max_blocks_per_request = max(1, (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE)
    num_blocks = request_batch * max_blocks_per_request
    kv_cache = torch.empty(
        (num_blocks, BLOCK_SIZE, 1, HEAD_DIM + torch.float32.itemsize),
        device=device,
        dtype=torch.uint8,
    )
    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, HEAD_DIM)
    kv_values.copy_(
        torch.randn(
            kv_values.shape,
            device=device,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
    )
    kv_scales.uniform_(0.001, 0.01)

    request_block_tables = torch.arange(
        num_blocks,
        device=device,
        dtype=torch.int32,
    ).view(request_batch, max_blocks_per_request)
    block_tables = request_block_tables.repeat_interleave(decode_width, dim=0)
    positions = torch.arange(max_model_len, device=device).view(1, -1)
    valid_logits = positions < context_lens.view(-1, 1)
    row_starts = torch.zeros(num_rows, device=device, dtype=torch.int32)
    row_ends = context_lens.view(-1)
    shape = {
        "name": (
            f"b{request_batch}-n{decode_width}-ctx{context_tokens}"
            f"-max{max_model_tokens}-c{compress_ratio}"
        ),
        "request_batch": request_batch,
        "draft_tokens": draft_tokens,
        "flattened_rows": num_rows,
        "context_tokens": context_tokens,
        "max_model_tokens": max_model_tokens,
        "compressed_context_min": int(context_lens.min().item()),
        "compressed_context_max": int(context_lens.max().item()),
        "compressed_logits_width": max_model_len,
        "compress_ratio": compress_ratio,
        "H": NUM_HEADS,
        "D": HEAD_DIM,
        "block_size": BLOCK_SIZE,
    }
    return (
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        valid_logits,
        row_starts,
        row_ends,
        shape,
    )


def _production_paged_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    return fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )


def _build_case(args: Mapping[str, Any], *, include_topk: bool, include_metadata: bool):
    from vllm.third_party import deep_gemm

    (
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        valid_logits,
        row_starts,
        row_ends,
        shape,
    ) = _make_inputs(args)
    max_model_len = int(shape["compressed_logits_width"])
    deep_gemm.set_pdl(True)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens,
        BLOCK_SIZE,
        deep_gemm.get_num_sms(),
    )
    schedule_buffer = torch.empty_like(schedule_metadata)

    def run_baseline_logits() -> torch.Tensor:
        return fp8_paged_mqa_logits_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        )

    def run_candidate_logits() -> torch.Tensor:
        metadata = schedule_metadata
        if include_metadata:
            schedule_buffer.copy_(
                get_paged_mqa_logits_metadata(
                    context_lens,
                    BLOCK_SIZE,
                    deep_gemm.get_num_sms(),
                )
            )
            metadata = schedule_buffer
        return _production_paged_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            metadata,
            max_model_len,
        )

    def mask_logits(output: torch.Tensor) -> torch.Tensor:
        return output.masked_fill(~valid_logits, 0.0)

    if not include_topk:
        shape["chain"] = "fp8-paged-mqa-logits"
        return ChainCase(
            baseline=Provider(
                "triton-rowwise",
                run_baseline_logits,
                {"mma": "tf32", "preallocated_output": False},
                correctness_transform=mask_logits,
            ),
            candidate=Provider(
                "deepgemm-sm120-paged",
                run_candidate_logits,
                {
                    "mma": "fp8",
                    "pdl": True,
                    "preallocated_output": False,
                    "metadata_in_timing": include_metadata,
                },
                correctness_transform=mask_logits,
            ),
            shape=shape,
            tolerances=CorrectnessTolerances(
                atol=0.02,
                rtol=0.01,
                max_mean_relative=5e-4,
                min_cosine=0.99999,
                require_allclose=False,
            ),
        )

    baseline_output = torch.empty(
        (q.shape[0], TOPK_TOKENS),
        device=q.device,
        dtype=torch.int32,
    )
    candidate_output = torch.empty_like(baseline_output)
    baseline_workspace = torch.empty(
        RADIX_TOPK_WORKSPACE_SIZE,
        device=q.device,
        dtype=torch.uint8,
    )
    candidate_workspace = torch.empty_like(baseline_workspace)
    reference_logits = torch.empty(
        (q.shape[0], max_model_len),
        device=q.device,
        dtype=torch.float32,
    )

    def run_topk(
        logits: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
    ) -> torch.Tensor:
        torch.ops._C.persistent_topk(
            logits,
            context_lens,
            output,
            workspace,
            TOPK_TOKENS,
            max_model_len,
        )
        return output

    def run_baseline() -> torch.Tensor:
        logits = run_baseline_logits()
        reference_logits.copy_(logits)
        return run_topk(logits, baseline_output, baseline_workspace)

    def run_candidate() -> torch.Tensor:
        logits = run_candidate_logits()
        return run_topk(logits, candidate_output, candidate_workspace)

    def compare_topk(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        return _compare_topk_with_ties(
            reference,
            candidate,
            tolerances,
            reference_logits,
            row_starts,
            row_ends,
        )

    shape["chain"] = "fp8-paged-mqa-logits-persistent-topk"
    shape["topk"] = TOPK_TOKENS
    return ChainCase(
        baseline=Provider(
            "triton-rowwise-persistent-topk",
            run_baseline,
            {"mma": "tf32", "topk": "persistent"},
        ),
        candidate=Provider(
            "deepgemm-sm120-paged-persistent-topk",
            run_candidate,
            {
                "mma": "fp8",
                "pdl": True,
                "topk": "persistent",
                "metadata_in_timing": include_metadata,
            },
            correctness_comparator=compare_topk,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=0.02,
            rtol=0.01,
            max_mean_relative=5e-4,
            min_cosine=0.99999,
            require_allclose=False,
        ),
    )


def build_a04_logits_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_topk=False, include_metadata=False)


def build_a04_topk_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_topk=True, include_metadata=False)


def build_a04_production_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_topk=True, include_metadata=True)
