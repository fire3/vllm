# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _can_use_global_topk_cutedsl,
    _compute_global_topk_indices_and_lens_kernel,
)
from vllm.utils.torch_utils import set_random_seed

COMPRESS_RATIO = 4
COMPRESSED_BLOCK_SIZE = 64
TOPK_TOKENS = 2048
RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024


def _load_cutedsl_candidate() -> Callable[..., None] | None:
    try:
        from vllm.models.deepseek_v4.nvidia.ops.global_topk_cutedsl import (
            launch_global_topk_indices_and_lens_cutedsl,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    return launch_global_topk_indices_and_lens_cutedsl


def _launch_triton(
    global_topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> None:
    num_tokens, topk = topk_indices.shape
    if num_tokens == 0:
        return
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )


def reference_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the C01 contract with ordinary indexed PyTorch operations."""
    valid_indices = topk_indices >= 0
    safe_indices = topk_indices.clamp_min(0)
    block_indices = torch.div(safe_indices, block_size, rounding_mode="floor")
    request_indices = token_to_req_indices[:, None].expand_as(block_indices)
    block_numbers = block_table[request_indices, block_indices]
    global_indices = block_numbers * block_size + safe_indices % block_size
    global_indices = torch.where(valid_indices, global_indices, -1)
    topk_lens = valid_indices.sum(dim=1, dtype=torch.int32)
    topk_lens = torch.where(is_valid_token, topk_lens, 0)
    return global_indices, topk_lens


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    request_batch = int(args.get("request_batch", 1))
    draft_tokens = int(args.get("draft_tokens", 7))
    num_tokens_arg = args.get("num_tokens")
    topk = int(args.get("topk", TOPK_TOKENS))
    compress_ratio = int(args.get("compress_ratio", COMPRESS_RATIO))
    block_size = int(args.get("block_size", COMPRESSED_BLOCK_SIZE))
    padding_every = int(args.get("padding_every", 0))
    candidate_threads = int(args.get("candidate_threads", 128))
    seed = int(args.get("seed", 0))
    if context_tokens <= 0:
        raise ValueError("C01 context_tokens must be positive")
    if request_batch not in (1, 4):
        raise ValueError("C01 request_batch must be 1 or 4")
    if draft_tokens < 0:
        raise ValueError("C01 draft_tokens must be non-negative")
    if topk <= 0:
        raise ValueError("C01 topk must be positive")
    if compress_ratio <= 0 or block_size <= 0:
        raise ValueError("C01 compression and block sizes must be positive")
    if padding_every < 0:
        raise ValueError("C01 padding_every must be non-negative")
    if candidate_threads not in (128, 256):
        raise ValueError("C01 candidate_threads must be 128 or 256")

    decode_width = draft_tokens + 1
    num_tokens = (
        request_batch * decode_width if num_tokens_arg is None else int(num_tokens_arg)
    )
    if num_tokens < 0:
        raise ValueError("C01 num_tokens must be non-negative")

    set_random_seed(seed)
    device = torch.device("cuda")
    compressed_context = (context_tokens + compress_ratio - 1) // compress_ratio
    if num_tokens == request_batch * decode_width:
        token_to_req_indices = torch.arange(
            request_batch, device=device, dtype=torch.int32
        ).repeat_interleave(decode_width)
        offsets = torch.arange(decode_width, device=device, dtype=torch.int32)
        valid_lens = (compressed_context - decode_width + 1 + offsets).clamp_min(0)
        valid_lens = valid_lens.repeat(request_batch)
    else:
        token_to_req_indices = (
            torch.arange(num_tokens, device=device, dtype=torch.int32) % request_batch
        )
        valid_lens = torch.full(
            (num_tokens,), compressed_context, device=device, dtype=torch.int32
        )
    valid_lens.clamp_(max=topk)

    max_blocks_per_request = max(1, (compressed_context + block_size - 1) // block_size)
    block_table = torch.arange(
        request_batch * max_blocks_per_request,
        device=device,
        dtype=torch.int32,
    ).view(request_batch, max_blocks_per_request)
    if request_batch > 1:
        block_table = block_table.flip(1).contiguous()

    columns = torch.arange(topk, device=device, dtype=torch.int32)
    if num_tokens:
        row_offsets = torch.arange(num_tokens, device=device, dtype=torch.int32)[
            :, None
        ]
        local_modulus = valid_lens.clamp_min(1)[:, None]
        topk_indices = (columns[None, :] * 67 + row_offsets * 131) % local_modulus
        topk_indices.masked_fill_(columns[None, :] >= valid_lens[:, None], -1)
    else:
        topk_indices = torch.empty((0, topk), device=device, dtype=torch.int32)

    is_valid_token = torch.ones(num_tokens, device=device, dtype=torch.bool)
    if padding_every:
        is_valid_token[padding_every - 1 :: padding_every] = False

    return {
        "topk_indices": topk_indices,
        "token_to_req_indices": token_to_req_indices,
        "block_table": block_table,
        "block_size": block_size,
        "is_valid_token": is_valid_token,
        "valid_lens": valid_lens,
        "shape": {
            "name": (
                f"t{num_tokens}-ctx{context_tokens}-b{request_batch}"
                f"-k{topk}-bs{block_size}"
            ),
            "num_tokens": num_tokens,
            "context_tokens": context_tokens,
            "compressed_context_tokens": compressed_context,
            "request_batch": request_batch,
            "draft_tokens": draft_tokens,
            "topk": topk,
            "compress_ratio": compress_ratio,
            "block_size": block_size,
            "padding_every": padding_every,
            "candidate_threads": candidate_threads,
        },
    }


def _exact_comparator(
    expected: torch.Tensor,
    baseline_lens: torch.Tensor,
    candidate_lens: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, CorrectnessTolerances], dict[str, Any]]:
    def compare(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        reference = torch.cat((reference, baseline_lens[:, None]), dim=1)
        candidate = torch.cat((candidate, candidate_lens[:, None]), dim=1)
        reference_exact = torch.equal(reference, expected)
        candidate_exact = torch.equal(candidate, expected)
        return {
            "passed": reference_exact and candidate_exact,
            "reference_exact": reference_exact,
            "candidate_exact": candidate_exact,
            "max_reference_mismatch": int((reference != expected).sum().item()),
            "max_candidate_mismatch": int((candidate != expected).sum().item()),
        }

    return compare


def _build_case(args: Mapping[str, Any], *, include_persistent_topk: bool) -> ChainCase:
    inputs = _make_inputs(args)
    source_indices = inputs["topk_indices"]
    token_to_req_indices = inputs["token_to_req_indices"]
    block_table = inputs["block_table"]
    block_size = inputs["block_size"]
    is_valid_token = inputs["is_valid_token"]
    shape = inputs["shape"]
    candidate_threads = int(shape["candidate_threads"])
    num_tokens, topk = source_indices.shape

    baseline_indices = torch.empty_like(source_indices)
    candidate_indices = torch.empty_like(source_indices)
    baseline_lens = torch.empty(num_tokens, device="cuda", dtype=torch.int32)
    candidate_lens = torch.empty_like(baseline_lens)
    candidate_impl = _load_cutedsl_candidate()
    candidate_active = candidate_impl is not None and _can_use_global_topk_cutedsl(
        topk=topk, num_tokens=num_tokens
    )

    baseline_workspace = None
    candidate_workspace = None
    baseline_topk_indices = source_indices
    candidate_topk_indices = source_indices
    logits = None
    persistent_lens = None
    if include_persistent_topk:
        if topk not in (512, 1024, 2048):
            raise ValueError("persistent_topk requires topk 512, 1024, or 2048")
        compressed_context = int(shape["compressed_context_tokens"])
        logits = torch.randn(
            (num_tokens, compressed_context), device="cuda", dtype=torch.float32
        )
        persistent_lens = inputs["valid_lens"].view(num_tokens, 1).contiguous()
        baseline_topk_indices = torch.empty_like(source_indices)
        candidate_topk_indices = torch.empty_like(source_indices)
        baseline_workspace = torch.empty(
            RADIX_TOPK_WORKSPACE_SIZE, device="cuda", dtype=torch.uint8
        )
        candidate_workspace = torch.empty_like(baseline_workspace)
        torch.ops._C.persistent_topk(
            logits,
            persistent_lens,
            baseline_topk_indices,
            baseline_workspace,
            topk,
            compressed_context,
        )
        source_indices = baseline_topk_indices

    expected_indices, expected_lens = reference_global_topk_indices_and_lens(
        source_indices,
        token_to_req_indices,
        block_table,
        block_size,
        is_valid_token,
    )
    expected = torch.cat((expected_indices, expected_lens[:, None]), dim=1)

    def run_baseline() -> torch.Tensor:
        if include_persistent_topk:
            assert logits is not None
            assert persistent_lens is not None
            assert baseline_workspace is not None
            torch.ops._C.persistent_topk(
                logits,
                persistent_lens,
                baseline_topk_indices,
                baseline_workspace,
                topk,
                logits.shape[1],
            )
        _launch_triton(
            baseline_indices,
            baseline_lens,
            baseline_topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token,
        )
        return baseline_indices

    def run_candidate() -> torch.Tensor:
        if include_persistent_topk:
            assert logits is not None
            assert persistent_lens is not None
            assert candidate_workspace is not None
            torch.ops._C.persistent_topk(
                logits,
                persistent_lens,
                candidate_topk_indices,
                candidate_workspace,
                topk,
                logits.shape[1],
            )
        if not candidate_active:
            _launch_triton(
                candidate_indices,
                candidate_lens,
                candidate_topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
            )
        else:
            assert candidate_impl is not None
            candidate_impl(
                candidate_indices,
                candidate_lens,
                candidate_topk_indices,
                token_to_req_indices,
                block_table,
                block_size,
                is_valid_token,
                threads=candidate_threads,
            )
        return candidate_indices

    shape["chain"] = (
        "persistent-topk-global-slot-mapping"
        if include_persistent_topk
        else "global-slot-mapping"
    )
    return ChainCase(
        baseline=Provider(
            "triton-global-topk",
            run_baseline,
            {
                "implementation": "triton",
                "persistent_topk": include_persistent_topk,
            },
        ),
        candidate=Provider(
            "cutedsl-global-topk" if candidate_active else "triton-mirror",
            run_candidate,
            {
                "implementation": "cutedsl" if candidate_active else "triton",
                "candidate_active": candidate_active,
                "persistent_topk": include_persistent_topk,
            },
            correctness_comparator=_exact_comparator(
                expected, baseline_lens, candidate_lens
            ),
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_c01_standalone_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_persistent_topk=False)


def build_c01_persistent_topk_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_persistent_topk=True)
