# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_prefill_metadata_kernel,
    _compute_swa_indices_and_lens_kernel,
)

WINDOW_SIZE = 128
KV_BLOCK_SIZE = 64


def _load_fused_candidate() -> Callable[..., Any] | None:
    from vllm.v1.attention.backends.mla import sparse_swa

    return getattr(sparse_swa, "build_swa_prefill_indices_and_metadata", None)


def _split_lengths(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    decode_request_batch = int(args.get("decode_request_batch", 0))
    draft_tokens = int(args.get("draft_tokens", 7))
    num_prefill_tokens = int(args["num_prefill_tokens"])
    prefill_request_batch = int(args.get("prefill_request_batch", 1))
    padding_every = int(args.get("padding_every", 0))
    seed = int(args.get("seed", 0))

    if context_tokens <= 0:
        raise ValueError("C03 context_tokens must be positive")
    if decode_request_batch not in (0, 1, 4):
        raise ValueError("C03 decode_request_batch must be 0, 1, or 4")
    if draft_tokens < 0:
        raise ValueError("C03 draft_tokens must be non-negative")
    if num_prefill_tokens <= 0:
        raise ValueError("C03 requires at least one prefill token")
    if prefill_request_batch not in (1, 4):
        raise ValueError("C03 prefill_request_batch must be 1 or 4")
    if num_prefill_tokens < prefill_request_batch:
        raise ValueError("C03 requires at least one token per prefill request")
    if num_prefill_tokens > context_tokens * prefill_request_batch:
        raise ValueError("C03 prefill query length exceeds context")
    if padding_every < 0:
        raise ValueError("C03 padding_every must be non-negative")

    set_random_seed(seed)
    device = torch.device("cuda")
    decode_query_len = draft_tokens + 1
    decode_query_lens = [decode_query_len] * decode_request_batch
    prefill_query_lens = _split_lengths(num_prefill_tokens, prefill_request_batch)
    query_lens = decode_query_lens + prefill_query_lens
    num_decodes = decode_request_batch
    num_prefills = prefill_request_batch
    num_decode_tokens = sum(decode_query_lens)
    num_tokens = num_decode_tokens + num_prefill_tokens
    num_requests = num_decodes + num_prefills

    query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.empty(num_requests + 1, dtype=torch.int32, device=device)
    query_start_loc[0] = 0
    torch.cumsum(query_lens_tensor, dim=0, out=query_start_loc[1:])
    seq_lens = torch.full(
        (num_requests,), context_tokens, dtype=torch.int32, device=device
    )
    token_to_req = torch.arange(
        num_requests, dtype=torch.int32, device=device
    ).repeat_interleave(query_lens_tensor)
    is_valid_token = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if padding_every:
        is_valid_token[padding_every - 1 :: padding_every] = False

    blocks_per_request = (context_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    block_table_storage = torch.empty(
        (num_requests, blocks_per_request + 3), dtype=torch.int32, device=device
    )
    logical_blocks = torch.arange(
        num_requests * blocks_per_request, dtype=torch.int32, device=device
    ).view(num_requests, blocks_per_request)
    block_table_storage[:, :blocks_per_request] = logical_blocks.flip(1)

    return {
        "seq_lens": seq_lens,
        "query_start_loc": query_start_loc,
        "token_to_req": token_to_req,
        "is_valid_token": is_valid_token,
        "block_table": block_table_storage[:, :blocks_per_request],
        "num_decodes": num_decodes,
        "num_prefills": num_prefills,
        "num_decode_tokens": num_decode_tokens,
        "num_prefill_tokens": num_prefill_tokens,
        "shape": {
            "name": (
                f"d{num_decode_tokens}-p{num_prefill_tokens}"
                f"-ctx{context_tokens}-r{num_requests}"
            ),
            "context_tokens": context_tokens,
            "decode_request_batch": decode_request_batch,
            "draft_tokens": draft_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_prefill_tokens": num_prefill_tokens,
            "prefill_request_batch": prefill_request_batch,
            "num_requests": num_requests,
            "num_tokens": num_tokens,
            "window_size": WINDOW_SIZE,
            "block_size": KV_BLOCK_SIZE,
            "padding_every": padding_every,
        },
    }


def _reference(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    num_decodes = int(inputs["num_decodes"])
    num_prefills = int(inputs["num_prefills"])
    num_decode_tokens = int(inputs["num_decode_tokens"])
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    query_start_loc = inputs["query_start_loc"]
    seq_lens = inputs["seq_lens"]

    prefill_req_ids = torch.arange(
        num_decodes,
        num_decodes + num_prefills,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    query_lens = query_start_loc[prefill_req_ids + 1] - query_start_loc[prefill_req_ids]
    prefix_lens = seq_lens[prefill_req_ids] - query_lens
    gather_lens = query_lens + torch.minimum(
        prefix_lens,
        torch.full_like(prefix_lens, WINDOW_SIZE - 1),
    )

    token_indices = torch.arange(
        num_decode_tokens,
        num_decode_tokens + num_prefill_tokens,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    request_indices = inputs["token_to_req"][token_indices]
    query_starts = query_start_loc[request_indices]
    request_query_lens = query_start_loc[request_indices + 1] - query_starts
    request_prefix_lens = seq_lens[request_indices] - request_query_lens
    positions = request_prefix_lens + token_indices - query_starts
    start_positions = torch.clamp(positions - WINDOW_SIZE + 1, min=0)
    swa_lens = positions + 1 - start_positions

    offsets = torch.arange(WINDOW_SIZE, dtype=torch.int32, device=seq_lens.device)
    source_positions = start_positions[:, None] + offsets[None, :]
    valid_entries = offsets[None, :] < swa_lens[:, None]
    block_indices = torch.div(source_positions, KV_BLOCK_SIZE, rounding_mode="floor")
    block_numbers = inputs["block_table"][request_indices[:, None], block_indices]
    swa_indices = block_numbers * KV_BLOCK_SIZE
    swa_indices += source_positions % KV_BLOCK_SIZE
    swa_indices = torch.where(valid_entries, swa_indices, -1)

    valid_tokens = inputs["is_valid_token"][token_indices]
    swa_lens = torch.where(valid_tokens, swa_lens, 0).to(torch.int32)
    swa_indices = torch.where(valid_tokens[:, None], swa_indices, -1)
    return {
        "gather_lens": gather_lens.to(torch.int32),
        "swa_indices": swa_indices.to(torch.int32),
        "swa_lens": swa_lens,
    }


def _launch_prefill_metadata(output: torch.Tensor, inputs: Mapping[str, Any]) -> None:
    num_prefills = int(inputs["num_prefills"])
    _compute_prefill_metadata_kernel[(1,)](
        output,
        inputs["seq_lens"],
        inputs["query_start_loc"],
        num_prefills,
        inputs["num_decodes"],
        WINDOW_SIZE,
        BLOCK_SIZE=triton.next_power_of_2(num_prefills),
    )


def _launch_swa_indices(
    output: torch.Tensor,
    output_lens: torch.Tensor,
    inputs: Mapping[str, Any],
) -> None:
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
        output,
        output.stride(0),
        output_lens,
        WINDOW_SIZE,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req"],
        inputs["is_valid_token"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        KV_BLOCK_SIZE,
        output_lens,
        inputs["num_decodes"],
        inputs["num_prefills"],
        token_offset=inputs["num_decode_tokens"],
        TRITON_BLOCK_SIZE=1024,
        PREFILL_METADATA_BLOCK_SIZE=1,
        WRITE_PREFILL_METADATA=False,
    )


def _exact_comparator(
    expected: Mapping[str, torch.Tensor],
    baseline_indices: torch.Tensor,
    baseline_lens: torch.Tensor,
    baseline_gather_lens: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_lens: torch.Tensor,
    candidate_gather_lens: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, CorrectnessTolerances], dict[str, Any]]:
    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "baseline_indices": torch.equal(baseline_indices, expected["swa_indices"]),
            "baseline_lens": torch.equal(baseline_lens, expected["swa_lens"]),
            "baseline_gather_lens": torch.equal(
                baseline_gather_lens, expected["gather_lens"]
            ),
            "candidate_indices": torch.equal(
                candidate_indices, expected["swa_indices"]
            ),
            "candidate_lens": torch.equal(candidate_lens, expected["swa_lens"]),
            "candidate_gather_lens": torch.equal(
                candidate_gather_lens, expected["gather_lens"]
            ),
        }
        return {"passed": all(checks.values()), "exact": checks}

    return compare


def build_c03_standalone_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    expected = _reference(inputs)
    num_prefills = int(inputs["num_prefills"])
    device = inputs["seq_lens"].device
    baseline = torch.empty(num_prefills, dtype=torch.int32, device=device)
    candidate = torch.empty_like(baseline)

    def run_baseline() -> torch.Tensor:
        _launch_prefill_metadata(baseline, inputs)
        return baseline

    def run_candidate() -> torch.Tensor:
        _launch_prefill_metadata(candidate, inputs)
        return candidate

    def compare(
        baseline_output: torch.Tensor,
        candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        return {
            "passed": torch.equal(baseline_output, expected["gather_lens"])
            and torch.equal(candidate_output, expected["gather_lens"]),
            "baseline_exact": torch.equal(baseline_output, expected["gather_lens"]),
            "candidate_exact": torch.equal(candidate_output, expected["gather_lens"]),
        }

    shape = dict(inputs["shape"])
    shape["chain"] = "prefill-gather-lens-producer"
    return ChainCase(
        baseline=Provider("triton-prefill-metadata", run_baseline),
        candidate=Provider(
            "triton-mirror",
            run_candidate,
            {"candidate_active": False},
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_c03_prefill_swa_chain_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    expected = _reference(inputs)
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    num_prefills = int(inputs["num_prefills"])
    device = inputs["seq_lens"].device

    baseline_storage = torch.empty(
        (num_prefill_tokens, WINDOW_SIZE + 5), dtype=torch.int32, device=device
    )
    baseline_indices = baseline_storage[:, :WINDOW_SIZE]
    baseline_lens = torch.empty(num_prefill_tokens, dtype=torch.int32, device=device)
    baseline_gather_lens = torch.empty(num_prefills, dtype=torch.int32, device=device)

    candidate_storage = torch.empty(
        (num_prefill_tokens, WINDOW_SIZE + 5), dtype=torch.int32, device=device
    )
    candidate_indices = candidate_storage[:, :WINDOW_SIZE]
    candidate_lens = torch.empty_like(baseline_lens)
    candidate_gather_lens = torch.empty_like(baseline_gather_lens)
    candidate_impl = _load_fused_candidate()
    candidate_active = candidate_impl is not None

    def run_baseline() -> torch.Tensor:
        _launch_swa_indices(baseline_indices, baseline_lens, inputs)
        _launch_prefill_metadata(baseline_gather_lens, inputs)
        return baseline_indices

    def run_candidate() -> torch.Tensor:
        if candidate_active:
            assert candidate_impl is not None
            candidate_impl(
                candidate_indices,
                candidate_lens,
                candidate_gather_lens,
                WINDOW_SIZE,
                inputs["query_start_loc"],
                inputs["seq_lens"],
                inputs["token_to_req"],
                inputs["is_valid_token"],
                inputs["block_table"],
                KV_BLOCK_SIZE,
                token_offset=inputs["num_decode_tokens"],
                num_decodes=inputs["num_decodes"],
            )
            return candidate_indices

        _launch_swa_indices(candidate_indices, candidate_lens, inputs)
        _launch_prefill_metadata(candidate_gather_lens, inputs)
        return candidate_indices

    shape = dict(inputs["shape"])
    shape["chain"] = "prefill-swa-indices-plus-gather-lens"
    return ChainCase(
        baseline=Provider(
            "triton-swa-then-prefill-metadata",
            run_baseline,
            {"launches": 2},
        ),
        candidate=Provider(
            "fused-swa-prefill-metadata" if candidate_active else "triton-mirror",
            run_candidate,
            {
                "candidate_active": candidate_active,
                "launches": 1 if candidate_active else 2,
            },
            correctness_comparator=_exact_comparator(
                expected,
                baseline_indices,
                baseline_lens,
                baseline_gather_lens,
                candidate_indices,
                candidate_lens,
                candidate_gather_lens,
            ),
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
