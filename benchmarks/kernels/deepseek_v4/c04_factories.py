# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import triton
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_swa_indices_and_lens_kernel,
    build_swa_prefill_indices_and_metadata,
)

WINDOW_SIZE = 128
KV_BLOCK_SIZE = 64


def _has_tiled_candidate() -> bool:
    from vllm.v1.attention.backends.mla import sparse_swa

    return hasattr(sparse_swa, "build_swa_indices_and_metadata_tiled")


def _split_lengths(total: int, count: int) -> list[int]:
    if count == 0:
        return []
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    decode_request_batch = int(args.get("decode_request_batch", 0))
    draft_tokens = int(args.get("draft_tokens", 7))
    num_prefill_tokens = int(args.get("num_prefill_tokens", 0))
    prefill_request_batch = int(args.get("prefill_request_batch", 0))
    request_padding = int(args.get("request_padding", 0))
    token_padding = int(args.get("token_padding", 0))
    padding_every = int(args.get("padding_every", 0))
    seed = int(args.get("seed", 0))

    if context_tokens <= 0:
        raise ValueError("C04 context_tokens must be positive")
    if decode_request_batch not in (0, 1, 4):
        raise ValueError("C04 decode_request_batch must be 0, 1, or 4")
    if draft_tokens < 0:
        raise ValueError("C04 draft_tokens must be non-negative")
    if prefill_request_batch not in (0, 1, 4):
        raise ValueError("C04 prefill_request_batch must be 0, 1, or 4")
    if (num_prefill_tokens == 0) != (prefill_request_batch == 0):
        raise ValueError("C04 prefill tokens and request batch must both be zero")
    if num_prefill_tokens and num_prefill_tokens < prefill_request_batch:
        raise ValueError("C04 requires at least one token per prefill request")
    if num_prefill_tokens > context_tokens * max(prefill_request_batch, 1):
        raise ValueError("C04 prefill query length exceeds context")
    if request_padding < 0 or token_padding < 0 or padding_every < 0:
        raise ValueError("C04 padding values must be non-negative")
    if request_padding and not num_prefill_tokens:
        raise ValueError("C04 request padding requires an actual prefill")
    if not decode_request_batch and not num_prefill_tokens:
        raise ValueError("C04 requires decode or prefill tokens")

    set_random_seed(seed)
    device = torch.device("cuda")
    decode_query_len = draft_tokens + 1
    decode_query_lens = [decode_query_len] * decode_request_batch
    prefill_query_lens = _split_lengths(num_prefill_tokens, prefill_request_batch)
    query_lens = decode_query_lens + prefill_query_lens + [0] * request_padding
    num_decodes = decode_request_batch
    num_prefills = prefill_request_batch + request_padding
    num_decode_tokens = sum(decode_query_lens)
    num_prefill_kernel_tokens = num_prefill_tokens + token_padding
    num_kernel_tokens = num_decode_tokens + num_prefill_kernel_tokens
    num_actual_tokens = num_decode_tokens + num_prefill_tokens
    num_requests = len(query_lens)

    query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.empty(num_requests + 1, dtype=torch.int32, device=device)
    query_start_loc[0] = 0
    torch.cumsum(query_lens_tensor, dim=0, out=query_start_loc[1:])
    seq_lens = torch.zeros(num_requests, dtype=torch.int32, device=device)
    num_actual_requests = decode_request_batch + prefill_request_batch
    seq_lens[:num_actual_requests] = context_tokens

    token_to_req = torch.zeros(num_kernel_tokens, dtype=torch.int32, device=device)
    if num_actual_tokens:
        actual_token_to_req = torch.arange(
            num_actual_requests, dtype=torch.int32, device=device
        ).repeat_interleave(query_lens_tensor[:num_actual_requests])
        token_to_req[: actual_token_to_req.shape[0]] = actual_token_to_req

    is_valid_token = torch.zeros(num_kernel_tokens, dtype=torch.bool, device=device)
    is_valid_token[:num_actual_tokens] = True
    if padding_every:
        is_valid_token[padding_every - 1 :: padding_every] = False

    blocks_per_request = (context_tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    block_table_storage = torch.zeros(
        (num_requests, blocks_per_request + 3), dtype=torch.int32, device=device
    )
    if num_actual_requests:
        logical_blocks = torch.arange(
            num_actual_requests * blocks_per_request,
            dtype=torch.int32,
            device=device,
        ).view(num_actual_requests, blocks_per_request)
        block_table_storage[:num_actual_requests, :blocks_per_request] = (
            logical_blocks.flip(1)
        )

    return {
        "seq_lens": seq_lens,
        "query_start_loc": query_start_loc,
        "token_to_req": token_to_req,
        "is_valid_token": is_valid_token,
        "block_table": block_table_storage[:, :blocks_per_request],
        "num_decodes": num_decodes,
        "num_prefills": num_prefills,
        "num_decode_tokens": num_decode_tokens,
        "num_actual_prefill_tokens": num_prefill_tokens,
        "num_prefill_tokens": num_prefill_kernel_tokens,
        "shape": {
            "name": (
                f"d{num_decode_tokens}-p{num_prefill_kernel_tokens}"
                f"-ctx{context_tokens}-r{num_requests}"
            ),
            "context_tokens": context_tokens,
            "decode_request_batch": decode_request_batch,
            "draft_tokens": draft_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_actual_prefill_tokens": num_prefill_tokens,
            "num_prefill_tokens": num_prefill_kernel_tokens,
            "prefill_request_batch": prefill_request_batch,
            "request_padding": request_padding,
            "token_padding": token_padding,
            "num_requests": num_requests,
            "num_kernel_tokens": num_kernel_tokens,
            "window_size": WINDOW_SIZE,
            "block_size": KV_BLOCK_SIZE,
            "padding_every": padding_every,
        },
    }


def _reference_segment(
    inputs: Mapping[str, Any], *, token_offset: int, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = inputs["seq_lens"].device
    if num_tokens == 0:
        return (
            torch.empty((0, WINDOW_SIZE), dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )

    token_indices = torch.arange(
        token_offset, token_offset + num_tokens, dtype=torch.int32, device=device
    )
    request_indices = inputs["token_to_req"][token_indices]
    query_start_loc = inputs["query_start_loc"]
    query_starts = query_start_loc[request_indices]
    query_lens = query_start_loc[request_indices + 1] - query_starts
    prefix_lens = inputs["seq_lens"][request_indices] - query_lens
    positions = prefix_lens + token_indices - query_starts
    start_positions = torch.clamp(positions - WINDOW_SIZE + 1, min=0)
    swa_lens = positions + 1 - start_positions

    offsets = torch.arange(WINDOW_SIZE, dtype=torch.int32, device=device)
    source_positions = start_positions[:, None] + offsets[None, :]
    valid_entries = offsets[None, :] < swa_lens[:, None]
    block_indices = torch.div(
        source_positions, KV_BLOCK_SIZE, rounding_mode="floor"
    ).clamp_(0, inputs["block_table"].shape[1] - 1)
    block_numbers = inputs["block_table"][request_indices[:, None], block_indices]
    swa_indices = block_numbers * KV_BLOCK_SIZE
    swa_indices += source_positions % KV_BLOCK_SIZE
    swa_indices = torch.where(valid_entries, swa_indices, -1)

    valid_tokens = inputs["is_valid_token"][token_indices]
    swa_lens = torch.where(valid_tokens, swa_lens, 0).to(torch.int32)
    swa_indices = torch.where(valid_tokens[:, None], swa_indices, -1)
    return swa_indices.to(torch.int32), swa_lens


def _reference_gather_lens(inputs: Mapping[str, Any]) -> torch.Tensor:
    num_decodes = int(inputs["num_decodes"])
    num_prefills = int(inputs["num_prefills"])
    device = inputs["seq_lens"].device
    if num_prefills == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    request_indices = torch.arange(
        num_decodes,
        num_decodes + num_prefills,
        dtype=torch.int32,
        device=device,
    )
    query_start_loc = inputs["query_start_loc"]
    query_lens = query_start_loc[request_indices + 1] - query_start_loc[request_indices]
    prefix_lens = inputs["seq_lens"][request_indices] - query_lens
    return (
        query_lens
        + torch.minimum(
            prefix_lens,
            torch.full_like(prefix_lens, WINDOW_SIZE - 1),
        )
    ).to(torch.int32)


def _launch_baseline_segment(
    output: torch.Tensor,
    output_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    inputs: Mapping[str, Any],
    *,
    token_offset: int,
    num_tokens: int,
    write_prefill_metadata: bool,
) -> None:
    if num_tokens == 0:
        return
    if write_prefill_metadata:
        _compute_swa_indices_and_lens_kernel[(num_tokens,)](
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
            gather_lens,
            inputs["num_decodes"],
            inputs["num_prefills"],
            token_offset=token_offset,
            TRITON_BLOCK_SIZE=1024,
            PREFILL_METADATA_BLOCK_SIZE=triton.next_power_of_2(inputs["num_prefills"]),
            WRITE_PREFILL_METADATA=True,
        )
        return
    _compute_swa_indices_and_lens_kernel[(num_tokens,)](
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
        gather_lens,
        inputs["num_decodes"],
        inputs["num_prefills"],
        token_offset=token_offset,
        TRITON_BLOCK_SIZE=1024,
        PREFILL_METADATA_BLOCK_SIZE=1,
        WRITE_PREFILL_METADATA=False,
    )


def _build_case(
    args: Mapping[str, Any], *, include_decode: bool, include_prefill: bool
) -> ChainCase:
    inputs = _make_inputs(args)
    num_decode_tokens = int(inputs["num_decode_tokens"]) if include_decode else 0
    num_prefill_tokens = int(inputs["num_prefill_tokens"]) if include_prefill else 0
    if include_decode and not num_decode_tokens:
        raise ValueError("C04 decode case requires decode tokens")
    if include_prefill and not num_prefill_tokens:
        raise ValueError("C04 prefill case requires prefill tokens")

    decode_expected = _reference_segment(
        inputs, token_offset=0, num_tokens=num_decode_tokens
    )
    prefill_expected = _reference_segment(
        inputs,
        token_offset=int(inputs["num_decode_tokens"]),
        num_tokens=num_prefill_tokens,
    )
    gather_expected = _reference_gather_lens(inputs) if include_prefill else None
    device = inputs["seq_lens"].device

    def make_matrix(rows: int) -> torch.Tensor:
        storage = torch.empty((rows, WINDOW_SIZE + 5), dtype=torch.int32, device=device)
        return storage[:, :WINDOW_SIZE]

    baseline_decode = make_matrix(num_decode_tokens)
    baseline_decode_lens = torch.empty(
        num_decode_tokens, dtype=torch.int32, device=device
    )
    baseline_prefill = make_matrix(num_prefill_tokens)
    baseline_prefill_lens = torch.empty(
        num_prefill_tokens, dtype=torch.int32, device=device
    )
    baseline_gather = torch.empty(
        int(inputs["num_prefills"]), dtype=torch.int32, device=device
    )
    candidate_decode = make_matrix(num_decode_tokens)
    candidate_decode_lens = torch.empty_like(baseline_decode_lens)
    candidate_prefill = make_matrix(num_prefill_tokens)
    candidate_prefill_lens = torch.empty_like(baseline_prefill_lens)
    candidate_gather = torch.empty_like(baseline_gather)
    candidate_active = _has_tiled_candidate()

    def run_baseline() -> torch.Tensor:
        if include_decode:
            _launch_baseline_segment(
                baseline_decode,
                baseline_decode_lens,
                baseline_gather,
                inputs,
                token_offset=0,
                num_tokens=num_decode_tokens,
                write_prefill_metadata=False,
            )
        if include_prefill:
            _launch_baseline_segment(
                baseline_prefill,
                baseline_prefill_lens,
                baseline_gather,
                inputs,
                token_offset=inputs["num_decode_tokens"],
                num_tokens=num_prefill_tokens,
                write_prefill_metadata=True,
            )
        return baseline_prefill if include_prefill else baseline_decode

    def run_candidate() -> torch.Tensor:
        if candidate_active:
            if include_decode:
                _launch_baseline_segment(
                    candidate_decode,
                    candidate_decode_lens,
                    candidate_gather,
                    inputs,
                    token_offset=0,
                    num_tokens=num_decode_tokens,
                    write_prefill_metadata=False,
                )
            if include_prefill:
                build_swa_prefill_indices_and_metadata(
                    candidate_prefill,
                    candidate_prefill_lens,
                    candidate_gather,
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
            return candidate_prefill if include_prefill else candidate_decode

        if include_decode:
            _launch_baseline_segment(
                candidate_decode,
                candidate_decode_lens,
                candidate_gather,
                inputs,
                token_offset=0,
                num_tokens=num_decode_tokens,
                write_prefill_metadata=False,
            )
        if include_prefill:
            _launch_baseline_segment(
                candidate_prefill,
                candidate_prefill_lens,
                candidate_gather,
                inputs,
                token_offset=inputs["num_decode_tokens"],
                num_tokens=num_prefill_tokens,
                write_prefill_metadata=True,
            )
        return candidate_prefill if include_prefill else candidate_decode

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks: dict[str, bool] = {}
        if include_decode:
            checks.update(
                baseline_decode=torch.equal(baseline_decode, decode_expected[0]),
                baseline_decode_lens=torch.equal(
                    baseline_decode_lens, decode_expected[1]
                ),
                candidate_decode=torch.equal(candidate_decode, decode_expected[0]),
                candidate_decode_lens=torch.equal(
                    candidate_decode_lens, decode_expected[1]
                ),
            )
        if include_prefill:
            assert gather_expected is not None
            checks.update(
                baseline_prefill=torch.equal(baseline_prefill, prefill_expected[0]),
                baseline_prefill_lens=torch.equal(
                    baseline_prefill_lens, prefill_expected[1]
                ),
                baseline_gather=torch.equal(baseline_gather, gather_expected),
                candidate_prefill=torch.equal(candidate_prefill, prefill_expected[0]),
                candidate_prefill_lens=torch.equal(
                    candidate_prefill_lens, prefill_expected[1]
                ),
                candidate_gather=torch.equal(candidate_gather, gather_expected),
            )
        return {"passed": all(checks.values()), "exact": checks}

    shape = dict(inputs["shape"])
    if include_decode and include_prefill:
        shape["chain"] = "causal-decode-then-prefill-swa-metadata"
    elif include_prefill:
        shape["chain"] = "causal-prefill-swa-metadata"
    else:
        shape["chain"] = "causal-decode-swa-metadata"
    launches = int(include_decode) + int(include_prefill)
    return ChainCase(
        baseline=Provider(
            "triton-token-per-program",
            run_baseline,
            {"launches": launches},
        ),
        candidate=Provider(
            "sm120-production-dispatch" if candidate_active else "triton-mirror",
            run_candidate,
            {"candidate_active": candidate_active, "launches": launches},
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_c04_decode_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_decode=True, include_prefill=False)


def build_c04_prefill_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_decode=False, include_prefill=True)


def build_c04_mixed_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_decode=True, include_prefill=True)
