# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _compute_global_topk_indices_and_lens_kernel,
)
from vllm.models.deepseek_v4.sparse_mla import (
    _build_c128a_topk_metadata_kernel,
)
from vllm.utils.torch_utils import set_random_seed

COMPRESS_RATIO = 128
MODEL_MAX_TOKENS = 1_048_576
MAX_COMPRESSED_TOKENS = MODEL_MAX_TOKENS // COMPRESS_RATIO
KV_BLOCK_SIZE = 256
COMPRESSED_BLOCK_SIZE = KV_BLOCK_SIZE // COMPRESS_RATIO


def _load_global_prefill_candidate() -> Callable[..., Any] | None:
    from vllm.models.deepseek_v4 import sparse_mla

    return getattr(sparse_mla, "build_c128a_global_prefill_metadata", None)


def _split_lengths(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    decode_request_batch = int(args.get("decode_request_batch", 0))
    draft_tokens = int(args.get("draft_tokens", 7))
    num_prefill_tokens = int(args.get("num_prefill_tokens", 0))
    prefill_request_batch = int(
        args.get("prefill_request_batch", int(num_prefill_tokens > 0))
    )
    max_compressed_tokens = int(
        args.get("max_compressed_tokens", MAX_COMPRESSED_TOKENS)
    )
    padding_every = int(args.get("padding_every", 0))
    seed = int(args.get("seed", 0))

    if context_tokens <= 0 or context_tokens > MODEL_MAX_TOKENS:
        raise ValueError("C02 context_tokens must be in (0, model_max_tokens]")
    if decode_request_batch not in (0, 1, 4):
        raise ValueError("C02 decode_request_batch must be 0, 1, or 4")
    if draft_tokens < 0:
        raise ValueError("C02 draft_tokens must be non-negative")
    if num_prefill_tokens < 0:
        raise ValueError("C02 num_prefill_tokens must be non-negative")
    if prefill_request_batch not in (0, 1, 4):
        raise ValueError("C02 prefill_request_batch must be 0, 1, or 4")
    if (num_prefill_tokens == 0) != (prefill_request_batch == 0):
        raise ValueError("C02 prefill request count must match prefill tokens")
    if num_prefill_tokens > context_tokens * max(prefill_request_batch, 1):
        raise ValueError("C02 prefill rows exceed the available request contexts")
    if max_compressed_tokens != MAX_COMPRESSED_TOKENS:
        raise ValueError("C02 uses the model-faithful compressed width 8192")
    if padding_every < 0:
        raise ValueError("C02 padding_every must be non-negative")

    decode_width = draft_tokens + 1
    num_decode_tokens = decode_request_batch * decode_width
    num_tokens = num_decode_tokens + num_prefill_tokens
    num_requests = decode_request_batch + prefill_request_batch
    if num_tokens == 0:
        raise ValueError("C02 benchmark requires at least one token")

    set_random_seed(seed)
    device = torch.device("cuda")
    decode_req_ids = torch.arange(
        decode_request_batch, dtype=torch.int32, device=device
    ).repeat_interleave(decode_width)
    if num_decode_tokens:
        decode_offsets = torch.arange(decode_width, dtype=torch.int64, device=device)
        decode_positions = (
            torch.full(
                (decode_request_batch, 1),
                context_tokens,
                dtype=torch.int64,
                device=device,
            )
            + decode_offsets
        ).reshape(-1)
    else:
        decode_positions = torch.empty(0, dtype=torch.int64, device=device)

    prefill_lengths = _split_lengths(num_prefill_tokens, prefill_request_batch)
    prefill_req_parts = []
    prefill_position_parts = []
    for request_index, query_len in enumerate(prefill_lengths):
        request_id = decode_request_batch + request_index
        prefill_req_parts.append(
            torch.full((query_len,), request_id, dtype=torch.int32, device=device)
        )
        prefill_position_parts.append(
            torch.arange(
                context_tokens - query_len,
                context_tokens,
                dtype=torch.int64,
                device=device,
            )
        )
    prefill_req_ids = (
        torch.cat(prefill_req_parts)
        if prefill_req_parts
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    prefill_positions = (
        torch.cat(prefill_position_parts)
        if prefill_position_parts
        else torch.empty(0, dtype=torch.int64, device=device)
    )
    token_to_req = torch.cat((decode_req_ids, prefill_req_ids))
    positions = torch.cat((decode_positions, prefill_positions))

    blocks_per_request = MAX_COMPRESSED_TOKENS // COMPRESSED_BLOCK_SIZE
    block_table_storage = torch.empty(
        (num_requests, blocks_per_request + 3), dtype=torch.int32, device=device
    )
    logical_block_table = torch.arange(
        num_requests * blocks_per_request, dtype=torch.int32, device=device
    ).view(num_requests, blocks_per_request)
    block_table_storage[:, :blocks_per_request] = logical_block_table.flip(1)
    block_table = block_table_storage[:, :blocks_per_request]

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    if padding_every:
        slot_mapping[padding_every - 1 :: padding_every] = -1

    return {
        "positions": positions,
        "token_to_req": token_to_req,
        "block_table": block_table,
        "slot_mapping": slot_mapping,
        "num_decode_tokens": num_decode_tokens,
        "num_prefill_tokens": num_prefill_tokens,
        "max_compressed_tokens": max_compressed_tokens,
        "shape": {
            "name": (
                f"d{num_decode_tokens}-p{num_prefill_tokens}"
                f"-ctx{context_tokens}-r{num_requests}-w{max_compressed_tokens}"
            ),
            "context_tokens": context_tokens,
            "decode_request_batch": decode_request_batch,
            "draft_tokens": draft_tokens,
            "num_decode_tokens": num_decode_tokens,
            "num_prefill_tokens": num_prefill_tokens,
            "prefill_request_batch": prefill_request_batch,
            "num_tokens": num_tokens,
            "num_requests": num_requests,
            "compress_ratio": COMPRESS_RATIO,
            "kv_block_size": KV_BLOCK_SIZE,
            "compressed_block_size": COMPRESSED_BLOCK_SIZE,
            "max_compressed_tokens": max_compressed_tokens,
            "padding_every": padding_every,
        },
    }


def _reference(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    positions = inputs["positions"]
    token_to_req = inputs["token_to_req"]
    block_table = inputs["block_table"]
    slot_mapping = inputs["slot_mapping"]
    num_decode_tokens = int(inputs["num_decode_tokens"])
    max_compressed_tokens = int(inputs["max_compressed_tokens"])

    num_compressed = torch.div(
        positions + 1, COMPRESS_RATIO, rounding_mode="floor"
    ).clamp(max=max_compressed_tokens)
    columns = torch.arange(
        max_compressed_tokens, dtype=torch.int32, device=positions.device
    )
    local_indices = torch.where(
        columns[None, :] < num_compressed[:, None], columns[None, :], -1
    )
    safe_indices = local_indices.clamp_min(0)
    request_indices = token_to_req[:, None].expand_as(local_indices)
    block_indices = torch.div(
        safe_indices, COMPRESSED_BLOCK_SIZE, rounding_mode="floor"
    )
    block_numbers = block_table[request_indices, block_indices]
    global_indices = block_numbers * COMPRESSED_BLOCK_SIZE
    global_indices += safe_indices % COMPRESSED_BLOCK_SIZE
    global_indices = torch.where(local_indices >= 0, global_indices, -1)
    lens = num_compressed.to(torch.int32)
    lens = torch.where(slot_mapping >= 0, lens, 0)
    return {
        "decode_global": global_indices[:num_decode_tokens],
        "decode_lens": lens[:num_decode_tokens],
        "prefill_local": local_indices[num_decode_tokens:],
        "prefill_global": global_indices[num_decode_tokens:],
        "prefill_lens": lens[num_decode_tokens:],
    }


def _launch_triton_metadata(
    global_decode: torch.Tensor,
    decode_lens: torch.Tensor,
    prefill_output: torch.Tensor,
    inputs: Mapping[str, Any],
) -> None:
    positions = inputs["positions"]
    num_tokens = positions.shape[0]
    _build_c128a_topk_metadata_kernel[(num_tokens,)](
        global_decode,
        global_decode.stride(0),
        decode_lens,
        prefill_output,
        prefill_output.stride(0),
        decode_lens,
        positions,
        COMPRESS_RATIO,
        inputs["max_compressed_tokens"],
        inputs["num_decode_tokens"],
        inputs["token_to_req"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        COMPRESSED_BLOCK_SIZE,
        inputs["slot_mapping"],
        BLOCK_SIZE=1024,
        PREFILL_GLOBAL=False,
    )


def _launch_global_mapping(
    output: torch.Tensor,
    output_lens: torch.Tensor,
    local_indices: torch.Tensor,
    inputs: Mapping[str, Any],
) -> None:
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    if num_prefill_tokens == 0:
        return
    num_decode_tokens = int(inputs["num_decode_tokens"])
    prefill_slice = slice(num_decode_tokens, num_decode_tokens + num_prefill_tokens)
    _compute_global_topk_indices_and_lens_kernel[(num_prefill_tokens,)](
        output,
        output.stride(0),
        output_lens,
        local_indices,
        local_indices.stride(0),
        local_indices.shape[-1],
        inputs["token_to_req"][prefill_slice],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        COMPRESSED_BLOCK_SIZE,
        inputs["slot_mapping"][prefill_slice] >= 0,
        TRITON_BLOCK_SIZE=1024,
    )


def _build_case(args: Mapping[str, Any], *, include_global_mapping: bool) -> ChainCase:
    inputs = _make_inputs(args)
    expected = _reference(inputs)
    shape = inputs["shape"]
    num_decode_tokens = int(inputs["num_decode_tokens"])
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    width = int(inputs["max_compressed_tokens"])
    device = inputs["positions"].device

    baseline_decode = torch.empty(
        (num_decode_tokens, width), dtype=torch.int32, device=device
    )
    baseline_decode_lens = torch.empty(
        num_decode_tokens, dtype=torch.int32, device=device
    )
    baseline_prefill = torch.empty(
        (num_prefill_tokens, width), dtype=torch.int32, device=device
    )
    baseline_prefill_global = torch.empty_like(baseline_prefill)
    baseline_prefill_lens = torch.empty(
        num_prefill_tokens, dtype=torch.int32, device=device
    )

    candidate_decode = torch.empty_like(baseline_decode)
    candidate_decode_lens = torch.empty_like(baseline_decode_lens)
    candidate_prefill = torch.empty_like(baseline_prefill)
    candidate_prefill_global = torch.empty_like(baseline_prefill)
    candidate_prefill_lens = torch.empty_like(baseline_prefill_lens)
    candidate_impl = _load_global_prefill_candidate()
    candidate_active = candidate_impl is not None and num_prefill_tokens > 0

    def run_baseline() -> torch.Tensor:
        _launch_triton_metadata(
            baseline_decode, baseline_decode_lens, baseline_prefill, inputs
        )
        if include_global_mapping:
            _launch_global_mapping(
                baseline_prefill_global,
                baseline_prefill_lens,
                baseline_prefill,
                inputs,
            )
            return baseline_prefill_global
        return baseline_prefill if num_prefill_tokens else baseline_decode

    def run_candidate() -> torch.Tensor:
        if candidate_active:
            assert candidate_impl is not None
            candidate_impl(
                inputs["positions"],
                COMPRESS_RATIO,
                num_decode_tokens,
                inputs["token_to_req"],
                inputs["block_table"],
                COMPRESSED_BLOCK_SIZE,
                inputs["slot_mapping"],
                candidate_decode,
                candidate_decode_lens,
                candidate_prefill,
                candidate_prefill_lens,
                max_compressed_tokens=width,
            )
            return candidate_prefill if num_prefill_tokens else candidate_decode

        _launch_triton_metadata(
            candidate_decode, candidate_decode_lens, candidate_prefill, inputs
        )
        if include_global_mapping:
            _launch_global_mapping(
                candidate_prefill_global,
                candidate_prefill_lens,
                candidate_prefill,
                inputs,
            )
            return candidate_prefill_global
        return candidate_prefill if num_prefill_tokens else candidate_decode

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        baseline_prefill_expected = (
            expected["prefill_global"]
            if include_global_mapping
            else expected["prefill_local"]
        )
        candidate_prefill_expected = (
            expected["prefill_global"]
            if include_global_mapping or candidate_active
            else expected["prefill_local"]
        )
        baseline_prefill_actual = (
            baseline_prefill_global if include_global_mapping else baseline_prefill
        )
        candidate_prefill_actual = (
            candidate_prefill
            if candidate_active
            else (
                candidate_prefill_global
                if include_global_mapping
                else candidate_prefill
            )
        )
        checks = {
            "baseline_decode": torch.equal(baseline_decode, expected["decode_global"]),
            "candidate_decode": torch.equal(
                candidate_decode, expected["decode_global"]
            ),
            "baseline_decode_lens": torch.equal(
                baseline_decode_lens, expected["decode_lens"]
            ),
            "candidate_decode_lens": torch.equal(
                candidate_decode_lens, expected["decode_lens"]
            ),
            "baseline_prefill": torch.equal(
                baseline_prefill_actual, baseline_prefill_expected
            ),
            "candidate_prefill": torch.equal(
                candidate_prefill_actual, candidate_prefill_expected
            ),
        }
        if include_global_mapping:
            checks["baseline_prefill_lens"] = torch.equal(
                baseline_prefill_lens, expected["prefill_lens"]
            )
        if include_global_mapping or candidate_active:
            checks["candidate_prefill_lens"] = torch.equal(
                candidate_prefill_lens, expected["prefill_lens"]
            )
        return {
            "passed": all(checks.values()),
            "exact": checks,
            "candidate_prefill_is_global": candidate_active,
        }

    shape["chain"] = (
        "c128a-metadata-global-prefill"
        if include_global_mapping
        else "c128a-metadata-producer"
    )
    return ChainCase(
        baseline=Provider(
            "triton-c128a-local-prefill",
            run_baseline,
            {
                "implementation": "triton",
                "prefill_output": "global" if include_global_mapping else "local",
                "includes_global_mapping": include_global_mapping,
            },
        ),
        candidate=Provider(
            "fused-c128a-global-prefill" if candidate_active else "triton-mirror",
            run_candidate,
            {
                "implementation": "triton-fused" if candidate_active else "triton",
                "candidate_active": candidate_active,
                "prefill_output": (
                    "global" if candidate_active or include_global_mapping else "local"
                ),
                "includes_global_mapping": include_global_mapping,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_c02_standalone_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_global_mapping=False)


def build_c02_global_prefill_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, include_global_mapping=True)
