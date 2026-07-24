# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.v1.attention.backends.mla.compressor_utils import (
    _compressed_slot_mapping_kernel,
    get_compressed_slot_mapping,
)

KV_BLOCK_SIZE = 256
MODEL_CONTEXT_TOKENS = (8_192, 32_768, 131_072)
MODEL_COMPRESS_RATIOS = (4, 128)
REQUEST_BATCHES = (1, 4)
DECODE_LENGTHS = (1, 8)
LEGAL_CANDIDATE_BLOCK_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)
LEGAL_CANDIDATE_NUM_WARPS = (1, 2, 4, 8)


def _make_query_start_loc(request_batch: int, decode_len: int) -> list[int]:
    starts = [0]
    for _ in range(request_batch):
        starts.append(starts[-1] + decode_len)
    return starts


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args.get("context_tokens", 8_192))
    compress_ratio = int(args.get("compress_ratio", 4))
    request_batch = int(args.get("request_batch", 1))
    decode_len = int(args.get("decode_len", 8))
    candidate_block_size = int(args.get("candidate_block_size", 1024))
    candidate_num_warps = args.get("candidate_num_warps")
    max_num_batched_tokens = int(args.get("max_num_batched_tokens", 8192))
    block_table_offset = int(args.get("block_table_offset", 8192))

    if context_tokens not in MODEL_CONTEXT_TOKENS:
        raise ValueError("F06 context_tokens must be 8K, 32K, or 128K")
    if compress_ratio not in MODEL_COMPRESS_RATIOS:
        raise ValueError("F06 compress_ratio must be 4 or 128")
    if request_batch not in REQUEST_BATCHES:
        raise ValueError("F06 request_batch must be 1 or 4")
    if decode_len not in DECODE_LENGTHS:
        raise ValueError("F06 decode_len must be 1 or 8")
    if KV_BLOCK_SIZE % compress_ratio != 0:
        raise ValueError("F06 KV block size must divide by compress_ratio")
    if candidate_block_size not in LEGAL_CANDIDATE_BLOCK_SIZES:
        raise ValueError("F06 candidate_block_size is not legal")
    if candidate_num_warps is not None:
        candidate_num_warps = int(candidate_num_warps)
        if candidate_num_warps not in LEGAL_CANDIDATE_NUM_WARPS:
            raise ValueError("F06 candidate_num_warps is not legal")
    if block_table_offset < 0:
        raise ValueError("F06 block_table_offset must be non-negative")

    storage_block_size = KV_BLOCK_SIZE // compress_ratio
    compressed_seq_len = context_tokens + decode_len + compress_ratio - 1
    compressed_seq_len //= compress_ratio
    max_blocks = (compressed_seq_len + storage_block_size - 1) // storage_block_size
    num_tokens = request_batch * decode_len
    if max_num_batched_tokens < num_tokens:
        raise ValueError("max_num_batched_tokens must cover all active tokens")
    device = torch.device("cuda")

    query_start_loc = torch.tensor(
        _make_query_start_loc(request_batch, decode_len),
        dtype=torch.int32,
        device=device,
    )
    seq_lens = torch.full(
        (request_batch,),
        context_tokens + decode_len,
        dtype=torch.int32,
        device=device,
    )
    block_table = (
        torch.arange(
            request_batch * max_blocks,
            dtype=torch.int32,
            device=device,
        ).view(request_batch, max_blocks)
        + block_table_offset
    )
    if request_batch > 1:
        row_offsets = torch.arange(
            request_batch, dtype=torch.int32, device=device
        ).view(request_batch, 1)
        block_table = block_table + row_offsets * (max_blocks + 19)

    baseline_out = torch.empty(max_num_batched_tokens, dtype=torch.int64, device=device)
    candidate_out = torch.empty_like(baseline_out)

    return {
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "block_table": block_table.contiguous(),
        "block_size": storage_block_size,
        "compress_ratio": compress_ratio,
        "num_tokens": num_tokens,
        "baseline_out": baseline_out,
        "candidate_out": candidate_out,
        "candidate_block_size": candidate_block_size,
        "candidate_num_warps": candidate_num_warps,
        "shape": {
            "name": (
                f"f06-c{compress_ratio}-ctx{context_tokens}"
                f"-d{decode_len}-b{request_batch}"
            ),
            "context_tokens": context_tokens,
            "decode_len": decode_len,
            "request_batch": request_batch,
            "num_tokens": num_tokens,
            "compress_ratio": compress_ratio,
            "kv_block_size": KV_BLOCK_SIZE,
            "storage_block_size": storage_block_size,
            "max_blocks": max_blocks,
            "max_num_batched_tokens": max_num_batched_tokens,
            "candidate_block_size": candidate_block_size,
            "candidate_num_warps": candidate_num_warps,
        },
    }


def _launch_baseline(inputs: Mapping[str, Any]) -> torch.Tensor:
    get_compressed_slot_mapping(
        int(inputs["num_tokens"]),
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["block_table"],
        int(inputs["block_size"]),
        int(inputs["compress_ratio"]),
        out=inputs["baseline_out"],
    )
    return inputs["baseline_out"]


def _launch_candidate(inputs: Mapping[str, Any]) -> torch.Tensor:
    out = inputs["candidate_out"]
    out.fill_(-1)
    kwargs: dict[str, Any] = {}
    candidate_num_warps = inputs["candidate_num_warps"]
    if candidate_num_warps is not None:
        kwargs["num_warps"] = int(candidate_num_warps)
    _compressed_slot_mapping_kernel[(inputs["block_table"].shape[0],)](
        out[: int(inputs["num_tokens"])],
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        int(inputs["block_size"]),
        int(inputs["compress_ratio"]),
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=int(inputs["candidate_block_size"]),
        **kwargs,
    )
    return out


def _compare_exact(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    _: CorrectnessTolerances,
) -> dict[str, Any]:
    matches = torch.equal(candidate, reference)
    mismatch_count = int((candidate != reference).sum().item())
    first_mismatch = -1
    if mismatch_count:
        first_mismatch = int((candidate != reference).nonzero()[0].item())
    return {
        "passed": bool(matches),
        "exact": bool(matches),
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
    }


def build_f06_compressed_slot_mapping_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    candidate_num_warps = inputs["candidate_num_warps"]
    candidate_suffix = f"block{inputs['candidate_block_size']}"
    if candidate_num_warps is not None:
        candidate_suffix += f"-warps{candidate_num_warps}"

    return ChainCase(
        baseline=Provider(
            "F06-production-wrapper",
            lambda: _launch_baseline(inputs),
            {
                "wrapper": "get_compressed_slot_mapping",
                "kernel": "_compressed_slot_mapping_kernel",
                "triton_block_size": 1024,
                "timed_provider_excludes_setup": True,
            },
            correctness_comparator=_compare_exact,
        ),
        candidate=Provider(
            f"F06-production-kernel-{candidate_suffix}",
            lambda: _launch_candidate(inputs),
            {
                "kernel": "_compressed_slot_mapping_kernel",
                "triton_block_size": inputs["candidate_block_size"],
                "num_warps": candidate_num_warps,
                "timed_provider_excludes_setup": True,
            },
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
