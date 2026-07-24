# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.f06_factories import (
    build_f06_compressed_slot_mapping_case,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    _compressed_slot_mapping_kernel,
    get_compressed_slot_mapping,
)

KV_BLOCK_SIZE = 256


def _reference(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    num_tokens: int,
) -> torch.Tensor:
    expected = torch.full(
        (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
    )
    for req_idx in range(seq_lens.numel()):
        query_start = int(query_start_loc[req_idx].item())
        query_end = int(query_start_loc[req_idx + 1].item())
        query_len = query_end - query_start
        start_pos = int(seq_lens[req_idx].item()) - query_len
        for offset in range(query_len):
            pos = start_pos + offset
            if (pos + 1) % compress_ratio != 0:
                continue
            pos_after_compress = pos // compress_ratio
            block_id = pos_after_compress // block_size
            block_number = int(block_table[req_idx, block_id].item())
            expected[query_start + offset] = (
                block_number * block_size + pos_after_compress % block_size
            )
    return expected


def _make_inputs(
    *,
    context_tokens: int,
    compress_ratio: int,
    request_batch: int,
    decode_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    device = torch.device("cuda")
    block_size = KV_BLOCK_SIZE // compress_ratio
    compressed_seq_len = context_tokens + decode_len + compress_ratio - 1
    compressed_seq_len //= compress_ratio
    max_blocks = (compressed_seq_len + block_size - 1) // block_size
    num_tokens = request_batch * decode_len

    query_start_loc = (
        torch.arange(request_batch + 1, dtype=torch.int32, device=device) * decode_len
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
        + 1000
    )
    if request_batch > 1:
        block_table = block_table + torch.arange(
            request_batch, dtype=torch.int32, device=device
        ).view(request_batch, 1) * (max_blocks + 11)
    return query_start_loc, seq_lens, block_table.contiguous(), block_size, num_tokens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize("context_tokens", [8_192, 32_768, 131_072])
@pytest.mark.parametrize("decode_len", [1, 8])
@pytest.mark.parametrize("request_batch", [1, 4])
def test_get_compressed_slot_mapping_matches_model_contract(
    compress_ratio: int,
    context_tokens: int,
    decode_len: int,
    request_batch: int,
) -> None:
    query_start_loc, seq_lens, block_table, block_size, num_tokens = _make_inputs(
        context_tokens=context_tokens,
        compress_ratio=compress_ratio,
        request_batch=request_batch,
        decode_len=decode_len,
    )
    out = torch.full((num_tokens + 13,), -777, dtype=torch.int64, device="cuda")

    actual = get_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        out=out,
    )

    expected = _reference(
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        num_tokens,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)
    assert torch.equal(out[:num_tokens], expected)
    assert (out[num_tokens:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_slot_mapping_uses_storage_block_size_at_boundary() -> None:
    device = torch.device("cuda")
    compress_ratio = 4
    block_size = KV_BLOCK_SIZE // compress_ratio
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([260], dtype=torch.int32, device=device)
    block_table = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    actual = get_compressed_slot_mapping(
        8,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
    )

    expected = torch.tensor(
        [-1, -1, -1, 5 * block_size + 63, -1, -1, -1, 7 * block_size],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_slot_mapping_kernel_candidate_matches_wrapper_exact() -> None:
    query_start_loc, seq_lens, block_table, block_size, num_tokens = _make_inputs(
        context_tokens=32_768,
        compress_ratio=4,
        request_batch=4,
        decode_len=8,
    )
    baseline = get_compressed_slot_mapping(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        4,
    )
    candidate = torch.full_like(baseline, -1)

    _compressed_slot_mapping_kernel[(4,)](
        candidate,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        4,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=256,
        num_warps=4,
    )

    assert torch.equal(candidate, baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_f06_benchmark_factory_matches_exact_and_is_cuda_graph_safe() -> None:
    case = build_f06_compressed_slot_mapping_case(
        {
            "context_tokens": 8_192,
            "compress_ratio": 128,
            "request_batch": 4,
            "decode_len": 8,
            "max_num_batched_tokens": 8192,
            "candidate_block_size": 256,
            "candidate_num_warps": 4,
        }
    )
    baseline = case.baseline.fn().detach().clone()
    candidate = case.candidate.fn().detach().clone()
    correctness = case.baseline.correctness_comparator(
        baseline,
        candidate,
        case.tolerances,
    )
    assert correctness["passed"]

    graph = torch.cuda.CUDAGraph()
    for _ in range(3):
        case.candidate.fn()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(3):
            case.candidate.fn()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(case.candidate.fn(), baseline)
