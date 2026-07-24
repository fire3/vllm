# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    _PREFILL_CHUNK_METADATA_PARALLEL_BLOCK_SIZE,
    _build_prefill_chunk_metadata_parallel_kernel,
    _use_parallel_prefill_chunk_metadata,
    build_prefill_chunk_metadata,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)


def _cumulative(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _local_length(global_length: int, rank: int, world: int, interleave: int) -> int:
    base = global_length // interleave // world * interleave
    remainder = global_length - base * world
    return base + min(max(remainder - rank * interleave, 0), interleave)


def _reference(
    query_lens: list[int],
    seq_lens: list[int],
    compress_ratio: int,
    *,
    query_slice: slice | None = None,
    dcp_rank: int = 0,
    dcp_world: int = 1,
    dcp_interleave: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compressed_lens = [seq_len // compress_ratio for seq_len in seq_lens]
    query_starts = _cumulative(query_lens)
    global_row_starts = _cumulative(compressed_lens)
    local_lens = [
        _local_length(length, dcp_rank, dcp_world, dcp_interleave)
        for length in compressed_lens
    ]
    local_row_starts = _cumulative(local_lens)
    slice_start = 0 if query_slice is None else query_slice.start
    slice_stop = query_starts[-1] if query_slice is None else query_slice.stop

    ks = [0] * (slice_stop - slice_start)
    ke = [0] * (slice_stop - slice_start)
    token_to_seq: list[int] = []
    for req_idx, (query_len, seq_len, compressed_len) in enumerate(
        zip(query_lens, seq_lens, compressed_lens)
    ):
        prefix_len = seq_len - query_len
        for offset in range(query_len):
            absolute_query = query_starts[req_idx] + offset
            if slice_start <= absolute_query < slice_stop:
                output_idx = absolute_query - slice_start
                global_context = (prefix_len + 1 + offset) // compress_ratio
                ks[output_idx] = local_row_starts[req_idx]
                ke[output_idx] = local_row_starts[req_idx] + _local_length(
                    global_context,
                    dcp_rank,
                    dcp_world,
                    dcp_interleave,
                )
        token_to_seq.extend([req_idx] * compressed_len)

    return (
        torch.tensor(global_row_starts, dtype=torch.int32, device="cuda"),
        torch.tensor(local_row_starts, dtype=torch.int32, device="cuda"),
        torch.tensor(token_to_seq, dtype=torch.int32, device="cuda"),
        torch.tensor(ks, dtype=torch.int32, device="cuda"),
        torch.tensor(ke, dtype=torch.int32, device="cuda"),
    )


def _build_chunk(
    query_lens: list[int],
    seq_lens: list[int],
    compress_ratio: int,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
    query_slice: slice | None = None,
    dcp_rank: int = 0,
    dcp_world: int = 1,
    dcp_interleave: int = 1,
):
    query_starts = _cumulative(query_lens)
    compressed_lens = [seq_len // compress_ratio for seq_len in seq_lens]
    query_start_loc_cpu = torch.tensor(query_starts, dtype=torch.int32)
    compressed_seq_lens_cpu = torch.tensor(compressed_lens, dtype=torch.int32)
    query_start_loc = query_start_loc_cpu.cuda()
    uncompressed_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    compressed_seq_lens = compressed_seq_lens_cpu.cuda()
    block_table = torch.zeros((len(query_lens), 1), dtype=torch.int32, device="cuda")
    return build_prefill_chunk_metadata(
        start_idx,
        len(query_lens) if end_idx is None else end_idx,
        query_start_loc,
        query_start_loc_cpu,
        uncompressed_seq_lens,
        compressed_seq_lens,
        compressed_seq_lens_cpu,
        block_table,
        compress_ratio,
        query_slice=query_slice,
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world,
        cp_kv_cache_interleave_size=dcp_interleave,
    )


@requires_sm120
@pytest.mark.parametrize(
    ("compress_ratio", "seq_lens"),
    [(4, [19, 4120, 39]), (128, [383, 131200, 1023])],
)
@torch.inference_mode()
def test_prefill_chunk_metadata_tiled_ragged_query_slice_matches_reference(
    compress_ratio: int, seq_lens: list[int]
):
    query_lens = [3, 1025, 7]
    query_slice = slice(1000, 1032)
    chunk = _build_chunk(
        query_lens,
        seq_lens,
        compress_ratio,
        start_idx=1,
        end_idx=3,
        query_slice=query_slice,
    )
    assert chunk is not None

    expected = _reference(
        query_lens[1:],
        seq_lens[1:],
        compress_ratio,
        query_slice=query_slice,
    )
    assert chunk.token_start == 1003
    assert chunk.token_end == 1035
    assert chunk.skip_kv_gather
    torch.testing.assert_close(chunk.cu_seq_lens, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(chunk.local_cu_seq_lens, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(chunk.token_to_seq, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(chunk.cu_seqlen_ks, expected[3], rtol=0, atol=0)
    torch.testing.assert_close(chunk.cu_seqlen_ke, expected[4], rtol=0, atol=0)


def test_prefill_chunk_metadata_parallel_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: capability == 120,
    )
    non_sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: False,
    )

    monkeypatch.setattr(indexer, "current_platform", sm120)
    assert not _use_parallel_prefill_chunk_metadata(1024, 1)
    assert _use_parallel_prefill_chunk_metadata(1025, 1)
    assert not _use_parallel_prefill_chunk_metadata(4096, 2)
    monkeypatch.setattr(indexer, "current_platform", non_sm120)
    assert not _use_parallel_prefill_chunk_metadata(4096, 1)


@requires_sm120
@torch.inference_mode()
def test_prefill_chunk_metadata_parallel_cuda_graph_replays_dynamic_input():
    query_len = 2048
    compress_ratio = 4
    compressed_len = 2048
    query_start_loc = torch.tensor([0, query_len], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([8192], dtype=torch.int32, device="cuda")
    cu_seq_lens = torch.tensor([0, compressed_len], dtype=torch.int32, device="cuda")
    token_to_seq = torch.empty(compressed_len, dtype=torch.int32, device="cuda")
    ks = torch.empty(query_len, dtype=torch.int32, device="cuda")
    ke = torch.empty(query_len, dtype=torch.int32, device="cuda")
    grid = (
        1,
        (compressed_len + _PREFILL_CHUNK_METADATA_PARALLEL_BLOCK_SIZE - 1)
        // _PREFILL_CHUNK_METADATA_PARALLEL_BLOCK_SIZE,
    )

    def run_kernel() -> None:
        _build_prefill_chunk_metadata_parallel_kernel[grid](
            query_start_loc,
            seq_lens,
            cu_seq_lens,
            cu_seq_lens,
            token_to_seq,
            ks,
            ke,
            0,
            query_len,
            0,
            1,
            1,
            BLOCK_SIZE=_PREFILL_CHUNK_METADATA_PARALLEL_BLOCK_SIZE,
            COMPRESS_RATIO=compress_ratio,
            num_warps=4,
        )

    run_kernel()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_kernel()

    seq_lens.add_(4)
    token_to_seq.fill_(-1)
    ks.fill_(-1)
    ke.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()

    offsets = torch.arange(query_len, dtype=torch.int32, device="cuda")
    expected_ke = (seq_lens - query_len + 1 + offsets) // compress_ratio
    torch.testing.assert_close(token_to_seq, torch.zeros_like(token_to_seq))
    torch.testing.assert_close(ks, torch.zeros_like(ks))
    torch.testing.assert_close(ke, expected_ke)


@requires_cuda
@torch.inference_mode()
def test_prefill_chunk_metadata_long_dcp_fallback_matches_reference():
    query_lens = [1025]
    seq_lens = [4100]
    compress_ratio = 4
    chunk = _build_chunk(
        query_lens,
        seq_lens,
        compress_ratio,
        dcp_rank=1,
        dcp_world=2,
    )
    assert chunk is not None

    expected = _reference(
        query_lens,
        seq_lens,
        compress_ratio,
        dcp_rank=1,
        dcp_world=2,
    )
    assert chunk.local_total_seq_lens == 512
    assert chunk.max_local_total_seq_lens == 513
    torch.testing.assert_close(chunk.cu_seq_lens, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(chunk.local_cu_seq_lens, expected[1], rtol=0, atol=0)
    torch.testing.assert_close(chunk.token_to_seq, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(chunk.cu_seqlen_ks, expected[3], rtol=0, atol=0)
    torch.testing.assert_close(chunk.cu_seqlen_ke, expected[4], rtol=0, atol=0)
