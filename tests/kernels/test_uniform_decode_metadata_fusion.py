# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.attention.backends.mla.indexer as indexer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    _launch_uniform_decode_metadata,
    _use_fused_uniform_decode_compression,
)

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)


def _run_metadata(
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    decode_len: int,
    compress_ratio: int,
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = seq_lens.shape[0] * decode_len
    output_seq_lens = torch.empty(num_tokens, dtype=torch.int32, device="cuda")
    output_block_table = torch.empty(
        (num_tokens, block_table.shape[1]), dtype=torch.int32, device="cuda"
    )
    output_decode_lens = torch.empty(num_tokens, dtype=torch.int32, device="cuda")
    fused = _launch_uniform_decode_metadata(
        seq_lens,
        output_seq_lens,
        block_table,
        output_block_table,
        output_decode_lens,
        num_tokens,
        decode_len,
        compress_ratio,
    )
    return fused, output_seq_lens, output_block_table, output_decode_lens


@requires_sm120
@pytest.mark.parametrize(
    ("request_batch", "decode_len", "compress_ratio"),
    [(1, 1, 1), (1, 8, 4), (4, 4, 4), (4, 8, 128)],
)
@torch.inference_mode()
def test_uniform_decode_metadata_fuses_compression_exactly(
    request_batch: int, decode_len: int, compress_ratio: int
):
    seq_lens = torch.arange(
        131072,
        131072 - request_batch,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    block_table = torch.arange(
        request_batch * 512, dtype=torch.int32, device="cuda"
    ).view(request_batch, 512)
    fused, output_seq_lens, output_block_table, output_decode_lens = _run_metadata(
        seq_lens, block_table, decode_len, compress_ratio
    )

    offsets = torch.arange(decode_len, dtype=torch.int32, device="cuda")
    expected_seq_lens = (
        (seq_lens[:, None] - decode_len + 1 + offsets[None, :]) // compress_ratio
    ).reshape(-1)
    expected_block_table = block_table.repeat_interleave(decode_len, dim=0)
    assert fused == (compress_ratio > 1)
    torch.testing.assert_close(output_seq_lens, expected_seq_lens, rtol=0, atol=0)
    torch.testing.assert_close(output_block_table, expected_block_table, rtol=0, atol=0)
    torch.testing.assert_close(
        output_decode_lens, torch.ones_like(output_decode_lens), rtol=0, atol=0
    )


@requires_sm120
@torch.inference_mode()
def test_uniform_decode_metadata_fusion_cuda_graph_replays_dynamic_inputs():
    decode_len = 8
    compress_ratio = 4
    seq_lens = torch.tensor([32768, 32761], dtype=torch.int32, device="cuda")
    block_table = torch.arange(1024, dtype=torch.int32, device="cuda").view(2, 512)
    num_tokens = seq_lens.shape[0] * decode_len
    output_seq_lens = torch.empty(num_tokens, dtype=torch.int32, device="cuda")
    output_block_table = torch.empty(
        (num_tokens, block_table.shape[1]), dtype=torch.int32, device="cuda"
    )
    output_decode_lens = torch.empty(num_tokens, dtype=torch.int32, device="cuda")

    def run_metadata() -> None:
        assert _launch_uniform_decode_metadata(
            seq_lens,
            output_seq_lens,
            block_table,
            output_block_table,
            output_decode_lens,
            num_tokens,
            decode_len,
            compress_ratio,
        )

    run_metadata()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_metadata()

    seq_lens.sub_(4)
    block_table.add_(17)
    output_seq_lens.fill_(-1)
    output_block_table.fill_(-1)
    output_decode_lens.fill_(-1)
    graph.replay()
    torch.accelerator.synchronize()

    offsets = torch.arange(decode_len, dtype=torch.int32, device="cuda")
    expected_seq_lens = (
        (seq_lens[:, None] - decode_len + 1 + offsets[None, :]) // compress_ratio
    ).reshape(-1)
    torch.testing.assert_close(output_seq_lens, expected_seq_lens, rtol=0, atol=0)
    torch.testing.assert_close(
        output_block_table,
        block_table.repeat_interleave(decode_len, dim=0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output_decode_lens, torch.ones_like(output_decode_lens), rtol=0, atol=0
    )


def test_uniform_decode_compression_guard_is_sm120_only(
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
    assert not _use_fused_uniform_decode_compression(1)
    assert _use_fused_uniform_decode_compression(4)
    assert _use_fused_uniform_decode_compression(128)
    monkeypatch.setattr(indexer, "current_platform", non_sm120)
    assert not _use_fused_uniform_decode_compression(4)


@requires_sm120
@torch.inference_mode()
def test_uniform_decode_builder_reports_compressed_output():
    buffer_capacity = 32
    tail_canary = -1234567
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.supports_varlen = False
    builder.decode_seq_lens_buffer = torch.full(
        (buffer_capacity,), tail_canary, dtype=torch.int32, device="cuda"
    )
    builder.expanded_block_table_buffer = torch.full(
        (buffer_capacity, 4), tail_canary, dtype=torch.int32, device="cuda"
    )
    builder.decode_lens_buffer = torch.full(
        (buffer_capacity,), tail_canary, dtype=torch.int32, device="cuda"
    )

    outputs = builder._prepare_decode_tensors(
        seq_lens=torch.tensor([32], dtype=torch.int32, device="cuda"),
        block_table=torch.tensor([[3, 5, 7, 11]], dtype=torch.int32, device="cuda"),
        decode_lens=torch.tensor([8], dtype=torch.int32, device="cuda"),
        decode_lens_cpu=torch.tensor([8], dtype=torch.int32),
        query_start_loc=torch.tensor([0], dtype=torch.int32, device="cuda"),
        num_decodes=1,
        num_decode_tokens=8,
        use_native=False,
        next_n=8,
        max_decode_len=8,
        compress_ratio=4,
    )
    seq_lens, block_table, decode_lens, batch_size, requires_padding, compressed = (
        outputs
    )
    torch.accelerator.synchronize()
    assert seq_lens.cpu().tolist() == [6, 6, 6, 7, 7, 7, 7, 8]
    assert block_table.cpu().tolist() == [[3, 5, 7, 11]] * 8
    assert decode_lens.cpu().tolist() == [1] * 8
    assert batch_size == 8
    assert not requires_padding
    assert compressed
    assert builder.decode_seq_lens_buffer[8:].cpu().tolist() == [0] * 24
    assert builder.expanded_block_table_buffer[8:].eq(tail_canary).all()
    assert builder.decode_lens_buffer[8:].eq(tail_canary).all()
