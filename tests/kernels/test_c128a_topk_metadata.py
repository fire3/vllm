# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import flashinfer_sparse
from vllm.models.deepseek_v4.sparse_mla import (
    build_c128a_global_prefill_metadata,
    build_c128a_topk_metadata,
)

COMPRESS_RATIO = 128
MAX_COMPRESSED_TOKENS = 128
COMPRESSED_BLOCK_SIZE = 2
NUM_DECODE_TOKENS = 4

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)


def _make_inputs() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    positions = torch.tensor(
        [255, 256, 511, 512, 0, 127, 128, 255, 384],
        dtype=torch.int64,
        device=device,
    )
    token_to_req = torch.tensor(
        [0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=torch.int32, device=device
    )
    blocks_per_request = MAX_COMPRESSED_TOKENS // COMPRESSED_BLOCK_SIZE
    block_table_storage = torch.empty(
        (4, blocks_per_request + 3), dtype=torch.int32, device=device
    )
    logical_blocks = torch.arange(
        4 * blocks_per_request, dtype=torch.int32, device=device
    ).view(4, blocks_per_request)
    block_table_storage[:, :blocks_per_request] = logical_blocks.flip(1)
    slot_mapping = torch.arange(positions.numel(), dtype=torch.int64, device=device)
    slot_mapping[1] = -1
    slot_mapping[6] = -1
    return {
        "positions": positions,
        "token_to_req": token_to_req,
        "block_table": block_table_storage[:, :blocks_per_request],
        "slot_mapping": slot_mapping,
    }


def _reference(
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = inputs["positions"]
    num_compressed = torch.div(
        positions + 1, COMPRESS_RATIO, rounding_mode="floor"
    ).clamp(max=MAX_COMPRESSED_TOKENS)
    columns = torch.arange(
        MAX_COMPRESSED_TOKENS, dtype=torch.int32, device=positions.device
    )
    local_indices = torch.where(
        columns[None, :] < num_compressed[:, None], columns[None, :], -1
    )
    safe_indices = local_indices.clamp_min(0)
    block_indices = torch.div(
        safe_indices, COMPRESSED_BLOCK_SIZE, rounding_mode="floor"
    )
    request_indices = inputs["token_to_req"][:, None].expand_as(block_indices)
    block_numbers = inputs["block_table"][request_indices, block_indices]
    global_indices = block_numbers * COMPRESSED_BLOCK_SIZE
    global_indices += safe_indices % COMPRESSED_BLOCK_SIZE
    global_indices = torch.where(local_indices >= 0, global_indices, -1)
    lens = num_compressed.to(torch.int32)
    lens = torch.where(inputs["slot_mapping"] >= 0, lens, 0)
    return local_indices, global_indices, lens


def _make_outputs(
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    num_prefill_tokens = num_tokens - NUM_DECODE_TOKENS
    decode_storage = torch.empty(
        (NUM_DECODE_TOKENS, MAX_COMPRESSED_TOKENS + 5),
        dtype=torch.int32,
        device=device,
    )
    prefill_storage = torch.empty(
        (num_prefill_tokens, MAX_COMPRESSED_TOKENS + 7),
        dtype=torch.int32,
        device=device,
    )
    return (
        decode_storage[:, :MAX_COMPRESSED_TOKENS],
        torch.empty(NUM_DECODE_TOKENS, dtype=torch.int32, device=device),
        prefill_storage[:, :MAX_COMPRESSED_TOKENS],
        torch.empty(num_prefill_tokens, dtype=torch.int32, device=device),
    )


@requires_cuda
@torch.inference_mode()
def test_c128a_local_and_global_prefill_metadata_match_reference():
    inputs = _make_inputs()
    expected_local, expected_global, expected_lens = _reference(inputs)

    local_outputs = _make_outputs(inputs["positions"].numel())
    local_decode, local_decode_lens, local_prefill, _ = local_outputs
    build_c128a_topk_metadata(
        inputs["positions"],
        COMPRESS_RATIO,
        NUM_DECODE_TOKENS,
        inputs["token_to_req"],
        inputs["block_table"],
        COMPRESSED_BLOCK_SIZE,
        inputs["slot_mapping"],
        local_decode,
        local_decode_lens,
        local_prefill,
        max_compressed_tokens=MAX_COMPRESSED_TOKENS,
    )

    global_outputs = _make_outputs(inputs["positions"].numel())
    global_decode, global_decode_lens, global_prefill, global_prefill_lens = (
        global_outputs
    )
    build_c128a_global_prefill_metadata(
        inputs["positions"],
        COMPRESS_RATIO,
        NUM_DECODE_TOKENS,
        inputs["token_to_req"],
        inputs["block_table"],
        COMPRESSED_BLOCK_SIZE,
        inputs["slot_mapping"],
        global_decode,
        global_decode_lens,
        global_prefill,
        global_prefill_lens,
        max_compressed_tokens=MAX_COMPRESSED_TOKENS,
    )

    assert local_decode.stride(0) != local_prefill.stride(0)
    assert torch.equal(local_decode, expected_global[:NUM_DECODE_TOKENS])
    assert torch.equal(local_decode_lens, expected_lens[:NUM_DECODE_TOKENS])
    assert torch.equal(local_prefill, expected_local[NUM_DECODE_TOKENS:])
    assert torch.equal(global_decode, expected_global[:NUM_DECODE_TOKENS])
    assert torch.equal(global_decode_lens, expected_lens[:NUM_DECODE_TOKENS])
    assert torch.equal(global_prefill, expected_global[NUM_DECODE_TOKENS:])
    assert torch.equal(global_prefill_lens, expected_lens[NUM_DECODE_TOKENS:])


@requires_cuda
@torch.inference_mode()
def test_c128a_global_prefill_metadata_cuda_graph_replay():
    inputs = _make_inputs()
    outputs = _make_outputs(inputs["positions"].numel())

    def launch() -> None:
        build_c128a_global_prefill_metadata(
            inputs["positions"],
            COMPRESS_RATIO,
            NUM_DECODE_TOKENS,
            inputs["token_to_req"],
            inputs["block_table"],
            COMPRESSED_BLOCK_SIZE,
            inputs["slot_mapping"],
            *outputs,
            max_compressed_tokens=MAX_COMPRESSED_TOKENS,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    inputs["positions"][-1] = 511
    inputs["slot_mapping"][-1] = -1
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()

    _, expected_global, expected_lens = _reference(inputs)
    decode, decode_lens, prefill, prefill_lens = outputs
    assert torch.equal(decode, expected_global[:NUM_DECODE_TOKENS])
    assert torch.equal(decode_lens, expected_lens[:NUM_DECODE_TOKENS])
    assert torch.equal(prefill, expected_global[NUM_DECODE_TOKENS:])
    assert torch.equal(prefill_lens, expected_lens[NUM_DECODE_TOKENS:])


def test_c128a_global_prefill_metadata_zero_tokens():
    empty_matrix = torch.empty((0, MAX_COMPRESSED_TOKENS), dtype=torch.int32)
    empty_vector = torch.empty(0, dtype=torch.int32)
    outputs = build_c128a_global_prefill_metadata(
        torch.empty(0, dtype=torch.int64),
        COMPRESS_RATIO,
        0,
        empty_vector,
        torch.empty((0, 0), dtype=torch.int32),
        COMPRESSED_BLOCK_SIZE,
        torch.empty(0, dtype=torch.int64),
        empty_matrix,
        empty_vector,
        empty_matrix,
        empty_vector,
        max_compressed_tokens=MAX_COMPRESSED_TOKENS,
    )
    assert all(output.shape[0] == 0 for output in outputs)


def test_flashinfer_c128a_global_prefill_arch_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    builder_cls = flashinfer_sparse.DeepseekV4FlashInferSparseMetadataBuilder
    assert flashinfer_sparse.DeepseekV4FlashInferMLASparseBackend.get_builder_cls() is (
        builder_cls
    )

    sm120 = SimpleNamespace(
        is_device_capability_family=lambda capability: capability == 120
    )
    monkeypatch.setattr(flashinfer_sparse, "current_platform", sm120)
    assert flashinfer_sparse._use_c128a_global_prefill_metadata()

    sm89 = SimpleNamespace(is_device_capability_family=lambda capability: False)
    monkeypatch.setattr(flashinfer_sparse, "current_platform", sm89)
    assert not flashinfer_sparse._use_c128a_global_prefill_metadata()


@pytest.mark.parametrize("precomputed_global", [False, True])
def test_flashinfer_prefill_selects_explicit_c128a_representation(
    monkeypatch: pytest.MonkeyPatch,
    precomputed_global: bool,
):
    local_indices = torch.tensor([[0, 1], [1, -1]], dtype=torch.int32)
    global_indices = torch.tensor([[9, 10], [11, -1]], dtype=torch.int32)
    global_lens = torch.tensor([2, 1], dtype=torch.int32)
    metadata = flashinfer_sparse.DeepseekV4FlashMLAMetadata(
        num_reqs=1,
        max_query_len=2,
        max_seq_len=2,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        block_table=torch.tensor([[0]], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
        block_size=256,
        topk_tokens=2,
        c128a_prefill_topk_indices=local_indices,
        c128a_global_prefill_topk_indices=(
            global_indices if precomputed_global else None
        ),
        c128a_global_prefill_topk_lens=(global_lens if precomputed_global else None),
    )
    swa_metadata = SimpleNamespace(
        num_prefills=1,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefill_tokens=2,
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0, 0], dtype=torch.int32),
        is_valid_token=torch.tensor([True, True]),
        prefill_swa_indices=torch.tensor([[0], [1]], dtype=torch.int32),
        prefill_swa_lens=torch.tensor([1, 1], dtype=torch.int32),
    )
    attention = SimpleNamespace(
        compress_ratio=128,
        PREFILL_CHUNK_SIZE=4,
        scale=1.0,
        attn_sink=None,
        _prepare_query=lambda query, _output: query,
        _as_sparse_cache=lambda cache: cache,
        _get_workspace=lambda _device: torch.empty(0, dtype=torch.uint8),
    )

    mappings: list[torch.Tensor] = []

    def map_local(indices, *_args, **_kwargs):
        mappings.append(indices)
        return global_indices, global_lens

    launches: list[dict[str, object]] = []
    monkeypatch.setattr(
        flashinfer_sparse,
        "compute_global_topk_indices_and_lens",
        map_local,
    )
    monkeypatch.setattr(
        flashinfer_sparse,
        "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
        lambda **kwargs: launches.append(kwargs),
    )

    flashinfer_sparse.DeepseekV4FlashInferSM120Attention._forward_prefill(
        attention,
        q=torch.empty((2, 1, 512), dtype=torch.bfloat16),
        compressed_k_cache=torch.empty((1, 2, 512), dtype=torch.bfloat16),
        swa_k_cache=torch.empty((1, 256, 512), dtype=torch.bfloat16),
        output=torch.empty((2, 1, 512), dtype=torch.bfloat16),
        attn_metadata=metadata,
        swa_metadata=swa_metadata,
    )

    assert len(launches) == 1
    assert len(mappings) == int(not precomputed_global)
    if mappings:
        assert torch.equal(mappings[0], local_indices)
    assert torch.equal(launches[0]["extra_sparse_indices"], global_indices)
    assert torch.equal(launches[0]["extra_sparse_topk_lens"], global_lens)
