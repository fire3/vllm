# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch

import vllm.v1.attention.backends.mla.sparse_swa as sparse_swa
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.sparse_swa import (
    _SM120_TILED_SWA_PREFILL_MIN_TOKENS,
    _compute_prefill_metadata_kernel,
    _compute_swa_indices_and_lens_kernel,
    _use_tiled_swa_prefill_metadata,
    build_swa_prefill_indices_and_metadata,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)

WINDOW_SIZE = 8
BLOCK_SIZE = 4


class Inputs(TypedDict):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    token_to_req: torch.Tensor
    is_valid_token: torch.Tensor
    block_table: torch.Tensor
    num_decodes: int
    num_prefills: int
    num_prefill_tokens: int


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _make_inputs(
    *,
    num_decodes: int,
    prefill_query_lens: list[int],
    invalid_absolute_tokens: set[int] | None = None,
    seq_len_delta: int = 0,
) -> Inputs:
    device = torch.device("cuda")
    invalid_absolute_tokens = invalid_absolute_tokens or set()
    query_lens = [1] * num_decodes + prefill_query_lens
    query_start_locs = [0]
    for query_len in query_lens:
        query_start_locs.append(query_start_locs[-1] + query_len)

    num_reqs = len(query_lens)
    seq_lens = []
    for req_idx, query_len in enumerate(query_lens):
        if req_idx < num_decodes:
            prefix_len = 3 + req_idx
        else:
            prefill_idx = req_idx - num_decodes
            prefix_len = 2 + prefill_idx * 3 + seq_len_delta
        seq_lens.append(prefix_len + query_len)

    max_blocks = max(16, (max(seq_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    block_table = torch.empty((num_reqs, max_blocks), dtype=torch.int32, device=device)
    for req_idx in range(num_reqs):
        block_table[req_idx] = torch.arange(
            req_idx * max_blocks,
            (req_idx + 1) * max_blocks,
            dtype=torch.int32,
            device=device,
        )

    total_tokens = query_start_locs[-1]
    token_to_req = torch.empty(total_tokens, dtype=torch.int32, device=device)
    for req_idx, (start, end) in enumerate(
        zip(query_start_locs[:-1], query_start_locs[1:])
    ):
        token_to_req[start:end] = req_idx

    is_valid_token = torch.ones(total_tokens, dtype=torch.bool, device=device)
    if invalid_absolute_tokens:
        invalid = torch.tensor(
            sorted(invalid_absolute_tokens), dtype=torch.int64, device=device
        )
        is_valid_token[invalid] = False

    return {
        "query_start_loc": torch.tensor(
            query_start_locs, dtype=torch.int32, device=device
        ),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=device),
        "token_to_req": token_to_req,
        "is_valid_token": is_valid_token,
        "block_table": block_table,
        "num_decodes": num_decodes,
        "num_prefills": len(prefill_query_lens),
        "num_prefill_tokens": sum(prefill_query_lens),
    }


def _reference(
    inputs: Inputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_start_loc = inputs["query_start_loc"].cpu()
    seq_lens = inputs["seq_lens"].cpu()
    token_to_req = inputs["token_to_req"].cpu()
    is_valid_token = inputs["is_valid_token"].cpu()
    block_table = inputs["block_table"].cpu()
    num_decodes = int(inputs["num_decodes"])
    num_prefills = int(inputs["num_prefills"])
    num_prefill_tokens = int(inputs["num_prefill_tokens"])

    ref_indices = torch.full((num_prefill_tokens, WINDOW_SIZE), -1, dtype=torch.int32)
    ref_lens = torch.zeros(num_prefill_tokens, dtype=torch.int32)
    ref_gather_lens = torch.empty(num_prefills, dtype=torch.int32)

    for prefill_idx in range(num_prefills):
        req_idx = num_decodes + prefill_idx
        query_len = query_start_loc[req_idx + 1] - query_start_loc[req_idx]
        prefix_len = seq_lens[req_idx] - query_len
        ref_gather_lens[prefill_idx] = query_len + min(prefix_len, WINDOW_SIZE - 1)

    token_offset = int(query_start_loc[num_decodes])
    for row in range(num_prefill_tokens):
        token_idx = token_offset + row
        if not bool(is_valid_token[token_idx]):
            continue
        req_idx = int(token_to_req[token_idx])
        query_start = int(query_start_loc[req_idx])
        query_len = int(query_start_loc[req_idx + 1] - query_start)
        prefix_len = int(seq_lens[req_idx]) - query_len
        pos = prefix_len + token_idx - query_start
        start_pos = max(pos - WINDOW_SIZE + 1, 0)
        end_pos = pos + 1
        ref_lens[row] = end_pos - start_pos
        for offset, pos_offset in enumerate(range(start_pos, end_pos)):
            block_number = block_table[req_idx, pos_offset // BLOCK_SIZE]
            ref_indices[row, offset] = block_number * BLOCK_SIZE + (
                pos_offset % BLOCK_SIZE
            )

    return ref_indices.cuda(), ref_lens.cuda(), ref_gather_lens.cuda()


def _run_fused(
    inputs: Inputs,
    *,
    output_stride: bool = False,
    gather_fill: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    num_prefills = int(inputs["num_prefills"])
    if output_stride:
        storage = torch.full(
            (num_prefill_tokens, 2, WINDOW_SIZE),
            -77,
            dtype=torch.int32,
            device="cuda",
        )
        swa_indices = storage[:, 1, :]
    else:
        storage = None
        swa_indices = torch.empty(
            (num_prefill_tokens, WINDOW_SIZE), dtype=torch.int32, device="cuda"
        )
    swa_lens = torch.empty(num_prefill_tokens, dtype=torch.int32, device="cuda")
    gather_lens = torch.empty(num_prefills, dtype=torch.int32, device="cuda")
    if gather_fill is not None:
        gather_lens.fill_(gather_fill)

    build_swa_prefill_indices_and_metadata(
        swa_indices,
        swa_lens,
        gather_lens,
        WINDOW_SIZE,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req"],
        inputs["is_valid_token"],
        inputs["block_table"],
        BLOCK_SIZE,
        token_offset=int(inputs["query_start_loc"][int(inputs["num_decodes"])]),
        num_decodes=int(inputs["num_decodes"]),
    )
    return swa_indices, swa_lens, gather_lens, storage


def _run_baseline(
    inputs: Inputs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    num_prefills = int(inputs["num_prefills"])
    swa_indices = torch.empty(
        (num_prefill_tokens, WINDOW_SIZE), dtype=torch.int32, device="cuda"
    )
    swa_lens = torch.empty(num_prefill_tokens, dtype=torch.int32, device="cuda")
    gather_lens = torch.empty(num_prefills, dtype=torch.int32, device="cuda")

    _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        WINDOW_SIZE,
        inputs["query_start_loc"],
        inputs["seq_lens"],
        inputs["token_to_req"],
        inputs["is_valid_token"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        BLOCK_SIZE,
        gather_lens,
        int(inputs["num_decodes"]),
        int(inputs["num_prefills"]),
        token_offset=int(inputs["query_start_loc"][int(inputs["num_decodes"])]),
        TRITON_BLOCK_SIZE=1024,
        PREFILL_METADATA_BLOCK_SIZE=1,
        WRITE_PREFILL_METADATA=False,
    )
    _compute_prefill_metadata_kernel[(1,)](
        gather_lens,
        inputs["seq_lens"],
        inputs["query_start_loc"],
        num_prefills,
        int(inputs["num_decodes"]),
        WINDOW_SIZE,
        BLOCK_SIZE=_next_power_of_2(num_prefills),
    )
    return swa_indices, swa_lens, gather_lens


@requires_sm120
@pytest.mark.parametrize("num_decodes", [0, 2], ids=["pure_prefill", "mixed"])
@pytest.mark.parametrize(
    "prefill_query_lens", [[5], [3, 4, 2, 5]], ids=["one_prefill", "four_prefills"]
)
@torch.inference_mode()
def test_fused_prefill_swa_metadata_matches_baseline_and_reference(
    num_decodes: int, prefill_query_lens: list[int]
):
    inputs = _make_inputs(
        num_decodes=num_decodes, prefill_query_lens=prefill_query_lens
    )
    fused_indices, fused_lens, fused_gather_lens, _ = _run_fused(inputs)
    baseline_indices, baseline_lens, baseline_gather_lens = _run_baseline(inputs)
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    torch.testing.assert_close(fused_lens, baseline_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_gather_lens, baseline_gather_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(fused_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_gather_lens, ref_gather_lens, rtol=0, atol=0)


@requires_sm120
@pytest.mark.parametrize(
    ("prefill_query_lens", "output_stride"),
    [([1025], False), ([255, 257, 256, 259], True)],
    ids=["one_prefill", "four_prefills"],
)
@torch.inference_mode()
def test_tiled_prefill_swa_metadata_matches_baseline_and_reference(
    prefill_query_lens: list[int],
    output_stride: bool,
):
    inputs = _make_inputs(num_decodes=2, prefill_query_lens=prefill_query_lens)
    tiled_indices, tiled_lens, tiled_gather_lens, _ = _run_fused(
        inputs, output_stride=output_stride
    )
    baseline_indices, baseline_lens, baseline_gather_lens = _run_baseline(inputs)
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(tiled_indices, baseline_indices, rtol=0, atol=0)
    torch.testing.assert_close(tiled_lens, baseline_lens, rtol=0, atol=0)
    torch.testing.assert_close(tiled_gather_lens, baseline_gather_lens, rtol=0, atol=0)
    torch.testing.assert_close(tiled_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(tiled_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(tiled_gather_lens, ref_gather_lens, rtol=0, atol=0)


@requires_sm120
@torch.inference_mode()
def test_fused_prefill_swa_metadata_uses_compact_prefill_gather_rows():
    inputs = _make_inputs(num_decodes=3, prefill_query_lens=[2, 3, 4, 5])
    _, _, fused_gather_lens, _ = _run_fused(inputs)
    _, _, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(fused_gather_lens, ref_gather_lens, rtol=0, atol=0)


@requires_sm120
@pytest.mark.parametrize(
    "first_query_len", [3, 1024], ids=["token_per_program", "tiled"]
)
@torch.inference_mode()
def test_fused_prefill_swa_metadata_overwrites_zero_query_padding_rows(
    first_query_len: int,
):
    inputs = _make_inputs(num_decodes=2, prefill_query_lens=[first_query_len, 0, 0])
    inputs["seq_lens"][-2:].zero_()
    fused_indices, fused_lens, fused_gather_lens, _ = _run_fused(
        inputs, gather_fill=-99
    )
    baseline_indices, baseline_lens, baseline_gather_lens = _run_baseline(inputs)
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(fused_indices, baseline_indices, rtol=0, atol=0)
    torch.testing.assert_close(fused_lens, baseline_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_gather_lens, baseline_gather_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(fused_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_gather_lens, ref_gather_lens, rtol=0, atol=0)
    assert torch.equal(fused_gather_lens[-2:], torch.zeros_like(fused_gather_lens[-2:]))


@requires_sm120
@torch.inference_mode()
def test_fused_prefill_swa_metadata_writes_gather_len_when_first_token_invalid():
    num_decodes = 2
    prefill_query_lens = [4]
    first_prefill_token = num_decodes
    inputs = _make_inputs(
        num_decodes=num_decodes,
        prefill_query_lens=prefill_query_lens,
        invalid_absolute_tokens={first_prefill_token},
    )
    fused_indices, fused_lens, fused_gather_lens, _ = _run_fused(inputs)
    _, _, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(fused_gather_lens, ref_gather_lens, rtol=0, atol=0)
    assert fused_lens[0].item() == 0
    assert torch.equal(
        fused_indices[0],
        torch.full((WINDOW_SIZE,), -1, dtype=torch.int32, device="cuda"),
    )


@requires_sm120
@torch.inference_mode()
def test_fused_prefill_swa_metadata_respects_independent_output_row_stride():
    inputs = _make_inputs(num_decodes=2, prefill_query_lens=[3, 2])
    fused_indices, fused_lens, fused_gather_lens, storage = _run_fused(
        inputs, output_stride=True
    )
    assert storage is not None
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)

    torch.testing.assert_close(fused_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(fused_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(fused_gather_lens, ref_gather_lens, rtol=0, atol=0)
    assert torch.equal(
        storage[:, 0, :],
        torch.full_like(storage[:, 0, :], -77),
    )


@requires_cuda
@torch.inference_mode()
def test_fused_prefill_swa_metadata_returns_for_zero_prefill_tokens():
    swa_indices = torch.empty((0, WINDOW_SIZE), dtype=torch.int32, device="cuda")
    swa_lens = torch.empty(0, dtype=torch.int32, device="cuda")
    gather_lens = torch.empty(1, dtype=torch.int32, device="cuda")
    query_start_loc = torch.tensor([0, 0], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
    token_to_req = torch.empty(0, dtype=torch.int32, device="cuda")
    is_valid_token = torch.empty(0, dtype=torch.bool, device="cuda")
    block_table = torch.empty((1, 1), dtype=torch.int32, device="cuda")

    build_swa_prefill_indices_and_metadata(
        swa_indices,
        swa_lens,
        gather_lens,
        WINDOW_SIZE,
        query_start_loc,
        seq_lens,
        token_to_req,
        is_valid_token,
        block_table,
        BLOCK_SIZE,
        token_offset=0,
        num_decodes=0,
    )

    assert swa_indices.numel() == 0
    assert swa_lens.numel() == 0


@requires_sm120
@pytest.mark.parametrize(
    "prefill_query_lens",
    [[3, 4], [1025]],
    ids=["token_per_program", "tiled_tail"],
)
@torch.inference_mode()
def test_fused_prefill_swa_metadata_cuda_graph_replays_dynamic_inputs(
    prefill_query_lens: list[int],
):
    inputs = _make_inputs(num_decodes=1, prefill_query_lens=prefill_query_lens)
    num_prefill_tokens = int(inputs["num_prefill_tokens"])
    num_prefills = int(inputs["num_prefills"])
    swa_indices = torch.empty(
        (num_prefill_tokens, WINDOW_SIZE), dtype=torch.int32, device="cuda"
    )
    swa_lens = torch.empty(num_prefill_tokens, dtype=torch.int32, device="cuda")
    gather_lens = torch.empty(num_prefills, dtype=torch.int32, device="cuda")
    num_decodes = int(inputs["num_decodes"])
    token_offset = int(inputs["query_start_loc"][num_decodes])

    def run_fused() -> None:
        build_swa_prefill_indices_and_metadata(
            swa_indices,
            swa_lens,
            gather_lens,
            WINDOW_SIZE,
            inputs["query_start_loc"],
            inputs["seq_lens"],
            inputs["token_to_req"],
            inputs["is_valid_token"],
            inputs["block_table"],
            BLOCK_SIZE,
            token_offset=token_offset,
            num_decodes=num_decodes,
        )

    run_fused()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fused()
    graph.replay()
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)
    torch.testing.assert_close(swa_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(swa_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(gather_lens, ref_gather_lens, rtol=0, atol=0)

    inputs["seq_lens"].add_(2)
    inputs["is_valid_token"][3] = False
    swa_indices.fill_(-99)
    swa_lens.fill_(-99)
    gather_lens.fill_(-99)
    graph.replay()
    torch.accelerator.synchronize()
    ref_indices, ref_lens, ref_gather_lens = _reference(inputs)
    torch.testing.assert_close(swa_indices, ref_indices, rtol=0, atol=0)
    torch.testing.assert_close(swa_lens, ref_lens, rtol=0, atol=0)
    torch.testing.assert_close(gather_lens, ref_gather_lens, rtol=0, atol=0)


def test_fused_prefill_swa_metadata_guard_selects_sm120_only(
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
    cpu = SimpleNamespace(
        is_cuda=lambda: False,
        is_device_capability_family=lambda capability: capability == 120,
    )

    monkeypatch.setattr(sparse_swa, "current_platform", sm120)
    assert sparse_swa._use_fused_prefill_metadata()
    monkeypatch.setattr(sparse_swa, "current_platform", non_sm120)
    assert not sparse_swa._use_fused_prefill_metadata()
    monkeypatch.setattr(sparse_swa, "current_platform", cpu)
    assert not sparse_swa._use_fused_prefill_metadata()


def test_tiled_prefill_swa_metadata_guard_selects_threshold_and_sm120(
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

    monkeypatch.setattr(sparse_swa, "current_platform", sm120)
    assert not _use_tiled_swa_prefill_metadata(_SM120_TILED_SWA_PREFILL_MIN_TOKENS - 1)
    assert _use_tiled_swa_prefill_metadata(_SM120_TILED_SWA_PREFILL_MIN_TOKENS)
    monkeypatch.setattr(sparse_swa, "current_platform", non_sm120)
    assert not _use_tiled_swa_prefill_metadata(_SM120_TILED_SWA_PREFILL_MIN_TOKENS)
