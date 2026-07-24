# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _GLOBAL_TOPK_CUTEDSL_MIN_TOKENS,
    compute_global_topk_indices_and_lens,
)
from vllm.platforms import current_platform

cache_utils_module = importlib.import_module(
    "vllm.models.deepseek_v4.common.ops.cache_utils"
)

TOPK = 512
BLOCK_SIZE = 64
NUM_REQUESTS = 4

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)


def _reference(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp_min(0)
    block_indices = torch.div(safe_indices, BLOCK_SIZE, rounding_mode="floor")
    request_indices = token_to_req_indices[:, None].expand_as(block_indices)
    block_numbers = block_table[request_indices, block_indices]
    global_indices = block_numbers * BLOCK_SIZE + safe_indices % BLOCK_SIZE
    global_indices = torch.where(valid, global_indices, -1)
    topk_lens = valid.sum(dim=1, dtype=torch.int32)
    return global_indices, torch.where(is_valid_token, topk_lens, 0)


def _make_inputs(
    num_tokens: int,
    *,
    input_row_padding: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    input_storage = torch.full(
        (num_tokens, TOPK + input_row_padding),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_indices = input_storage[:, :TOPK]
    token_to_req = torch.arange(num_tokens, dtype=torch.int32, device=device)
    token_to_req.remainder_(NUM_REQUESTS)

    blocks_per_request = TOPK // BLOCK_SIZE
    block_table_storage = torch.empty(
        (NUM_REQUESTS, blocks_per_request + 3),
        dtype=torch.int32,
        device=device,
    )
    logical_block_table = torch.arange(
        NUM_REQUESTS * blocks_per_request,
        dtype=torch.int32,
        device=device,
    ).view(NUM_REQUESTS, blocks_per_request)
    block_table_storage[:, :blocks_per_request] = logical_block_table.flip(1)
    block_table = block_table_storage[:, :blocks_per_request]

    if num_tokens:
        columns = torch.arange(TOPK, dtype=torch.int32, device=device)
        rows = torch.arange(num_tokens, dtype=torch.int32, device=device)[:, None]
        topk_indices.copy_((columns[None, :] * 67 + rows * 131) % TOPK)
        boundary_values = torch.tensor(
            [0, 63, 64, 65, 127, 128, 129, 65],
            dtype=torch.int32,
            device=device,
        )
        topk_indices[:, : boundary_values.numel()] = boundary_values
        valid_counts = torch.tensor(
            [0, 1, 127, 128, 129, TOPK],
            dtype=torch.int32,
            device=device,
        )
        valid_counts = valid_counts[
            torch.arange(num_tokens, device=device) % valid_counts.numel()
        ]
        topk_indices.masked_fill_(columns[None, :] >= valid_counts[:, None], -1)

    is_valid_token = torch.ones(num_tokens, dtype=torch.bool, device=device)
    if num_tokens > 3:
        is_valid_token[3::5] = False
    return topk_indices, token_to_req, block_table, is_valid_token


def _launch_cutedsl(
    output: torch.Tensor,
    topk_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    is_valid_token: torch.Tensor,
    *,
    threads: int,
) -> None:
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v4.nvidia.ops.global_topk_cutedsl import (
        launch_global_topk_indices_and_lens_cutedsl,
    )

    launch_global_topk_indices_and_lens_cutedsl(
        output,
        topk_lens,
        topk_indices,
        token_to_req,
        block_table,
        BLOCK_SIZE,
        is_valid_token,
        threads=threads,
    )


def test_global_topk_cutedsl_dispatch_guard(monkeypatch: pytest.MonkeyPatch):
    sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: capability == 120,
    )
    monkeypatch.setattr(cache_utils_module, "current_platform", sm120)
    monkeypatch.setattr(cache_utils_module, "has_cutedsl", lambda: True)

    can_use = cache_utils_module._can_use_global_topk_cutedsl
    assert can_use(topk=TOPK, num_tokens=_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS)
    assert not can_use(topk=TOPK, num_tokens=_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS - 1)
    assert not can_use(topk=1024, num_tokens=_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS)

    monkeypatch.setattr(cache_utils_module, "has_cutedsl", lambda: False)
    assert not can_use(topk=TOPK, num_tokens=_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS)
    monkeypatch.setattr(
        cache_utils_module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: False,
        ),
    )
    assert not can_use(topk=TOPK, num_tokens=_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS)


def test_compute_global_topk_zero_tokens_returns_empty():
    topk_indices = torch.empty((0, TOPK), dtype=torch.int32)
    output, topk_lens = compute_global_topk_indices_and_lens(
        topk_indices,
        torch.empty(0, dtype=torch.int32),
        torch.empty((0, 0), dtype=torch.int32),
        BLOCK_SIZE,
        torch.empty(0, dtype=torch.bool),
    )
    assert output.shape == (0, TOPK)
    assert topk_lens.shape == (0,)


@requires_sm120
@pytest.mark.parametrize("threads", [128, 256])
@torch.inference_mode()
def test_global_topk_cutedsl_matches_reference_with_padded_rows(threads: int):
    inputs = _make_inputs(11, input_row_padding=4)
    topk_indices, token_to_req, block_table, is_valid_token = inputs
    output_storage = torch.empty(
        (topk_indices.shape[0], TOPK + 12), dtype=torch.int32, device="cuda"
    )
    output = output_storage[:, :TOPK]
    topk_lens = torch.empty(topk_indices.shape[0], dtype=torch.int32, device="cuda")

    _launch_cutedsl(
        output,
        topk_lens,
        topk_indices,
        token_to_req,
        block_table,
        is_valid_token,
        threads=threads,
    )
    expected_output, expected_lens = _reference(
        topk_indices, token_to_req, block_table, is_valid_token
    )
    assert output.stride(0) != topk_indices.stride(0)
    assert torch.equal(output, expected_output)
    assert torch.equal(topk_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_compute_global_topk_sm120_dispatch_matches_reference():
    inputs = _make_inputs(_GLOBAL_TOPK_CUTEDSL_MIN_TOKENS)
    topk_indices, token_to_req, block_table, is_valid_token = inputs
    mixed_token_to_req = torch.cat(
        (token_to_req, torch.tensor([0], dtype=torch.int32, device="cuda"))
    )
    mixed_is_valid_token = torch.cat(
        (is_valid_token, torch.tensor([False], dtype=torch.bool, device="cuda"))
    )
    output, topk_lens = compute_global_topk_indices_and_lens(
        topk_indices,
        mixed_token_to_req,
        block_table,
        BLOCK_SIZE,
        mixed_is_valid_token,
    )
    expected_output, expected_lens = _reference(
        topk_indices, token_to_req, block_table, is_valid_token
    )
    assert torch.equal(output, expected_output)
    assert torch.equal(topk_lens, expected_lens)


@requires_sm120
@torch.inference_mode()
def test_global_topk_cutedsl_cuda_graph():
    inputs = _make_inputs(64, input_row_padding=4)
    topk_indices, token_to_req, block_table, is_valid_token = inputs
    output = torch.empty_like(topk_indices)
    topk_lens = torch.empty(topk_indices.shape[0], dtype=torch.int32, device="cuda")

    _launch_cutedsl(
        output,
        topk_lens,
        topk_indices,
        token_to_req,
        block_table,
        is_valid_token,
        threads=128,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_cutedsl(
            output,
            topk_lens,
            topk_indices,
            token_to_req,
            block_table,
            is_valid_token,
            threads=128,
        )
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()

    expected_output, expected_lens = _reference(
        topk_indices, token_to_req, block_table, is_valid_token
    )
    assert torch.equal(output, expected_output)
    assert torch.equal(topk_lens, expected_lens)
