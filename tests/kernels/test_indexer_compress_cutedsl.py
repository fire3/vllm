# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
)
from vllm.platforms import current_platform

compressor_module = importlib.import_module("vllm.models.deepseek_v4.compressor")

HEAD_DIM = 128
ROPE_DIM = 64
STATE_WIDTH = 256
STATE_BLOCK_SIZE = 4
COMPRESS_RATIO = 4
TOKEN_STRIDE = 128
MXFP4_TOKEN_STRIDE = HEAD_DIM // 2
MXFP4_QUANT_BLOCK = 32
SCALE_DIM = 4
RMS_EPS = 1e-6
FP8_MAX = 448.0
SENTINEL = 0xA5
PAGE_ALIGNMENT_BYTES = 576

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="SM120 CUDA device required",
)


class _Inputs(TypedDict):
    state_cache: torch.Tensor
    token_to_req: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    rms_weight: torch.Tensor
    cos_sin_cache: torch.Tensor
    kv_slot_mapping: torch.Tensor
    baseline_cache: torch.Tensor
    candidate_cache: torch.Tensor
    baseline_storage: torch.Tensor
    candidate_storage: torch.Tensor
    kv_block_size: int
    kv_page_stride: int
    logical_kv_page_size: int
    token_stride: int
    quant_block: int
    use_mxfp4: bool


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _make_inputs(
    num_tokens: int,
    kv_block_size: int,
    *,
    pad_every: int = 0,
    num_reqs: int = 1,
    padded_pages: bool = False,
    use_mxfp4: bool = False,
    seed: int = 0,
) -> _Inputs:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    token_ids = torch.arange(num_tokens, dtype=torch.int64, device=device)
    token_to_req = (token_ids % num_reqs).to(torch.int32)
    positions = (token_ids // num_reqs) * COMPRESS_RATIO + COMPRESS_RATIO - 1
    max_position = int(positions.max().item())
    state_blocks_per_req = max_position // STATE_BLOCK_SIZE + 2
    num_state_blocks = num_reqs * state_blocks_per_req
    logical_state_page_size = STATE_BLOCK_SIZE * 2 * STATE_WIDTH
    state_page_stride = (
        _round_up(
            logical_state_page_size * torch.float32.itemsize,
            PAGE_ALIGNMENT_BYTES,
        )
        // torch.float32.itemsize
        if padded_pages
        else logical_state_page_size
    )
    state_storage = torch.randn(
        num_state_blocks * state_page_stride,
        dtype=torch.float32,
        device=device,
    )
    state_cache = torch.as_strided(
        state_storage,
        size=(num_state_blocks, STATE_BLOCK_SIZE, 2 * STATE_WIDTH),
        stride=(state_page_stride, 2 * STATE_WIDTH, 1),
    )
    logical_block_table = (
        torch.arange(num_state_blocks, dtype=torch.int32, device=device)
        .reshape(num_reqs, state_blocks_per_req)
        .flip(1)
        .contiguous()
    )
    if padded_pages:
        block_table_storage = torch.empty(
            (num_reqs, state_blocks_per_req + 3),
            dtype=torch.int32,
            device=device,
        )
        block_table_storage[:, :state_blocks_per_req] = logical_block_table
        block_table = block_table_storage[:, :state_blocks_per_req]
    else:
        block_table = logical_block_table
    slot_mapping = token_ids + 1
    kv_slot_mapping = torch.arange(
        13,
        13 + num_tokens,
        dtype=torch.int64,
        device=device,
    )
    if pad_every:
        padded = torch.arange(num_tokens, device=device) % pad_every == pad_every - 1
        slot_mapping[padded] = -1
        kv_slot_mapping[padded] = -1

    last_kv_slot = 13 + num_tokens - 1
    num_kv_blocks = last_kv_slot // kv_block_size + 2
    cache_shape = (num_kv_blocks, kv_block_size, TOKEN_STRIDE + SCALE_DIM)
    logical_kv_page_size = kv_block_size * (TOKEN_STRIDE + SCALE_DIM)
    kv_page_stride = (
        _round_up(logical_kv_page_size, PAGE_ALIGNMENT_BYTES)
        if padded_pages
        else logical_kv_page_size
    )
    baseline_storage = torch.full(
        (num_kv_blocks * kv_page_stride,),
        SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    candidate_storage = baseline_storage.clone()
    rms_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos_sin_cache = torch.randn(
        max_position + 1,
        ROPE_DIM,
        dtype=torch.float32,
        device=device,
    )
    return {
        "state_cache": state_cache,
        "token_to_req": token_to_req,
        "positions": positions,
        "slot_mapping": slot_mapping,
        "block_table": block_table,
        "rms_weight": rms_weight,
        "cos_sin_cache": cos_sin_cache,
        "kv_slot_mapping": kv_slot_mapping,
        "baseline_cache": torch.as_strided(
            baseline_storage,
            size=cache_shape,
            stride=(kv_page_stride, TOKEN_STRIDE + SCALE_DIM, 1),
        ),
        "candidate_cache": torch.as_strided(
            candidate_storage,
            size=cache_shape,
            stride=(kv_page_stride, TOKEN_STRIDE + SCALE_DIM, 1),
        ),
        "baseline_storage": baseline_storage,
        "candidate_storage": candidate_storage,
        "kv_block_size": kv_block_size,
        "kv_page_stride": kv_page_stride,
        "logical_kv_page_size": logical_kv_page_size,
        "token_stride": MXFP4_TOKEN_STRIDE if use_mxfp4 else TOKEN_STRIDE,
        "quant_block": MXFP4_QUANT_BLOCK if use_mxfp4 else HEAD_DIM,
        "use_mxfp4": use_mxfp4,
    }


def _run_triton(inputs: _Inputs) -> None:
    state_cache = inputs["state_cache"]
    block_table = inputs["block_table"]
    cache = inputs["baseline_cache"]
    assert isinstance(state_cache, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    kv_block_size = inputs["kv_block_size"]
    assert isinstance(kv_block_size, int)
    token_stride = inputs["token_stride"]
    quant_block = inputs["quant_block"]
    kernel = (
        _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
        if inputs["use_mxfp4"]
        else _fused_kv_compress_norm_rope_insert_indexer_attn
    )
    num_tokens = inputs["slot_mapping"].shape[0]
    kernel[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        inputs["token_to_req"],
        inputs["positions"],
        inputs["slot_mapping"],
        block_table,
        block_table.stride(0),
        STATE_BLOCK_SIZE,
        inputs["rms_weight"],
        RMS_EPS,
        inputs["cos_sin_cache"],
        inputs["cos_sin_cache"].stride(0),
        cache,
        inputs["kv_slot_mapping"],
        kv_block_size,
        HEAD_SIZE=HEAD_DIM,
        TRITON_BLOCK_SIZE=HEAD_DIM,
        STATE_WIDTH=STATE_WIDTH,
        COMPRESS_RATIO=COMPRESS_RATIO,
        OVERLAP=True,
        ROPE_HEAD_DIM=ROPE_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=cache.stride(0),
        num_warps=1,
        launch_pdl=False,
    )


def _run_cutedsl(inputs: _Inputs) -> None:
    pytest.importorskip("cutlass")
    from vllm.models.deepseek_v4.nvidia.ops.indexer_compress_cutedsl import (
        compress_norm_rope_store_indexer_cutedsl,
    )

    cache = inputs["candidate_cache"]
    assert isinstance(cache, torch.Tensor)
    compress_norm_rope_store_indexer_cutedsl(
        state_cache=inputs["state_cache"],
        num_actual=inputs["slot_mapping"].shape[0],
        token_to_req_indices=inputs["token_to_req"],
        positions=inputs["positions"],
        slot_mapping=inputs["slot_mapping"],
        block_table=inputs["block_table"],
        block_size=STATE_BLOCK_SIZE,
        state_width=STATE_WIDTH,
        cos_sin_cache=inputs["cos_sin_cache"],
        kv_cache=cache,
        k_cache_metadata=SimpleNamespace(slot_mapping=inputs["kv_slot_mapping"]),
        pdl_kwargs={"launch_pdl": False},
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_DIM,
        compress_ratio=COMPRESS_RATIO,
        overlap=True,
        use_fp4_cache=inputs["use_mxfp4"],
        rms_norm_weight=inputs["rms_weight"],
        rms_norm_eps=RMS_EPS,
        quant_block=inputs["quant_block"],
        token_stride=inputs["token_stride"],
        scale_dim=SCALE_DIM,
    )


def _assert_cache_matches(inputs: _Inputs) -> None:
    baseline = inputs["baseline_cache"]
    candidate = inputs["candidate_cache"]
    kv_slot_mapping = inputs["kv_slot_mapping"]
    kv_block_size = inputs["kv_block_size"]
    assert isinstance(baseline, torch.Tensor)
    assert isinstance(candidate, torch.Tensor)
    assert isinstance(kv_slot_mapping, torch.Tensor)
    assert isinstance(kv_block_size, int)
    token_stride = inputs["token_stride"]
    quant_block = inputs["quant_block"]
    use_mxfp4 = inputs["use_mxfp4"]

    active_slots = torch.unique(kv_slot_mapping[kv_slot_mapping >= 0])
    pages = active_slots // kv_block_size
    offsets = active_slots % kv_block_size
    page_stride = inputs["logical_kv_page_size"]
    value_indices = (
        pages[:, None] * page_stride
        + offsets[:, None] * token_stride
        + torch.arange(token_stride, device=baseline.device)[None, :]
    ).flatten()
    scale_indices = (
        pages[:, None] * page_stride
        + kv_block_size * token_stride
        + offsets[:, None] * SCALE_DIM
        + torch.arange(SCALE_DIM, device=baseline.device)[None, :]
    ).flatten()
    untouched = torch.ones(baseline.numel(), dtype=torch.bool, device=baseline.device)
    untouched[value_indices] = False
    untouched[scale_indices] = False

    baseline_flat = baseline.flatten()
    candidate_flat = candidate.flatten()
    assert torch.equal(baseline_flat[untouched], candidate_flat[untouched])
    baseline_scale_bytes = baseline_flat[scale_indices].reshape(-1, SCALE_DIM)
    candidate_scale_bytes = candidate_flat[scale_indices].reshape(-1, SCALE_DIM)
    baseline_scale = (
        baseline_scale_bytes if use_mxfp4 else baseline_scale_bytes.view(torch.float32)
    )
    candidate_scale = (
        candidate_scale_bytes
        if use_mxfp4
        else candidate_scale_bytes.view(torch.float32)
    )
    assert torch.equal(baseline_scale, candidate_scale)

    baseline_u8 = baseline_flat[value_indices].reshape(-1, token_stride)
    candidate_u8 = candidate_flat[value_indices].reshape(-1, token_stride)
    mismatch_count = int((baseline_u8 != candidate_u8).sum().item())
    assert mismatch_count <= max(1, baseline_u8.numel() // 10_000)
    if use_mxfp4:
        fp4_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=baseline.device,
        )

        def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1)
            nibbles = nibbles.reshape(-1, SCALE_DIM, quant_block)
            magnitudes = fp4_values[(nibbles & 0x7).long()]
            values = torch.where((nibbles & 0x8) != 0, -magnitudes, magnitudes)
            scale_values = torch.exp2(scales.to(torch.int16).float() - 127.0)
            return (values * scale_values[..., None]).reshape(-1, HEAD_DIM)

        baseline_dequant = dequantize(baseline_u8, baseline_scale)
        candidate_dequant = dequantize(candidate_u8, candidate_scale)
        error_scale = torch.exp2(
            torch.maximum(baseline_scale, candidate_scale).to(torch.int16).float()
            - 127.0
        ).repeat_interleave(quant_block, dim=1)
        max_error_in_scales = 2.0
        min_cosine = 0.99999
    else:
        baseline_dequant = (
            baseline_u8.view(torch.float8_e4m3fn).float() * baseline_scale
        )
        candidate_dequant = (
            candidate_u8.view(torch.float8_e4m3fn).float() * candidate_scale
        )
        error_scale = baseline_scale
        max_error_in_scales = 32.0
        min_cosine = 0.999999
    error_in_scales = (baseline_dequant - candidate_dequant).abs() / error_scale
    assert torch.isfinite(candidate_dequant).all()
    assert error_in_scales.max().item() <= max_error_in_scales
    cosine = torch.nn.functional.cosine_similarity(
        baseline_dequant.flatten().double(),
        candidate_dequant.flatten().double(),
        dim=0,
    )
    assert cosine.item() >= min_cosine

    kv_page_stride = inputs["kv_page_stride"]
    logical_kv_page_size = inputs["logical_kv_page_size"]
    if kv_page_stride > logical_kv_page_size:
        num_blocks = baseline.shape[0]
        for storage in (
            inputs["baseline_storage"],
            inputs["candidate_storage"],
        ):
            padding = storage.view(num_blocks, kv_page_stride)[:, logical_kv_page_size:]
            assert torch.all(padding == SENTINEL)


@requires_sm120
@pytest.mark.parametrize("use_mxfp4", [False, True], ids=["fp8", "mxfp4"])
@pytest.mark.parametrize(
    ("num_tokens", "kv_block_size", "pad_every", "num_reqs", "padded_pages"),
    [
        (1, 16, 0, 1, False),
        (7, 64, 0, 1, True),
        (32, 16, 5, 1, True),
        (65, 64, 7, 3, True),
        (256, 64, 0, 1, False),
    ],
)
@torch.inference_mode()
def test_indexer_compress_cutedsl_matches_triton(
    num_tokens: int,
    kv_block_size: int,
    pad_every: int,
    num_reqs: int,
    padded_pages: bool,
    use_mxfp4: bool,
):
    inputs = _make_inputs(
        num_tokens,
        kv_block_size,
        pad_every=pad_every,
        num_reqs=num_reqs,
        padded_pages=padded_pages,
        use_mxfp4=use_mxfp4,
        seed=num_tokens + kv_block_size,
    )
    _run_triton(inputs)
    _run_cutedsl(inputs)
    _assert_cache_matches(inputs)


@requires_sm120
@pytest.mark.parametrize("use_mxfp4", [False, True], ids=["fp8", "mxfp4"])
@torch.inference_mode()
def test_indexer_compress_cutedsl_cuda_graph(use_mxfp4: bool):
    inputs = _make_inputs(
        256,
        64,
        pad_every=11,
        padded_pages=True,
        use_mxfp4=use_mxfp4,
        seed=17,
    )
    _run_triton(inputs)
    _run_cutedsl(inputs)
    candidate_cache = inputs["candidate_cache"]
    assert isinstance(candidate_cache, torch.Tensor)
    candidate_cache.fill_(SENTINEL)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_cutedsl(inputs)
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    _assert_cache_matches(inputs)


def test_indexer_compress_cutedsl_dispatch_guard(monkeypatch: pytest.MonkeyPatch):
    sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: capability == 120,
    )
    monkeypatch.setattr(compressor_module, "current_platform", sm120)
    monkeypatch.setattr(compressor_module, "has_cutedsl", lambda: True)

    def can_use(
        num_tokens: int = 8192,
        head_dim: int = 128,
        compress_ratio: int = 4,
        use_fp4_cache: bool = False,
    ) -> bool:
        return compressor_module._can_use_indexer_compress_cutedsl(
            head_dim=head_dim,
            compress_ratio=compress_ratio,
            use_fp4_cache=use_fp4_cache,
            num_tokens=num_tokens,
        )

    assert can_use()
    assert not can_use(num_tokens=8191)
    assert not can_use(head_dim=512)
    assert not can_use(compress_ratio=128)
    assert can_use(use_fp4_cache=True)
    assert not can_use(num_tokens=8191, use_fp4_cache=True)

    monkeypatch.setattr(compressor_module, "has_cutedsl", lambda: False)
    assert not can_use()
    monkeypatch.setattr(
        compressor_module,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda capability: False,
        ),
    )
    assert not can_use()
