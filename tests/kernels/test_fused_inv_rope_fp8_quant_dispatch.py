# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device required",
)

inv_rope_module = importlib.import_module(
    "vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant"
)


def _make_inputs(num_tokens: int, num_groups: int, device: str = "cuda"):
    input = torch.randn(
        num_tokens,
        num_groups * 8,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
    cache = torch.zeros(1024, 64, dtype=torch.float32, device=device)
    cache[:, :32] = 1.0
    output = torch.empty(
        num_groups,
        num_tokens,
        4096,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    scales = torch.empty(
        num_groups,
        32,
        num_tokens,
        dtype=torch.float32,
        device=device,
    )
    return input, positions, cache, output, scales


def _fake_sm120_platform():
    return SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: capability == 120,
        is_arch_support_pdl=lambda: False,
    )


def _can_use(inputs) -> bool:
    input, positions, cache, output, scales = inputs
    num_tokens = input.shape[0]
    num_groups = input.shape[1] // 8
    return inv_rope_module._can_use_inverse_rope_fp8_quant_cute(
        input,
        positions,
        cache,
        output,
        scales,
        8,
        128,
        4,
        64,
        32,
        False,
        448.0,
        num_tokens,
        num_groups,
        4096,
        32,
    )


@torch.inference_mode()
def test_sm120_cute_dispatch(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def fake_cute(input, positions, cache, output, scales, *, enable_pdl):
        nonlocal calls
        calls += 1
        assert input.shape == (256, 16, 512)
        assert positions.shape == (256,)
        assert cache.shape == (1024, 64)
        assert output.shape == (2, 256, 4096)
        assert scales.shape == (2, 32, 256)
        assert scales.is_contiguous()
        assert enable_pdl is False
        output.zero_()
        scales.fill_(1.0)

    monkeypatch.setattr(inv_rope_module, "current_platform", _fake_sm120_platform())
    monkeypatch.setattr(
        inv_rope_module,
        "_get_inverse_rope_fp8_quant_cute_impl",
        lambda: fake_cute,
    )
    input, positions, cache, _, _ = _make_inputs(256, 2)
    output, scales = inv_rope_module.fused_inv_rope_fp8_quant(
        input,
        positions,
        cache,
        2,
        8,
        compact_scales=True,
    )

    assert calls == 1
    assert torch.count_nonzero(output.float()) == 0
    assert torch.all(scales == 1.0)


@torch.inference_mode()
def test_cute_dispatch_threshold_falls_back(monkeypatch: pytest.MonkeyPatch):
    def unexpected_cute_lookup():
        raise AssertionError("CuTe lookup must not run below the token threshold")

    monkeypatch.setattr(inv_rope_module, "current_platform", _fake_sm120_platform())
    monkeypatch.setattr(
        inv_rope_module,
        "_get_inverse_rope_fp8_quant_cute_impl",
        unexpected_cute_lookup,
    )
    input, positions, cache, _, _ = _make_inputs(255, 2)
    output, scales = inv_rope_module.fused_inv_rope_fp8_quant(
        input,
        positions,
        cache,
        2,
        8,
        compact_scales=True,
    )

    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(scales).all()


@torch.inference_mode()
def test_missing_cute_symbol_falls_back(monkeypatch: pytest.MonkeyPatch):
    lookups = 0

    def missing_cute_symbol():
        nonlocal lookups
        lookups += 1
        return None

    monkeypatch.setattr(inv_rope_module, "current_platform", _fake_sm120_platform())
    monkeypatch.setattr(
        inv_rope_module,
        "_get_inverse_rope_fp8_quant_cute_impl",
        missing_cute_symbol,
    )
    input, positions, cache, _, _ = _make_inputs(256, 2)
    output, scales = inv_rope_module.fused_inv_rope_fp8_quant(
        input,
        positions,
        cache,
        2,
        8,
        compact_scales=True,
    )

    assert lookups == 1
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(scales).all()


def test_cute_guard_rejects_cpu_and_non_sm120(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(inv_rope_module, "current_platform", _fake_sm120_platform())
    assert not _can_use(_make_inputs(256, 2, device="cpu"))

    non_sm120 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda capability: False,
    )
    monkeypatch.setattr(inv_rope_module, "current_platform", non_sm120)
    assert not _can_use(_make_inputs(256, 2))
