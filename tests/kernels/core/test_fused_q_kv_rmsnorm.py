# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness + large-token-count launch tests for fused_q_kv_rmsnorm.

Before the grid-dim fix the kernel used grid ``(2, num_tokens)``, which hit
CUDA's 65535 grid-y cap for ``num_tokens >= 65536`` and failed with
``Triton Error [CUDA]: invalid argument`` at every large chunked-prefill
profile run. These tests pin the new grid layout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm as rmsnorm_module
from vllm.models.deepseek_v4.common.ops import fused_q_kv_rmsnorm
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused_q_kv_rmsnorm requires a CUDA/ROCm device",
)


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    y = x_f32 * torch.rsqrt(variance + eps) * w.to(torch.float32)
    return y.to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 17, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_q_kv_rmsnorm_correctness(num_tokens: int, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)
    eps = 1e-6

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, eps)

    qr_ref = _ref_rmsnorm(qr, qw, eps)
    kv_ref = _ref_rmsnorm(kv, kvw, eps)

    tol = dict(rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(qr_out, qr_ref, **tol)
    torch.testing.assert_close(kv_out, kv_ref, **tol)


@pytest.mark.parametrize("num_tokens", [65535, 65536, 131072])
def test_fused_q_kv_rmsnorm_launches_past_grid_y_cap(num_tokens: int):
    """Regression guard: grid used to be (2, num_tokens), hitting CUDA's
    65535 grid-y cap at num_tokens >= 65536. The new grid (num_tokens, 2)
    lifts that bound to 2**31-1."""
    device = "cuda"
    dtype = torch.bfloat16
    q_size, kv_size = 192, 576
    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qw = torch.randn(q_size, dtype=dtype, device=device)
    kvw = torch.randn(kv_size, dtype=dtype, device=device)

    qr_out, kv_out = fused_q_kv_rmsnorm(qr, kv, qw, kvw, 1e-6)
    # spot-check a couple of rows against the torch reference
    for row in (0, num_tokens // 2, num_tokens - 1):
        torch.testing.assert_close(
            qr_out[row],
            _ref_rmsnorm(qr[row : row + 1], qw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            kv_out[row],
            _ref_rmsnorm(kv[row : row + 1], kvw, 1e-6)[0],
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.parametrize(
    ("num_tokens", "parallel_tasks"),
    [(16, True), (8192, False)],
)
def test_fused_q_kv_rmsnorm_dual_cute_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    parallel_tasks: bool,
):
    q_size, kv_size = 1536, 512
    qr_kv = torch.randn(
        num_tokens,
        q_size + kv_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qr, kv = qr_kv.split([q_size, kv_size], dim=-1)
    q_weight = torch.randn(q_size, dtype=torch.bfloat16, device="cuda")
    kv_weight = torch.randn(kv_size, dtype=torch.bfloat16, device="cuda")
    calls = []

    def fake_dual_rmsnorm(
        q,
        k,
        qw,
        kw,
        q_out,
        k_out,
        eps,
        *,
        parallel_tasks,
    ):
        calls.append(parallel_tasks)
        q_out.copy_(_ref_rmsnorm(q, qw, eps))
        k_out.copy_(_ref_rmsnorm(k, kw, eps))

    monkeypatch.setattr(
        rmsnorm_module,
        "_get_dual_rmsnorm_cute_impl",
        lambda: fake_dual_rmsnorm,
    )
    monkeypatch.setattr(
        rmsnorm_module,
        "_can_use_dual_rmsnorm_cute",
        lambda *args: True,
    )

    qr_out, kv_out = fused_q_kv_rmsnorm(
        qr,
        kv,
        q_weight,
        kv_weight,
        1e-6,
    )

    assert calls == [parallel_tasks]
    torch.testing.assert_close(qr_out, _ref_rmsnorm(qr, q_weight, 1e-6))
    torch.testing.assert_close(kv_out, _ref_rmsnorm(kv, kv_weight, 1e-6))


def test_fused_q_kv_rmsnorm_small_bucket_retains_triton(
    monkeypatch: pytest.MonkeyPatch,
):
    q_size, kv_size = 1536, 512
    qr = torch.randn(8, q_size, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(8, kv_size, dtype=torch.bfloat16, device="cuda")
    q_weight = torch.randn(q_size, dtype=torch.bfloat16, device="cuda")
    kv_weight = torch.randn(kv_size, dtype=torch.bfloat16, device="cuda")
    monkeypatch.setattr(
        rmsnorm_module,
        "_get_dual_rmsnorm_cute_impl",
        lambda: pytest.fail("small buckets must not resolve the CuTe kernel"),
    )

    qr_out, kv_out = fused_q_kv_rmsnorm(
        qr,
        kv,
        q_weight,
        kv_weight,
        1e-6,
    )

    torch.testing.assert_close(
        qr_out,
        _ref_rmsnorm(qr, q_weight, 1e-6),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        kv_out,
        _ref_rmsnorm(kv, kv_weight, 1e-6),
        rtol=1e-2,
        atol=1e-2,
    )


def test_dual_cute_guard_is_scoped_to_sm120_production_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeSM120Platform:
        @staticmethod
        def is_cuda():
            return True

        @staticmethod
        def is_device_capability_family(capability):
            return capability == 120

    monkeypatch.setattr(rmsnorm_module, "current_platform", FakeSM120Platform())
    q_size, kv_size = 1536, 512
    qr_kv = torch.empty(
        16,
        q_size + kv_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    qr, kv = qr_kv.split([q_size, kv_size], dim=-1)
    q_weight = torch.empty(q_size, dtype=torch.bfloat16, device="cuda")
    kv_weight = torch.empty(kv_size, dtype=torch.bfloat16, device="cuda")
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)

    assert rmsnorm_module._can_use_dual_rmsnorm_cute(
        qr,
        kv,
        q_weight,
        kv_weight,
        qr_out,
        kv_out,
    )
    assert not rmsnorm_module._can_use_dual_rmsnorm_cute(
        qr[:8],
        kv[:8],
        q_weight,
        kv_weight,
        qr_out[:8],
        kv_out[:8],
    )
    assert not rmsnorm_module._can_use_dual_rmsnorm_cute(
        qr.float(),
        kv.float(),
        q_weight.float(),
        kv_weight.float(),
        qr_out.float(),
        kv_out.float(),
    )

    cpu_tensors = (
        torch.empty(16, q_size, dtype=torch.bfloat16),
        torch.empty(16, kv_size, dtype=torch.bfloat16),
        torch.empty(q_size, dtype=torch.bfloat16),
        torch.empty(kv_size, dtype=torch.bfloat16),
        torch.empty(16, q_size, dtype=torch.bfloat16),
        torch.empty(16, kv_size, dtype=torch.bfloat16),
    )
    assert not rmsnorm_module._can_use_dual_rmsnorm_cute(*cpu_tensors)


def test_dual_cute_impl_missing_symbol_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    rmsnorm_module._get_dual_rmsnorm_cute_impl.cache_clear()
    monkeypatch.setattr(rmsnorm_module, "has_cutedsl", lambda: True)
    monkeypatch.setattr(
        rmsnorm_module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )
    try:
        assert rmsnorm_module._get_dual_rmsnorm_cute_impl() is None
    finally:
        rmsnorm_module._get_dual_rmsnorm_cute_impl.cache_clear()


def test_dual_cute_impl_surfaces_unexpected_import_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken_import(_name: str):
        raise ImportError("broken FlashInfer installation")

    rmsnorm_module._get_dual_rmsnorm_cute_impl.cache_clear()
    monkeypatch.setattr(rmsnorm_module, "has_cutedsl", lambda: True)
    monkeypatch.setattr(rmsnorm_module.importlib, "import_module", broken_import)
    try:
        with pytest.raises(ImportError, match="broken FlashInfer installation"):
            rmsnorm_module._get_dual_rmsnorm_cute_impl()
    finally:
        rmsnorm_module._get_dual_rmsnorm_cute_impl.cache_clear()
