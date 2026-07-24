# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch


def test_sm120_hc_prenorm_uses_vendored_deepgemm(monkeypatch):
    import vllm.utils.deep_gemm as deep_gemm

    backend = Mock(return_value=None)
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "_can_use_sm120_deep_gemm_hc_prenorm",
        lambda *args: True,
    )
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_hc_prenorm_impl",
        lambda: backend,
    )

    x = torch.empty((1, 64), dtype=torch.bfloat16)
    fn = torch.empty((8, 64), dtype=torch.float32)
    out = torch.empty((1, 1, 8), dtype=torch.float32)
    sqrsum = torch.empty((1, 1), dtype=torch.float32)
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 1)

    backend.assert_called_once_with(x, fn, out, sqrsum, 1)


def test_sm120_hc_prenorm_keeps_portable_fallback(monkeypatch):
    import vllm.utils.deep_gemm as deep_gemm
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    fallback = Mock(return_value=None)
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "_can_use_sm120_deep_gemm_hc_prenorm",
        lambda *args: False,
    )
    monkeypatch.setattr(
        sm12x_deep_gemm_fallbacks,
        "_tf32_hc_prenorm_gemm_sm12x",
        fallback,
    )

    x = torch.empty((1, 64), dtype=torch.bfloat16)
    fn = torch.empty((8, 64), dtype=torch.float32)
    out = torch.empty((1, 1, 8), dtype=torch.float32)
    sqrsum = torch.empty((1, 1), dtype=torch.float32)
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 1)

    fallback.assert_called_once_with(x, fn, out, sqrsum, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("num_tokens", "expected"), ((32, False), (64, True)))
def test_sm120_hc_prenorm_cuda_graph_bucket_guard(monkeypatch, num_tokens, expected):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        Mock(
            is_cuda=Mock(return_value=True),
            is_device_capability_family=Mock(return_value=True),
        ),
    )
    monkeypatch.setattr(deep_gemm.envs, "VLLM_USE_DEEP_GEMM", True)
    backend = Mock()
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_hc_prenorm_impl",
        backend,
    )

    x = torch.empty((num_tokens, 256), device="cuda", dtype=torch.bfloat16)
    fn = torch.empty((8, 256), device="cuda", dtype=torch.float32)
    out = torch.empty((1, num_tokens, 8), device="cuda", dtype=torch.float32)
    sqrsum = torch.empty((1, num_tokens), device="cuda", dtype=torch.float32)

    assert (
        deep_gemm._can_use_sm120_deep_gemm_hc_prenorm(x, fn, out, sqrsum, 1) is expected
    )
    assert backend.call_count == int(expected)
