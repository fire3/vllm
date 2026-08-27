# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for the DSv4 Triton sparse-MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str = "deepseek_v4") -> SimpleNamespace:
    return SimpleNamespace(
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.TRITON_MLA_SPARSE_DSV4
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_triton_mla_sparse_dsv4_backend_registered() -> None:
    from vllm.models.deepseek_v4.nvidia.triton_sparse import (
        DeepseekV4TritonMLASparseBackend,
    )

    assert (
        DeepseekV4TritonMLASparseBackend.get_name() == "TRITON_MLA_SPARSE_DSV4"
    )
    assert (
        AttentionBackendEnum.TRITON_MLA_SPARSE_DSV4.get_class()
        is DeepseekV4TritonMLASparseBackend
    )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        (DeviceCapability(12, 0), []),
        (DeviceCapability(8, 9), []),
        (DeviceCapability(10, 0), ["requires SM89 or SM120"]),
    ],
)
def test_select_dsv4_attn_cls_gates_capability(
    monkeypatch,
    capability,
    expected,
) -> None:
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls
    from vllm.models.deepseek_v4.nvidia.triton_sparse import (
        DeepseekV4TritonMLAAttention,
    )

    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: capability
    )
    vllm_config = _fake_vllm_config()
    if expected:
        with pytest.raises(ValueError, match=expected[0]):
            _select_dsv4_attn_cls(vllm_config)
    else:
        assert _select_dsv4_attn_cls(vllm_config) is DeepseekV4TritonMLAAttention


def test_triton_backend_combination_does_not_require_flashinfer(
    monkeypatch,
) -> None:
    from vllm.models.deepseek_v4.nvidia.triton_sparse import (
        DeepseekV4TritonMLASparseBackend,
    )

    monkeypatch.setattr(
        "vllm.utils.flashinfer.has_flashinfer_sparse_mla_sm89", lambda: False
    )
    monkeypatch.setattr(
        "vllm.utils.flashinfer.has_flashinfer_sparse_mla_sm120", lambda: False
    )

    with set_current_vllm_config(_fake_vllm_config()):
        invalid = DeepseekV4TritonMLASparseBackend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=256,
            use_mla=True,
            has_sink=True,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 9),
            attn_type="decoder",
        )
    assert invalid == []


def test_flashinfer_backend_remains_default(monkeypatch) -> None:
    """FLASHINFER_MLA_SPARSE_DSV4 must keep selecting the FlashInfer class."""
    from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
        DeepseekV4FlashInferMLAAttention,
    )
    from vllm.models.deepseek_v4.nvidia.model import _select_dsv4_attn_cls

    vllm_config = _fake_vllm_config()
    vllm_config.attention_config.backend = (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
    )
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda: DeviceCapability(8, 9)
    )
    # On SM89 upstream v0.28.0 still routes FLASHINFER_MLA_SPARSE_DSV4 to the
    # generic FlashInfer MLA attention (the FlashInfer-on-SM89 enablement was
    # intentionally not part of this Triton port). The assertion guards that
    # the Triton backend never hijacks the FlashInfer selection.
    assert (
        _select_dsv4_attn_cls(vllm_config) is DeepseekV4FlashInferMLAAttention
    )
