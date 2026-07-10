# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 selects the FlashInfer sparse MLA path ported from SM120."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)


def _fake_vllm_config(model_type: str = "deepseek_v4") -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm89_capability_accepted(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: True)

    with set_current_vllm_config(_fake_vllm_config()):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 9),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm86_capability_rejected(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config()):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 6),
            attn_type="decoder",
        )

    assert "compute capability not supported" in invalid_reasons


def test_sm89_dsv4_defaults_to_ported_sm120_attention(monkeypatch) -> None:
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: True)
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert (
        dsv4_model._select_dsv4_attn_cls(vllm_config)
        is DeepseekV4FlashInferSM120Attention
    )


def test_sm89_dsv4_rejects_unpatched_flashinfer(monkeypatch) -> None:
    monkeypatch.setattr(
        dsv4_model.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm89", lambda: False)
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    with pytest.raises(RuntimeError, match="sparse MLA SM89 JIT patch"):
        dsv4_model._select_dsv4_attn_cls(vllm_config)
