# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 selects the FlashInfer sparse MLA path ported from SM120."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia import model as dsv4_model
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadataBuilder
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla import sparse_swa
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops import flashmla as flashmla_ops


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
    vllm_config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    assert (
        dsv4_model._select_dsv4_attn_cls(vllm_config)
        is DeepseekV4FlashInferSM120Attention
    )


def test_sm89_dspark_uses_uniform_cudagraphs(monkeypatch) -> None:
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability: False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dspark",
            num_speculative_tokens=7,
        )
    )

    builders = (
        DeepseekV4FlashMLAMetadataBuilder,
        DeepseekV32IndexerMetadataBuilder,
        DeepseekSparseSWAMetadataBuilder,
    )
    for builder in builders:
        assert (
            builder.get_cudagraph_support(vllm_config, None)
            is AttentionCGSupport.UNIFORM_BATCH
        )


def test_sm89_swa_metadata_skips_flashmla_scheduler(monkeypatch) -> None:
    sm89 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda _: False,
    )
    monkeypatch.setattr(flashmla_ops, "current_platform", sm89)
    monkeypatch.setattr(flashmla_ops, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla_ops, "_flashmla_extension_C_AVAILABLE", True)
    get_mla_metadata = Mock()
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", get_mla_metadata)
    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder._layer_types = {"swaonly"}

    metadata = builder.build_tile_scheduler(num_decode_tokens=1)

    assert all(value is None for value in metadata.values())
    get_mla_metadata.assert_not_called()


def test_hopper_swa_metadata_keeps_flashmla_scheduler(monkeypatch) -> None:
    hopper = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 90,
    )
    monkeypatch.setattr(flashmla_ops, "current_platform", hopper)
    monkeypatch.setattr(flashmla_ops, "_flashmla_C_AVAILABLE", True)
    monkeypatch.setattr(flashmla_ops, "_flashmla_extension_C_AVAILABLE", True)
    scheduler = object()
    get_mla_metadata = Mock(return_value=(scheduler, None))
    monkeypatch.setattr(sparse_swa, "get_mla_metadata", get_mla_metadata)
    builder = object.__new__(DeepseekSparseSWAMetadataBuilder)
    builder._layer_types = {"swaonly"}

    metadata = builder.build_tile_scheduler(num_decode_tokens=1)

    assert metadata["swaonly"] is scheduler
    get_mla_metadata.assert_called_once_with()
