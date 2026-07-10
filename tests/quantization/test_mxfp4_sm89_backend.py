# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability

SM89_CAPABILITY = DeviceCapability(major=8, minor=9)


def _sm89_has_device_capability(capability: tuple[int, int] | int) -> bool:
    if isinstance(capability, tuple):
        return capability <= SM89_CAPABILITY
    return SM89_CAPABILITY.to_int() >= capability


def _sm89_is_device_capability(capability: tuple[int, int] | int) -> bool:
    if isinstance(capability, tuple):
        return capability == SM89_CAPABILITY
    return SM89_CAPABILITY.to_int() == capability


def _sm89_is_device_capability_family(capability: int) -> bool:
    return (SM89_CAPABILITY.to_int() // 10) == (capability // 10)


def _make_mxfp4_moe_config(
    *,
    moe_backend: str = "auto",
    use_batched_activation_format: bool = False,
) -> FusedMoEConfig:
    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    if use_batched_activation_format:
        parallel_config = replace(
            parallel_config,
            dp_size=2,
            ep_size=2,
            use_ep=True,
            all2all_backend="deepep_low_latency",
        )
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=parallel_config,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


def test_sm89_auto_mxfp4_backend_selects_marlin():
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
        ) as mock_get_config,
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(
            current_platform,
            "has_device_capability",
            side_effect=_sm89_has_device_capability,
        ),
        patch.object(
            current_platform,
            "is_device_capability",
            side_effect=_sm89_is_device_capability,
        ),
        patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=_sm89_is_device_capability_family,
        ),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=SM89_CAPABILITY,
        ),
    ):
        mock_get_config.return_value.model_config.quantization_config = None

        backend, experts_cls = select_mxfp4_moe_backend(_make_mxfp4_moe_config())

    assert backend == Mxfp4MoeBackend.MARLIN
    assert backend != Mxfp4MoeBackend.DEEPGEMM_MXFP4
    assert experts_cls is not None


def test_sm89_auto_mxfp4_batched_format_selects_batched_marlin():
    with (
        patch(
            "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
        ) as mock_get_config,
        patch.object(current_platform, "is_cuda", return_value=True),
        patch.object(current_platform, "is_rocm", return_value=False),
        patch.object(current_platform, "is_cpu", return_value=False),
        patch.object(current_platform, "is_xpu", return_value=False),
        patch.object(
            current_platform,
            "has_device_capability",
            side_effect=_sm89_has_device_capability,
        ),
        patch.object(
            current_platform,
            "is_device_capability",
            side_effect=_sm89_is_device_capability,
        ),
        patch.object(
            current_platform,
            "is_device_capability_family",
            side_effect=_sm89_is_device_capability_family,
        ),
        patch.object(
            current_platform,
            "get_device_capability",
            return_value=SM89_CAPABILITY,
        ),
    ):
        mock_get_config.return_value.model_config.quantization_config = None

        backend, experts_cls = select_mxfp4_moe_backend(
            _make_mxfp4_moe_config(use_batched_activation_format=True)
        )

    assert backend == Mxfp4MoeBackend.BATCHED_MARLIN
    assert backend != Mxfp4MoeBackend.DEEPGEMM_MXFP4
    assert experts_cls is not None
