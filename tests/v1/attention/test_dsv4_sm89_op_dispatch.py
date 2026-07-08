# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89 DSv4 auxiliary ops must avoid CuTe-DSL-only implementations."""

from pathlib import Path

from vllm.platforms.interface import DeviceCapability
from vllm.utils import import_utils


def test_has_cutedsl_false_on_sm89(monkeypatch) -> None:
    monkeypatch.setattr(import_utils, "_has_module", lambda _: True)

    class FakePlatform:
        @staticmethod
        def is_cuda() -> bool:
            return True

        @staticmethod
        def get_device_capability() -> DeviceCapability:
            return DeviceCapability(8, 9)

        @staticmethod
        def is_device_capability(capability: tuple[int, int]) -> bool:
            return capability == (8, 9)

    monkeypatch.setattr("vllm.platforms.current_platform", FakePlatform)

    assert not import_utils.has_cutedsl()


def test_compressor_cutedsl_dispatch_is_gated_for_sm89() -> None:
    compressor = Path("vllm/models/deepseek_v4/compressor.py").read_text()

    assert "has_cutedsl()" in compressor
    assert "current_platform.is_cuda() and self.head_dim == 512 and has_cutedsl()" in (
        compressor.replace("\n", " ")
    )
