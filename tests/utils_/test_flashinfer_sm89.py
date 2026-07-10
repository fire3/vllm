# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType

import torch

from vllm.utils import flashinfer as fi_utils


def _install_flashinfer_resolver(monkeypatch, resolver) -> None:
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    mla = ModuleType("flashinfer.mla")
    mla.__path__ = []
    core = ModuleType("flashinfer.mla._core")
    core.__dict__["_resolve_dsv4_sparse_mla_backend"] = resolver
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.mla", mla)
    monkeypatch.setitem(sys.modules, "flashinfer.mla._core", core)


def test_sm89_sparse_mla_probe_accepts_patched_flashinfer(monkeypatch) -> None:
    fi_utils.has_flashinfer_sparse_mla_sm89.cache_clear()
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    _install_flashinfer_resolver(monkeypatch, lambda _: "sparse")

    assert fi_utils.has_flashinfer_sparse_mla_sm89()
    fi_utils.has_flashinfer_sparse_mla_sm89.cache_clear()


def test_sm89_sparse_mla_probe_rejects_unpatched_flashinfer(monkeypatch) -> None:
    fi_utils.has_flashinfer_sparse_mla_sm89.cache_clear()
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)

    def reject_sm89(_: torch.device) -> str:
        raise ValueError("unsupported compute capability")

    _install_flashinfer_resolver(monkeypatch, reject_sm89)

    assert not fi_utils.has_flashinfer_sparse_mla_sm89()
    fi_utils.has_flashinfer_sparse_mla_sm89.cache_clear()
