# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not has_cutedsl(),
    reason="This test requires CUDA and CuTe DSL",
)


@pytest.mark.parametrize(
    ("max_gather_tokens", "num_reqs", "expected"),
    [
        (1, 1, 1024),
        (64, 1, 1024),
        (1, 4, 1),
        (4, 4, 1),
        (5, 4, 2),
        (64, 4, 16),
        (256, 4, 64),
        (1024, 4, 256),
        (4096, 4, 1024),
        (8192, 4, 1024),
    ],
)
def test_select_worker_ctas(
    max_gather_tokens: int,
    num_reqs: int,
    expected: int,
) -> None:
    from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
        _select_worker_ctas,
    )

    assert _select_worker_ctas(max_gather_tokens, num_reqs) == expected


@pytest.mark.parametrize(
    (
        "is_sm120",
        "num_reqs",
        "output_capacity",
        "max_gather_tokens",
        "offset",
        "expected_worker_ctas",
    ),
    [
        (True, 1, 8256, 64, 0, 1024),
        (True, 4, 8256, 64, 0, 16),
        (True, 4, 8448, 256, 5, 64),
        (True, 4, 4096, 4096, 5, 1024),
        (True, 4, 64, None, 0, 16),
        (False, 4, 8256, 64, 0, 1024),
    ],
)
def test_dequant_gather_worker_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    is_sm120: bool,
    num_reqs: int,
    output_capacity: int,
    max_gather_tokens: int | None,
    offset: int,
    expected_worker_ctas: int,
) -> None:
    from vllm.models.deepseek_v4.nvidia.ops import dequant_gather_k_cutedsl as module

    compile_calls: list[dict[str, object]] = []

    def fake_compile(**kwargs: object):
        compile_calls.append(kwargs)
        return lambda *_args: None

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(is_device_capability_family=lambda _capability: is_sm120),
    )
    monkeypatch.setattr(
        module.DequantGatherKCacheKernel,
        "compile",
        staticmethod(fake_compile),
    )

    out = torch.empty(num_reqs, offset + output_capacity, 512)
    module.dequantize_and_gather_k_cache_cutedsl(
        out,
        torch.empty(0),
        torch.empty(0),
        None,
        torch.empty(0),
        block_size=64,
        offset=offset,
        max_gather_tokens=max_gather_tokens,
    )

    assert compile_calls == [
        {
            "block_size": 64,
            "has_gather_lens": False,
            "num_worker_ctas": expected_worker_ctas,
        }
    ]


@pytest.mark.parametrize("max_gather_tokens", [-1, 65])
def test_dequant_gather_rejects_invalid_host_bound(max_gather_tokens: int) -> None:
    from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
        dequantize_and_gather_k_cache_cutedsl,
    )

    with pytest.raises(ValueError, match="must fit in the output capacity"):
        dequantize_and_gather_k_cache_cutedsl(
            torch.empty(4, 64, 512),
            torch.empty(0),
            torch.empty(0),
            None,
            torch.empty(0),
            block_size=64,
            offset=0,
            max_gather_tokens=max_gather_tokens,
        )
