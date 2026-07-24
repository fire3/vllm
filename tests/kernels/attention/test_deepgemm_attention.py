# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    _ceil_to_ue8m0,
    calc_diff,
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.math_utils import cdiv


@pytest.mark.parametrize(
    ("has_extension", "capability", "expected"),
    [
        (False, (8, 9), True),
        (False, (9, 0), False),
        (False, (12, 0), True),
        (True, (9, 0), True),
    ],
)
def test_mqa_backend_availability(
    monkeypatch: pytest.MonkeyPatch,
    has_extension: bool,
    capability: tuple[int, int],
    expected: bool,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: has_extension)
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda family: capability[0] == family // 10,
            is_device_capability=lambda requested: capability == requested,
        ),
    )

    assert deep_gemm.is_mqa_backend_available() is expected


def test_sm89_sparse_indexer_allows_mqa_fallback_without_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
    import vllm.utils.deep_gemm as deep_gemm

    sm89 = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda _: False,
        is_device_capability=lambda capability: capability == (8, 9),
    )
    default_vllm_config.parallel_config.decode_context_parallel_size = 1
    default_vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    monkeypatch.setattr(deep_gemm, "current_platform", sm89)
    monkeypatch.setattr(deep_gemm, "has_deep_gemm", lambda: False)
    monkeypatch.setattr(sparse_indexer, "current_platform", sm89)

    sparse_indexer.SparseAttnIndexer(
        k_cache=torch.empty(0),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=2048,
        head_dim=128,
        max_model_len=4096,
        max_total_seq_len=4096,
        topk_indices_buffer=torch.empty(0, dtype=torch.int32),
    )


def test_sm12x_direct_mqa_topk_dispatch(monkeypatch: pytest.MonkeyPatch):
    import vllm.utils.deep_gemm as deep_gemm
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability_family=lambda _: False,
            is_device_capability=lambda capability: capability == (8, 9),
        ),
    )
    backend = Mock(return_value=True)
    monkeypatch.setattr(sm12x_deep_gemm_fallbacks, "fp8_fp4_mqa_topk_indices", backend)

    q = (torch.empty(0), None)
    kv = (torch.empty(0), torch.empty(0))
    weights = torch.empty(0)
    cu_seqlen_ks = torch.empty(0, dtype=torch.int32)
    cu_seqlen_ke = torch.empty(0, dtype=torch.int32)
    topk_indices = torch.empty(0, dtype=torch.int32)

    assert deep_gemm.fp8_fp4_mqa_topk_indices(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
    )
    backend.assert_called_once_with(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
    )


def test_sm120_deepgemm_mqa_skips_portable_direct_topk(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(
        deep_gemm, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    monkeypatch.setattr(deep_gemm, "_can_use_sm120_deep_gemm_mqa", lambda *args: True)

    q = (torch.empty(0), None)
    kv = (torch.empty(0), torch.empty(0))
    weights = torch.empty(0)
    cu_seqlen_ks = torch.empty(0, dtype=torch.int32)
    cu_seqlen_ke = torch.empty(0, dtype=torch.int32)
    topk_indices = torch.empty(0, dtype=torch.int32)

    assert not deep_gemm.fp8_fp4_mqa_topk_indices(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
    )


@pytest.mark.parametrize("clean_logits", [False, True])
def test_sm120_deepgemm_mqa_logits_uses_vendored_backend(
    monkeypatch: pytest.MonkeyPatch,
    clean_logits: bool,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "_can_use_sm120_deep_gemm_mqa", lambda *args: True)
    backend_output = torch.empty(0)
    vendored_backend = Mock(return_value=backend_output)
    external_backend = Mock()
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_mqa_logits_impl",
        lambda: vendored_backend,
    )
    monkeypatch.setattr(deep_gemm, "_fp8_fp4_mqa_logits_impl", external_backend)

    q = (torch.empty(0), None)
    kv = (torch.empty(0), torch.empty(0))
    weights = torch.empty(0)
    cu_seqlen_ks = torch.empty(0, dtype=torch.int32)
    cu_seqlen_ke = torch.empty(0, dtype=torch.int32)

    output = deep_gemm.fp8_fp4_mqa_logits(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )

    assert output is backend_output
    vendored_backend.assert_called_once_with(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )
    external_backend.assert_not_called()


def test_sm120_mqa_resolver_imports_vendored_directly(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    vendored_backend = Mock()
    vendored = SimpleNamespace(fp8_fp4_mqa_logits=vendored_backend)
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        assert name == "vllm.third_party.deep_gemm"
        return vendored

    monkeypatch.setattr(deep_gemm.importlib, "import_module", import_module)
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        SimpleNamespace(is_arch_support_pdl=lambda: False),
    )
    deep_gemm._get_vendored_sm120_mqa_logits_impl.cache_clear()
    try:
        assert deep_gemm._get_vendored_sm120_mqa_logits_impl() is vendored_backend
        assert imported == ["vllm.third_party.deep_gemm"]
    finally:
        deep_gemm._get_vendored_sm120_mqa_logits_impl.cache_clear()


def test_sm120_mqa_resolver_propagates_vendored_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    def import_module(name: str):
        assert name == "vllm.third_party.deep_gemm"
        raise RuntimeError("broken vendored extension")

    monkeypatch.setattr(deep_gemm.importlib, "import_module", import_module)
    deep_gemm._get_vendored_sm120_mqa_logits_impl.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="broken vendored extension"):
            deep_gemm._get_vendored_sm120_mqa_logits_impl()
    finally:
        deep_gemm._get_vendored_sm120_mqa_logits_impl.cache_clear()


def test_sm120_paged_mqa_resolver_imports_vendored_directly(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    metadata_backend = Mock()
    logits_backend = Mock()
    vendored = SimpleNamespace(
        get_paged_mqa_logits_metadata=metadata_backend,
        fp8_fp4_paged_mqa_logits=logits_backend,
    )
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        assert name == "vllm.third_party.deep_gemm"
        return vendored

    monkeypatch.setattr(deep_gemm.importlib, "import_module", import_module)
    monkeypatch.setattr(
        deep_gemm,
        "current_platform",
        SimpleNamespace(is_arch_support_pdl=lambda: False),
    )
    deep_gemm._get_vendored_sm120_paged_mqa_impls.cache_clear()
    try:
        assert deep_gemm._get_vendored_sm120_paged_mqa_impls() == (
            metadata_backend,
            logits_backend,
        )
        assert imported == ["vllm.third_party.deep_gemm"]
    finally:
        deep_gemm._get_vendored_sm120_paged_mqa_impls.cache_clear()


def test_sm120_paged_mqa_logits_uses_vendored_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    backend_output = torch.empty(0)
    vendored_backend = Mock(return_value=backend_output)
    external_backend = Mock()
    fallback = Mock()
    monkeypatch.setattr(
        deep_gemm,
        "_can_use_sm120_deep_gemm_paged_mqa",
        lambda *args: True,
    )
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_paged_mqa_impls",
        lambda: (Mock(), vendored_backend),
    )
    monkeypatch.setattr(deep_gemm, "_fp8_fp4_paged_mqa_logits_impl", external_backend)
    monkeypatch.setattr(deep_gemm, "_fp8_paged_mqa_logits_sm12x", fallback)

    q = (torch.empty(0), None)
    kv_cache = torch.empty(0)
    weights = torch.empty(0)
    context_lens = torch.empty(0, dtype=torch.int32)
    block_tables = torch.empty(0, dtype=torch.int32)
    schedule_metadata = torch.empty(0, dtype=torch.int32)
    output = deep_gemm.fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        1024,
        clean_logits=False,
    )

    assert output is backend_output
    vendored_backend.assert_called_once_with(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        1024,
        clean_logits=False,
        logits_dtype=torch.float32,
    )
    external_backend.assert_not_called()
    fallback.assert_not_called()


def test_sm120_paged_mqa_logits_falls_back_without_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    backend_output = torch.empty(0)
    fallback = Mock(return_value=backend_output)
    monkeypatch.setattr(
        deep_gemm,
        "_can_use_sm120_deep_gemm_paged_mqa",
        lambda *args: False,
    )
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    monkeypatch.setattr(deep_gemm, "_fp8_paged_mqa_logits_sm12x", fallback)

    q = (torch.empty(0), None)
    kv_cache = torch.empty(0)
    weights = torch.empty(0)
    context_lens = torch.empty(0, dtype=torch.int32)
    block_tables = torch.empty(0, dtype=torch.int32)
    schedule_metadata = torch.empty(0, dtype=torch.int32)
    output = deep_gemm.fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        1024,
        clean_logits=False,
    )

    assert output is backend_output
    fallback.assert_called_once_with(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        1024,
    )


def test_sm120_paged_mqa_metadata_uses_vendored_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    backend_output = torch.empty((1, 2), dtype=torch.int32)
    vendored_backend = Mock(return_value=backend_output)
    external_backend = Mock()
    monkeypatch.setattr(
        deep_gemm,
        "is_sm120_deep_gemm_paged_mqa_supported",
        lambda: True,
    )
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_paged_mqa_impls",
        lambda: (vendored_backend, Mock()),
    )
    monkeypatch.setattr(
        deep_gemm, "_get_paged_mqa_logits_metadata_impl", external_backend
    )
    context_lens = torch.ones((2, 1), dtype=torch.int32)

    output = deep_gemm.get_paged_mqa_logits_metadata(context_lens, 64, 188)

    assert output is backend_output
    vendored_backend.assert_called_once_with(context_lens, 64, 188)
    external_backend.assert_not_called()


def test_sm120_paged_mqa_enables_scheduler_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.attention.backends.mla import indexer

    monkeypatch.setattr(indexer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        indexer.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(
        indexer,
        "is_sm120_deep_gemm_paged_mqa_supported",
        lambda: True,
    )
    assert indexer._uses_deep_gemm_scheduler_metadata()

    monkeypatch.setattr(
        indexer,
        "is_sm120_deep_gemm_paged_mqa_supported",
        lambda: False,
    )
    assert not indexer._uses_deep_gemm_scheduler_metadata()


@pytest.mark.parametrize(("value", "expected"), [("0", False), ("1", True)])
def test_sm120_deepgemm_mqa_policy_honors_global_opt_out(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: bool,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", value)
    assert deep_gemm._sm120_deep_gemm_mqa_enabled() is expected


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="requires SM120",
)
def test_sm120_deepgemm_mqa_eligibility_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "_sm120_deep_gemm_mqa_enabled", lambda: True)
    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_mqa_logits_impl",
        lambda: Mock(),
    )

    m, n, h, d = 2, 4, 16, 32
    q_values = torch.empty((m, h, d), device="cuda", dtype=torch.float8_e4m3fn)
    k_values = torch.empty((n, d), device="cuda", dtype=torch.float8_e4m3fn)
    k_scale = torch.empty(n, device="cuda", dtype=torch.float32)
    weights = torch.empty((m, h), device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.zeros(m, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.full((m,), n, device="cuda", dtype=torch.int32)

    def eligible(
        q: tuple[torch.Tensor, torch.Tensor | None] = (q_values, None),
        kv: tuple[torch.Tensor, torch.Tensor] = (k_values, k_scale),
        query_weights: torch.Tensor = weights,
    ) -> bool:
        return deep_gemm._can_use_sm120_deep_gemm_mqa(
            q,
            kv,
            query_weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )

    assert eligible()

    invalid_q = torch.empty((m, 8, d), device="cuda", dtype=torch.float8_e4m3fn)
    invalid_weights = torch.empty((m, 8), device="cuda", dtype=torch.float32)
    assert not eligible((invalid_q, None), query_weights=invalid_weights)

    noncontiguous_k = torch.empty((n, d * 2), device="cuda", dtype=torch.float8_e4m3fn)[
        :, ::2
    ]
    assert not eligible(kv=(noncontiguous_k, k_scale))

    monkeypatch.setattr(
        deep_gemm,
        "_get_vendored_sm120_mqa_logits_impl",
        lambda: None,
    )
    assert not eligible()

    monkeypatch.setattr(deep_gemm, "_sm120_deep_gemm_mqa_enabled", lambda: False)
    assert not eligible()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not current_platform.is_device_capability_family(120),
    reason="requires SM120",
)
def test_sm120_deepgemm_paged_mqa_eligibility_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(
        deep_gemm,
        "is_sm120_deep_gemm_paged_mqa_supported",
        lambda: True,
    )
    batch_size, next_n, num_heads, head_dim = 2, 1, 16, 32
    q_values = torch.empty(
        (batch_size, next_n, num_heads, head_dim),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    kv_cache = torch.empty(
        (4, 64, 1, head_dim + 4),
        device="cuda",
        dtype=torch.uint8,
    )
    weights = torch.empty(
        (batch_size * next_n, num_heads),
        device="cuda",
        dtype=torch.float32,
    )
    context_lens = torch.ones(
        (batch_size, next_n),
        device="cuda",
        dtype=torch.int32,
    )
    block_tables = torch.zeros((batch_size, 1), device="cuda", dtype=torch.int32)
    schedule_metadata = torch.zeros((189, 2), device="cuda", dtype=torch.int32)

    def eligible(
        query: tuple[torch.Tensor, torch.Tensor | None] = (q_values, None),
        cache: torch.Tensor = kv_cache,
        indices: torch.Tensor | None = None,
    ) -> bool:
        return deep_gemm._can_use_sm120_deep_gemm_paged_mqa(
            query,
            cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            64,
            indices,
        )

    assert eligible()
    assert not eligible((q_values, torch.empty(0, device="cuda")))
    assert not eligible(
        indices=torch.zeros(batch_size, device="cuda", dtype=torch.int32)
    )
    assert not eligible(cache=kv_cache[:, :32])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_mqa_logits_rejects_output_on_wrong_device() -> None:
    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import fp8_mqa_logits_triton

    q = torch.empty((0, 16, 32), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((0, 32), device="cuda", dtype=torch.float8_e4m3fn)
    k_scale = torch.empty(0, device="cuda", dtype=torch.float32)
    weights = torch.empty((0, 16), device="cuda", dtype=torch.float32)
    cu_seqlen_ks = torch.empty(0, device="cuda", dtype=torch.int32)
    cu_seqlen_ke = torch.empty(0, device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="device cuda"):
        fp8_mqa_logits_triton(
            q,
            (k, k_scale),
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out=torch.empty((0, 0), dtype=torch.float32),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sm12x_direct_topk_handles_topk_wider_than_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.models.deepseek_v4.nvidia.ops import (
        sm12x_deep_gemm_fallbacks,
        sm12x_mqa,
    )

    logits = torch.randn((4, 2), device="cuda", dtype=torch.float32)
    monkeypatch.setattr(sm12x_mqa, "fp8_mqa_logits_triton", lambda *args: logits)
    q = torch.empty((4, 16, 32), device="cuda", dtype=torch.float8_e4m3fn)
    k = torch.empty((2, 32), device="cuda", dtype=torch.float8_e4m3fn)
    k_scale = torch.ones(2, device="cuda", dtype=torch.float32)
    weights = torch.empty((4, 16), device="cuda", dtype=torch.float32)
    row_starts = torch.zeros(4, device="cuda", dtype=torch.int32)
    row_ends = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32)
    output = torch.empty((4, 4), device="cuda", dtype=torch.int32)

    assert sm12x_deep_gemm_fallbacks._fp8_mqa_logits_topk_triton(
        (q, None),
        (k, k_scale),
        weights,
        row_starts,
        row_ends,
        output,
    )
    actual = output.sort(dim=1).values
    expected = torch.tensor(
        [
            [-1, -1, -1, -1],
            [-1, -1, -1, 0],
            [-1, -1, 0, 1],
            [-1, -1, 0, 1],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)


def test_sm120_mqa_logits_falls_back_without_deepgemm(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "_can_use_sm120_deep_gemm_mqa", lambda *args: False)
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    backend_output = torch.empty(0)
    fallback = Mock(return_value=backend_output)
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_sm12x", fallback)

    q = (torch.empty(0), None)
    kv = (torch.empty(0), torch.empty(0))
    weights = torch.empty(0)
    cu_seqlen_ks = torch.empty(0, dtype=torch.int32)
    cu_seqlen_ke = torch.empty(0, dtype=torch.int32)

    output = deep_gemm.fp8_fp4_mqa_logits(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )

    assert output is backend_output
    fallback.assert_called_once_with(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        False,
    )


def test_sm12x_direct_paged_mqa_topk_dispatch(monkeypatch: pytest.MonkeyPatch):
    import vllm.utils.deep_gemm as deep_gemm
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    monkeypatch.setattr(
        deep_gemm, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(deep_gemm, "_use_sm12x_mqa_fallback", lambda: True)
    backend = Mock(return_value=True)
    monkeypatch.setattr(
        sm12x_deep_gemm_fallbacks,
        "fp8_fp4_paged_mqa_topk_indices",
        backend,
    )

    q = (torch.empty(0), None)
    kv_cache = torch.empty(0)
    weights = torch.empty(0)
    context_lens = torch.empty(0, dtype=torch.int32)
    block_tables = torch.empty(0, dtype=torch.int32)
    topk_indices = torch.empty(0, dtype=torch.int32)

    assert deep_gemm.fp8_fp4_paged_mqa_topk_indices(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        1024,
        topk_indices,
    )
    backend.assert_called_once_with(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        1024,
        topk_indices,
    )


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    ke = torch.zeros(seq_len, dtype=torch.int, device="cuda")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
):
    seq_len_kv = kv.shape[0]

    k = kv
    q = q.float()
    k = k.float()

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi
    score = torch.einsum("mhd,nd->hmn", q, k)
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
@pytest.mark.parametrize("clean_logits", [True, False])
def test_deepgemm_fp8_mqa_logits(clean_logits: bool):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128
    for seq_len in (512,):
        for seq_len_kv in (1024,):
            for disable_cp in (False, True):
                q = torch.randn(
                    seq_len,
                    num_heads,
                    head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv = torch.randn(
                    seq_len_kv, head_dim, device="cuda", dtype=torch.bfloat16
                )
                weights = torch.randn(
                    seq_len, num_heads, device="cuda", dtype=torch.float32
                )

                if disable_cp:
                    ks = torch.zeros(seq_len, dtype=torch.int, device="cuda")
                    ke = torch.arange(seq_len, dtype=torch.int, device="cuda") + (
                        seq_len_kv - seq_len
                    )
                else:
                    ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)
                logits = fp8_fp4_mqa_logits(
                    (q_fp8, None), kv_fp8, weights, ks, ke, clean_logits=clean_logits
                )

                ref_logits = _ref_fp8_mqa_logits(
                    q=q,
                    kv=kv,
                    weights=weights,
                    cu_seqlen_ks=ks,
                    cu_seqlen_ke=ke,
                )
                ref_neginf_mask = ref_logits == float("-inf")

                if clean_logits:
                    neginf_mask = logits == float("-inf")
                    assert torch.equal(neginf_mask, ref_neginf_mask)

                ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
                logits = logits.masked_fill(ref_neginf_mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"


def _ref_fp8_fp4_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size,
                (block_rk + 1) * block_size,
                device="cuda",
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGEMM not available")
@pytest.mark.skipif(
    not current_platform.has_device_capability(90), reason="SM90 and SM100 only"
)
def test_deepgemm_fp8_fp4_paged_mqa_logits():
    # NOTE: clean_logits=True is incompatible with the 2D context_lens
    # required by csrc/apis/attention.hpp; only the False path is exercised.
    clean_logits = False
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096
    for batch_size, next_n in [(4, 1), (2, 2)]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                num_blocks, blocksize = max_model_len * 2, 64

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device="cuda",
                    dtype=torch.float32,
                )

                context_lens = (
                    torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                    .cuda()
                    .to(torch.int32)
                )
                max_block_len = (
                    (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros(
                    (batch_size, max_block_len),
                    device="cuda",
                    dtype=torch.int32,
                )

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    for j in range((ctx_len + blocksize - 1) // blocksize):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                # deep_gemm paged MQA logits requires 2D context_lens of
                # shape (B, next_n) (csrc/apis/attention.hpp:332-335);
                # see indexer.py:607-608. For each batch/next_n token, the
                # effective context length is context_lens[b] - next_n + j + 1.
                next_n_arange = torch.arange(next_n, device="cuda", dtype=torch.int32)
                context_lens_2d = (
                    context_lens.unsqueeze(-1) - next_n + 1 + next_n_arange
                ).contiguous()
                schedule_metadata = get_paged_mqa_logits_metadata(
                    context_lens_2d, blocksize, get_num_sms()
                )
                logits = fp8_fp4_paged_mqa_logits(
                    (q_fp8, None),
                    kv_cache_fp8,
                    weights,
                    context_lens_2d,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                    clean_logits=clean_logits,
                )

                ref_logits = _ref_fp8_fp4_paged_mqa_logits(
                    q,
                    kv_cache,
                    weights,
                    context_lens,
                    block_tables,
                    max_model_len,
                )

                positions = (
                    torch.arange(max_model_len, device="cuda")
                    .unsqueeze(0)
                    .expand(batch_size * next_n, -1)
                )
                row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
                next_n_offset = (
                    torch.arange(batch_size * next_n, device="cuda") % next_n
                )
                mask = positions <= (
                    context_lens[row_indices] - next_n + next_n_offset
                ).unsqueeze(1)

                logits = logits.masked_fill(~mask, 0)
                ref_logits = ref_logits.masked_fill(~mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"
