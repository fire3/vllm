# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the phase-2A tiled Triton sparse-MLA prefill kernel.

The kernel fuses the SWA + compressed (c4/c128) sources into a single launch
with CSR (flat indices + indptr) metadata and one shared online-softmax
accumulator. Tests compare against the torch reference and, when available,
the FlashInfer DSv4 launcher on identical inputs.
"""

from typing import Optional

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
    _pack_sparse_rows,
)
from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_vllm,
)
from vllm.models.deepseek_v4.nvidia.triton_sparse import (
    DeepseekV4TritonMLAAttention,
)

_PAGE_SIZE = 64
_DATA_STRIDE = 576
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM
_ROPE_BYTES = 2 * _ROPE_DIM


def _make_packed_cache(
    num_blocks: int,
    device: torch.device,
    seed: int,
    page_size: int = _PAGE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a packed fp8_ds_mla cache plus the dequantized reference KV."""
    torch.manual_seed(seed)
    block_bytes = page_size * 584
    scale_off = page_size * _DATA_STRIDE
    num_tokens = num_blocks * page_size
    kv = torch.randn(num_tokens, _D, device=device) * 0.05
    kv_nope = torch.zeros(num_tokens, _NOPE_DIM, dtype=torch.float32, device=device)
    kv_rope = kv[:, _NOPE_DIM:].to(torch.bfloat16)

    cache_flat = torch.zeros(
        num_blocks * block_bytes, dtype=torch.uint8, device=device
    )
    exp = torch.randint(-8, 9, (num_tokens, 7), device=device)
    scale = torch.exp2(exp.to(torch.float32))

    for t in range(num_tokens):
        block = t // page_size
        off = t % page_size
        data_base = block * block_bytes + off * _DATA_STRIDE
        nope = kv[t, :_NOPE_DIM]
        quant = torch.zeros_like(nope)
        for g in range(7):
            group = nope[g * 64:(g + 1) * 64]
            fp8 = (group / scale[t, g]).clamp(-448.0, 448.0).to(
                torch.float8_e4m3fn
            )
            cache_flat[data_base + g * 64:data_base + (g + 1) * 64] = (
                fp8.view(torch.uint8)
            )
            quant[g * 64:(g + 1) * 64] = fp8.to(torch.float32) * scale[t, g]
        kv_nope[t] = quant
        cache_flat[data_base + _NOPE_DIM:data_base + _NOPE_DIM + _ROPE_BYTES] = (
            kv_rope[t].view(torch.uint8)
        )
        scale_base = block * block_bytes + scale_off + off * 8
        cache_flat[scale_base:scale_base + 7] = (exp[t] + 127).to(torch.uint8)

    return cache_flat, kv_nope, kv_rope


def _cache_view(
    cache_flat: torch.Tensor, num_blocks: int, page_size: int = _PAGE_SIZE
) -> torch.Tensor:
    return cache_flat.view(num_blocks, page_size, 584)


def _reference_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    indices: torch.Tensor,  # [B, T] int32 physical slots
    topk_length: torch.Tensor,  # [B] int32
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-sink per-source reference attention (out [B,1,H,D], lse [B,1,H])."""
    B, _, H, D = q.shape
    T = indices.shape[1]
    q_f = q.float()
    valid = (indices >= 0) & (
        torch.arange(T, device=q.device)[None, :] < topk_length[:, None]
    )
    safe = indices.clamp(min=0)
    gathered = torch.cat(
        [kv_nope[safe], kv_rope[safe].float()], dim=-1
    )  # [B, T, D]
    scores = torch.einsum("bhd,btd->bht", q_f[:, 0], gathered) * softmax_scale
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, H]
    weights = torch.exp(scores - lse[:, :, None])
    weights = torch.where(valid[:, None, :], weights, 0.0)
    out = torch.einsum("bht,btd->bhd", weights, gathered)
    lonely = lse == float("-inf")
    out[lonely] = 0.0
    return out.to(torch.bfloat16).unsqueeze(1), lse.unsqueeze(1)


def _reference_prefill(
    q: torch.Tensor,  # [T, H, D] bf16
    sinks: Optional[torch.Tensor],
    main_nope: torch.Tensor,
    main_rope: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_nope: Optional[torch.Tensor],
    extra_rope: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_lens: Optional[torch.Tensor],
    softmax_scale: float,
) -> torch.Tensor:
    """LSE-merge the per-source references, then apply the sink."""
    q4 = q.unsqueeze(1)
    merged, lse = _reference_decode(
        q4, main_nope, main_rope, swa_indices, swa_lens, softmax_scale
    )
    if extra_indices is not None:
        assert extra_nope is not None and extra_rope is not None
        assert extra_lens is not None
        ref_extra, lse_extra = _reference_decode(
            q4, extra_nope, extra_rope, extra_indices, extra_lens, softmax_scale
        )
        max_lse = torch.maximum(lse, lse_extra)
        w1 = torch.exp(lse - max_lse)
        w2 = torch.exp(lse_extra - max_lse)
        total = (w1 + w2).clamp(min=1e-20)
        merged = (
            w1.unsqueeze(-1) * merged.float()
            + w2.unsqueeze(-1) * ref_extra.float()
        ) / total.unsqueeze(-1)
        lse = max_lse + torch.log(total)
    if sinks is not None:
        combined = torch.logaddexp(lse, sinks[None, :].expand_as(lse))
        w = torch.exp(lse - combined)
        merged = (merged * w.unsqueeze(-1)).to(torch.bfloat16)
    return merged.squeeze(1)


def _make_prefill_inputs(
    device: torch.device,
    T: int = 32,
    H: int = 16,
    W: int = 128,
    E: int = 128,
) -> tuple[torch.Tensor, ...]:
    """Multi-token prefill inputs: causal SWA windows + growing c4 top-k."""
    main_blocks, extra_blocks = 16, 64
    main_page, extra_page = 64, 2

    torch.manual_seed(3)
    q = (torch.randn(T, H, _D, device=device) * 0.05).to(torch.bfloat16)
    sinks = torch.randn(H, device=device) * 2 - 4

    main_flat, main_nope, main_rope = _make_packed_cache(
        main_blocks, device, seed=4, page_size=main_page
    )
    extra_flat, extra_nope, extra_rope = _make_packed_cache(
        extra_blocks, device, seed=5, page_size=extra_page
    )

    pos = torch.arange(T, device=device)
    swa_lens = torch.minimum(pos + 1, torch.tensor(W, device=device)).to(
        torch.int32
    )
    swa_indices = torch.full((T, W), -1, dtype=torch.int32, device=device)
    for i in range(T):
        swa_indices[i, :swa_lens[i]] = torch.arange(
            max(0, i + 1 - W), i + 1, dtype=torch.int32, device=device
        )

    extra_lens = torch.minimum(
        (pos + 1) // 4, torch.tensor(E, device=device)
    ).to(torch.int32)
    extra_indices = torch.full((T, E), -1, dtype=torch.int32, device=device)
    for i in range(T):
        extra_indices[i, :extra_lens[i]] = torch.arange(
            extra_lens[i], dtype=torch.int32, device=device
        )

    main_cache = _cache_view(main_flat, main_blocks, main_page).unsqueeze(-2)
    extra_cache = _cache_view(extra_flat, extra_blocks, extra_page).unsqueeze(-2)
    return (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
    )


def _make_attn(
    sinks: torch.Tensor, softmax_scale: float
) -> DeepseekV4TritonMLAAttention:
    attn = object.__new__(DeepseekV4TritonMLAAttention)
    attn.scale = softmax_scale
    attn.attn_sink = sinks
    return attn


def test_pack_sparse_rows_csr():
    """CSR packing must copy exactly each row's valid prefix."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    T, W = 5, 8
    torch.manual_seed(0)
    dense = torch.randint(0, 100, (T, W), device=device, dtype=torch.int32)
    lens = torch.tensor([0, 1, 3, 8, 2], device=device, dtype=torch.int32)

    flat, indptr = _pack_sparse_rows(dense, lens)

    expected_indptr = torch.zeros(T + 1, dtype=torch.int32, device=device)
    torch.cumsum(lens, dim=0, out=expected_indptr[1:])
    expected_flat = torch.cat(
        [dense[i, :lens[i]] for i in range(T)]
    ).to(torch.int32)
    assert torch.equal(indptr, expected_indptr)
    assert torch.equal(flat[:expected_flat.shape[0]], expected_flat)


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("with_extra", [False, True])
def test_tiled_prefill_matches_reference(with_sink, with_extra):
    """Fused tiled kernel vs torch reference (LSE merge + Python sink)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
    ) = _make_prefill_inputs(device)
    T, H, _ = q.shape
    softmax_scale = _D ** -0.5
    sink = (
        sinks
        if with_sink
        else torch.full((H,), float("-inf"), device=device)
    )
    attn = _make_attn(sink, softmax_scale)
    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=swa_indices.unsqueeze(1),
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache if with_extra else None,
        extra_indices=extra_indices if with_extra else None,
        extra_lens=extra_lens if with_extra else None,
        out=out,
    )
    ref = _reference_prefill(
        q,
        sinks if with_sink else None,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_nope if with_extra else None,
        extra_rope if with_extra else None,
        extra_indices if with_extra else None,
        extra_lens if with_extra else None,
        softmax_scale,
    )
    assert out.shape == ref.shape
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_tiled_prefill_edge_cases():
    """Empty extra region, sentinels inside the prefix, and T=1."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    softmax_scale = _D ** -0.5

    # Extra cache present but every extra row empty: fused kernel must reduce
    # exactly to the SWA-only result.
    (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        _,
        _,
        _,
        extra_lens,
    ) = _make_prefill_inputs(device)
    T, H, _ = q.shape
    zero_extra_lens = torch.zeros(T, dtype=torch.int32, device=device)
    attn = _make_attn(sinks, softmax_scale)
    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=swa_indices.unsqueeze(1),
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=torch.full(
            (T, 128), -1, dtype=torch.int32, device=device
        ),
        extra_lens=zero_extra_lens,
        out=out,
    )
    ref = _reference_prefill(
        q,
        sinks,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        None,
        None,
        None,
        None,
        softmax_scale,
    )
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    # -1 sentinels sprinkled inside the valid prefix are skipped.
    torch.manual_seed(7)
    T2, H2 = 8, 16
    q2 = (torch.randn(T2, H2, _D, device=device) * 0.05).to(torch.bfloat16)
    sinks2 = torch.randn(H2, device=device) * 2 - 4
    flat2, nope2, rope2 = _make_packed_cache(8, device, seed=9)
    cache2 = _cache_view(flat2, 8).unsqueeze(-2)
    lens2 = torch.randint(2, 17, (T2,), device=device, dtype=torch.int32)
    idx2 = torch.randint(0, 512, (T2, 16), device=device, dtype=torch.int32)
    idx2[:, 2::5] = -1
    attn2 = _make_attn(sinks2, softmax_scale)
    out2 = torch.zeros(T2, H2, _D, dtype=torch.bfloat16, device=device)
    attn2._launch_sparse_mla_prefill(
        q=q2,
        swa_kv_cache=cache2,
        swa_indices=idx2.unsqueeze(1),
        swa_lens=lens2,
        extra_kv_cache=None,
        extra_indices=None,
        extra_lens=None,
        out=out2,
    )
    ref2, ref2_lse = _reference_decode(
        q2.unsqueeze(1), nope2, rope2, idx2, lens2, softmax_scale
    )
    combined = torch.logaddexp(ref2_lse[:, 0], sinks2[None, :])
    w2_ = torch.exp(ref2_lse[:, 0] - combined)
    ref2_sunk = (ref2[:, 0].float() * w2_.unsqueeze(-1)).to(torch.bfloat16)
    torch.testing.assert_close(
        out2.float(),
        ref2_sunk.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    # Single query token with a one-entry SWA row.
    q3 = (torch.randn(1, H2, _D, device=device) * 0.05).to(torch.bfloat16)
    out3 = torch.zeros(1, H2, _D, dtype=torch.bfloat16, device=device)
    attn2._launch_sparse_mla_prefill(
        q=q3,
        swa_kv_cache=cache2,
        swa_indices=torch.tensor([[7]], device=device, dtype=torch.int32).unsqueeze(1),
        swa_lens=torch.ones(1, dtype=torch.int32, device=device),
        extra_kv_cache=None,
        extra_indices=None,
        extra_lens=None,
        out=out3,
    )
    ref3 = _reference_prefill(
        q3,
        sinks2,
        nope2,
        rope2,
        torch.tensor([[7]], device=device, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32, device=device),
        None,
        None,
        None,
        None,
        softmax_scale,
    )
    torch.testing.assert_close(
        out3.float(),
        ref3.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_tiled_prefill_packed_slab_view():
    """The launcher must read through vLLM's packed fp8_ds_mla slab views
    (nonzero storage offset, 576B-padded page stride)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    num_blocks, page_size = 4, 64
    slab_stride = 37440  # ceil(64*584 / 576) * 576
    layer_offset = 37440  # second layer inside the slab
    layer_bytes = num_blocks * page_size * 584

    storage = torch.zeros(
        layer_offset + num_blocks * slab_stride,
        dtype=torch.uint8,
        device=device,
    )
    flat, kv_nope, kv_rope = _make_packed_cache(num_blocks, device, seed=11)
    slab = storage.view(-1, slab_stride)
    slab[1:, :layer_bytes // num_blocks] = flat.view(num_blocks, -1)

    layer = torch.as_strided(
        storage,
        (num_blocks, page_size, 584),
        (slab_stride, 584, 1),
        storage_offset=layer_offset,
    ).unsqueeze(-2)
    assert layer.storage_offset() == layer_offset
    assert layer.stride(0) == slab_stride

    T, H = 4, 16
    softmax_scale = _D ** -0.5
    torch.manual_seed(6)
    q = (torch.randn(T, H, _D, device=device) * 0.05).to(torch.bfloat16)
    sinks = torch.randn(H, device=device) * 2 - 4
    indices = torch.randint(
        0, num_blocks * page_size, (T, 16), device=device, dtype=torch.int32
    )
    lens = torch.randint(1, 17, (T,), device=device, dtype=torch.int32)
    indices[:, 2::5] = -1

    attn = _make_attn(sinks, softmax_scale)
    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=layer,
        swa_indices=indices.unsqueeze(1),
        swa_lens=lens,
        extra_kv_cache=None,
        extra_indices=None,
        extra_lens=None,
        out=out,
    )
    ref, ref_lse = _reference_decode(
        q.unsqueeze(1), kv_nope, kv_rope, indices, lens, softmax_scale
    )
    combined = torch.logaddexp(ref_lse[:, 0], sinks[None, :])
    w = torch.exp(ref_lse[:, 0] - combined)
    ref_sunk = (ref[:, 0].float() * w.unsqueeze(-1)).to(torch.bfloat16)
    torch.testing.assert_close(
        out.float(),
        ref_sunk.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_tiled_prefill_matches_flashinfer():
    """Cross-check against the FlashInfer DSv4 launcher when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    (
        q,
        sinks,
        main_cache,
        _,
        _,
        swa_indices,
        swa_lens,
        extra_cache,
        _,
        _,
        extra_indices,
        extra_lens,
    ) = _make_prefill_inputs(device)
    T, H, _ = q.shape
    softmax_scale = _D ** -0.5
    attn = _make_attn(sinks, softmax_scale)
    out_tiled = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=swa_indices.unsqueeze(1),
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        out=out_tiled,
    )

    from vllm.utils.flashinfer import (
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4,
    )

    out_fi = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    try:
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=q.unsqueeze(1),
            swa_kv_cache=main_cache,
            workspace_buffer=torch.empty(1, dtype=torch.int8, device=device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache,
            out=out_fi,
            bmm1_scale=softmax_scale,
            sinks=sinks,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_indices,
            extra_sparse_topk_lens=extra_lens,
        )
    except RuntimeError as exc:
        pytest.skip(f"FlashInfer sparse MLA DSV4 unavailable: {exc}")
    torch.testing.assert_close(
        out_tiled.float(),
        out_fi.float(),
        atol=5e-2,
        rtol=5e-2,
    )


def test_triton_decode_vllm_tiled_matches_reference():
    """The decode entry now routes through the tiled fused kernel; it must
    match the torch reference on decode-style rows with extra + sink."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
    ) = _make_prefill_inputs(device)
    T, H, _ = q.shape
    softmax_scale = _D ** -0.5

    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    triton_sparse_mla_decode_vllm(
        q=q.unsqueeze(1),
        swa_kv_cache=main_cache,
        swa_indices=swa_indices.unsqueeze(1),
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        attn_sink=sinks,
        softmax_scale=softmax_scale,
        out=out,
    )
    ref = _reference_prefill(
        q,
        sinks,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
        softmax_scale,
    )
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_stale_padding_lens_are_masked_by_consumer():
    """Stale lens on padding rows must not leak into output.

    FULL decode graphs replay a fixed captured row prefix, so rows beyond the
    runtime token count can carry stale lens values if a metadata builder
    forgot its tail zeroing. The consumer side (pack lens clamp + PAD slot
    mask) must keep such rows from producing non-zero output, and the pack
    clamp must bound a stale oversized lens to the row capacity.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    T, W, E = 8, 128, 128
    (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
    ) = _make_prefill_inputs(device, T=T, H=16, W=W, E=E)
    _, H, _ = q.shape
    softmax_scale = _D ** -0.5

    # Simulate a producer that forgot to zero the decode tail: rows past the
    # real token count keep a stale, oversized lens (clamped by the pack to
    # the row capacity) and PAD slots (-1), which the fused kernel masks.
    num_real = 1
    stale_swa_lens = swa_lens.clone()
    stale_swa_lens[num_real:] = 4096
    stale_swa_indices = swa_indices.clone()
    stale_swa_indices[num_real:] = -1
    stale_extra_lens = extra_lens.clone()
    stale_extra_lens[num_real:] = 4096
    stale_extra_indices = extra_indices.clone()
    stale_extra_indices[num_real:] = -1

    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn = _make_attn(sinks, softmax_scale)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=stale_swa_indices.unsqueeze(1),
        swa_lens=stale_swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=stale_extra_indices,
        extra_lens=stale_extra_lens,
        out=out,
    )

    # Padding rows must produce zero output even with stale positive lens.
    assert torch.equal(out[num_real:], torch.zeros_like(out[num_real:])), (
        "stale padding lens leaked into output"
    )

    # The valid row must still match the reference computed from its own lens.
    ref = _reference_prefill(
        q[:num_real],
        sinks,
        main_nope,
        main_rope,
        swa_indices[:num_real],
        swa_lens[:num_real],
        extra_nope,
        extra_rope,
        extra_indices[:num_real],
        extra_lens[:num_real],
        softmax_scale,
    )
    torch.testing.assert_close(
        out[:num_real].float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    # The pack clamp must bound the stale oversized lens to the row capacity.
    flat, indptr = _pack_sparse_rows(
        stale_swa_indices.unsqueeze(1), stale_swa_lens, slot=0
    )
    deltas = indptr[1:] - indptr[:-1]
    assert bool((deltas <= W).all()), "stale lens not clamped to row capacity"


def test_shared_partial_buffer_reuse_does_not_leak_stale_chunks():
    """Reused eager partial buffers must never leak previously written chunks.

    Eager K-split calls now share one grow-only [64, S, H, D] partial buffer
    across every row count T instead of allocating a fresh slab per T. A
    padding row whose metadata still carries a stale positive lens (PAD slots
    only, so the chunk CTA computes no data) must not pick up finite LSE /
    partial output left by an earlier call that used the same buffer slots.
    The consumer-side mask relies on every chunk slot the merge can read
    having been (re)written this step, so this is a two-call regression test:
    the first call dirties the shared slots with real data, the second call
    replays the stale-padding scenario over the same buffer.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    T, W, E = 8, 128, 128
    (
        q,
        sinks,
        main_cache,
        main_nope,
        main_rope,
        swa_indices,
        swa_lens,
        extra_cache,
        extra_nope,
        extra_rope,
        extra_indices,
        extra_lens,
    ) = _make_prefill_inputs(device, T=T, H=16, W=W, E=E)
    _, H, _ = q.shape
    softmax_scale = _D ** -0.5
    attn = _make_attn(sinks, softmax_scale)

    # First call: all rows valid. This writes finite LSE / partial output into
    # the shared buffer slots that the second call will (incorrectly, per its
    # stale metadata) still believe belong to padding rows.
    out_first = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=swa_indices.unsqueeze(1),
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        out=out_first,
    )

    # Second call over the same eager buffer: rows beyond num_real are stale
    # padding (PAD slots + oversized lens) and must produce zero output.
    num_real = 1
    stale_swa_lens = swa_lens.clone()
    stale_swa_lens[num_real:] = 4096
    stale_swa_indices = swa_indices.clone()
    stale_swa_indices[num_real:] = -1
    stale_extra_lens = extra_lens.clone()
    stale_extra_lens[num_real:] = 4096
    stale_extra_indices = extra_indices.clone()
    stale_extra_indices[num_real:] = -1

    out = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=stale_swa_indices.unsqueeze(1),
        swa_lens=stale_swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=stale_extra_indices,
        extra_lens=stale_extra_lens,
        out=out,
    )
    assert torch.equal(out[num_real:], torch.zeros_like(out[num_real:])), (
        "stale chunks from a previous call leaked through the shared partial "
        "buffer"
    )

    ref = _reference_prefill(
        q[:num_real],
        sinks,
        main_nope,
        main_rope,
        swa_indices[:num_real],
        swa_lens[:num_real],
        extra_nope,
        extra_rope,
        extra_indices[:num_real],
        extra_lens[:num_real],
        softmax_scale,
    )
    torch.testing.assert_close(
        out[:num_real].float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_ksplit_eager_scratch_converges_and_is_bounded():
    """Eager K-split scratch must converge instead of stacking per-T slabs.

    Before the cache-key convergence, visiting every decode row count 1..64
    created 64 live [T, S, H, D] partial buffers whose sizes summed to ~1.1
    GiB at S=68/H=8 (TP=8 rank) -- the source of sudden per-card reserved
    growth during serving. The eager cache must instead hold one grow-only
    slab per (S, H) and slice it down per call.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.models.deepseek_v4.nvidia.ops import (
        triton_sparse_mla_prefill as _mod,
    )

    device = torch.device("cuda")
    S, H = 68, 8
    cache = _mod._KSPLIT_PARTIAL_BUFFERS
    cache.clear()
    try:
        reserved_before = torch.cuda.memory_reserved()
        for T in range(1, 65):
            out, lse = _mod._get_ksplit_partial_buffers(T, S, H, device)
            assert out.shape == (T, S, H, _D)
            assert lse.shape == (T, S, H)
        eager_keys = [key for key in cache if key[-1] == 0]
        assert len(eager_keys) == 1, (
            f"expected one shared eager entry, found {len(eager_keys)}"
        )
        out_base, lse_base = cache[eager_keys[0]]
        assert out_base.shape[0] == _mod._KSPLIT_MAX_T, (
            "eager partial buffer should be sized to the K-split capacity"
        )
        held_bytes = (
            out_base.numel() * out_base.element_size()
            + lse_base.numel() * lse_base.element_size()
        )
        grown = torch.cuda.memory_reserved() - reserved_before
        assert grown <= held_bytes + (16 << 20), (
            "eager scratch reserved far more than one capacity-bounded slab: "
            f"grown={grown}, held={held_bytes}"
        )
    finally:
        cache.clear()
        torch.cuda.empty_cache()


def test_csr_eager_scratch_converges_per_slot():
    """Eager CSR flat buffers must converge per slot instead of per capacity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vllm.models.deepseek_v4.nvidia.ops import (
        triton_sparse_mla_prefill as _mod,
    )

    device = torch.device("cuda")
    cache = _mod._CSR_FLAT_BUFFERS
    cache.clear()
    try:
        requested = []
        for T in range(1, 65):
            capacity = T * 2048
            requested.append(capacity)
            buf = _mod._get_csr_flat_buffer(capacity, device, slot=1)
            assert buf.numel() == capacity
        eager_keys = [key for key in cache if key[-1] == 0]
        assert len(eager_keys) == 1, (
            f"expected one shared eager CSR entry, found {len(eager_keys)}"
        )
        assert cache[eager_keys[0]].numel() == max(requested), (
            "eager CSR buffer should hold the largest capacity requested"
        )
    finally:
        cache.clear()
        torch.cuda.empty_cache()
