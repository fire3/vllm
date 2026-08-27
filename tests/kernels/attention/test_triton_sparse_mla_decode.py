# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical tests for the DSv4 Triton sparse-MLA decode kernel.

The kernel is ported from SGLang's SM120 FlashMLA implementation and consumes
the packed ``fp8_ds_mla`` page layout: per 64-token page, 64*576 bytes of
token data (448 FP8 NoPE + 128B BF16 RoPE per token) followed by 64*8 bytes of
UE8M0 scale exponents.
"""

from typing import Optional

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
    flash_mla_sparse_decode_triton,
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
    """Build a packed fp8_ds_mla cache plus the original dequantized KV.

    Returns ``(cache_flat, kv_nope, kv_rope)`` where ``kv_nope``/``kv_rope``
    are the reference dequantized values of shape ``[num_blocks*page_size, D]``.
    """
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
    # Per-token group scales: 7 groups of 64 NoPE dims, random exponents.
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


def _reference_decode(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    kv_nope: torch.Tensor,  # [num_tokens, 448] f32
    kv_rope: torch.Tensor,  # [num_tokens, 64] bf16
    indices: torch.Tensor,  # [B, T] int32 physical slots
    topk_length: torch.Tensor,  # [B] int32
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, _, H, D = q.shape
    T = indices.shape[1]
    q_f = q.float()
    valid = (indices >= 0) & (
        torch.arange(T, device=q.device)[None, :] < topk_length[:, None]
    )
    safe = indices.clamp(min=0)
    gathered = torch.cat(
        [
            kv_nope[safe],
            kv_rope[safe].float(),
        ],
        dim=-1,
    )  # [B, T, D]
    scores = torch.einsum("bhd,btd->bht", q_f[:, 0], gathered) * softmax_scale
    scores = torch.where(valid[:, None, :], scores, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [B, H]
    if attn_sink is not None:
        lse_out = torch.logaddexp(lse, attn_sink[None, :].expand_as(lse))
    else:
        lse_out = lse.clone()
    weights = torch.exp(scores - lse_out[:, :, None])
    weights = torch.where(valid[:, None, :], weights, 0.0)
    out = torch.einsum("bht,btd->bhd", weights, gathered)
    lonely = lse == float("-inf")
    out[lonely] = 0.0
    return out.to(torch.bfloat16).unsqueeze(1), lse.unsqueeze(1)


def _cache_view(
    cache_flat: torch.Tensor, num_blocks: int, page_size: int = _PAGE_SIZE
) -> torch.Tensor:
    return cache_flat.view(num_blocks, page_size, 584)


@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("with_extra", [False, True])
def test_triton_sparse_mla_decode_matches_reference(with_sink, with_extra):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    torch.manual_seed(0)

    num_blocks = 8
    B = 4
    H = 4
    T = 32
    softmax_scale = _D ** -0.5

    cache_flat, kv_nope, kv_rope = _make_packed_cache(num_blocks, device, seed=1)
    k_cache = _cache_view(cache_flat, num_blocks)

    q = (torch.randn(B, 1, H, _D, device=device) * 0.05).to(torch.bfloat16)
    indices = torch.randint(0, num_blocks * _PAGE_SIZE, (B, T), device=device)
    topk_length = torch.randint(1, T + 1, (B,), device=device, dtype=torch.int32)
    # Sprinkle invalid slots to exercise masking.
    indices[:, 3::7] = -1
    attn_sink = (
        torch.randn(H, device=device) * 2 - 4 if with_sink else None
    )

    out, _ = flash_mla_sparse_decode_triton(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
        head_dim_v=_D,
        softmax_scale=softmax_scale,
    )
    ref, ref_lse = _reference_decode(
        q=q,
        kv_nope=kv_nope,
        kv_rope=kv_rope,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
    )
    assert out.shape == ref.shape == (B, 1, H, _D)
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=2e-2,
        rtol=2e-2,
    )

    if with_extra:
        # Extra (compressed) cache with a different page size (c128: 2/page).
        extra_blocks = 16
        extra_page_size = 2
        extra_flat, e_nope, e_rope = _make_packed_cache(
            extra_blocks, device, seed=7, page_size=extra_page_size
        )
        extra_num_tokens = extra_blocks * extra_page_size
        extra_cache = _cache_view(extra_flat, extra_blocks, extra_page_size)
        extra_indices = torch.randint(
            0, extra_num_tokens, (B, 16), device=device
        )
        extra_lens = torch.randint(1, 17, (B,), device=device, dtype=torch.int32)

        out_merged, _ = flash_mla_sparse_decode_triton(
            q=q,
            k_cache=k_cache,
            indices=indices,
            topk_length=topk_length,
            attn_sink=attn_sink,
            head_dim_v=_D,
            softmax_scale=softmax_scale,
            extra_k_cache=extra_cache,
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
        )
        ref_main, ref_main_lse = _reference_decode(
            q=q,
            kv_nope=kv_nope,
            kv_rope=kv_rope,
            indices=indices,
            topk_length=topk_length,
            softmax_scale=softmax_scale,
            attn_sink=None,
        )
        ref_extra, ref_extra_lse = _reference_decode(
            q=q,
            kv_nope=e_nope,
            kv_rope=e_rope,
            indices=extra_indices,
            topk_length=extra_lens,
            softmax_scale=softmax_scale,
            attn_sink=None,
        )
        # LSE-weighted merge, then sink, mirrors the kernel wrapper.
        max_lse = torch.maximum(ref_main_lse, ref_extra_lse)
        w1 = torch.exp(ref_main_lse - max_lse)
        w2 = torch.exp(ref_extra_lse - max_lse)
        total = (w1 + w2).clamp(min=1e-20)
        merged = (
            w1.unsqueeze(-1) * ref_main.float()
            + w2.unsqueeze(-1) * ref_extra.float()
        ) / total.unsqueeze(-1)
        merged = merged.to(torch.bfloat16)
        if attn_sink is not None:
            lse_merged = max_lse + torch.log(total)
            combined = torch.logaddexp(lse_merged, attn_sink[None, :])
            w = torch.exp(lse_merged - combined)
            merged = (merged.float() * w.unsqueeze(-1)).to(torch.bfloat16)
        assert out_merged.shape == ref_main.shape
        torch.testing.assert_close(
            out_merged.float(),
            merged.float(),
            atol=3e-2,
            rtol=3e-2,
        )


def test_triton_sparse_mla_decode_empty_topk():
    """Zero valid tokens should produce a zero output without crashing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    cache_flat, _, _ = _make_packed_cache(2, device, seed=2)
    k_cache = _cache_view(cache_flat, 2)
    q = torch.randn(1, 1, 2, _D, device=device).to(torch.bfloat16)
    indices = torch.full((1, 16), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(1, dtype=torch.int32, device=device)
    out, _ = flash_mla_sparse_decode_triton(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=lens,
        attn_sink=None,
        head_dim_v=_D,
        softmax_scale=_D ** -0.5,
    )
    assert torch.all(out == 0)


def test_triton_sparse_mla_decode_packed_slab_view():
    """The kernel must read through vLLM's packed fp8_ds_mla slab views.

    Attention layers alias one shared block slab: each layer is a strided view
    with nonzero storage offset and a 576B-padded page stride (37440B for a
    64-token page). Regression test for the ``as_strided`` flatten OOB error
    seen during engine warmup with TRITON_MLA_SPARSE_DSV4.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    num_blocks = 4
    page_size = 64
    slab_stride = 37440  # ceil(64*584 / 576) * 576
    layer_offset = 37440  # second layer inside the slab
    layer_bytes = num_blocks * page_size * 584

    storage = torch.zeros(
        layer_offset + num_blocks * slab_stride,
        dtype=torch.uint8,
        device=device,
    )
    flat, kv_nope, kv_rope = _make_packed_cache(num_blocks, device, seed=11)

    # Place the layer's page data at its slab offset; each page uses only the
    # unpadded 37376 bytes of its 37440-byte slot.
    slab = storage.view(-1, slab_stride)
    slab[1:, :layer_bytes // num_blocks] = flat.view(num_blocks, -1)

    layer = torch.as_strided(
        storage,
        (num_blocks, page_size, 584),
        (slab_stride, 584, 1),
        storage_offset=layer_offset,
    )
    assert layer.storage_offset() == layer_offset
    assert layer.stride(0) == slab_stride

    B, H, T = 3, 4, 32
    softmax_scale = _D ** -0.5
    torch.manual_seed(6)
    q = (torch.randn(B, 1, H, _D, device=device) * 0.05).to(torch.bfloat16)
    indices = torch.randint(0, num_blocks * page_size, (B, T), device=device)
    topk_length = torch.randint(1, T + 1, (B,), device=device, dtype=torch.int32)
    indices[:, 2::5] = -1

    out, _ = flash_mla_sparse_decode_triton(
        q=q,
        k_cache=layer,
        indices=indices,
        topk_length=topk_length,
        attn_sink=None,
        head_dim_v=_D,
        softmax_scale=softmax_scale,
    )
    ref, _ = _reference_decode(
        q=q,
        kv_nope=kv_nope,
        kv_rope=kv_rope,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        attn_sink=None,
    )
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=2e-2,
        rtol=2e-2,
    )


def _make_prefill_inputs(
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build prefill-style inputs: multi-token rows with per-token causal
    SWA windows and growing compressed top-k counts.

    Mirrors the DSv4 ``_forward_prefill`` data flow: each prefill query token
    becomes one decode row with its own SWA + compressed indices.
    """
    T, H, W, E = 32, 16, 128, 128
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

    # Compressed top-k: ~(pos+1)/4 valid entries (c4 semantics), capped at E.
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


def test_triton_prefill_launcher_matches_reference_and_flashinfer():
    """Phase-1 prefill: the Triton launcher hook must match a torch reference
    and (when FlashInfer is available) the FlashInfer DSv4 prefill launcher on
    identical multi-token inputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda")
    softmax_scale = _D ** -0.5

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

    # Torch reference: main + extra rows merged by LSE, then sink applied.
    ref_main, ref_main_lse = _reference_decode(
        q=q.unsqueeze(1),
        kv_nope=main_nope,
        kv_rope=main_rope,
        indices=swa_indices,
        topk_length=swa_lens,
        softmax_scale=softmax_scale,
        attn_sink=None,
    )
    ref_extra, ref_extra_lse = _reference_decode(
        q=q.unsqueeze(1),
        kv_nope=extra_nope,
        kv_rope=extra_rope,
        indices=extra_indices,
        topk_length=extra_lens,
        softmax_scale=softmax_scale,
        attn_sink=None,
    )
    max_lse = torch.maximum(ref_main_lse, ref_extra_lse)
    w1 = torch.exp(ref_main_lse - max_lse)
    w2 = torch.exp(ref_extra_lse - max_lse)
    total = (w1 + w2).clamp(min=1e-20)
    merged = (
        w1.unsqueeze(-1) * ref_main.float() + w2.unsqueeze(-1) * ref_extra.float()
    ) / total.unsqueeze(-1)
    lse_merged = max_lse + torch.log(total)
    combined = torch.logaddexp(lse_merged, sinks[None, :].expand_as(lse_merged))
    w = torch.exp(lse_merged - combined)
    ref = (merged * w.unsqueeze(-1)).to(torch.bfloat16).squeeze(1)

    # Triton launcher hook (no FlashInfer involved).
    from vllm.models.deepseek_v4.nvidia.triton_sparse import (
        DeepseekV4TritonMLAAttention,
    )

    attn = object.__new__(DeepseekV4TritonMLAAttention)
    attn.scale = softmax_scale
    attn.attn_sink = sinks
    out_t = torch.zeros(T, H, _D, dtype=torch.bfloat16, device=device)
    attn._launch_sparse_mla_prefill(
        q=q,
        swa_kv_cache=main_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_kv_cache=extra_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        out=out_t,
    )
    torch.testing.assert_close(
        out_t.float(),
        ref.float(),
        atol=3e-2,
        rtol=3e-2,
    )

    # Cross-check against the FlashInfer DSv4 launcher on the same inputs.
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
        # FlashInfer unavailable (or its fallback fired): the Triton-vs-reference
        # check above is the portable part of this test.
        pytest.skip(f"FlashInfer sparse MLA DSV4 unavailable: {exc}")
    torch.testing.assert_close(
        out_t.float(),
        out_fi.float(),
        atol=5e-2,
        rtol=5e-2,
    )
