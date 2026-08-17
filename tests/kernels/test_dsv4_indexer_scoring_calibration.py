# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibration tests for the SM89 Triton indexer scoring fallback.

The SM89 sparse-indexer fallback (``fp8_mqa_logits_triton`` /
``fp8_paged_mqa_logits_triton``) must stay numerically aligned with the
reference implementations used on other SMs:

* DeepGEMM ``fp8_fp4_mqa_logits`` / ``fp8_fp4_paged_mqa_logits``
  (SM90/SM100/SM120): ``logit = (sum_h relu(q.k_h) * w_h) * s_kv`` — the
  per-token KV scale is applied once, after the weighted head sum.
* CuteDSL ``fused_indexer_q_rope_quant_fp8_cutedsl`` (SM100/SM120): weight
  fold order ``(w * float(softmax_scale * head_scale)) * q_scale``.

The two scale-application orders are mathematically equivalent and, with the
ue8m0 power-of-two scales used by the DSv4 indexer cache, bit-identical;
these tests pin the SM89 kernels to the DeepGEMM/CuteDSL op order and verify
the residual error is fp8-MMA accumulation-order noise only.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_fp8_mqa_logits import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="The SM89 Triton indexer fallback requires CUDA",
)


def _quant_fp8(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(row/group) ue8m0 FP8 quantizer matching the indexer kernels."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


def _ref_deepgemm_order(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_scale: torch.Tensor,
    cu_ks: torch.Tensor,
    cu_ke: torch.Tensor,
) -> torch.Tensor:
    """(sum_h relu(q.k_h) * w_h) * s_kv, -inf outside [start, end)."""
    seq_len_kv = k_fp8.shape[0]
    dot = torch.einsum("mhd,nd->mhn", q_fp8.float(), k_fp8.float())
    logits = torch.einsum("mh,mhn->mn", weights, torch.relu(dot))
    logits = logits * k_scale[None, :]
    cols = torch.arange(seq_len_kv, device=q_fp8.device)
    mask = (cols[None, :] >= cu_ks[:, None]) & (cols[None, :] < cu_ke[:, None])
    return logits.masked_fill(~mask, float("-inf"))


def _ref_deepgemm_paged(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Paged reference in DeepGEMM order for ``next_n == 1`` batches."""
    rows = q_fp8.shape[0]
    max_model_len = k_fp8.shape[0]
    logits = torch.full(
        (rows, max_model_len), float("-inf"), dtype=torch.float32, device=q_fp8.device
    )
    for r in range(rows):
        length = int(context_lens[r].item())
        num_blocks = (length + block_size - 1) // block_size
        pages = block_tables[r, :num_blocks]
        k_flat = k_fp8.view(-1, block_size, k_fp8.shape[-1])[pages].reshape(
            -1, k_fp8.shape[-1]
        )
        s_flat = k_scale.view(-1, block_size)[pages].reshape(-1)
        dots = torch.einsum("hd,nd->hn", q_fp8[r].float(), k_flat.float())
        scores = torch.einsum("h,hn->n", weights[r], torch.relu(dots))
        logits[r, :length] = scores[:length] * s_flat[:length]
    return logits


def test_fp8_mqa_logits_triton_matches_deepgemm_order() -> None:
    set_random_seed(0)
    torch.set_default_device("cuda:0")
    H, D, M, N = 64, 128, 16, 2048
    softmax_scale = D**-0.5
    head_scale = torch.randn(H, dtype=torch.float32)

    q_bf16 = torch.randn(M, H, D, dtype=torch.bfloat16) / 10
    k_bf16 = torch.randn(N, D, dtype=torch.bfloat16) / 10
    q_fp8, q_scale = _quant_fp8(q_bf16.float(), dim=-1)
    k_fp8, k_scale = _quant_fp8(k_bf16.float(), dim=-1)
    k_scale = k_scale.squeeze(-1)
    weights = (q_scale.squeeze(-1) * softmax_scale * head_scale).float()

    # Non-zero row starts + mixed window widths exercise the same shape the
    # multi-request prefill chunks produce (gathered K covers all requests).
    cu_ks = torch.zeros(M, dtype=torch.int32)
    cu_ke = torch.full((M,), N, dtype=torch.int32)
    for m in range(1, M):
        start = (m * 137) % N
        end = min(start + 256 + m * 64, N)
        cu_ks[m] = start
        cu_ke[m] = end

    logits = fp8_mqa_logits_triton(q_fp8, k_fp8, k_scale, weights, cu_ks, cu_ke)
    ref = _ref_deepgemm_order(q_fp8, k_fp8, weights, k_scale, cu_ks, cu_ke)
    cols = torch.arange(N, device="cuda")
    in_window = (cols[None, :] >= cu_ks[:, None]) & (cols[None, :] < cu_ke[:, None])
    abs_err = (logits - ref).abs()[in_window]
    # fp8 MMA accumulation-order noise only (fp8 products are exact in fp32).
    assert abs_err.max().item() < 1e-3, f"max_abs={abs_err.max().item():.3e}"


def test_fp8_paged_mqa_logits_triton_matches_deepgemm_order() -> None:
    set_random_seed(1)
    torch.set_default_device("cuda:0")
    H, D, rows, N = 64, 128, 4, 4096
    block_size = 64
    num_blocks = N // block_size
    softmax_scale = D**-0.5
    head_scale = torch.randn(H, dtype=torch.float32)

    q_bf16 = torch.randn(rows, H, D, dtype=torch.bfloat16) / 10
    k_bf16 = torch.randn(N, D, dtype=torch.bfloat16) / 10
    q_fp8, q_scale = _quant_fp8(q_bf16.float(), dim=-1)
    k_fp8, k_scale = _quant_fp8(k_bf16.float(), dim=-1)
    k_scale = k_scale.squeeze(-1)
    weights = (q_scale.squeeze(-1) * softmax_scale * head_scale).float()

    # Flat packed cache layout: [bs*D] FP8 data bytes then [bs*4] fp32 scale
    # bytes per block (matches indexer_k_quant_and_cache / Triton compressor).
    kv_cache = torch.zeros(num_blocks, block_size, 1, D + 4, dtype=torch.uint8)
    flat = kv_cache.view(num_blocks, -1)
    flat[:, : block_size * D].view(num_blocks, block_size, D).copy_(
        k_fp8.view(torch.uint8).view(num_blocks, block_size, D)
    )
    flat[:, block_size * D : block_size * (D + 4)].view(
        num_blocks, block_size, 4
    ).copy_(
        k_scale.view(num_blocks, block_size, 1, 1)
        .contiguous()
        .view(torch.uint8)
        .view(num_blocks, block_size, 4)
    )
    block_tables = torch.arange(num_blocks, dtype=torch.int32).view(1, -1).repeat(rows, 1)
    context_lens = torch.tensor(
        [768, 2048, 3072, 4096], dtype=torch.int32, device="cuda"
    )

    logits = fp8_paged_mqa_logits_triton(
        q_fp8.unsqueeze(1),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len=N,
    )
    ref = _ref_deepgemm_paged(
        q_fp8, k_fp8, k_scale, weights, context_lens, block_tables, block_size
    )
    for r in range(rows):
        length = int(context_lens[r].item())
        abs_err = (logits[r, :length] - ref[r, :length]).abs()
        assert abs_err.max().item() < 1e-3, f"row {r} max_abs={abs_err.max().item():.3e}"


def test_fused_indexer_q_weight_fold_matches_cutedsl_order() -> None:
    """Triton weight fold must be bit-identical to the CuteDSL reference:
    ``weights_out = (w * float(softmax_scale * head_scale)) * q_scale``."""
    from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
        fused_indexer_q_rope_quant,
    )

    set_random_seed(2)
    torch.set_default_device("cuda:0")
    H, D, M = 8, 128, 4
    rope_dim = 64
    softmax_scale = D**-0.5
    head_scale = H**-0.5
    max_pos = 65536
    positions = torch.tensor([0, 5, 100, max_pos - 1], dtype=torch.int64)
    q = torch.randn(M, H, D, dtype=torch.bfloat16) / 5
    cos_sin = torch.randn(max_pos, rope_dim, dtype=torch.bfloat16) * 0.1
    w = torch.randn(M, H, dtype=torch.float32)

    _, w_out = fused_indexer_q_rope_quant(
        positions, q, cos_sin, w, softmax_scale, head_scale
    )

    # Rebuild q_scale per (token, head) exactly like the kernel: GPT-J
    # interleaved RoPE on the trailing rope_dim, bf16 roundtrip before amax,
    # then scale = 2^ceil(log2(max(amax, 1e-4) / 448)).
    nope_dim = D - rope_dim
    half = rope_dim // 2
    qf = q.float()
    x_nope = qf[..., :nope_dim]
    rot = qf[..., nope_dim:].reshape(M, H, half, 2)
    x_even = rot[..., 0]
    x_odd = rot[..., 1]
    pos = positions[:, None, None].expand(M, H, half)
    cos = cos_sin[pos, torch.arange(half, device="cuda")].to(torch.float32)
    sin = cos_sin[pos, torch.arange(half, device="cuda") + half].to(torch.float32)
    r_even = (x_even * cos - x_odd * sin).to(torch.bfloat16).to(torch.float32)
    r_odd = (x_odd * cos + x_even * sin).to(torch.bfloat16).to(torch.float32)
    amax = torch.maximum(
        torch.maximum(x_nope.abs().amax(dim=-1), r_even.abs().amax(dim=-1)),
        r_odd.abs().amax(dim=-1),
    )
    q_scale = torch.exp2(
        torch.ceil(torch.log2(torch.maximum(amax, 1e-4) / 448.0))
    )

    expected = (w * float(softmax_scale * head_scale)) * q_scale
    assert torch.equal(w_out.view(torch.int32), expected.view(torch.int32))
