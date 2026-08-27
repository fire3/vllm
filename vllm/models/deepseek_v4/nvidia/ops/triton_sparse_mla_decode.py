# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA decode entry point for DeepSeek V4 (Flash).

Decode routes through the tiled dual-source fused kernel shared with prefill
(``triton_sparse_mla_prefill_vllm``): each decode row is a one-token query
with the same packed ``fp8_ds_mla`` page layout and sparse-index metadata as
prefill, so a single fused kernel launch handles both the SWA and compressed
(c4/c128) sources with one shared online-softmax state.
"""

from typing import Optional

import torch

from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
    triton_sparse_mla_prefill_vllm,
)


def triton_sparse_mla_decode_vllm(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_kv_cache: torch.Tensor,  # [num_pages, page_size, 1, bpt] uint8
    swa_indices: torch.Tensor,  # [B, window_size] int32 physical slots
    swa_lens: Optional[torch.Tensor],  # [B] int32
    extra_kv_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_lens: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],  # [H] f32
    softmax_scale: float,
    out: torch.Tensor,  # [B, H, D] bf16, written in place
) -> None:
    """vLLM entry point for the DSv4 Triton sparse-MLA decode path.

    Runs the tiled dual-source fused kernel shared with prefill
    (head-blocked, per-64-dim-group ``tl.dot``, single launch with a shared
    online-softmax state across the SWA and compressed sources).
    """
    triton_sparse_mla_prefill_vllm(
        q=q.squeeze(1),
        swa_kv_cache=swa_kv_cache,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        extra_kv_cache=extra_kv_cache,
        extra_indices=extra_indices,
        extra_lens=extra_lens,
        attn_sink=attn_sink,
        softmax_scale=softmax_scale,
        out=out,
    )
