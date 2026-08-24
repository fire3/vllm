# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse-MLA backend for DeepSeek V4 (Flash).

This backend replaces FlashInfer's DSv4 sparse-MLA *decode* kernel with the
Triton kernel ported from SGLang's SM120 FlashMLA implementation
(``flash_mla_sparse_decode_triton``). It consumes the same packed
``fp8_ds_mla`` page layout and the same sparse-index metadata as
``FLASHINFER_MLA_SPARSE_DSV4``, so it is a drop-in swap for the decode path:

    --attention-backend TRITON_MLA_SPARSE_DSV4

The DSv4 prefill launcher is not covered by the ported operator (it is a
decode-only kernel); prefill continues to use FlashInfer when prefill tokens
are present, so FlashInfer is still required for mixed-batch steps.
"""

from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.envs import VLLM_TRITON_SPARSE_MLA_PREFILL_DECODE_WRAPPER
from vllm.platforms.interface import DeviceCapability
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
    DeepseekV4FlashInferSM120Attention,
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_vllm,
)
from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
    triton_sparse_mla_prefill_vllm,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLAMetadata


class DeepseekV4TritonMLASparseBackend(DeepseekV4FlashInferMLASparseBackend):
    """DeepSeek-V4 sparse-MLA backend with a Triton decode kernel."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8",
        "fp8_e4m3",
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 12 or (
            capability.major == 8 and capability.minor == 9
        )

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # The Triton kernel reads the packed fp8_ds_mla layout, so unlike the
        # FlashInfer backend there is no plain bf16/per-tensor-fp8 KV path.
        if dtype != torch.bfloat16:
            return "TRITON_MLA_SPARSE_DSV4 requires bf16 query dtype"
        if kv_cache_dtype not in (None, "auto", "fp8", "fp8_e4m3", "fp8_ds_mla"):
            return "kv_cache_dtype not supported"
        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            index_topk = getattr(
                vllm_config.model_config.hf_text_config, "index_topk", None
            )
            if index_topk is None:
                return (
                    "TRITON_MLA_SPARSE_DSV4 requires a model with index_topk "
                    "config"
                )
        return None


class DeepseekV4TritonMLAAttention(DeepseekV4FlashInferSM120Attention):
    """DeepSeek V4 sparse MLA attention fully backed by the Triton kernel.

    Both decode and prefill run the SGLang-ported Triton kernel: prefill
    tokens are expanded to decode-style rows (one query token per row) with
    per-token SWA/top-k indices, mirroring SGLang's SM120 extend path. No
    FlashInfer kernel is required by this backend.
    """

    backend_cls = DeepseekV4TritonMLASparseBackend
    _require_flashinfer_capability: ClassVar[bool] = False

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                        output_buffers=self._global_topk_output_buffers(
                            self.topk_indices_buffer[:num_decode_tokens]
                        ),
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
        swa_cache = self._as_sparse_cache(self.swa_cache_layer.kv_cache)
        extra_cache = self._as_sparse_cache(kv_cache) if kv_cache is not None else None
        if extra_cache is not None and extra_sparse_indices is None:
            raise RuntimeError(
                "Compressed sparse MLA decode requires compressed sparse indices."
            )
        triton_sparse_mla_decode_vllm(
            q=q.unsqueeze(1),
            swa_kv_cache=swa_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            extra_kv_cache=extra_cache,
            extra_indices=extra_sparse_indices,
            extra_lens=extra_sparse_lengths,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            out=output,
        )

    def _launch_sparse_mla_prefill(
        self,
        q: torch.Tensor,  # [T, H, D] bf16, query padded to backend head count
        swa_kv_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        extra_kv_cache: torch.Tensor | None,
        extra_indices: torch.Tensor | None,
        extra_lens: torch.Tensor | None,
        out: torch.Tensor,  # [T, H, D], written in place
    ) -> None:
        if VLLM_TRITON_SPARSE_MLA_PREFILL_DECODE_WRAPPER:
            # Phase-1 path: decode-style rows, two kernel passes + LSE merge.
            triton_sparse_mla_decode_vllm(
                q=q.unsqueeze(1),
                swa_kv_cache=swa_kv_cache,
                swa_indices=swa_indices,
                swa_lens=swa_lens,
                extra_kv_cache=extra_kv_cache,
                extra_indices=extra_indices,
                extra_lens=extra_lens,
                attn_sink=self.attn_sink,
                softmax_scale=self.scale,
                out=out,
            )
            return
        triton_sparse_mla_prefill_vllm(
            q=q,
            swa_kv_cache=swa_kv_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            extra_kv_cache=extra_kv_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            out=out,
        )

    def _reserve_empty_forward_workspace(self) -> None:
        # The Triton path has no FlashInfer workspace to reserve.
        return None
