# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA backend for SM89 (Ada, e.g. L40S) NoPE models.

GLM-5.3-Flash's DSA layers (pure indexer top-k attention, no SWA) run on
FlashInfer FA3 on SM90 and FlashInfer TRTLLM-gen on SM120, but there is no
FlashInfer/FlashMLA kernel for SM89. This backend implements the same
semantics with a Triton kernel (``triton_topk_mla``):

* page_size=1: the per-token top-k slot indices ARE the page table, so each
  query token becomes one varlen row whose ``kv_indices`` slice is its
  top-k rows and whose ``kv_len`` is its valid count. Causality is already
  encoded by the indexer's selection, so ``causal=False``.
* KV cache is plain contiguous ``[num_blocks, block_size, 512]`` bf16 or fp8
  e4m3 (per-tensor ``k_scale``), identical to the SM90 FlashInfer layout.
* CUDA graphs: all per-step content (top-k slots and lens) is written into
  fixed builder-owned device buffers by kernels inside the captured forward,
  so replay reads refreshed addresses; no host-side planning is needed.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.models.glm5next.nvidia.ops.triton_topk_mla import (
    triton_topk_mla_forward,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import KVCacheLayout

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import AttentionSpec

_FP8_KV_DTYPES = ("fp8", "fp8_e4m3")


class TritonMLASparseSM89Backend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_SM89"

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseSM89Builder"]:
        return TritonMLASparseSM89Builder

    @staticmethod
    def get_impl_cls() -> type[MLAAttentionImpl]:
        return TritonMLASparseSM89Impl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 512 = ckv 512 + kpe 0 (NoPE); the Triton kernel is NoPE-only.
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8 and capability.minor == 9

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
        if not use_sparse:
            return "TRITON_MLA_SPARSE_SM89 requires sparse MLA"
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        if vllm_config.model_config is not None:
            hf = vllm_config.model_config.hf_text_config
            if hf.kv_lora_rank != 512:
                return "TRITON_MLA_SPARSE_SM89 requires kv_lora_rank=512"
            if hf.qk_rope_head_dim != 0:
                return (
                    "TRITON_MLA_SPARSE_SM89 requires qk_rope_head_dim=0 "
                    "(NoPE); rope MLA is not supported by the Triton kernel"
                )
            if hf.index_topk is None:
                return "TRITON_MLA_SPARSE_SM89 requires index_topk"
        return None

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)


class _SM89State:
    """Builder-owned capture-stable device buffers for the Triton kernel.

    One instance serves every MLA layer in an attention group: the buffers
    are sized to ``max_num_batched_tokens`` and refreshed in place every
    step, so their addresses never move and captured CUDA graphs read the
    current contents on replay.
    """

    def __init__(
        self,
        device: torch.device,
        max_tokens: int,
        topk_width: int,
    ) -> None:
        self.kv_indices = torch.zeros(
            max_tokens * topk_width, dtype=torch.int32, device=device
        )
        self.kv_lens = torch.zeros(max_tokens, dtype=torch.int32, device=device)


@dataclass
class TritonMLASparseSM89Metadata(FlashInferMLASparseMetadata):
    state: _SM89State | None = None


class TritonMLASparseSM89Builder(FlashInferMLASparseMetadataBuilder):
    """Reuse the common sparse metadata (req ids, topk buffer access)."""

    metadata_cls = TritonMLASparseSM89Metadata

    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        attention_layer = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ]
        impl = attention_layer.impl
        if not isinstance(impl, TritonMLASparseSM89Impl):
            raise TypeError(
                "TritonMLASparseSM89Builder requires a Triton SM89 "
                f"implementation, got {type(impl).__name__}."
            )
        topk_indices_buffer = impl.topk_indices_buffer
        assert topk_indices_buffer is not None
        self.state = _SM89State(
            device,
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_indices_buffer.shape[1],
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonMLASparseSM89Metadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        assert isinstance(metadata, TritonMLASparseSM89Metadata)
        metadata.state = self.state
        return metadata


class TritonMLASparseSM89Impl(SparseMLACommonImpl[TritonMLASparseSM89Metadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: Any | None = None,
        **mla_args: Any,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "TritonMLASparseSM89Impl does not support alibi, sliding "
                "window, or logits soft cap."
            )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        assert self.qk_rope_head_dim == 0, (
            "TritonMLASparseSM89Impl is NoPE-only "
            f"(qk_rope_head_dim={self.qk_rope_head_dim})"
        )
        assert self.head_size == self.kv_lora_rank, (
            f"head_size ({self.head_size}) must equal kv_lora_rank "
            f"({self.kv_lora_rank}) for NoPE MLA"
        )
        assert self.topk_indices_buffer is not None
        self.supports_quant_query_input = False
        self.use_fp8_kv_cache = self.kv_cache_dtype in _FP8_KV_DTYPES

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonMLASparseSM89Metadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(q, tuple):
            raise NotImplementedError(
                "TritonMLASparseSM89Impl expects split (q_nope, q_rope)."
            )
        q_nope, q_rope = q
        num_tokens = q_rope.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_tokens]
        # return_valid_counts=True keeps the compacted-prefix layout: valid
        # entries at [0, valid_count), -1 past it — exactly the rows the
        # kernel iterates (bounded by kv_lens).
        topk_slots, valid_counts = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_tokens],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )
        state = attn_metadata.state
        assert state is not None
        width = topk_slots.shape[1]
        # Refresh top-k rows in graph; clamp masked tails to a valid slot and
        # zero the lens buffer so padded rows never read.
        state.kv_indices[: num_tokens * width].copy_(
            topk_slots.reshape(-1).clamp_(min=0).to(torch.int32)
        )
        state.kv_lens.zero_()
        state.kv_lens[:num_tokens].copy_(valid_counts.to(torch.int32))

        # Plain [num_blocks, block_size, head_size] cache -> flat row view
        # (rows beyond the block stride are never indexed by the convert
        # kernel, matching the SM90 backend's flat row addressing).
        rows, _ = flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
        if self.use_fp8_kv_cache:
            rows = rows.view(torch.float8_e4m3fn)

        out = torch.empty(
            (num_tokens, self.num_heads, self.head_size),
            dtype=torch.bfloat16,
            device=q_nope.device,
        )
        kv_indices = state.kv_indices[: num_tokens * width].view(num_tokens, width)
        triton_topk_mla_forward(
            q_nope,
            rows,
            kv_indices,
            state.kv_lens[:num_tokens],
            out,
            sm_scale=self.scale,
            kv_scale=float(layer._k_scale_float or 1.0)
            if self.use_fp8_kv_cache
            else 1.0,
            num_heads=self.num_heads,
        )
        return out, None
