# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.utils.torch_utils import set_random_seed

HEAD_DIM = 128
ROPE_DIM = 64
COMPRESS_RATIO = 4
STATE_WIDTH = 2 * HEAD_DIM
STATE_BLOCK_SIZE = 4
KV_BLOCK_SIZE = 64
QUANT_BLOCK = HEAD_DIM
TOKEN_STRIDE = HEAD_DIM
SCALE_DIM = torch.float32.itemsize
ALLOCATED_CACHE_ROW_BYTES = HEAD_DIM + torch.float32.itemsize
FP8_MAX = 448.0
RMS_EPS = 1e-6
_SENTINEL = 0xA5
PAGE_ALIGNMENT_BYTES = 576


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _load_cutedsl_candidate(*, use_mxfp4: bool) -> Callable[..., None] | None:
    try:
        if use_mxfp4:
            from vllm.models.deepseek_v4.nvidia.ops.indexer_compress_cutedsl import (
                fused_kv_compress_norm_rope_insert_indexer_mxfp4_cutedsl,
            )

            return fused_kv_compress_norm_rope_insert_indexer_mxfp4_cutedsl
        from vllm.models.deepseek_v4.nvidia.ops.indexer_compress_cutedsl import (
            fused_kv_compress_norm_rope_insert_indexer_fp8_cutedsl,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    return fused_kv_compress_norm_rope_insert_indexer_fp8_cutedsl


def _launch_triton(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    use_mxfp4: bool,
    quant_block: int,
    token_stride: int,
) -> None:
    kernel = (
        _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
        if use_mxfp4
        else _fused_kv_compress_norm_rope_insert_indexer_attn
    )
    kernel[(slot_mapping.shape[0],)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        STATE_BLOCK_SIZE,
        rms_norm_weight,
        RMS_EPS,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        KV_BLOCK_SIZE,
        HEAD_SIZE=HEAD_DIM,
        TRITON_BLOCK_SIZE=HEAD_DIM,
        STATE_WIDTH=STATE_WIDTH,
        COMPRESS_RATIO=COMPRESS_RATIO,
        OVERLAP=True,
        ROPE_HEAD_DIM=ROPE_DIM,
        FP8_MAX=FP8_MAX,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=SCALE_DIM,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=1,
        launch_pdl=False,
    )


def _build_b04_case(
    args: Mapping[str, Any],
    *,
    include_state_store: bool,
    use_mxfp4: bool = False,
    candidate_min_tokens: int = 8192,
) -> ChainCase:
    num_tokens = int(args.get("num_tokens", 256))
    position_offset = int(args.get("position_offset", 0))
    pad_every = int(args.get("pad_every", 0))
    seed = int(args.get("seed", 0))
    if num_tokens <= 0:
        raise ValueError("B04 num_tokens must be positive")
    if position_offset < 0:
        raise ValueError("B04 position_offset must be non-negative")
    if pad_every < 0:
        raise ValueError("B04 pad_every must be non-negative")
    quant_block = 32 if use_mxfp4 else QUANT_BLOCK
    token_stride = HEAD_DIM // 2 if use_mxfp4 else TOKEN_STRIDE
    cache_format = "mxfp4" if use_mxfp4 else "fp8"

    set_random_seed(seed)
    device = torch.device("cuda")
    positions = torch.arange(
        position_offset,
        position_offset + num_tokens,
        dtype=torch.int64,
        device=device,
    )
    max_position = position_offset + num_tokens - 1
    num_state_blocks = max_position // STATE_BLOCK_SIZE + 2
    state_page_elements = STATE_BLOCK_SIZE * 2 * STATE_WIDTH
    state_page_stride = (
        _round_up(
            state_page_elements * torch.float32.itemsize,
            PAGE_ALIGNMENT_BYTES,
        )
        // torch.float32.itemsize
    )
    state_storage = torch.randn(
        num_state_blocks * state_page_stride,
        dtype=torch.float32,
        device=device,
    )
    state_cache = torch.as_strided(
        state_storage,
        size=(num_state_blocks, STATE_BLOCK_SIZE, 2 * STATE_WIDTH),
        stride=(state_page_stride, 2 * STATE_WIDTH, 1),
    )
    block_table = torch.arange(
        num_state_blocks,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    token_to_req_indices = torch.zeros(
        num_tokens,
        dtype=torch.int32,
        device=device,
    )
    slot_mapping = positions.clone()
    boundary = (positions + 1) % COMPRESS_RATIO == 0
    kv_slot_mapping = torch.full_like(slot_mapping, -1)
    kv_slot_mapping[boundary] = positions[boundary] // COMPRESS_RATIO
    if pad_every:
        padded = torch.arange(num_tokens, device=device) % pad_every == pad_every - 1
        slot_mapping[padded] = -1
        kv_slot_mapping[padded] = -1

    max_kv_slot = max(0, max_position // COMPRESS_RATIO)
    num_kv_blocks = max_kv_slot // KV_BLOCK_SIZE + 2
    logical_kv_page_bytes = KV_BLOCK_SIZE * ALLOCATED_CACHE_ROW_BYTES
    kv_page_stride = _round_up(logical_kv_page_bytes, PAGE_ALIGNMENT_BYTES)
    k_storage = torch.full(
        (num_kv_blocks * kv_page_stride,),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    k_cache = torch.as_strided(
        k_storage,
        size=(num_kv_blocks, KV_BLOCK_SIZE, ALLOCATED_CACHE_ROW_BYTES),
        stride=(kv_page_stride, ALLOCATED_CACHE_ROW_BYTES, 1),
    )
    rms_norm_weight = torch.randn(
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    cos_sin_cache = torch.randn(
        (max_position + 1, ROPE_DIM),
        dtype=torch.float32,
        device=device,
    )
    kv_score = torch.randn(
        (num_tokens, 2 * STATE_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    kv, score = kv_score.split(STATE_WIDTH, dim=-1)
    ape = torch.randn(
        (COMPRESS_RATIO, STATE_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    candidate_impl = _load_cutedsl_candidate(use_mxfp4=use_mxfp4)
    candidate_uses_cutedsl = (
        candidate_impl is not None and num_tokens >= candidate_min_tokens
    )
    active_slots = torch.unique(kv_slot_mapping[kv_slot_mapping >= 0])
    active_pages = active_slots // KV_BLOCK_SIZE
    active_offsets = active_slots % KV_BLOCK_SIZE
    block_byte_stride = logical_kv_page_bytes
    value_indices = (
        active_pages[:, None] * block_byte_stride
        + active_offsets[:, None] * token_stride
        + torch.arange(token_stride, device=device)[None, :]
    ).flatten()
    scale_indices = (
        active_pages[:, None] * block_byte_stride
        + KV_BLOCK_SIZE * token_stride
        + active_offsets[:, None] * SCALE_DIM
        + torch.arange(SCALE_DIM, device=device)[None, :]
    ).flatten()
    untouched_mask = torch.ones(k_cache.numel(), dtype=torch.bool, device=device)
    untouched_mask[value_indices] = False
    untouched_mask[scale_indices] = False

    baseline_needs_reset = True
    candidate_needs_reset = True

    def prepare_state() -> None:
        if include_state_store:
            save_partial_states(
                kv,
                score,
                ape,
                positions,
                state_cache,
                slot_mapping,
                STATE_BLOCK_SIZE,
                STATE_WIDTH,
                COMPRESS_RATIO,
                pdl_kwargs={"launch_pdl": False},
            )

    def run_baseline() -> torch.Tensor:
        nonlocal baseline_needs_reset
        if baseline_needs_reset:
            k_cache.fill_(_SENTINEL)
            baseline_needs_reset = False
        prepare_state()
        _launch_triton(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            use_mxfp4=use_mxfp4,
            quant_block=quant_block,
            token_stride=token_stride,
        )
        return k_cache

    def run_candidate() -> torch.Tensor:
        nonlocal candidate_needs_reset
        if candidate_needs_reset:
            k_cache.fill_(_SENTINEL)
            candidate_needs_reset = False
        prepare_state()
        if not candidate_uses_cutedsl:
            _launch_triton(
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                rms_norm_weight,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
                use_mxfp4=use_mxfp4,
                quant_block=quant_block,
                token_stride=token_stride,
            )
        else:
            candidate_impl(
                state_cache,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                STATE_BLOCK_SIZE,
                rms_norm_weight,
                RMS_EPS,
                cos_sin_cache,
                k_cache,
                kv_slot_mapping,
                KV_BLOCK_SIZE,
                k_cache.stride(0),
                head_size=HEAD_DIM,
                state_width=STATE_WIDTH,
                rope_head_dim=ROPE_DIM,
                fp8_max=FP8_MAX,
                quant_block=quant_block,
                token_stride=token_stride,
                scale_dim=SCALE_DIM,
                compress_ratio=COMPRESS_RATIO,
                overlap=True,
            )
        return k_cache

    def compare_indexer_cache(
        reference: torch.Tensor,
        candidate: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        del tolerances
        reference_flat = reference.flatten()
        candidate_flat = candidate.flatten()
        untouched_exact = torch.equal(
            reference_flat[untouched_mask],
            candidate_flat[untouched_mask],
        )
        reference_scale_bytes = reference_flat[scale_indices].reshape(-1, SCALE_DIM)
        candidate_scale_bytes = candidate_flat[scale_indices].reshape(-1, SCALE_DIM)
        reference_scale = (
            reference_scale_bytes
            if use_mxfp4
            else reference_scale_bytes.view(torch.float32)
        )
        candidate_scale = (
            candidate_scale_bytes
            if use_mxfp4
            else candidate_scale_bytes.view(torch.float32)
        )
        scale_exact = torch.equal(reference_scale, candidate_scale)
        reference_values_u8 = reference_flat[value_indices].reshape(-1, token_stride)
        candidate_values_u8 = candidate_flat[value_indices].reshape(-1, token_stride)
        if use_mxfp4:
            fp4_values = torch.tensor(
                [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                dtype=torch.float32,
                device=device,
            )

            def dequantize_mxfp4(
                packed: torch.Tensor,
                scales: torch.Tensor,
            ) -> torch.Tensor:
                nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1)
                nibbles = nibbles.reshape(-1, SCALE_DIM, quant_block)
                magnitudes = fp4_values[(nibbles & 0x7).long()]
                values = torch.where((nibbles & 0x8) != 0, -magnitudes, magnitudes)
                scale_values = torch.exp2(scales.to(torch.int16).float() - 127.0)
                return (values * scale_values[..., None]).reshape(-1, HEAD_DIM)

            reference_dequant = dequantize_mxfp4(
                reference_values_u8,
                reference_scale,
            )
            candidate_dequant = dequantize_mxfp4(
                candidate_values_u8,
                candidate_scale,
            )
            scale_denom = torch.exp2(
                torch.maximum(reference_scale, candidate_scale).to(torch.int16).float()
                - 127.0
            ).repeat_interleave(quant_block, dim=1)
            max_allowed_error_in_scales = 2.0
            min_cosine = 0.99999
        else:
            reference_values = reference_values_u8.view(torch.float8_e4m3fn).float()
            candidate_values = candidate_values_u8.view(torch.float8_e4m3fn).float()
            reference_dequant = reference_values * reference_scale
            candidate_dequant = candidate_values * candidate_scale
            scale_denom = torch.maximum(reference_scale.abs(), candidate_scale.abs())
            max_allowed_error_in_scales = 32.0
            min_cosine = 0.999999
        finite = bool(torch.isfinite(candidate_dequant).all().item())
        diff = (reference_dequant - candidate_dequant).abs()
        max_error_in_scales = float((diff / scale_denom).max().item())
        byte_mismatch_fraction = float(
            (reference_values_u8 != candidate_values_u8).float().mean().item()
        )
        reference_f64 = reference_dequant.flatten().double()
        candidate_f64 = candidate_dequant.flatten().double()
        denominator = reference_f64.abs().clamp_min(1e-12)
        relative = (candidate_f64 - reference_f64).abs() / denominator
        cosine = float(
            torch.nn.functional.cosine_similarity(
                reference_f64,
                candidate_f64,
                dim=0,
                eps=1e-12,
            ).item()
        )
        passed = (
            untouched_exact
            and scale_exact
            and finite
            and max_error_in_scales <= max_allowed_error_in_scales
            and byte_mismatch_fraction <= 1e-4
            and cosine >= min_cosine
        )
        return {
            "passed": passed,
            "finite": finite,
            "allclose": bool(torch.equal(reference, candidate)),
            "atol": 0.0,
            "rtol": 0.0,
            "max_abs": float(diff.max().item()),
            "max_rel": float(relative.max().item()),
            "mean_abs": float(diff.double().mean().item()),
            "mean_relative": float(relative.mean().item()),
            "cosine": cosine,
            "max_mean_relative": None,
            "min_cosine": min_cosine,
            "untouched_exact": untouched_exact,
            "scale_exact": scale_exact,
            "value_byte_mismatch_fraction": byte_mismatch_fraction,
            "max_dequant_error_in_scales": max_error_in_scales,
            "max_allowed_dequant_error_in_scales": max_allowed_error_in_scales,
            "max_allowed_value_byte_mismatch_fraction": 1e-4,
        }

    chain_name = (
        f"save-compress-norm-rope-{cache_format}-cache"
        if include_state_store
        else f"compress-norm-rope-{cache_format}-cache"
    )
    shape_name = f"t{num_tokens}-o{position_offset}-indexer-{cache_format}"
    if include_state_store:
        shape_name += "-save-chain"
    active_tokens = int((boundary & (slot_mapping >= 0)).sum().item())
    return ChainCase(
        baseline=Provider(
            f"triton-{chain_name}",
            run_baseline,
            {
                "compress_backend": "triton",
                "launches": 2 if include_state_store else 1,
                "preallocated_output": True,
                "shared_output_with_candidate": True,
            },
        ),
        candidate=Provider(
            f"candidate-{chain_name}",
            run_candidate,
            {
                "compress_backend": (
                    "cutedsl" if candidate_uses_cutedsl else "triton-mirror"
                ),
                "launches": 2 if include_state_store else 1,
                "preallocated_output": True,
                "shared_output_with_baseline": True,
            },
            correctness_comparator=compare_indexer_cache,
        ),
        shape={
            "name": shape_name,
            "T": num_tokens,
            "position_offset": position_offset,
            "active_tokens": active_tokens,
            "pad_every": pad_every,
            "head_dim": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "state_width": STATE_WIDTH,
            "state_block_size": STATE_BLOCK_SIZE,
            "kv_block_size": KV_BLOCK_SIZE,
            "kv_block_stride": k_cache.stride(0),
            "allocated_cache_row_bytes": ALLOCATED_CACHE_ROW_BYTES,
            "payload_bytes": token_stride,
            "scale_bytes": SCALE_DIM,
            "quant_block": quant_block,
            "cache_format": cache_format,
            "state_block_stride": state_cache.stride(0),
            "page_alignment_bytes": PAGE_ALIGNMENT_BYTES,
            "compress_ratio": COMPRESS_RATIO,
            "chain": chain_name,
        },
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_b04_indexer_fp8_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b04_case(args, include_state_store=False)


def build_b04_indexer_fp8_save_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b04_case(args, include_state_store=True)


def build_b05_indexer_mxfp4_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b04_case(
        args,
        include_state_store=False,
        use_mxfp4=True,
        candidate_min_tokens=8192,
    )


def build_b05_indexer_mxfp4_save_chain_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_b04_case(
        args,
        include_state_store=True,
        use_mxfp4=True,
        candidate_min_tokens=8192,
    )


__all__ = [
    "build_b04_indexer_fp8_case",
    "build_b04_indexer_fp8_save_chain_case",
    "build_b05_indexer_mxfp4_case",
    "build_b05_indexer_mxfp4_save_chain_case",
]
