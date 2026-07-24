# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import cvt, recast_val

_TORCH_TO_CUTE = {
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
}


class IndexerCompressNormRopeStoreFp8Kernel:
    """Fuse the fixed C4 indexer compressor into one warp per token."""

    head_dim = 128
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    state_width = 256
    compress_ratio = 4
    state_block_size = 4
    window = 8
    token_stride = 128
    scale_dim = 4
    allocated_row_bytes = token_stride + scale_dim
    mxfp4_token_stride = head_dim // 2
    mxfp4_min_absmax = float.fromhex("0x6p-126")
    fp8_max = 448.0
    min_scale = 1.0e-4
    rcp_ln2 = 1.4426950408889634
    warps_per_cta = 4
    threads = warps_per_cta * 32
    elems_per_lane = head_dim // 32

    def __init__(
        self,
        kv_cache_block_size: int,
        norm_weight_dtype: type[cutlass.Numeric],
        use_mxfp4: bool = False,
    ):
        self.kv_cache_block_size = kv_cache_block_size
        self.norm_weight_dtype = norm_weight_dtype
        self.use_mxfp4 = use_mxfp4

    @cute.jit
    def __call__(
        self,
        state_cache: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        block_table: cute.Tensor,
        rms_norm_weight: cute.Tensor,
        rms_norm_eps: Float32,
        cos_sin_cache: cute.Tensor,
        k_cache: cute.Tensor,
        kv_slot_mapping: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            rms_norm_eps,
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
        ).launch(
            grid=(cute.ceil_div(slot_mapping.shape[0], self.warps_per_cta), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        state_cache: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        positions: cute.Tensor,
        slot_mapping: cute.Tensor,
        block_table: cute.Tensor,
        rms_norm_weight: cute.Tensor,
        rms_norm_eps: Float32,
        cos_sin_cache: cute.Tensor,
        k_cache: cute.Tensor,
        kv_slot_mapping: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        token_idx = block_idx * self.warps_per_cta + warp_id
        elem_base = lane_id * self.elems_per_lane

        slot_id = Int64(-1)
        position = Int64(0)
        req_idx = Int32(0)
        kv_slot_idx = Int64(-1)
        in_bounds = token_idx < slot_mapping.shape[0]
        has_position = in_bounds and token_idx < positions.shape[0]
        has_req_idx = in_bounds and token_idx < token_to_req_indices.shape[0]
        has_kv_slot = in_bounds and token_idx < kv_slot_mapping.shape[0]
        if lane_id == 0 and in_bounds:
            slot_id = slot_mapping[token_idx]
            if has_position:
                position = positions[token_idx]
            if has_req_idx:
                req_idx = token_to_req_indices[token_idx]
            if has_kv_slot:
                kv_slot_idx = kv_slot_mapping[token_idx]
        slot_id = cute.arch.shuffle_sync(slot_id, offset=0)
        position = cute.arch.shuffle_sync(position, offset=0)
        req_idx = cute.arch.shuffle_sync(req_idx, offset=0)
        kv_slot_idx = cute.arch.shuffle_sync(kv_slot_idx, offset=0)

        boundary = has_position and (
            (position + Int64(1)) % Int64(self.compress_ratio) == Int64(0)
        )
        active = (
            slot_id >= Int64(0)
            and has_req_idx
            and has_kv_slot
            and boundary
            and kv_slot_idx >= Int64(0)
        )

        if active:
            x_max = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            x_sum = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            x_product = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            for elem in cutlass.range_constexpr(self.elems_per_lane):
                x_max[elem] = -Float32.inf
                x_sum[elem] = Float32(0.0)
                x_product[elem] = Float32(0.0)

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                Float32,
                num_bits_per_copy=128,
            )
            kv_values = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            score_values = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            start = position - Int64(self.window - 1)

            for row in cutlass.range_constexpr(self.window):
                source_pos = start + Int64(row)
                if source_pos >= Int64(0):
                    block_number = Int32(0)
                    if lane_id == 0:
                        block_index = source_pos // Int64(self.state_block_size)
                        block_number = block_table[req_idx, block_index]
                    block_number = cute.arch.shuffle_sync(block_number, offset=0)
                    block_offset = source_pos % Int64(self.state_block_size)
                    head_offset = Int64((row // self.compress_ratio) * self.head_dim)
                    row_tensor = state_cache[block_number.to(Int64), block_offset, None]
                    kv_tile = cute.local_tile(
                        row_tensor,
                        tiler=(self.elems_per_lane,),
                        coord=((head_offset + elem_base) // self.elems_per_lane,),
                    )
                    score_tile = cute.local_tile(
                        row_tensor,
                        tiler=(self.elems_per_lane,),
                        coord=(
                            (head_offset + Int64(self.state_width) + elem_base)
                            // self.elems_per_lane,
                        ),
                    )
                    cute.copy(copy_atom, kv_tile, kv_values)
                    cute.copy(copy_atom, score_tile, score_values)

                    for elem in cutlass.range_constexpr(self.elems_per_lane):
                        score = score_values[elem]
                        new_max = cute.arch.fmax(x_max[elem], score)
                        old_scale = cute.math.exp2(
                            (x_max[elem] - new_max) * Float32(self.rcp_ln2),
                            fastmath=True,
                        )
                        new_scale = cute.math.exp2(
                            (score - new_max) * Float32(self.rcp_ln2),
                            fastmath=True,
                        )
                        x_sum[elem] = x_sum[elem] * old_scale + new_scale
                        x_product[elem] = (
                            x_product[elem] * old_scale + kv_values[elem] * new_scale
                        )
                        x_max[elem] = new_max

            x = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            local_sumsq = Float32(0.0)
            for elem in cutlass.range_constexpr(self.elems_per_lane):
                x[elem] = x_product[elem] / x_sum[elem]
                local_sumsq += x[elem] * x[elem]

            total_sumsq = local_sumsq
            for step in cutlass.range_constexpr(5):
                total_sumsq += cute.arch.shuffle_sync_bfly(
                    total_sumsq,
                    16 >> step,
                )
            rrms = cute.math.rsqrt(
                total_sumsq / Float32(self.head_dim) + rms_norm_eps,
                fastmath=True,
            )
            for elem in cutlass.range_constexpr(self.elems_per_lane):
                weight = rms_norm_weight[elem_base + elem].to(Float32)
                x[elem] = x[elem] * rrms * weight

            if elem_base >= self.nope_dim:
                compressed_pos = (position // Int64(self.compress_ratio)) * Int64(
                    self.compress_ratio
                )
                pair_base = (elem_base - self.nope_dim) // 2
                for pair in cutlass.range_constexpr(self.elems_per_lane // 2):
                    elem = pair * 2
                    pair_idx = pair_base + pair
                    cos_v = cos_sin_cache[compressed_pos, pair_idx]
                    sin_v = cos_sin_cache[compressed_pos, pair_idx + self.rope_dim // 2]
                    real = x[elem] * cos_v - x[elem + 1] * sin_v
                    imag = x[elem] * sin_v + x[elem + 1] * cos_v
                    x[elem] = real
                    x[elem + 1] = imag

            q = cute.make_rmem_tensor((self.elems_per_lane,), Float32)
            local_absmax = Float32(0.0)
            for pair in cutlass.range_constexpr(self.elems_per_lane // 2):
                elem = pair * 2
                packed_bf16 = cvt.fp32x2_to_bf16x2(x[elem], x[elem + 1])
                q0, q1 = cvt.bf16x2_to_fp32x2(packed_bf16)
                q[elem] = q0
                q[elem + 1] = q1
                local_absmax = cute.arch.fmax(
                    local_absmax,
                    cute.arch.fmax(cute.math.absf(q0), cute.math.absf(q1)),
                )

            page = kv_slot_idx // Int64(self.kv_cache_block_size)
            kv_offset = kv_slot_idx - page * Int64(self.kv_cache_block_size)
            if cutlass.const_expr(self.use_mxfp4):
                absmax = local_absmax
                for step in cutlass.range_constexpr(3):
                    absmax = cute.arch.fmax(
                        absmax,
                        cute.arch.shuffle_sync_bfly(absmax, 4 >> step),
                    )
                fp4_scale = cute.arch.fmax(
                    absmax,
                    Float32(self.mxfp4_min_absmax),
                ) * Float32(1.0 / 6.0)
                scale_bits = recast_val(fp4_scale, Uint32)
                scale_exp = ((scale_bits + Uint32(0x7FFFFF)) >> Uint32(23)) & Uint32(
                    0xFF
                )
                inv_scale = recast_val(
                    (Uint32(254) - scale_exp) << Uint32(23),
                    Float32,
                )
                value_base = page * k_cache.stride[0] + kv_offset * Int64(
                    self.mxfp4_token_stride
                )
                scale_base = (
                    page * k_cache.stride[0]
                    + Int64(self.kv_cache_block_size * self.mxfp4_token_stride)
                    + kv_offset * Int64(self.scale_dim)
                )
                packed_fp4 = cvt.fp32x4_to_fp4x4(
                    q[0] * inv_scale,
                    q[1] * inv_scale,
                    q[2] * inv_scale,
                    q[3] * inv_scale,
                )
                k_cache_u16 = cute.recast_tensor(k_cache, Uint16)
                k_cache_u16.iterator[
                    (value_base + elem_base.to(Int64) // Int64(2)) // Int64(2)
                ] = Uint16(packed_fp4)
                if lane_id % 8 == 0:
                    k_cache.iterator[scale_base + lane_id.to(Int64) // Int64(8)] = (
                        Uint8(scale_exp)
                    )
            else:
                absmax = local_absmax
                for step in cutlass.range_constexpr(5):
                    absmax = cute.arch.fmax(
                        absmax,
                        cute.arch.shuffle_sync_bfly(absmax, 16 >> step),
                    )
                raw_scale = cute.arch.fmax(
                    absmax,
                    Float32(self.min_scale),
                ) * Float32(1.0 / self.fp8_max)
                scale_bits = recast_val(raw_scale, Uint32)
                scale_exp = ((scale_bits + Uint32(0x7FFFFF)) >> Uint32(23)) & Uint32(
                    0xFF
                )
                inv_scale = recast_val(
                    (Uint32(254) - scale_exp) << Uint32(23),
                    Float32,
                )
                value_base = page * k_cache.stride[0] + kv_offset * Int64(
                    self.token_stride
                )
                scale_base = (
                    page * k_cache.stride[0]
                    + Int64(self.kv_cache_block_size * self.token_stride)
                    + kv_offset * Int64(self.scale_dim)
                )
                k_cache_u32 = cute.recast_tensor(k_cache, Uint32)
                packed_fp8 = cvt.fp32x4_to_fp8x4(
                    q[0] * inv_scale,
                    q[1] * inv_scale,
                    q[2] * inv_scale,
                    q[3] * inv_scale,
                )
                k_cache_u32.iterator[(value_base + elem_base.to(Int64)) // Int64(4)] = (
                    packed_fp8
                )
                if lane_id == 0:
                    k_cache_u32.iterator[scale_base // Int64(4)] = scale_exp << Uint32(
                        23
                    )

    @cache
    @staticmethod
    def compile(
        state_block_stride: int,
        state_row_stride: int,
        kv_cache_block_size: int,
        kv_block_stride: int,
        norm_weight_dtype: type[cutlass.Numeric],
        use_mxfp4: bool = False,
    ):
        if state_block_stride <= 0 or state_row_stride <= 0:
            raise ValueError("state cache strides must be positive")
        if kv_cache_block_size <= 0:
            raise ValueError("kv_cache_block_size must be positive")
        num_tokens = cute.sym_int()
        num_state_blocks = cute.sym_int()
        block_table_width = cute.sym_int()
        max_position = cute.sym_int()
        num_kv_blocks = cute.sym_int()

        state_cache = cute.runtime.make_fake_tensor(
            Float32,
            (
                num_state_blocks,
                IndexerCompressNormRopeStoreFp8Kernel.state_block_size,
                2 * IndexerCompressNormRopeStoreFp8Kernel.state_width,
            ),
            stride=(
                state_block_stride,
                state_row_stride,
                1,
            ),
            assumed_align=16,
        )
        token_to_req_indices = make_fake_tensor(
            Int32,
            (num_tokens,),
            divisibility=4,
        )
        positions = make_fake_tensor(Int64, (num_tokens,), divisibility=8)
        slot_mapping = make_fake_tensor(Int64, (num_tokens,), divisibility=8)
        block_table = make_fake_tensor(
            Int32,
            (cute.sym_int(), block_table_width),
            divisibility=1,
        )
        rms_norm_weight = make_fake_tensor(
            norm_weight_dtype,
            (IndexerCompressNormRopeStoreFp8Kernel.head_dim,),
            divisibility=4,
        )
        cos_sin_cache = cute.runtime.make_fake_tensor(
            Float32,
            (max_position, IndexerCompressNormRopeStoreFp8Kernel.rope_dim),
            stride=(IndexerCompressNormRopeStoreFp8Kernel.rope_dim, 1),
            assumed_align=16,
        )
        k_cache = cute.runtime.make_fake_tensor(
            Uint8,
            (
                num_kv_blocks,
                kv_cache_block_size,
                IndexerCompressNormRopeStoreFp8Kernel.allocated_row_bytes,
            ),
            stride=(
                kv_block_stride,
                IndexerCompressNormRopeStoreFp8Kernel.allocated_row_bytes,
                1,
            ),
            assumed_align=16,
        )
        kv_slot_mapping = make_fake_tensor(
            Int64,
            (num_tokens,),
            divisibility=8,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = IndexerCompressNormRopeStoreFp8Kernel(
            kv_cache_block_size,
            norm_weight_dtype,
            use_mxfp4,
        )
        return cute.compile(
            kernel,
            state_cache,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            rms_norm_weight,
            Float32(0.0),
            cos_sin_cache,
            k_cache,
            kv_slot_mapping,
            stream,
            options="--enable-tvm-ffi",
        )


def _fused_kv_compress_norm_rope_insert_indexer_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 128,
    state_width: int = 256,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 128,
    token_stride: int = 128,
    scale_dim: int = 4,
    compress_ratio: int = 4,
    overlap: bool = True,
    use_mxfp4: bool = False,
) -> None:
    """Run the fixed DeepSeek-V4 C4 indexer compressor."""
    layout = (
        head_size,
        state_width,
        rope_head_dim,
        fp8_max,
        quant_block,
        token_stride,
        scale_dim,
        compress_ratio,
        overlap,
        block_size,
    )
    expected = (
        (128, 256, 64, 448.0, 32, 64, 4, 4, True, 4)
        if use_mxfp4
        else (128, 256, 64, 448.0, 128, 128, 4, 4, True, 4)
    )
    if layout != expected:
        raise ValueError(f"unsupported indexer compressor layout: {layout}")
    num_tokens = slot_mapping.shape[0]
    if positions.shape != (num_tokens,):
        raise ValueError("positions must have shape [num_tokens]")
    if token_to_req_indices.shape != (num_tokens,):
        raise ValueError("token_to_req_indices must have shape [num_tokens]")
    if kv_slot_mapping.shape != (num_tokens,):
        raise ValueError("kv_slot_mapping must have shape [num_tokens]")
    if state_cache.ndim != 3 or state_cache.shape[1:] != (4, 512):
        raise ValueError("state_cache must have shape [num_blocks, 4, 512]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    if rms_norm_weight.shape != (128,):
        raise ValueError("rms_norm_weight must have shape [128]")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
        raise ValueError("cos_sin_cache must have shape [max_position, 64]")
    if k_cache.ndim != 3 or k_cache.shape[1:] != (
        kv_cache_block_size,
        132,
    ):
        raise ValueError("k_cache must have shape [num_blocks, block_size, 132]")
    if kv_block_stride != k_cache.stride(0):
        raise ValueError("kv_block_stride must match k_cache.stride(0)")
    tensors = (
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if any(tensor.device != state_cache.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if state_cache.dtype != torch.float32:
        raise ValueError("state_cache must have dtype float32")
    if token_to_req_indices.dtype != torch.int32:
        raise ValueError("token_to_req_indices must have dtype int32")
    if positions.dtype != torch.int64 or slot_mapping.dtype != torch.int64:
        raise ValueError("positions and slot_mapping must have dtype int64")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must have dtype int32")
    if rms_norm_weight.dtype not in _TORCH_TO_CUTE:
        raise ValueError("rms_norm_weight must have dtype bfloat16 or float32")
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache must have dtype float32")
    if k_cache.dtype != torch.uint8 or kv_slot_mapping.dtype != torch.int64:
        raise ValueError("k_cache and kv_slot_mapping have incompatible dtypes")
    state_block_stride = state_cache.stride(0)
    state_row_stride = state_cache.stride(1)
    if state_cache.stride(2) != 1:
        raise ValueError("state_cache rows must be contiguous")
    if state_row_stride < state_cache.shape[2]:
        raise ValueError("state_cache rows must not overlap")
    min_state_block_stride = (
        state_cache.shape[1] - 1
    ) * state_row_stride + state_cache.shape[2]
    if state_block_stride < min_state_block_stride:
        raise ValueError("state_cache blocks must not overlap")
    if state_row_stride % 4 or state_block_stride % 4:
        raise ValueError("state_cache rows and blocks must be 16-byte aligned")
    if k_cache.stride(2) != 1 or k_cache.stride(1) != 132:
        raise ValueError("k_cache must have contiguous 132-byte logical rows")
    if kv_block_stride < kv_cache_block_size * 132:
        raise ValueError("k_cache blocks must not overlap")
    if kv_block_stride % 16:
        raise ValueError("k_cache blocks must be 16-byte aligned")
    if block_table.stride(1) != 1:
        raise ValueError("block_table rows must be contiguous")
    contiguous_tensors = (
        token_to_req_indices,
        positions,
        slot_mapping,
        rms_norm_weight,
        cos_sin_cache,
        kv_slot_mapping,
    )
    if any(not tensor.is_contiguous() for tensor in contiguous_tensors):
        raise ValueError("metadata, norm weight, and RoPE cache must be contiguous")
    aligned_tensors = (state_cache, cos_sin_cache, k_cache)
    if any(tensor.data_ptr() % 16 for tensor in aligned_tensors):
        raise ValueError("state, RoPE, and KV cache pointers must be 16-byte aligned")
    if num_tokens == 0:
        return

    compiled = IndexerCompressNormRopeStoreFp8Kernel.compile(
        state_block_stride,
        state_row_stride,
        kv_cache_block_size,
        kv_block_stride,
        _TORCH_TO_CUTE[rms_norm_weight.dtype],
        use_mxfp4,
    )
    compiled(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
    )


def fused_kv_compress_norm_rope_insert_indexer_fp8_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 128,
    state_width: int = 256,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 128,
    token_stride: int = 128,
    scale_dim: int = 4,
    compress_ratio: int = 4,
    overlap: bool = True,
) -> None:
    """Run the fixed DeepSeek-V4 C4 FP8 indexer compressor."""
    _fused_kv_compress_norm_rope_insert_indexer_cutedsl(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        kv_block_stride,
        head_size,
        state_width,
        rope_head_dim,
        fp8_max,
        quant_block,
        token_stride,
        scale_dim,
        compress_ratio,
        overlap,
        use_mxfp4=False,
    )


def fused_kv_compress_norm_rope_insert_indexer_mxfp4_cutedsl(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    kv_block_stride: int,
    head_size: int = 128,
    state_width: int = 256,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 32,
    token_stride: int = 64,
    scale_dim: int = 4,
    compress_ratio: int = 4,
    overlap: bool = True,
) -> None:
    """Run the fixed DeepSeek-V4 C4 MXFP4 indexer compressor."""
    _fused_kv_compress_norm_rope_insert_indexer_cutedsl(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        kv_block_stride,
        head_size,
        state_width,
        rope_head_dim,
        fp8_max,
        quant_block,
        token_stride,
        scale_dim,
        compress_ratio,
        overlap,
        use_mxfp4=True,
    )


def compress_norm_rope_store_indexer_cutedsl(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Adapt the production compressor call to the fixed CuTe kernel."""
    if num_actual != slot_mapping.shape[0]:
        raise ValueError("num_actual must match slot_mapping")
    if bool(pdl_kwargs.get("launch_pdl", False)):
        raise ValueError("the indexer compressor requires PDL to be disabled")
    kernel = (
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_cutedsl
        if use_fp4_cache
        else fused_kv_compress_norm_rope_insert_indexer_fp8_cutedsl
    )
    kernel(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],
        kv_cache.stride(0),
        head_size=head_dim,
        state_width=state_width,
        rope_head_dim=rope_head_dim,
        fp8_max=448.0,
        quant_block=quant_block,
        token_stride=token_stride,
        scale_dim=scale_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )


__all__ = [
    "IndexerCompressNormRopeStoreFp8Kernel",
    "compress_norm_rope_store_indexer_cutedsl",
    "fused_kv_compress_norm_rope_insert_indexer_fp8_cutedsl",
    "fused_kv_compress_norm_rope_insert_indexer_mxfp4_cutedsl",
]
