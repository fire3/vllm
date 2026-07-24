# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Boolean, Int32
from quack.compile_utils import make_fake_tensor


class GlobalTopKIndicesAndLensKernel:
    """Map local top-k rows with one CTA per token."""

    def __init__(self, topk: int, block_size: int, threads: int):
        self.topk = topk
        self.block_size = block_size
        self.threads = threads
        self.num_warps = threads // 32
        self.iterations = (topk + threads - 1) // threads

    @cute.jit
    def __call__(
        self,
        global_topk_indices: cute.Tensor,
        topk_lens: cute.Tensor,
        topk_indices: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        block_table: cute.Tensor,
        is_valid_token: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(
            global_topk_indices,
            topk_lens,
            topk_indices,
            token_to_req_indices,
            block_table,
            is_valid_token,
        ).launch(
            grid=(topk_indices.shape[0], 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        global_topk_indices: cute.Tensor,
        topk_lens: cute.Tensor,
        topk_indices: cute.Tensor,
        token_to_req_indices: cute.Tensor,
        block_table: cute.Tensor,
        is_valid_token: cute.Tensor,
    ):
        token_idx, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane_id = tid % 32
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        req_idx = token_to_req_indices[token_idx]

        local_count = Int32(0)
        for iteration in cutlass.range_constexpr(self.iterations):
            offset = tid + iteration * self.threads
            if offset < self.topk:
                local_idx = topk_indices[token_idx, offset]
                if local_idx >= Int32(0):
                    page_idx = local_idx // Int32(self.block_size)
                    block_number = block_table[req_idx, page_idx]
                    block_offset = local_idx % Int32(self.block_size)
                    global_topk_indices[token_idx, offset] = (
                        block_number * Int32(self.block_size) + block_offset
                    )
                    local_count += Int32(1)
                else:
                    global_topk_indices[token_idx, offset] = Int32(-1)

        for step in cutlass.range_constexpr(5):
            local_count += cute.arch.shuffle_sync_bfly(local_count, 16 >> step)

        smem = cutlass.utils.SmemAllocator()
        warp_counts = smem.allocate_tensor(
            Int32,
            cute.make_layout((self.num_warps,)),
            byte_alignment=16,
        )
        if lane_id == 0:
            warp_counts[warp_id] = local_count
        cute.arch.sync_threads()

        if warp_id == 0:
            total = Int32(0)
            if lane_id < self.num_warps:
                total = warp_counts[lane_id]
            for step in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total, 16 >> step)
            if lane_id == 0:
                topk_lens[token_idx] = total if is_valid_token[token_idx] else Int32(0)

    @cache
    @staticmethod
    def compile(topk: int, block_size: int, threads: int):
        num_tokens = cute.sym_int()
        num_requests = cute.sym_int()
        block_table_width = cute.sym_int()
        output_row_stride = cute.sym_int64(divisibility=4)
        input_row_stride = cute.sym_int64(divisibility=4)
        block_table_stride = cute.sym_int64(divisibility=1)

        global_topk_indices = cute.runtime.make_fake_tensor(
            Int32,
            (num_tokens, topk),
            stride=(output_row_stride, 1),
            assumed_align=16,
        )
        topk_lens = make_fake_tensor(Int32, (num_tokens,), divisibility=4)
        topk_indices = cute.runtime.make_fake_tensor(
            Int32,
            (num_tokens, topk),
            stride=(input_row_stride, 1),
            assumed_align=16,
        )
        token_to_req_indices = make_fake_tensor(Int32, (num_tokens,), divisibility=4)
        block_table = cute.runtime.make_fake_tensor(
            Int32,
            (num_requests, block_table_width),
            stride=(block_table_stride, 1),
            assumed_align=4,
        )
        is_valid_token = cute.runtime.make_fake_tensor(
            Boolean,
            (num_tokens,),
            stride=(1,),
            assumed_align=1,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = GlobalTopKIndicesAndLensKernel(topk, block_size, threads)
        return cute.compile(
            kernel,
            global_topk_indices,
            topk_lens,
            topk_indices,
            token_to_req_indices,
            block_table,
            is_valid_token,
            stream,
            options="--enable-tvm-ffi",
        )


def launch_global_topk_indices_and_lens_cutedsl(
    global_topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    *,
    threads: int = 128,
) -> None:
    """Launch the CuTe DSL local-to-global top-k mapping kernel."""
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be two-dimensional")
    num_tokens, topk = topk_indices.shape
    if global_topk_indices.shape != topk_indices.shape:
        raise ValueError("global_topk_indices must match topk_indices")
    if topk_lens.shape != (num_tokens,):
        raise ValueError("topk_lens must have shape [num_tokens]")
    if token_to_req_indices.shape != (num_tokens,):
        raise ValueError("token_to_req_indices must have shape [num_tokens]")
    if is_valid_token.shape != (num_tokens,):
        raise ValueError("is_valid_token must have shape [num_tokens]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if threads not in (128, 256):
        raise ValueError("threads must be 128 or 256")
    tensors = (
        global_topk_indices,
        topk_lens,
        topk_indices,
        token_to_req_indices,
        block_table,
        is_valid_token,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if any(tensor.device != topk_indices.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    integer_tensors = (
        global_topk_indices,
        topk_lens,
        topk_indices,
        token_to_req_indices,
        block_table,
    )
    if any(tensor.dtype != torch.int32 for tensor in integer_tensors):
        raise ValueError("index and length tensors must have dtype int32")
    if is_valid_token.dtype != torch.bool:
        raise ValueError("is_valid_token must have dtype bool")
    if topk_indices.stride(1) != 1 or global_topk_indices.stride(1) != 1:
        raise ValueError("top-k rows must be contiguous in the last dimension")
    if block_table.stride(1) != 1:
        raise ValueError("block_table rows must be contiguous")
    if num_tokens == 0:
        return

    GlobalTopKIndicesAndLensKernel.compile(topk, block_size, threads)(
        global_topk_indices,
        topk_lens,
        topk_indices,
        token_to_req_indices,
        block_table,
        is_valid_token,
    )
