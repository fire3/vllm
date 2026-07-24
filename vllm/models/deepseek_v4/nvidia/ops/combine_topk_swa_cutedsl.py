# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Int32

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


class CombineTopKSwaIndicesKernel:
    """Build each FlashMLA sparse-index row with one warp."""

    warps_per_cta = 4
    threads = warps_per_cta * 32

    def __init__(self, topk: int, window_size: int, num_reqs: int):
        self.topk = topk
        self.window_size = window_size
        self.num_reqs = num_reqs
        self.combined_width = (
            (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
            // _SPARSE_PREFILL_TOPK_ALIGNMENT
            * _SPARSE_PREFILL_TOPK_ALIGNMENT
        )

    @cute.jit
    def __call__(
        self,
        combined_indices: cute.Tensor,
        combined_lens: cute.Tensor,
        topk_indices: cute.Tensor,
        query_start_loc: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor,
        m: Int32,
        n: Int32,
        compress_ratio: Int32,
        stream: CUstream,
    ):
        self.kernel(
            combined_indices,
            combined_lens,
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            m,
            n,
            compress_ratio,
        ).launch(
            grid=(
                cute.ceil_div(topk_indices.shape[0], self.warps_per_cta),
                1,
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        combined_indices: cute.Tensor,
        combined_lens: cute.Tensor,
        topk_indices: cute.Tensor,
        query_start_loc: cute.Tensor,
        seq_lens: cute.Tensor,
        gather_lens: cute.Tensor,
        m: Int32,
        n: Int32,
        compress_ratio: Int32,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32
        token_idx = block_idx * self.warps_per_cta + warp_id

        if token_idx < topk_indices.shape[0]:
            base = query_start_loc[0]
            req_idx = Int32(0)
            for request in cutlass.range_constexpr(1, self.num_reqs):
                request_start = query_start_loc[request] - base
                if token_idx >= request_start:
                    req_idx = Int32(request)

            query_start = query_start_loc[req_idx] - base
            query_end = query_start_loc[req_idx + 1] - base
            query_len = query_end - query_start
            seq_len = seq_lens[req_idx]
            gather_len = gather_lens[req_idx]
            pos = seq_len - query_len + token_idx - query_start
            topk_len = cute.arch.make_warp_uniform(
                min((pos + Int32(1)) // compress_ratio, Int32(self.topk))
            )
            swa_len = cute.arch.make_warp_uniform(
                min(pos + Int32(1), Int32(self.window_size))
            )
            request_offset = m * req_idx
            gather_start = seq_len - gather_len

            if lane_id == 0:
                combined_lens[token_idx] = topk_len + swa_len

            for column in range(lane_id, self.combined_width, 32):
                value = Int32(-1)
                if column < topk_len:
                    value = topk_indices[token_idx, column] + request_offset
                elif column < topk_len + swa_len:
                    swa_offset = column - topk_len
                    value = (
                        request_offset
                        + n
                        + swa_offset
                        + pos
                        - swa_len
                        + Int32(1)
                        - gather_start
                    )
                combined_indices[token_idx, column] = value

    @cache
    @staticmethod
    def compile(topk: int, window_size: int, num_reqs: int):
        num_tokens = cute.sym_int()
        source_width = cute.sym_int()
        source_row_stride = cute.sym_int64(divisibility=1)
        combined_width = (
            (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
            // _SPARSE_PREFILL_TOPK_ALIGNMENT
            * _SPARSE_PREFILL_TOPK_ALIGNMENT
        )

        combined_indices = cute.runtime.make_fake_tensor(
            Int32,
            (num_tokens, combined_width),
            stride=(combined_width, 1),
            assumed_align=16,
        )
        combined_lens = cute.runtime.make_fake_tensor(
            Int32,
            (num_tokens,),
            stride=(1,),
            assumed_align=4,
        )
        topk_indices = cute.runtime.make_fake_tensor(
            Int32,
            (num_tokens, source_width),
            stride=(source_row_stride, 1),
            assumed_align=4,
        )
        query_start_loc = cute.runtime.make_fake_tensor(
            Int32,
            (num_reqs + 1,),
            stride=(1,),
            assumed_align=4,
        )
        seq_lens = cute.runtime.make_fake_tensor(
            Int32,
            (num_reqs,),
            stride=(1,),
            assumed_align=4,
        )
        gather_lens = cute.runtime.make_fake_tensor(
            Int32,
            (num_reqs,),
            stride=(1,),
            assumed_align=4,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = CombineTopKSwaIndicesKernel(topk, window_size, num_reqs)
        return cute.compile(
            kernel,
            combined_indices,
            combined_lens,
            topk_indices,
            query_start_loc,
            seq_lens,
            gather_lens,
            Int32(0),
            Int32(0),
            Int32(1),
            stream,
            options="--enable-tvm-ffi",
        )


def combine_topk_swa_indices_cutedsl(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    m: int,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded FlashMLA gathered-workspace indices in one kernel."""
    if topk_indices.ndim != 2:
        raise ValueError("topk_indices must be two-dimensional")
    if query_start_loc.ndim != 1 or seq_lens.ndim != 1 or gather_lens.ndim != 1:
        raise ValueError("C10 metadata tensors must be one-dimensional")
    num_tokens, source_width = topk_indices.shape
    num_reqs = seq_lens.shape[0]
    if num_reqs <= 0 or query_start_loc.shape != (num_reqs + 1,):
        raise ValueError("query_start_loc must delimit every request")
    if gather_lens.shape != (num_reqs,):
        raise ValueError("gather_lens must match seq_lens")
    if topk < 0 or topk > source_width:
        raise ValueError("topk must fit the source row")
    if window_size <= 0 or compress_ratio <= 0:
        raise ValueError("window_size and compress_ratio must be positive")
    if m < 0 or n < 0:
        raise ValueError("M and N must be non-negative")

    tensors = (topk_indices, query_start_loc, seq_lens, gather_lens)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("all C10 tensors must be CUDA tensors")
    if any(tensor.device != topk_indices.device for tensor in tensors):
        raise ValueError("all C10 tensors must be on the same CUDA device")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise ValueError("all C10 tensors must have dtype int32")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("C10 tensors must be contiguous in their last dimension")

    combined_width = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.empty(
        (num_tokens, combined_width),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if num_tokens == 0:
        return combined_indices, combined_lens

    CombineTopKSwaIndicesKernel.compile(topk, window_size, num_reqs)(
        combined_indices,
        combined_lens,
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        m,
        n,
        compress_ratio,
    )
    return combined_indices, combined_lens
