# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for DSv4 sparse MLA benchmark inputs."""

from __future__ import annotations

import math

import torch

KV_NOPE_DIM = 448
QK_ROPE_DIM = 64
HEAD_DIM = KV_NOPE_DIM + QK_ROPE_DIM
ROPE_BYTES = QK_ROPE_DIM * 2
SCALE_BYTES = 8
TOKEN_DATA_BYTES = KV_NOPE_DIM + ROPE_BYTES
BYTES_PER_TOKEN = TOKEN_DATA_BYTES + SCALE_BYTES
PAGE_BLOCK_SIZE = 64
SWA_WIDTH = 128


def make_footer_kv_cache(
    num_slots: int,
    *,
    block_size: int = PAGE_BLOCK_SIZE,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Build a DSv4 footer-layout sparse MLA cache.

    The logical token payload is 584 bytes, but both FlashInfer SM120 DSV4 and
    the old Triton reference use a block footer layout:

    - token data: ``block_base + local_idx * 576``
    - token scales: ``block_base + block_size * 576 + local_idx * 8``

    The returned tensor has shape ``[num_blocks, block_size, 584]`` and custom
    strides so kernel pointer arithmetic sees the footer layout.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    num_blocks = max(1, math.ceil(num_slots / block_size))
    storage = torch.empty(
        num_blocks,
        block_size * BYTES_PER_TOKEN,
        device=device,
        dtype=torch.uint8,
    )

    fp8_values = torch.randn(
        num_blocks,
        block_size,
        KV_NOPE_DIM,
        device=device,
        dtype=torch.float16,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    rope_values = torch.randn(
        num_blocks,
        block_size,
        QK_ROPE_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )

    for block_idx in range(num_blocks):
        block = storage[block_idx]
        for local_idx in range(block_size):
            data_offset = local_idx * TOKEN_DATA_BYTES
            scale_offset = block_size * TOKEN_DATA_BYTES + local_idx * SCALE_BYTES
            block[data_offset : data_offset + KV_NOPE_DIM] = fp8_values[
                block_idx, local_idx
            ].view(torch.uint8)
            block[
                data_offset + KV_NOPE_DIM : data_offset + TOKEN_DATA_BYTES
            ] = rope_values[block_idx, local_idx].view(torch.uint8)
            # UE8M0 encoded scale: 127 means scale 1.0.
            block[scale_offset : scale_offset + SCALE_BYTES].fill_(127)

    return torch.as_strided(
        storage,
        size=(num_blocks, block_size, BYTES_PER_TOKEN),
        stride=(block_size * BYTES_PER_TOKEN, TOKEN_DATA_BYTES, 1),
    )


def make_topk_indices(
    num_tokens: int,
    num_slots: int,
    topk: int,
    *,
    device: str = "cuda",
    seed: int = 1,
) -> torch.Tensor:
    """Build deterministic per-token global slot ids."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.stack(
        [
            torch.randperm(num_slots, device=device, generator=generator)[:topk]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)


def make_queries(
    num_tokens: int,
    num_heads: int,
    *,
    device: str = "cuda",
    seed: int = 2,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(
        num_tokens,
        num_heads,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )


def bench_cuda(fn, *, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters
