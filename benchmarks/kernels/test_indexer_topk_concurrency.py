# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-batch / concurrency oracle for the SM89 DSv4 indexer top-k chain.

The sparse-MLA backends (Triton and FlashInfer) share one model-level
``topk_indices_buffer`` plus the per-token C128A compressed lens, both
produced by the indexer top-k chain:

* decode: ``fp8_paged_mqa_logits_triton`` + ``persistent_topk``
  (topk in 512/1024/2048, the SM89 production path) or
  ``top_k_per_row_decode`` (other widths);
* prefill: ``fp8_mqa_logits_triton`` + ``top_k_per_row_prefill``;
* C128A: ``build_c128a_topk_metadata`` (``c128a_decode_topk_lens`` +
  global/prefill compressed indices).

This suite stresses the exact conditions a production multi-batch step
exercises and checks every output row against an independent fp32 torch
reference:

* varying per-row context lengths (0..W), -1 padding, capacity collisions;
* chunked decode (native MTP 2D lengths, SM89 flatten expansion, and the
  ``requires_padding`` pack/unpack round-trip);
* prefill chunks with non-zero ``cu_seqlen_ks``, window lengths on both
  sides of ``topk`` (shortcut vs histogram paths), and empty windows;
* the shared ``topk_indices_buffer`` reused sequentially by several
  "layers" with varying row counts, and concurrent disjoint writes from
  multiple CUDA streams (async races on the shared buffer);
* a side-stream reader snapshot of the buffer joined before the next
  "layer" overwrites the same rows (the production MLA join protocol);
* CUDA-graph capture + replay with mutated inputs each replay;
* ``max_context_len`` clipping and indexer block-column variants;
* the C128A lens/indices kernel including slot-mapping invalidation and
  ``max_compressed_tokens`` clipping.

The kernel top-k output order is unspecified (insertion sort / radix /
atomic scatter), so rows are compared as sets: no strictly-better column
may be omitted, no strictly-worse column may be selected, and the valid
count must equal ``min(row_len, topk)``.

Run on gserver (L40S)::

    python benchmarks/kernels/test_indexer_topk_concurrency.py --scenario all
"""

import argparse
import math
import os

import torch

_HEAD_DIM = 128
_BLOCK_SIZE = 64
_FP8_MIN, _FP8_MAX = -448.0, 448.0
_RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
_PERSISTENT_TOPK = (512, 1024, 2048)
_INDEXER_BLOCK_COL_ENV = "VLLM_SM89_INDEXER_BLOCK_COL"


def _quant_fp8(x: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-group ue8m0 quantizer matching the indexer Q/K kernels."""
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))
    q = (x / scale).clamp(_FP8_MIN, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def _make_indexer_cache(
    num_blocks: int,
    device: torch.device,
    seed: int,
    score_scale: float = 1.0,
    exp_range: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed indexer KV cache + fp8 values + per-token scales.

    Layout matches ``indexer_k_quant_and_cache`` / ``fp8_paged_mqa_logits_triton``:
    per block ``[block_size * head_dim]`` fp8 data bytes followed by
    ``[block_size * 4]`` fp32 scale bytes. Torch shape
    ``[num_blocks, block_size, 1, head_dim + 4]`` uint8.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = num_blocks * _BLOCK_SIZE
    k_bf16 = torch.randn(n, _HEAD_DIM, generator=gen) * 0.05 * score_scale
    # ue8m0 pow2 per-token scales (indexer cache stores fp32 values).
    exp = torch.randint(-exp_range, exp_range + 1, (n,), generator=gen)
    k_scale = torch.exp2(exp.to(torch.float32))
    k_fp8 = (k_bf16 / k_scale[:, None]).clamp(_FP8_MIN, _FP8_MAX).to(
        torch.float8_e4m3fn
    )

    cache = torch.zeros(
        num_blocks, _BLOCK_SIZE, 1, _HEAD_DIM + 4, dtype=torch.uint8
    )
    flat = cache.view(num_blocks, -1)
    flat[:, : _BLOCK_SIZE * _HEAD_DIM].view(
        num_blocks, _BLOCK_SIZE, _HEAD_DIM
    ).copy_(k_fp8.view(torch.uint8).view(num_blocks, _BLOCK_SIZE, _HEAD_DIM))
    flat[:, _BLOCK_SIZE * _HEAD_DIM :].view(num_blocks, _BLOCK_SIZE, 4).copy_(
        k_scale.view(num_blocks, _BLOCK_SIZE, 1, 1)
        .contiguous()
        .view(torch.uint8)
        .view(num_blocks, _BLOCK_SIZE, 4)
    )
    return cache.to(device), k_fp8.to(device), k_scale.to(device)


def _make_q(
    rows: int,
    H: int,
    device: torch.device,
    seed: int,
    score_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer Q: fp8 [rows, H, D] + folded weights [rows, H]."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_bf16 = (torch.randn(rows, H, _HEAD_DIM, generator=gen) * 0.05 * score_scale)
    q_fp8, q_scale = _quant_fp8(q_bf16.float(), dim=-1)
    w = torch.randn(rows, H, generator=gen)
    weights = (w * q_scale.squeeze(-1) * float(_HEAD_DIM**-0.5)).float()
    return q_fp8.to(device), weights.to(device)


def _ref_decode_windows(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens2d: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-row fp32 score windows for the paged decode contract."""
    B, next_n, H, _ = q_fp8.shape
    qf = q_fp8.float()
    wf = weights.float().reshape(B, next_n, H)
    windows: list[torch.Tensor] = []
    for b in range(B):
        max_l = int(seq_lens2d[b].max().item())
        if max_l <= 0:
            windows.extend(
                [torch.empty(0, dtype=torch.float32, device=q_fp8.device)]
                * next_n
            )
            continue
        nb = math.ceil(max_l / _BLOCK_SIZE)
        pages = block_table[b, :nb].long()
        token_offs = (
            pages[:, None] * _BLOCK_SIZE
            + torch.arange(_BLOCK_SIZE, device=pages.device)[None, :]
        )
        k_eff = k_fp8[token_offs.reshape(-1)].float()
        s_eff = k_scale[token_offs.reshape(-1)]
        dots = torch.einsum("jhd,nd->jhn", qf[b], k_eff)
        scores = torch.einsum("jh,jhn->jn", wf[b], torch.relu(dots)) * s_eff[None, :]
        for j in range(next_n):
            length = int(seq_lens2d[b, j].item())
            windows.append(scores[j, :length])
    return windows


def _ref_prefill_windows(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    cu_ks: torch.Tensor,
    cu_ke: torch.Tensor,
) -> list[torch.Tensor]:
    """Per-row fp32 score windows for the dense prefill contract."""
    qf = q_fp8.float()
    dots = torch.einsum("rhd,nd->rhn", qf, k_fp8.float())
    scores = (
        torch.einsum("rh,rhn->rn", weights.float(), torch.relu(dots))
        * k_scale[None, :]
    )
    return [
        scores[r, int(cu_ks[r].item()) : int(cu_ke[r].item())]
        for r in range(q_fp8.shape[0])
    ]


def _check_row(
    scores: torch.Tensor,
    out_row: torch.Tensor,
    topk: int,
    eps_rel: float = 1e-3,
) -> tuple[bool, str]:
    """Robust top-k set equivalence for one row (unspecified output order)."""
    out_row = out_row[:topk]
    valid = out_row >= 0
    n = int(scores.numel())
    kk = min(n, topk)
    if int(valid.sum().item()) != kk:
        return False, f"valid count {int(valid.sum().item())} != ref {kk}"
    have = out_row[valid]
    if n == 0:
        return True, ""
    if n <= topk:
        got = set(have.tolist())
        if got != set(range(n)) or len(got) != kk:
            return False, f"shortcut set mismatch (got {sorted(got)[:8]})"
        return True, ""
    chosen = scores[have]
    thr = torch.topk(scores, topk).values[-1]
    scale = float(scores.abs().max())
    eps = eps_rel * max(scale, 1e-6)
    if bool((chosen < thr - eps).any().item()):
        worst = float((thr - chosen).clamp(min=0).max().item())
        return False, f"selected below threshold by {worst:.3e}"
    got = set(have.tolist())
    must = set((scores > thr + eps).nonzero().flatten().tolist())
    if not must <= got:
        return False, f"omitted strictly-better indices {sorted(must - got)[:8]}"
    if len(got) != kk:
        return False, "duplicate indices"
    return True, ""


def _run_decode_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_lens2d: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
    buf_view: torch.Tensor,
    max_model_len: int,
    *,
    max_context_len: int | None = None,
    workspace: torch.Tensor | None = None,
    block_col: int | None = None,
) -> None:
    """Indexer decode top-k, mirroring sparse_attn_indexer's decode branch."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.triton_fp8_mqa_logits import (
        fp8_paged_mqa_logits_triton,
    )

    old_env = os.environ.get(_INDEXER_BLOCK_COL_ENV)
    if block_col is not None:
        os.environ[_INDEXER_BLOCK_COL_ENV] = str(block_col)
    try:
        logits = fp8_paged_mqa_logits_triton(
            q_fp8,
            kv_cache,
            weights,
            seq_lens2d,
            block_table,
            max_model_len,
            max_context_len=max_context_len,
        )
    finally:
        if block_col is not None:
            if old_env is None:
                os.environ.pop(_INDEXER_BLOCK_COL_ENV, None)
            else:
                os.environ[_INDEXER_BLOCK_COL_ENV] = old_env

    rows = logits.shape[0]
    next_n = seq_lens2d.shape[1]
    if topk in _PERSISTENT_TOPK:
        if workspace is None:
            workspace = torch.empty(
                _RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=logits.device
            )
        torch.ops._C.persistent_topk(
            logits,
            seq_lens2d.reshape(-1).contiguous(),
            buf_view,
            workspace,
            topk,
            logits.shape[1],
        )
    else:
        ops.top_k_per_row_decode(
            logits,
            next_n,
            seq_lens2d,
            buf_view,
            rows,
            logits.stride(0),
            logits.stride(1),
            topk,
        )


def _run_prefill_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    cu_ks: torch.Tensor,
    cu_ke: torch.Tensor,
    topk: int,
    buf_view: torch.Tensor,
) -> None:
    """Indexer prefill top-k, mirroring sparse_attn_indexer's prefill branch."""
    from vllm import _custom_ops as ops
    from vllm.v1.attention.ops.triton_fp8_mqa_logits import fp8_mqa_logits_triton

    logits = fp8_mqa_logits_triton(
        q_fp8, k_fp8, k_scale, weights, cu_ks, cu_ke
    )
    rows = logits.shape[0]
    ops.top_k_per_row_prefill(
        logits,
        cu_ks,
        cu_ke,
        buf_view,
        rows,
        logits.stride(0),
        logits.stride(1),
        topk,
    )


class DecodeCase:
    """One decode indexer step (logits + top-k) with a per-row reference."""

    def __init__(
        self,
        seed: int,
        *,
        B: int,
        next_n: int,
        N: int,
        topk: int,
        H: int = 16,
        lens_mode: str = "random",
        score_scale: float = 1.0,
    ):
        self.dev = torch.device("cuda:0")
        self.B, self.next_n, self.N, self.topk, self.H = B, next_n, N, topk, H
        self.rows = B * next_n
        self.max_model_len = N
        self.lens_mode = lens_mode
        g = torch.Generator(device="cpu").manual_seed(seed)

        num_blocks = math.ceil(N / _BLOCK_SIZE)
        self.cache, self.k_fp8, self.k_scale = _make_indexer_cache(
            num_blocks, self.dev, seed, score_scale
        )

        seq_lens = self._sample_seq_lens(g)
        self.seq_lens2d = seq_lens.to(self.dev)

        bt = torch.zeros(B, num_blocks, dtype=torch.int32)
        for b in range(B):
            nb = math.ceil(int(seq_lens[b].max().item()) / _BLOCK_SIZE)
            if nb > 0:
                bt[b, :nb] = torch.randperm(num_blocks, generator=g)[:nb]
        self.block_table = bt.to(self.dev)

        q_fp8, weights = _make_q(self.rows, H, self.dev, seed + 11, score_scale)
        self.q_fp8 = q_fp8.reshape(B, next_n, H, _HEAD_DIM)
        self.weights = weights
        self.ref_windows = _ref_decode_windows(
            self.q_fp8,
            self.weights,
            self.k_fp8,
            self.k_scale,
            self.block_table,
            self.seq_lens2d,
        )
        self.g = g

    def _sample_seq_lens(self, g: torch.Generator) -> torch.Tensor:
        """Per-token context lengths for the configured lens mode (CPU)."""
        B, next_n, N, topk = self.B, self.next_n, self.N, self.topk
        if self.lens_mode == "full":
            lens = torch.full((B,), N, dtype=torch.int64)
        elif self.lens_mode == "short":
            lens = torch.randint(0, min(topk, N) + 1, (B,), generator=g)
        elif self.lens_mode == "native":
            lens = torch.randint(next_n, N + 1, (B,), generator=g)
            return (
                lens[:, None] - next_n + 1 + torch.arange(next_n)[None, :]
            ).to(torch.int32)
        else:
            lens = torch.randint(0, N + 1, (B,), generator=g)
        return lens[:, None].expand(B, next_n).clone().to(torch.int32)

    def run(
        self,
        buf_view: torch.Tensor,
        *,
        workspace: torch.Tensor | None = None,
        max_context_len: int | None = None,
        block_col: int | None = None,
    ) -> None:
        _run_decode_topk(
            self.q_fp8,
            self.weights,
            self.cache,
            self.seq_lens2d,
            self.block_table,
            self.topk,
            buf_view,
            self.max_model_len,
            max_context_len=max_context_len,
            workspace=workspace,
            block_col=block_col,
        )

    def refresh(self, seed: int) -> None:
        """Mutate contents in-place (same addresses) + recompute reference."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        B, next_n, N = self.B, self.next_n, self.N
        num_blocks = math.ceil(N / _BLOCK_SIZE)
        cache, k_fp8, k_scale = _make_indexer_cache(num_blocks, self.dev, seed, 1.0)
        self.cache.view(-1).copy_(cache.view(-1))
        self.k_fp8.copy_(k_fp8)
        self.k_scale.copy_(k_scale)

        seq_lens = self._sample_seq_lens(g)
        bt = torch.zeros(B, num_blocks, dtype=torch.int32)
        for b in range(B):
            nb = math.ceil(int(seq_lens[b].max().item()) / _BLOCK_SIZE)
            if nb > 0:
                bt[b, :nb] = torch.randperm(num_blocks, generator=g)[:nb]
        self.block_table.copy_(bt)
        self.seq_lens2d.copy_(seq_lens)

        q_fp8, weights = _make_q(self.rows, self.H, self.dev, seed + 11, 1.0)
        self.q_fp8.copy_(q_fp8.reshape(B, next_n, self.H, _HEAD_DIM))
        self.weights.copy_(weights)
        self.ref_windows = _ref_decode_windows(
            self.q_fp8,
            self.weights,
            self.k_fp8,
            self.k_scale,
            self.block_table,
            self.seq_lens2d,
        )

    def check(self, buf_view: torch.Tensor, tag: str, max_print: int = 3) -> int:
        nfail = 0
        for r in range(self.rows):
            ok, why = _check_row(self.ref_windows[r], buf_view[r], self.topk)
            if not ok:
                nfail += 1
                if nfail <= max_print:
                    print(f"  [FAIL {tag}] row {r}: {why}")
        return nfail


class PrefillCase:
    """One prefill indexer step with per-row reference windows."""

    def __init__(
        self,
        seed: int,
        *,
        rows: int,
        N: int,
        topk: int,
        H: int = 16,
        window_max: int = 2048,
    ):
        self.dev = torch.device("cuda:0")
        self.rows, self.N, self.topk, self.H = rows, N, topk, H
        g = torch.Generator(device="cpu").manual_seed(seed)
        k_bf16 = torch.randn(N, _HEAD_DIM, generator=g) * 0.05
        k_fp8, k_scale = _quant_fp8(k_bf16.float(), dim=-1)
        self.k_fp8 = k_fp8.to(self.dev)
        self.k_scale = k_scale.squeeze(-1).to(self.dev)

        # Windows: mix of >topk, <topk, ==topk and empty rows, non-zero starts.
        starts = torch.zeros(rows, dtype=torch.int64)
        ends = torch.zeros(rows, dtype=torch.int64)
        cursor = 0
        for r in range(rows):
            length = int(
                torch.randint(0, window_max + 1, (1,), generator=g).item()
                if r % 3
                else torch.randint(topk // 2, topk * 2, (1,), generator=g).item()
            )
            length = min(length, N - cursor)
            starts[r] = cursor
            ends[r] = cursor + length
            cursor += length + int(
                torch.randint(0, 33, (1,), generator=g).item()
            )  # gaps
            if cursor >= N:
                break
        ends = ends.clamp(max=N)
        self.cu_ks = starts.to(torch.int32).to(self.dev)
        self.cu_ke = ends.to(torch.int32).to(self.dev)

        q_fp8, weights = _make_q(rows, H, self.dev, seed + 31, 1.0)
        self.q_fp8 = q_fp8
        self.weights = weights
        self.ref_windows = _ref_prefill_windows(
            self.q_fp8, self.weights, self.k_fp8, self.k_scale, self.cu_ks, self.cu_ke
        )

    def run(self, buf_view: torch.Tensor) -> None:
        _run_prefill_topk(
            self.q_fp8,
            self.weights,
            self.k_fp8,
            self.k_scale,
            self.cu_ks,
            self.cu_ke,
            self.topk,
            buf_view,
        )

    def check(self, buf_view: torch.Tensor, tag: str, max_print: int = 3) -> int:
        nfail = 0
        for r in range(self.rows):
            ok, why = _check_row(self.ref_windows[r], buf_view[r], self.topk)
            if not ok:
                nfail += 1
                if nfail <= max_print:
                    print(f"  [FAIL {tag}] row {r}: {why}")
        return nfail


def scenario_multibatch(n_rep: int = 2) -> None:
    """Repeated steps with varying lens/-1 padding/capacity collisions."""
    nfail = 0
    specs = [
        dict(B=16, next_n=1, N=8192, topk=512, lens_mode="random"),
        dict(B=8, next_n=1, N=16384, topk=64, lens_mode="random"),  # decode radix
        dict(B=8, next_n=1, N=8192, topk=512, lens_mode="full"),
        dict(B=16, next_n=1, N=8192, topk=128, lens_mode="short"),
        dict(B=4, next_n=4, N=4096, topk=512, lens_mode="native"),
        dict(B=8, next_n=1, N=4096, topk=1024, lens_mode="random"),
        dict(B=16, next_n=1, N=4096, topk=2048, lens_mode="random"),
    ]
    for seed in range(4):
        for spec in specs:
            c = DecodeCase(seed * 100 + 7, **spec)
            buf = torch.full((c.rows, c.topk), -1, dtype=torch.int32, device=c.dev)
            for rep in range(n_rep):
                if rep > 0:
                    c.refresh(seed * 100 + rep * 17 + 3)
                c.run(buf, max_context_len=c.N)
                torch.cuda.synchronize()
                nfail += c.check(buf, f"seed={seed} rep={rep} {spec}")
        if seed % 2 == 0:
            print(f"  multibatch seed={seed} ok")
    print(f"multibatch: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_radix() -> None:
    """Cooperative radix branch of persistent_topk (max seq > 32768)."""
    nfail = 0
    for seed in range(2):
        c = DecodeCase(
            900 + seed, B=2, next_n=1, N=65536, topk=512, H=8, lens_mode="random"
        )
        buf = torch.full((c.rows, c.topk), -1, dtype=torch.int32, device=c.dev)
        c.run(buf, max_context_len=c.N)
        torch.cuda.synchronize()
        nfail += c.check(buf, f"radix seed={seed}")
    print(f"radix: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_padding() -> None:
    """requires_padding pack/unpack round-trip (chunked decode alignment)."""
    from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

    nfail = 0
    dev = torch.device("cuda:0")
    for seed in range(3):
        B_real = 4
        B_pad = 2
        B = B_real + B_pad
        topk = 512
        N = 4096
        g = torch.Generator(device="cpu").manual_seed(seed)
        lens = torch.randint(1, N + 1, (B_real,), generator=g)
        decode_lens = torch.cat(
            [
                torch.ones(B_real, dtype=torch.int64),
                torch.zeros(B_pad, dtype=torch.int64),
            ]
        )
        seq_lens = torch.cat([lens, torch.zeros(B_pad, dtype=torch.int64)])
        seq_lens2d = seq_lens[:, None].to(torch.int32).to(dev)

        num_blocks = math.ceil(int(lens.max().item()) / _BLOCK_SIZE)
        cache, k_fp8, k_scale = _make_indexer_cache(num_blocks, dev, seed + 40, 1.0)
        bt = torch.zeros(B, num_blocks, dtype=torch.int32)
        for b in range(B_real):
            nb = math.ceil(int(lens[b].item()) / _BLOCK_SIZE)
            bt[b, :nb] = torch.randperm(num_blocks, generator=g)[:nb]
        block_table = bt.to(dev)

        q_fp8, weights = _make_q(B_real, 16, dev, seed + 50, 1.0)
        packed = pack_seq_triton(
            q_fp8, decode_lens.to(torch.int32).to(dev), pad_value=0
        ).reshape(B, 16, _HEAD_DIM)
        packed_weights = torch.zeros(B, 16, dtype=torch.float32, device=dev)
        packed_weights[:B_real] = weights

        buf = torch.full((B * topk,), -1, dtype=torch.int32, device=dev).view(B, topk)
        _run_decode_topk(
            packed.reshape(B, 1, 16, _HEAD_DIM),
            packed_weights,
            cache,
            seq_lens2d,
            block_table,
            topk,
            buf,
            N,
            max_context_len=int(lens.max().item()),
        )
        unpacked = unpack_seq_triton(
            buf.reshape(B, 1, topk), decode_lens.to(torch.int32).to(dev)
        )
        buf[:B_real].copy_(unpacked)
        torch.cuda.synchronize()

        # Reference: per-real-token windows over [0, L).
        ref = _ref_decode_windows(
            q_fp8.reshape(B_real, 1, 16, _HEAD_DIM),
            weights,
            k_fp8,
            k_scale,
            block_table[:B_real],
            seq_lens2d[:B_real],
        )
        fails = 0
        for r in range(B_real):
            ok, why = _check_row(ref[r], buf[r], topk)
            if not ok:
                fails += 1
                if fails <= 3:
                    print(f"  [FAIL padding seed={seed}] row {r}: {why}")
        nfail += fails
    print(f"padding: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_spec_flatten() -> None:
    """SM89 flatten path: multi-token decode requests expanded per token."""
    nfail = 0
    dev = torch.device("cuda:0")
    for seed in range(3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        reqs = 4
        topk = 512
        N = 4096
        seq_lens = torch.randint(2, N + 1, (reqs,), generator=g)
        decode_lens = torch.randint(1, 4, (reqs,), generator=g)
        decode_lens = torch.minimum(decode_lens, seq_lens)
        query_start_loc = torch.zeros(reqs + 1, dtype=torch.int64)
        torch.cumsum(decode_lens, dim=0, out=query_start_loc[1:])
        num_decode_tokens = int(query_start_loc[-1].item())

        expanded_offsets = torch.repeat_interleave(
            seq_lens - decode_lens - query_start_loc[:reqs], decode_lens
        )
        per_token_lens = expanded_offsets + torch.arange(num_decode_tokens) + 1
        seq_lens2d = per_token_lens[:, None].to(torch.int32).to(dev)

        num_blocks = math.ceil(N / _BLOCK_SIZE)
        cache, k_fp8, k_scale = _make_indexer_cache(num_blocks, dev, seed + 60, 1.0)
        bt = torch.zeros(reqs, num_blocks, dtype=torch.int32)
        for b in range(reqs):
            nb = math.ceil(int(seq_lens[b].item()) / _BLOCK_SIZE)
            bt[b, :nb] = torch.randperm(num_blocks, generator=g)[:nb]
        block_table = torch.repeat_interleave(
            bt, decode_lens, dim=0, output_size=num_decode_tokens
        ).to(dev)

        q_fp8, weights = _make_q(num_decode_tokens, 16, dev, seed + 70, 1.0)
        buf = torch.full((num_decode_tokens, topk), -1, dtype=torch.int32, device=dev)
        _run_decode_topk(
            q_fp8.reshape(num_decode_tokens, 1, 16, _HEAD_DIM),
            weights,
            cache,
            seq_lens2d,
            block_table,
            topk,
            buf,
            N,
            max_context_len=int(per_token_lens.max().item()),
        )
        torch.cuda.synchronize()

        ref = _ref_decode_windows(
            q_fp8.reshape(num_decode_tokens, 1, 16, _HEAD_DIM),
            weights,
            k_fp8,
            k_scale,
            block_table,
            seq_lens2d,
        )
        fails = 0
        for r in range(num_decode_tokens):
            ok, why = _check_row(ref[r], buf[r], topk)
            if not ok:
                fails += 1
                if fails <= 3:
                    print(f"  [FAIL flatten seed={seed}] row {r}: {why}")
        nfail += fails
    print(f"spec-flatten: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_prefill() -> None:
    """Prefill chunks: non-zero windows, shortcut/main paths, shared buffer."""
    nfail = 0
    dev = torch.device("cuda:0")
    for seed in range(3):
        c = PrefillCase(800 + seed, rows=8, N=16384, topk=512)
        buf = torch.full((16, c.topk), -1, dtype=torch.int32, device=dev)
        c.run(buf[:8])
        torch.cuda.synchronize()
        nfail += c.check(buf[:8], f"prefill seed={seed} chunk0")

        # Second chunk at a later token offset (chunked prefill).
        c2 = PrefillCase(900 + seed, rows=6, N=16384, topk=512, window_max=1024)
        c2.run(buf[8:14])
        torch.cuda.synchronize()
        nfail += c2.check(buf[8:14], f"prefill seed={seed} chunk1")

        # Empty chunk: sparse_attn_indexer fills -1 explicitly.
        buf[14:16].fill_(-1)
        if not bool((buf[14:16] == -1).all().item()):
            nfail += 1
            print(f"  [FAIL prefill seed={seed}] empty chunk not -1")
    print(f"prefill: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_streams(n_rounds: int = 4) -> None:
    """Concurrent decode top-k on 4 streams, disjoint shared-buffer regions."""
    dev = torch.device("cuda:0")
    streams = [torch.cuda.Stream(device=dev) for _ in range(4)]
    B_each, topk, N = 8, 512, 8192
    # JIT-warm the kernels on one stream before racing four streams on them.
    warm = DecodeCase(1, B=1, next_n=1, N=256, topk=topk, lens_mode="short")
    warm_buf = torch.full((1, topk), -1, dtype=torch.int32, device=dev)
    warm_ws = torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
    warm.run(warm_buf, workspace=warm_ws, max_context_len=256)
    torch.cuda.synchronize()

    buf = torch.full((4 * B_each, topk), -1, dtype=torch.int32, device=dev)
    cases = [
        DecodeCase(1000 + i, B=B_each, next_n=1, N=N, topk=topk, lens_mode="random")
        for i in range(4)
    ]
    workspaces = [
        torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
        for _ in range(4)
    ]
    nfail = 0
    for rnd in range(n_rounds):
        evs = []
        for i, (s, c) in enumerate(zip(streams, cases)):
            with torch.cuda.stream(s):
                ev = torch.cuda.Event()
                c.run(
                    buf[i * B_each : (i + 1) * B_each],
                    workspace=workspaces[i],
                    max_context_len=N,
                )
                ev.record()
                evs.append(ev)
        torch.cuda.synchronize()
        for i, c in enumerate(cases):
            nfail += c.check(
                buf[i * B_each : (i + 1) * B_each], f"stream rnd={rnd} i={i}"
            )
        for i, c in enumerate(cases):
            c.refresh(5000 + rnd * 100 + i)
        torch.cuda.synchronize()
    print(f"streams: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_layers() -> None:
    """Sequential cross-layer reuse of the shared buffer with varying rows."""
    nfail = 0
    dev = torch.device("cuda:0")
    topk, N = 512, 8192
    buf = torch.full((64, topk), -1, dtype=torch.int32, device=dev)
    # Row counts vary across "layers"; a later layer must not see stale rows.
    row_counts = [16, 8, 16, 4, 16, 0]
    for seed in range(3):
        for i, rows in enumerate(row_counts):
            if rows == 0:
                continue
            c = DecodeCase(
                7000 + seed * 100 + i,
                B=rows,
                next_n=1,
                N=N,
                topk=topk,
                lens_mode="random",
            )
            c.run(buf[:rows], max_context_len=N)
            torch.cuda.synchronize()
            nfail += c.check(buf[:rows], f"layers seed={seed} layer={i}")
    print(f"layers: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_join() -> None:
    """Side-stream reader of topk must join before the next layer overwrites."""
    dev = torch.device("cuda:0")
    nfail = 0
    for seed in range(2):
        topk, N = 512, 8192
        buf = torch.full((8, topk), -1, dtype=torch.int32, device=dev)
        # Initialize snap before a.run so its writes are stream-ordered ahead
        # of the side-stream copy (no cross-stream init/copy race).
        snap = torch.full_like(buf, -1)
        a = DecodeCase(
            6000 + seed, B=8, next_n=1, N=N, topk=topk, lens_mode="random"
        )
        a.run(buf, max_context_len=N)
        main_ev = torch.cuda.Event()
        main_ev.record()

        s1 = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s1):
            s1.wait_event(main_ev)
            snap.copy_(buf)
        torch.cuda.synchronize()
        nfail += a.check(snap, f"join seed={seed} snapshot")

        b = DecodeCase(
            6100 + seed, B=8, next_n=1, N=N, topk=topk, lens_mode="random"
        )
        b.run(buf, max_context_len=N)
        torch.cuda.synchronize()
        nfail += b.check(buf, f"join seed={seed} layer B")
    print(f"join: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_graph() -> None:
    """CUDA-graph capture + replay with mutated inputs each replay."""
    dev = torch.device("cuda:0")
    nfail = 0
    for seed in range(2):
        c = DecodeCase(
            2000 + seed, B=8, next_n=1, N=4096, topk=512, lens_mode="random"
        )
        buf = torch.full((c.rows, c.topk), -1, dtype=torch.int32, device=dev)
        ws = torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)

        def fn() -> None:
            c.run(buf, workspace=ws, max_context_len=c.N)

        fn()  # warmup / JIT outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        for rep in range(3):
            c.refresh(3000 + seed * 10 + rep)
            g.replay()
            torch.cuda.synchronize()
            nfail += c.check(buf, f"graph seed={seed} rep={rep}")
    print(f"graph: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_clipping() -> None:
    """max_context_len / block-col variants must not change the top-k output."""
    nfail = 0
    dev = torch.device("cuda:0")
    c = DecodeCase(4000, B=8, next_n=1, N=8192, topk=512, lens_mode="random")
    ws = torch.empty(_RADIX_TOPK_WORKSPACE_SIZE, dtype=torch.uint8, device=dev)
    outs = {}
    for key, kwargs in [
        ("exact", dict(max_context_len=c.N, block_col=None)),
        ("unclipped", dict(max_context_len=None, block_col=None)),
        ("bc128", dict(max_context_len=c.N, block_col=128)),
        ("bc256", dict(max_context_len=c.N, block_col=256)),
    ]:
        buf = torch.full((c.rows, c.topk), -1, dtype=torch.int32, device=dev)
        c.run(buf, workspace=ws, **kwargs)
        torch.cuda.synchronize()
        outs[key] = buf.clone()
        nfail += c.check(buf, f"clipping {key}")
    for key, out in outs.items():
        if key == "exact":
            continue
        # Output order is unspecified; compare per-row index sets.
        exact_sets = [set(r[r >= 0].tolist()) for r in outs["exact"]]
        out_sets = [set(r[r >= 0].tolist()) for r in out]
        if exact_sets != out_sets:
            nfail += 1
            print(f"  [FAIL clipping] {key} differs from exact")
    print(f"clipping: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def _ref_c128a(
    positions: torch.Tensor,
    compress_ratio: int,
    num_decode_tokens: int,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    max_compressed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for build_c128a_topk_metadata."""
    num_tokens = positions.numel()
    num = ((positions + 1) // compress_ratio).clamp(max=max_compressed)
    decode_lens = torch.zeros(num_decode_tokens, dtype=torch.int32)
    global_decode = torch.full(
        (num_decode_tokens, max_compressed), -1, dtype=torch.int32
    )
    for t in range(num_decode_tokens):
        if slot_mapping[t] >= 0:
            decode_lens[t] = num[t]
        req = int(token_to_req[t].item())
        for col in range(int(num[t].item())):
            block_numbers = int(block_table[req, col // block_size].item())
            global_decode[t, col] = block_numbers * block_size + col % block_size
    prefill_local = torch.full(
        (num_tokens - num_decode_tokens, max_compressed), -1, dtype=torch.int32
    )
    for t in range(num_decode_tokens, num_tokens):
        p = t - num_decode_tokens
        prefill_local[p, : int(num[t].item())] = torch.arange(
            int(num[t].item()), dtype=torch.int32
        )
    return global_decode, decode_lens, prefill_local


def scenario_c128a() -> None:
    """C128A compressed topk lens/indices against the torch reference."""
    from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

    nfail = 0
    dev = torch.device("cuda:0")
    for seed in range(3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        num_decode, num_prefill = 7, 5
        num_tokens = num_decode + num_prefill
        compress_ratio, block_size = 128, 16
        max_pos = 20000
        positions = torch.randint(0, max_pos, (num_tokens,), generator=g)
        slot_mapping = torch.cat(
            [
                torch.randint(0, 50000, (num_decode,), generator=g),
                torch.full((num_prefill,), -1, dtype=torch.int64),
            ]
        )
        token_to_req = torch.cat(
            [
                torch.randint(0, num_decode, (num_decode,), generator=g),
                torch.randint(0, num_decode, (num_prefill,), generator=g),
            ]
        )
        width = math.ceil(max_pos / compress_ratio / block_size) + 1
        block_table = torch.randint(0, 4000, (num_decode, width), generator=g)
        token_to_req = token_to_req.to(torch.int32)
        block_table = block_table.to(torch.int32)
        # Invalidate one decode row via slot_mapping=-1.
        slot_mapping[3] = -1
        max_compressed = 160

        global_buf = torch.zeros(
            num_decode, max_compressed, dtype=torch.int32, device=dev
        )
        lens_buf = torch.zeros(num_decode, dtype=torch.int32, device=dev)
        prefill_buf = torch.zeros(
            num_prefill, max_compressed, dtype=torch.int32, device=dev
        )
        build_c128a_topk_metadata(
            positions.to(dev),
            compress_ratio,
            num_decode,
            token_to_req.to(dev),
            block_table.to(dev),
            block_size,
            slot_mapping.to(dev),
            global_buf,
            lens_buf,
            prefill_buf,
            max_compressed_tokens=max_compressed,
        )
        torch.cuda.synchronize()
        g_ref, lens_ref, p_ref = _ref_c128a(
            positions,
            compress_ratio,
            num_decode,
            token_to_req,
            block_table,
            block_size,
            slot_mapping,
            max_compressed,
        )
        if not torch.equal(global_buf.cpu(), g_ref):
            nfail += 1
            print(f"  [FAIL c128a seed={seed}] global indices")
        if not torch.equal(lens_buf.cpu(), lens_ref):
            nfail += 1
            print(f"  [FAIL c128a seed={seed}] decode lens")
        if not torch.equal(prefill_buf.cpu(), p_ref):
            nfail += 1
            print(f"  [FAIL c128a seed={seed}] prefill local")

        # Sequential reuse with different positions must overwrite fully.
        positions2 = torch.randint(max_pos // 2, max_pos, (num_tokens,), generator=g)
        build_c128a_topk_metadata(
            positions2.to(dev),
            compress_ratio,
            num_decode,
            token_to_req.to(dev),
            block_table.to(dev),
            block_size,
            slot_mapping.to(dev),
            global_buf,
            lens_buf,
            prefill_buf,
            max_compressed_tokens=max_compressed,
        )
        torch.cuda.synchronize()
        g_ref2, lens_ref2, p_ref2 = _ref_c128a(
            positions2,
            compress_ratio,
            num_decode,
            token_to_req,
            block_table,
            block_size,
            slot_mapping,
            max_compressed,
        )
        if not (
            torch.equal(global_buf.cpu(), g_ref2)
            and torch.equal(lens_buf.cpu(), lens_ref2)
            and torch.equal(prefill_buf.cpu(), p_ref2)
        ):
            nfail += 1
            print(f"  [FAIL c128a seed={seed}] sequential reuse")
    print(f"c128a: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        choices=[
            "multibatch",
            "radix",
            "padding",
            "flatten",
            "prefill",
            "streams",
            "layers",
            "join",
            "graph",
            "clipping",
            "c128a",
            "all",
        ],
        default="all",
    )
    args = ap.parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9), (
        "this oracle targets the SM89 indexer fallback"
    )
    # Register the compiled vLLM ops (torch.ops._C) before probing for them.
    import vllm._C_stable_libtorch  # noqa: F401

    assert hasattr(torch.ops._C, "persistent_topk"), "persistent_topk missing in _C"
    sc = args.scenario
    if sc in ("multibatch", "all"):
        print("== multibatch ==")
        scenario_multibatch()
    if sc in ("radix", "all"):
        print("== radix ==")
        scenario_radix()
    if sc in ("padding", "all"):
        print("== padding ==")
        scenario_padding()
    if sc in ("flatten", "all"):
        print("== spec-flatten ==")
        scenario_spec_flatten()
    if sc in ("prefill", "all"):
        print("== prefill ==")
        scenario_prefill()
    if sc in ("streams", "all"):
        print("== streams ==")
        scenario_streams()
    if sc in ("layers", "all"):
        print("== layers ==")
        scenario_layers()
    if sc in ("join", "all"):
        print("== join ==")
        scenario_join()
    if sc in ("graph", "all"):
        print("== graph ==")
        scenario_graph()
    if sc in ("clipping", "all"):
        print("== clipping ==")
        scenario_clipping()
    if sc in ("c128a", "all"):
        print("== c128a ==")
        scenario_c128a()


if __name__ == "__main__":
    main()
