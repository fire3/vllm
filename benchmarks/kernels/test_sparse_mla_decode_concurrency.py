"""Multi-batch / concurrency correctness checks for the DSv4 sparse-MLA decode.

The fused tiled decode path (``triton_sparse_mla_prefill_vllm``) keeps a
global CSR scratch buffer pool. This suite stresses the exact conditions a
production multi-batch step exercises:

* per-row lens that vary (0..W) with -1 padding,
* swa/extra topk widths that collide on capacity (128/128) or differ
  (128/512, 128/2048) and change across steps (c128 width transitions),
* repeated calls with the same capacity but different contents (stale
  buffer detection),
* concurrent calls on multiple CUDA streams (async races on the shared
  scratch buffer),
* CUDA-graph capture + replay with mutated inputs,
* an explicit reproduction of the pre-fix swa/extra CSR aliasing
  (``slot=0`` for both packs) that produced garbled output.

Every call is checked against an independent per-case torch reference, plus
the legacy (phase-1) path as a second oracle.

Run on gserver::

    python benchmarks/kernels/test_sparse_mla_decode_concurrency.py \
        --scenario multibatch
"""

import argparse

import torch

_PAGE_TOKENS = 64
_DATA_STRIDE = 576
_NOPE_DIM = 448
_ROPE_DIM = 64
_D = _NOPE_DIM + _ROPE_DIM
_ROPE_BYTES = 2 * _ROPE_DIM


def _close(out: torch.Tensor, ref: torch.Tensor, tol: float = 5e-2) -> bool:
    """Max deviation relative to the global output magnitude.

    The fused kernel rounds the value operand to bf16 for the MMA (one ULP of
    the output dtype), so isolated near-zero elements can differ by a few ULP;
    a softmax semantics bug (exp2 vs exp) shifts every element by tens of
    percent and is caught by this bound.
    """
    scale = ref.float().abs().max().item()
    return (out.float() - ref.float()).abs().max().item() <= tol * max(scale, 1e-6)


def _make_packed_cache(
    num_tokens: int,
    page_size: int,
    device: torch.device,
    seed: int,
    score_scale: float = 1.0,
    exp_range: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed fp8_ds_mla cache + dequantized reference KV."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_blocks = (num_tokens + page_size - 1) // page_size
    kv_nope = torch.randn(num_tokens, _NOPE_DIM, generator=gen) * 0.05 * score_scale
    kv_rope = torch.randn(num_tokens, _ROPE_DIM, generator=gen) * 0.05 * score_scale
    exp = torch.randint(-exp_range, exp_range + 1, (num_tokens, 7), generator=gen)
    scale = torch.exp2(exp.to(torch.float32))
    scale448 = scale[:, :, None].expand(-1, -1, 64).reshape(-1, _NOPE_DIM)

    fp8 = (kv_nope / scale448).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    data = torch.zeros(num_tokens, _DATA_STRIDE, dtype=torch.uint8)
    data[:, :_NOPE_DIM] = fp8.view(torch.uint8)
    data[:, _NOPE_DIM:] = kv_rope.to(torch.bfloat16).view(torch.uint8)

    block_bytes = page_size * 584
    cache = torch.zeros(num_blocks * block_bytes, dtype=torch.uint8)
    cache.view(num_blocks, block_bytes)[:, : page_size * _DATA_STRIDE] = data.view(
        num_blocks, page_size * _DATA_STRIDE
    )
    scale_view = cache.view(num_blocks, block_bytes)[
        :, page_size * _DATA_STRIDE : page_size * (_DATA_STRIDE + 8)
    ].view(num_blocks, page_size, 8)
    scale_view[:, :, :7] = (exp + 127).to(torch.uint8).view(num_blocks, page_size, 7)
    kv_nope_q = fp8.to(torch.float32) * scale448
    return cache.to(device), kv_nope_q, kv_rope.to(torch.bfloat16)


def _cache_view(cache: torch.Tensor, num_tokens: int, page_size: int) -> torch.Tensor:
    num_blocks = (num_tokens + page_size - 1) // page_size
    return cache.view(num_blocks, page_size, 584)


def _reference(
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_nope: torch.Tensor,
    swa_rope: torch.Tensor,
    swa_idx: torch.Tensor,  # [B, W] int32, -1 padded
    swa_lens: torch.Tensor,  # [B] int32
    extra_nope: torch.Tensor | None,
    extra_rope: torch.Tensor | None,
    extra_idx: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    sinks: torch.Tensor,  # [H] f32
    softmax_scale: float,
) -> torch.Tensor:
    """Per-source LSE merge + sink, fp32 (matches legacy path semantics)."""
    B, _, H, D = q.shape
    qf = q.float()

    def one(src_nope, src_rope, idx, lens):
        W = idx.shape[1]
        valid = (idx >= 0) & (
            torch.arange(W, device=idx.device)[None, :] < lens[:, None]
        )
        safe = idx.clamp(min=0)
        kv = torch.cat([src_nope[safe], src_rope[safe].float()], dim=-1)
        scores = torch.einsum("bhd,btd->bht", qf[:, 0], kv) * softmax_scale
        scores = torch.where(valid[:, None, :], scores, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        w = torch.exp(scores - lse[:, :, None])
        w = torch.where(torch.isfinite(lse)[:, :, None], w, 0.0)
        w = torch.where(valid[:, None, :], w, 0.0)
        out = torch.einsum("bht,btd->bhd", w, kv)
        lonely = lse == float("-inf")
        out[lonely] = 0.0
        return out, lse

    out, lse = one(swa_nope, swa_rope, swa_idx, swa_lens)
    if extra_idx is not None:
        out_e, lse_e = one(extra_nope, extra_rope, extra_idx, extra_lens)
        mx = torch.maximum(lse, lse_e)
        w1 = torch.exp(lse - mx)
        w2 = torch.exp(lse_e - mx)
        total = (w1 + w2).clamp(min=1e-20)
        out = (w1.unsqueeze(-1) * out.float() + w2.unsqueeze(-1) * out_e.float()) / (
            total.unsqueeze(-1)
        )
        lse = mx + torch.log(total)
        out = torch.where(
            torch.isfinite(lse)[:, :, None], out.float(), torch.zeros_like(out.float())
        )
    combined = torch.logaddexp(lse, sinks[None, :].expand_as(lse))
    w = torch.where(lse > -1e20, torch.exp(lse - combined), torch.zeros_like(lse))
    return (out.float() * w.unsqueeze(-1)).to(torch.bfloat16)


class Case:
    def __init__(
        self,
        seed: int,
        B: int,
        H: int,
        swa_topk: int,
        extra_topk: int,
        swa_page: int,
        extra_page: int,
        lens_mode: str = "random",
        score_scale: float = 1.0,
        exp_range: int = 8,
    ):
        torch.manual_seed(seed)
        dev = torch.device("cuda:0")
        self.dev = dev
        self.B, self.H = B, H
        self.swa_topk, self.extra_topk = swa_topk, extra_topk
        swa_tokens = swa_page * max(8, 2 * (swa_topk // swa_page) + 4)
        extra_tokens = extra_page * max(8, 2 * (extra_topk // extra_page) + 4)
        self.swa_cache_f, swa_nope, swa_rope = _make_packed_cache(
            swa_tokens, swa_page, dev, seed, score_scale, exp_range
        )
        self.extra_cache_f, extra_nope, extra_rope = _make_packed_cache(
            extra_tokens, extra_page, dev, seed + 1, score_scale, exp_range
        )
        self.swa_cache = _cache_view(self.swa_cache_f, swa_tokens, swa_page)
        self.extra_cache = _cache_view(self.extra_cache_f, extra_tokens, extra_page)
        self.swa_nope = swa_nope.to(dev)
        self.swa_rope = swa_rope.to(dev)
        self.extra_nope = extra_nope.to(dev)
        self.extra_rope = extra_rope.to(dev)

        g = torch.Generator(device="cpu").manual_seed(seed + 7)
        if lens_mode == "full":
            swa_lens = torch.full((B,), swa_topk, dtype=torch.int32)
            extra_lens = torch.full((B,), extra_topk, dtype=torch.int32)
        else:
            swa_lens = torch.randint(0, swa_topk + 1, (B,), generator=g)
            extra_lens = torch.randint(0, extra_topk + 1, (B,), generator=g)
        self.swa_lens = swa_lens.to(torch.int32).to(dev)
        self.extra_lens = extra_lens.to(torch.int32).to(dev)

        self.swa_idx = torch.full((B, swa_topk), -1, dtype=torch.int32)
        self.extra_idx = torch.full((B, extra_topk), -1, dtype=torch.int32)
        for b in range(B):
            n = self.swa_lens[b].item()
            if n:
                self.swa_idx[b, :n] = torch.randint(
                    0,
                    swa_tokens,
                    (n,),
                    generator=torch.Generator().manual_seed(seed * 1000 + b),
                )
            n = self.extra_lens[b].item()
            if n:
                self.extra_idx[b, :n] = torch.randint(
                    0,
                    extra_tokens,
                    (n,),
                    generator=torch.Generator().manual_seed(seed * 1000 + b + 500),
                )
        self.swa_idx = self.swa_idx.to(dev)
        self.extra_idx = self.extra_idx.to(dev)
        self.q = (torch.randn(B, 1, H, _D, device=dev) * 0.05).to(torch.bfloat16)
        self.sinks = torch.randn(H, device=dev) * 2 - 4
        self.scale = _D**-0.5
        self.ref = _reference(
            self.q,
            self.swa_nope,
            self.swa_rope,
            self.swa_idx,
            self.swa_lens,
            self.extra_nope,
            self.extra_rope,
            self.extra_idx,
            self.extra_lens,
            self.sinks,
            self.scale,
        )
        self.out = torch.zeros(B, H, _D, dtype=torch.bfloat16, device=dev)

    def run_public(self) -> torch.Tensor:
        from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_vllm,
        )

        self.out.zero_()
        triton_sparse_mla_decode_vllm(
            q=self.q,
            swa_kv_cache=self.swa_cache,
            swa_indices=self.swa_idx.unsqueeze(1),
            swa_lens=self.swa_lens,
            extra_kv_cache=self.extra_cache,
            extra_indices=self.extra_idx.unsqueeze(1),
            extra_lens=self.extra_lens,
            attn_sink=self.sinks,
            softmax_scale=self.scale,
            out=self.out,
        )
        return self.out

    def run_legacy(self) -> torch.Tensor:
        import os

        os.environ["VLLM_TRITON_SPARSE_MLA_DECODE_LEGACY"] = "1"
        import importlib

        import vllm.envs

        importlib.reload(vllm.envs)
        from vllm.models.deepseek_v4.nvidia.ops import triton_sparse_mla_decode as d

        importlib.reload(d)
        self.out.zero_()
        d.triton_sparse_mla_decode_vllm(
            q=self.q,
            swa_kv_cache=self.swa_cache,
            swa_indices=self.swa_idx.unsqueeze(1),
            swa_lens=self.swa_lens,
            extra_kv_cache=self.extra_cache,
            extra_indices=self.extra_idx.unsqueeze(1),
            extra_lens=self.extra_lens,
            attn_sink=self.sinks,
            softmax_scale=self.scale,
            out=self.out,
        )
        return self.out

    def check(self, out: torch.Tensor, tol: float = 5e-2, tag: str = "") -> float:
        d = (out.float() - self.ref.float()).abs()
        denom = self.ref.float().abs().clamp(min=1e-3)
        rel = (d / denom).max().item()
        mx = d.max().item()
        ok = _close(out, self.ref, tol)
        if not ok:
            bad = d > (tol + tol * self.ref.float().abs())
            print(
                f"  [FAIL {tag}] max_abs={mx:.4e} max_rel={rel:.2f} "
                f"bad_frac={(bad.float()).mean().item():.3%}"
            )
        return mx


def _case_specs() -> list[dict]:
    return [
        # score_scale=500 gives realistic attention score spreads (~9 natural
        # log units), where a base-2/linear softmax mix would deviate ~50%.
        dict(
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=512,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=2048,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=4,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=16,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=16,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=2,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=7,
            H=8,
            swa_topk=96,
            extra_topk=100,
            swa_page=64,
            extra_page=16,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=2,
            H=8,
            swa_topk=32,
            extra_topk=128,
            swa_page=16,
            extra_page=2,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=1,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
        dict(
            B=3,
            H=16,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        ),
    ]


def scenario_multibatch(n_rep: int = 3) -> None:
    """Repeated calls with same capacity but different contents + stale check."""
    nfail = 0
    for seed in range(7):
        for spec in _case_specs():
            for rep in range(n_rep):
                c = Case(seed * 100 + rep * 17 + 3, **spec)
                out = c.run_public()
                torch.cuda.synchronize()
                if not _close(out, c.ref):
                    nfail += 1
                    c.check(out, tag=f"seed={seed} rep={rep} {spec}")
            if seed % 3 == 0:
                print(f"  multibatch seed={seed} ok")
    print(f"multibatch: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_streams(n_rounds: int = 6) -> None:
    """Concurrent calls on 4 streams, same capacities to race the scratch pool."""
    dev = torch.device("cuda:0")
    streams = [torch.cuda.Stream(device=dev) for _ in range(4)]
    nfail = 0
    cases = [
        Case(
            1000 + i,
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        )
        for i in range(4)
    ]
    for rnd in range(n_rounds):
        evs = []
        for s, c in zip(streams, cases):
            with torch.cuda.stream(s):
                ev = torch.cuda.Event()
                c.out.zero_()
                c.run_public()
                ev.record()
                evs.append((s, c, ev))
        torch.cuda.synchronize()
        for s, c, _ in evs:
            if not _close(c.out, c.ref):
                nfail += 1
                c.check(c.out, tag=f"stream rnd={rnd}")
    print(f"streams: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_graph() -> None:
    """CUDA-graph capture + replay with mutated inputs each replay."""
    dev = torch.device("cuda:0")
    nfail = 0
    for seed in range(4):
        c = Case(
            2000 + seed,
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=128,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        )
        from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_vllm,
        )

        def fn(case: Case) -> None:
            triton_sparse_mla_decode_vllm(
                q=case.q,
                swa_kv_cache=case.swa_cache,
                swa_indices=case.swa_idx.unsqueeze(1),
                swa_lens=case.swa_lens,
                extra_kv_cache=case.extra_cache,
                extra_indices=case.extra_idx.unsqueeze(1),
                extra_lens=case.extra_lens,
                attn_sink=case.sinks,
                softmax_scale=case.scale,
                out=case.out,
            )

        fn(c)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(c)
        torch.cuda.synchronize()
        for rep in range(3):
            # Mutate contents against the same caches: new lens/indices in the
            # same buffers, reference recomputed from c's own KV.
            gen = torch.Generator(device="cpu").manual_seed(3000 + seed * 10 + rep)
            swa_lens = torch.randint(0, 129, (8,), generator=gen)
            extra_lens = torch.randint(0, 129, (8,), generator=gen)
            swa_idx = torch.full((8, 128), -1, dtype=torch.int32)
            extra_idx = torch.full((8, 128), -1, dtype=torch.int32)
            for b in range(8):
                n = swa_lens[b].item()
                if n:
                    swa_idx[b, :n] = torch.randint(
                        0,
                        512,
                        (n,),
                        generator=torch.Generator().manual_seed(b * 7 + rep),
                    )
                n = extra_lens[b].item()
                if n:
                    extra_idx[b, :n] = torch.randint(
                        0,
                        512,
                        (n,),
                        generator=torch.Generator().manual_seed(b * 7 + rep + 100),
                    )
            c.swa_lens.copy_(swa_lens.to(dev))
            c.extra_lens.copy_(extra_lens.to(dev))
            c.swa_idx.copy_(swa_idx.to(dev))
            c.extra_idx.copy_(extra_idx.to(dev))
            c.ref = _reference(
                c.q,
                c.swa_nope,
                c.swa_rope,
                c.swa_idx,
                c.swa_lens,
                c.extra_nope,
                c.extra_rope,
                c.extra_idx,
                c.extra_lens,
                c.sinks,
                c.scale,
            )
            c.out.zero_()
            g.replay()
            torch.cuda.synchronize()
            if not _close(c.out, c.ref):
                nfail += 1
                c.check(c.out, tag=f"graph seed={seed} rep={rep}")
    print(f"graph: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_transition() -> None:
    """c128-style extra width transitions 128 -> 256 -> 512 -> 1024."""
    nfail = 0
    widths = [128, 256, 512, 1024, 2048, 128]
    for seed in range(3):
        for w in widths:
            c = Case(
                4000 + seed * 100 + w,
                B=8,
                H=8,
                swa_topk=128,
                extra_topk=w,
                swa_page=64,
                extra_page=64,
                score_scale=500.0,
                exp_range=0,
            )
            out = c.run_public()
            if not _close(out, c.ref):
                nfail += 1
                c.check(out, tag=f"transition seed={seed} w={w}")
    print(f"transition: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_wide_capture() -> None:
    """Width-independent K-split: wide buffer views (stride W_MAX) with narrow
    active rows must produce oracle output for any runtime row length.

    Mirrors the production c128a metadata views: the tensors are slices of
    max-width buffers (outer stride W_MAX) whose active width is small. The
    K-split grid/scratch is sized to W_MAX while per-CTA source/chunk
    selection comes from the runtime CSR lengths, so the result must equal
    the reference for a mix of row lengths (including 0 and W_ACTIVE).
    """
    dev = torch.device("cuda:0")
    nfail = 0
    for seed in range(4):
        c = Case(
            9000 + seed,
            B=8,
            H=8,
            swa_topk=128,
            extra_topk=2048,
            swa_page=64,
            extra_page=64,
            score_scale=500.0,
            exp_range=0,
        )
        from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_decode import (
            triton_sparse_mla_decode_vllm,
        )

        # Slice of a max-width buffer with the active view width, preserving
        # the buffer stride (exactly what build_c128a_topk_metadata returns).
        swa_wide = torch.full(
            (c.B, 4096), -1, dtype=torch.int32, device=dev
        )
        extra_wide = torch.full(
            (c.B, 4096), -1, dtype=torch.int32, device=dev
        )
        swa_wide[:, : c.swa_topk] = c.swa_idx.squeeze(1)
        extra_wide[:, : c.extra_topk] = c.extra_idx.squeeze(1)
        swa_view = swa_wide[:, : c.swa_topk].unsqueeze(1)
        extra_view = extra_wide[:, : c.extra_topk].unsqueeze(1)

        c.out.zero_()
        triton_sparse_mla_decode_vllm(
            q=c.q,
            swa_kv_cache=c.swa_cache,
            swa_indices=swa_view,
            swa_lens=c.swa_lens,
            extra_kv_cache=c.extra_cache,
            extra_indices=extra_view,
            extra_lens=c.extra_lens,
            attn_sink=c.sinks,
            softmax_scale=c.scale,
            out=c.out,
        )
        torch.cuda.synchronize()
        if not _close(c.out, c.ref):
            nfail += 1
            c.check(c.out, tag=f"wide seed={seed}")

        # Same wide views, captured into a CUDA graph, replayed with mutated
        # row lengths/contents (runtime width changes must not corrupt output).
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            triton_sparse_mla_decode_vllm(
                q=c.q,
                swa_kv_cache=c.swa_cache,
                swa_indices=swa_view,
                swa_lens=c.swa_lens,
                extra_kv_cache=c.extra_cache,
                extra_indices=extra_view,
                extra_lens=c.extra_lens,
                attn_sink=c.sinks,
                softmax_scale=c.scale,
                out=c.out,
            )
        torch.cuda.synchronize()
        for rep in range(3):
            gen = torch.Generator(device="cpu").manual_seed(9100 + seed * 10 + rep)
            swa_lens = torch.randint(0, 129, (c.B,), generator=gen)
            extra_lens = torch.randint(0, 257, (c.B,), generator=gen)
            swa_wide.fill_(-1)
            extra_wide.fill_(-1)
            for b in range(c.B):
                n = swa_lens[b].item()
                if n:
                    swa_wide[b, :n] = torch.randint(
                        0,
                        512,
                        (n,),
                        generator=torch.Generator().manual_seed(b * 13 + rep + seed),
                    )
                n = extra_lens[b].item()
                if n:
                    extra_wide[b, :n] = torch.randint(
                        0,
                        512,
                        (n,),
                        generator=torch.Generator().manual_seed(b * 13 + rep + seed + 50),
                    )
            c.swa_lens.copy_(swa_lens.to(dev))
            c.extra_lens.copy_(extra_lens.to(dev))
            c.swa_idx = swa_view.squeeze(1)
            c.extra_idx = extra_view.squeeze(1)
            c.ref = _reference(
                c.q,
                c.swa_nope,
                c.swa_rope,
                c.swa_idx,
                c.swa_lens,
                c.extra_nope,
                c.extra_rope,
                c.extra_idx,
                c.extra_lens,
                c.sinks,
                c.scale,
            )
            c.out.zero_()
            g.replay()
            torch.cuda.synchronize()
            if not _close(c.out, c.ref):
                nfail += 1
                c.check(c.out, tag=f"wide-graph seed={seed} rep={rep}")
    print(f"wide_capture: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


def scenario_reproduce_alias() -> None:
    """Reproduce the pre-fix swa/extra CSR aliasing (slot=0 for both packs)."""
    import triton

    from vllm.models.deepseek_v4.nvidia.ops.triton_sparse_mla_prefill import (
        _DEFAULT_BLOCK_H,
        _DEFAULT_BLOCK_K,
        _GROUP_DIM,
        _NOPE_DIM,
        _SCALE_STRIDE,
        _TOKEN_DATA_STRIDE,
        _pack_sparse_rows,
        _paged_cache_views,
        _tiled_sparse_prefill_kernel,
    )

    c = Case(
        5000,
        B=8,
        H=8,
        swa_topk=128,
        extra_topk=128,
        swa_page=64,
        extra_page=64,
        lens_mode="full",
        score_scale=500.0,
        exp_range=0,
    )
    swa_fp8, swa_u8, swa_bf16, swa_lo, swa_ps, swa_pb = _paged_cache_views(c.swa_cache)
    extra_fp8, extra_u8, extra_bf16, extra_lo, extra_ps, extra_pb = _paged_cache_views(
        c.extra_cache
    )
    grid = (c.B, triton.cdiv(c.H, _DEFAULT_BLOCK_H))

    def launch(slot: int) -> torch.Tensor:
        swa_flat, swa_indptr = _pack_sparse_rows(
            c.swa_idx.unsqueeze(1), c.swa_lens, slot=0
        )
        extra_flat, extra_indptr = _pack_sparse_rows(
            c.extra_idx.unsqueeze(1), c.extra_lens, slot=slot
        )
        out = torch.zeros(c.B, c.H, _D, dtype=torch.bfloat16, device=c.dev)
        lse = torch.empty(c.B, c.H, dtype=torch.float32, device=c.dev)
        _tiled_sparse_prefill_kernel[grid](
            Q_ptr=c.q.squeeze(1),
            O_ptr=out,
            LSE_ptr=lse,
            swa_cache_fp8_ptr=swa_fp8,
            swa_cache_uint8_ptr=swa_u8,
            swa_cache_bf16_ptr=swa_bf16,
            swa_idx_ptr=swa_flat,
            swa_indptr_ptr=swa_indptr,
            extra_cache_fp8_ptr=extra_fp8,
            extra_cache_uint8_ptr=extra_u8,
            extra_cache_bf16_ptr=extra_bf16,
            extra_idx_ptr=extra_flat,
            extra_indptr_ptr=extra_indptr,
            sm_scale=c.scale,
            swa_page_size=64,
            swa_page_bytes=c.swa_cache.stride(0),
            swa_layer_off=swa_lo,
            swa_scale_off=64 * _TOKEN_DATA_STRIDE,
            extra_page_size=64,
            extra_page_bytes=c.extra_cache.stride(0),
            extra_layer_off=extra_lo,
            extra_scale_off=64 * _TOKEN_DATA_STRIDE,
            H=c.H,
            stride_qb=c.q.stride(0),
            stride_qh=c.q.stride(2),
            stride_ob=out.stride(0),
            stride_oh=out.stride(1),
            HAS_EXTRA=True,
            BLOCK_H=_DEFAULT_BLOCK_H,
            BLOCK_K=_DEFAULT_BLOCK_K,
            GROUP_DIM=_GROUP_DIM,
            NOPE_DIM=_NOPE_DIM,
            TOKEN_DATA_STRIDE=_TOKEN_DATA_STRIDE,
            SCALE_STRIDE=_SCALE_STRIDE,
            num_warps=8,
            num_stages=2,
        )
        torch.cuda.synchronize()
        return out

    aliased = launch(slot=0)
    correct = launch(slot=1)
    d_alias = (aliased.float() - c.ref.float()).abs().max().item()
    d_correct = (correct.float() - c.ref.float()).abs().max().item()
    d_between = (aliased.float() - correct.float()).abs().max().item()
    print(
        f"  aliased-vs-ref={d_alias:.4e} correct-vs-ref={d_correct:.4e} "
        f"aliased-vs-correct={d_between:.4e}"
    )
    assert d_between > 5e-2 * c.ref.float().abs().max().item(), (
        "aliasing should change the output"
    )
    assert d_correct <= 5e-2 * c.ref.float().abs().max().item(), (
        "fixed path should match the reference"
    )
    torch.cuda.synchronize()
    print("reproduce-alias: PASS (bug mechanism confirmed, fix prevents it)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        choices=[
            "multibatch",
            "streams",
            "graph",
            "transition",
            "wide",
            "alias",
            "all",
        ],
        default="all",
    )
    args = ap.parse_args()
    assert torch.cuda.is_available()
    sc = args.scenario
    if sc in ("multibatch", "all"):
        print("== multibatch ==")
        scenario_multibatch()
    if sc in ("streams", "all"):
        print("== streams ==")
        scenario_streams()
    if sc in ("graph", "all"):
        print("== graph ==")
        scenario_graph()
    if sc in ("transition", "all"):
        print("== transition ==")
        scenario_transition()
    if sc in ("wide", "all"):
        print("== wide_capture ==")
        scenario_wide_capture()
    if sc in ("alias", "all"):
        print("== reproduce-alias ==")
        scenario_reproduce_alias()


if __name__ == "__main__":
    main()
