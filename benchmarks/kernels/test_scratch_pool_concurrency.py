# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-stream / reuse oracle for the DSv4 eager-scratch pool aliasing.

``DeepseekV4EagerScratchPool`` keeps one byte ``storage`` and hands out three
view groups -- fp4-indexer (values/scales/weights), global-topk
(indices/lens) and compressor (fp32 scratch) -- all starting at offset 0,
deliberately aliasing the same bytes. The design comment says this is safe
because "C4 uses fp4+global, C128 uses compressor, layers run sequentially
on one stream"; there is no mechanism inside the pool that prevents two
purposes from overwriting each other if they ever run concurrently on
different streams.

On the current SM89 tree the aliasing is mostly dormant:
* ``use_fp4_indexer_cache`` requires sm_100 (fp4 views never handed out);
* ``has_cutedsl()`` is False and ``_prefer_two_stage_compressor()`` is
  ROCm-only, so the head=512 C128 compressor takes the single-pass Triton
  path and the compressor scratch is never handed out on CUDA;
* the only live pool consumer on SM89 is
  ``compute_global_topk_indices_and_lens`` (C4 global views) on the default
  stream inside ``forward_mqa``.

This suite verifies the mechanism regardless of which views are live today:

* aliasing: all three view groups start at storage offset 0 and the storage
  is sized as the max of the group byte sizes (not the sum);
* global_topk: the live SM89 pool consumer checked against an independent
  torch reference over multi-step reuse with varying token counts;
* streams_overlap: concurrent writes into the global-topk and compressor
  views on two unjoined CUDA streams corrupt the shared storage, while the
  same writes joined by events are exactly last-writer-wins -- i.e. the
  aliasing is real and only the layer/stream protocol protects it;
* q_out_reuse: the separate q buffer reused sequentially across "layers"
  with a side-stream snapshot join (the MLA read protocol).

Run on gserver (L40S, remember LD_LIBRARY_PATH, see
``test_indexer_topk_concurrency.py``)::

    python benchmarks/kernels/test_scratch_pool_concurrency.py --scenario all
"""

import argparse
from math import prod

import torch

_BLOCK_SIZE = 64


def _make_pool(
    max_tokens: int,
    device: torch.device,
    *,
    index_topk: int = 512,
    q_head_dim: int = 512,
    index_q_heads: int = 64,
    index_q_head_dim: int = 128,
    padded_q_heads: int = 8,
):
    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool

    return DeepseekV4EagerScratchPool(
        max_tokens,
        padded_q_heads,
        q_head_dim,
        index_q_heads,
        index_q_head_dim,
        index_topk,
        device,
    )


def _ref_global_topk(
    topk_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch reference for compute_global_topk_indices_and_lens (CPU)."""
    valid = topk_indices >= 0
    safe = topk_indices.clamp(min=0)
    block_idx = safe // block_size
    blocks = block_table[token_to_req[:, None].expand_as(block_idx), block_idx]
    slot_ids = blocks * block_size + safe % block_size
    out = torch.where(valid, slot_ids, -1)
    counts = valid.sum(dim=-1)
    lens = torch.where(is_valid_token.bool(), counts, torch.zeros_like(counts))
    return out, lens


def _run_global_topk(
    topk_indices: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor],
) -> None:
    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        compute_global_topk_indices_and_lens,
    )

    compute_global_topk_indices_and_lens(
        topk_indices,
        token_to_req,
        block_table,
        block_size,
        is_valid_token,
        output_buffers=output_buffers,
    )


def scenario_aliasing() -> int:
    """All three view groups alias storage at offset 0; size is max not sum."""
    nfail = 0
    dev = torch.device("cuda:0")
    max_tokens = 16384
    index_topk = 512
    q_head_dim = 512
    index_q_heads = 64
    index_q_head_dim = 128
    pool = _make_pool(
        max_tokens,
        dev,
        index_topk=index_topk,
        q_head_dim=q_head_dim,
        index_q_heads=index_q_heads,
        index_q_head_dim=index_q_head_dim,
    )
    storage = pool._storage
    fp4_values, fp4_scales, fp4_weights = pool._fp4_template
    global_indices, global_lens = pool._global_template
    compressor = pool._compressor_template

    if not (
        fp4_values.data_ptr()
        == global_indices.data_ptr()
        == compressor.data_ptr()
        == storage.data_ptr()
    ):
        nfail += 1
        print("  [FAIL aliasing] first views do not start at storage offset 0")

    def group_bytes(specs) -> int:
        offset = 0
        for shape, dtype in specs:
            offset = (offset + 255) // 256 * 256 + prod(shape) * dtype.itemsize
        return (offset + 255) // 256 * 256

    fp4_specs = (
        ((max_tokens, index_q_heads, index_q_head_dim // 2), torch.uint8),
        ((max_tokens, index_q_heads, index_q_head_dim // 32), torch.uint8),
        ((max_tokens, index_q_heads), torch.float32),
    )
    global_specs = (
        ((max_tokens, index_topk), torch.int32),
        ((max_tokens,), torch.int32),
    )
    compressor_specs = (((max_tokens, q_head_dim), torch.float32),)
    expected = max(
        group_bytes(s) for s in (fp4_specs, global_specs, compressor_specs)
    )
    if storage.numel() != expected:
        nfail += 1
        print(
            f"  [FAIL aliasing] storage {storage.numel()} != max group {expected}"
        )

    # Byte-level alias: a write through the global view must be visible through
    # the compressor view (and vice versa).
    global_indices.fill_(0x11223344 & 0x7FFFFFFF)
    torch.cuda.synchronize()
    comp_bytes = compressor.view(torch.uint8)[: 4 * 8]
    g_bytes = global_indices.view(torch.uint8)[: 4 * 8]
    if not torch.equal(comp_bytes.cpu(), g_bytes.cpu()):
        nfail += 1
        print("  [FAIL aliasing] global/compressor views are not byte-alias")

    # q_out is a separate buffer, not storage.
    if pool.q_out(8).data_ptr() == storage.data_ptr():
        nfail += 1
        print("  [FAIL aliasing] q_out aliases storage")

    print(
        f"  storage={storage.numel()/1e6:.1f}MB "
        f"(max group={expected/1e6:.1f}MB, "
        f"fp4={group_bytes(fp4_specs)/1e6:.1f}MB, "
        f"global={group_bytes(global_specs)/1e6:.1f}MB, "
        f"compressor={group_bytes(compressor_specs)/1e6:.1f}MB)"
    )
    return nfail


def scenario_global_topk() -> int:
    """Live SM89 pool consumer vs reference, multi-step reuse."""
    nfail = 0
    dev = torch.device("cuda:0")
    index_topk = 512
    block_size = _BLOCK_SIZE
    max_tokens = 64
    num_reqs = 8
    width = 512
    pool = _make_pool(max_tokens, dev, index_topk=index_topk)

    for seed in range(4):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for step, num_tokens in enumerate([16, 8, 16, 4, 16, 0]):
            if num_tokens == 0:
                continue
            topk_indices = torch.randint(
                -1,
                width * block_size,
                (num_tokens, index_topk),
                generator=g,
            )
            # Ensure a mix of valid / -1 entries.
            mask = torch.rand(num_tokens, index_topk, generator=g) < 0.75
            topk_indices = torch.where(mask, topk_indices, torch.tensor(-1))
            token_to_req = torch.randint(0, num_reqs, (num_tokens,), generator=g)
            block_table = torch.randint(
                0, 4000, (num_reqs, width), generator=g
            ).to(torch.int32)
            is_valid = torch.randint(0, 2, (num_tokens,), generator=g).to(
                torch.int32
            )

            topk_dev = topk_indices.to(dev)
            ttr_dev = token_to_req.to(dev)
            bt_dev = block_table.to(dev)
            valid_dev = is_valid.to(dev)
            indices_view, lens_view = pool.global_topk_outputs(topk_dev)
            indices_view.fill_(-7)
            lens_view.fill_(-7)
            _run_global_topk(
                topk_dev,
                ttr_dev,
                bt_dev,
                block_size,
                valid_dev,
                (indices_view, lens_view),
            )
            torch.cuda.synchronize()

            ref_indices, ref_lens = _ref_global_topk(
                topk_indices,
                token_to_req,
                block_table,
                block_size,
                is_valid,
            )
            if not torch.equal(indices_view.cpu(), ref_indices):
                nfail += 1
                print(f"  [FAIL global_topk seed={seed} step={step}] indices")
            if not torch.equal(lens_view.cpu(), ref_lens):
                nfail += 1
                print(f"  [FAIL global_topk seed={seed} step={step}] lens")

            # The smaller-step view must be a prefix of the larger view: a
            # later 16-token step fully overwrites the rows a prior 8-token
            # step wrote (stale-tail check is implicit: only n rows are read).
            if num_tokens == 16:
                tail = indices_view[8:16]
                if bool((tail < 0).all().item()):
                    nfail += 1
                    print(f"  [FAIL global_topk seed={seed} step={step}] stale tail")
    return nfail


def _global_topk_inputs(
    num_tokens: int,
    index_topk: int,
    width: int,
    block_size: int,
    num_reqs: int,
    seed: int,
    dev: torch.device,
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    topk_indices = torch.randint(
        0, width * block_size, (num_tokens, index_topk), generator=g
    )
    token_to_req = torch.randint(0, num_reqs, (num_tokens,), generator=g)
    block_table = torch.randint(0, 4000, (num_reqs, width), generator=g).to(
        torch.int32
    )
    is_valid = torch.ones(num_tokens, dtype=torch.int32)
    return (
        topk_indices.to(dev),
        token_to_req.to(dev),
        block_table.to(dev),
        is_valid.to(dev),
    )


def scenario_streams_overlap(n_rounds: int = 5) -> int:
    """Unjoined cross-stream writes corrupt the shared storage; joined don't."""
    nfail = 0
    dev = torch.device("cuda:0")
    free_mb, _ = torch.cuda.mem_get_info(dev)
    if free_mb < 256e6:
        print(f"  [skip streams_overlap] only {free_mb/1e6:.0f}MB free")
        return 0
    num_tokens = 8192
    index_topk = 512
    block_size = _BLOCK_SIZE
    width = 512
    num_reqs = 64
    pool = _make_pool(
        num_tokens,
        dev,
        index_topk=index_topk,
        padded_q_heads=1,
        index_q_heads=16,
    )
    indices_view, lens_view = pool.global_topk_outputs(
        torch.empty(num_tokens, index_topk, dtype=torch.int32, device=dev)
    )
    comp_view = pool.compressor_scratch(num_tokens)
    prefix_bytes = num_tokens * index_topk * 4

    topk_cpu, ttr_cpu, bt_cpu, valid_cpu = _global_topk_inputs(
        num_tokens, index_topk, width, block_size, num_reqs, 777, torch.device("cpu")
    )
    ref_indices, _ = _ref_global_topk(
        topk_cpu, ttr_cpu, bt_cpu, block_size, valid_cpu
    )
    topk_dev, ttr_dev, bt_dev, valid_dev = (
        topk_cpu.to(dev),
        ttr_cpu.to(dev),
        bt_cpu.to(dev),
        valid_cpu.to(dev),
    )

    # Pure A: global-topk kernel alone.
    _run_global_topk(
        topk_dev, ttr_dev, bt_dev, block_size, valid_dev, (indices_view, lens_view)
    )
    torch.cuda.synchronize()
    region = pool._storage[:prefix_bytes]
    if not torch.equal(region.view(torch.int32).cpu(), ref_indices.reshape(-1)):
        nfail += 1
        print("  [FAIL streams_overlap] pure A does not match reference")
        return nfail

    # Pure B: compressor view fill alone.
    comp_view.fill_(-3.25)
    torch.cuda.synchronize()
    region = pool._storage[:prefix_bytes]
    if not bool((region.view(torch.float32) == -3.25).all().item()):
        nfail += 1
        print("  [FAIL streams_overlap] pure B fill not observed")
        return nfail

    def check_a() -> bool:
        return torch.equal(
            pool._storage[:prefix_bytes].view(torch.int32).cpu(),
            ref_indices.reshape(-1),
        )

    def check_b() -> bool:
        return bool(
            (pool._storage[:prefix_bytes].view(torch.float32) == -3.25).all().item()
        )

    # Unjoined: global-topk on stream A, compressor fill on stream B, no event
    # between them. The two views alias the same bytes -> storage is a mix.
    s_a = torch.cuda.Stream(device=dev)
    s_b = torch.cuda.Stream(device=dev)
    hazard_observed = 0
    for rnd in range(n_rounds):
        # Alternate launch order; no events between the two streams.
        if rnd % 2 == 0:
            with torch.cuda.stream(s_a):
                _run_global_topk(
                    topk_dev,
                    ttr_dev,
                    bt_dev,
                    block_size,
                    valid_dev,
                    (indices_view, lens_view),
                )
            with torch.cuda.stream(s_b):
                comp_view.fill_(-3.25)
        else:
            with torch.cuda.stream(s_b):
                comp_view.fill_(-3.25)
            with torch.cuda.stream(s_a):
                _run_global_topk(
                    topk_dev,
                    ttr_dev,
                    bt_dev,
                    block_size,
                    valid_dev,
                    (indices_view, lens_view),
                )
        torch.cuda.synchronize()
        pure_a_ok = check_a()
        pure_b_ok = check_b()
        if not (pure_a_ok or pure_b_ok):
            hazard_observed += 1
        else:
            print(
                f"  [info streams_overlap] rnd={rnd} serialized "
                f"(pure_a={pure_a_ok} pure_b={pure_b_ok})"
            )
    print(
        f"  unjoined race: corruption in {hazard_observed}/{n_rounds} rounds "
        "(serialization is a valid outcome; the join protocol below is the "
        "deterministic guarantee)"
    )

    # Joined: event after A, B waits -> storage is exactly last-writer-wins.
    for order in range(2):
        indices_view.fill_(-1)
        ev = torch.cuda.Event()
        if order == 0:
            with torch.cuda.stream(s_a):
                _run_global_topk(
                    topk_dev,
                    ttr_dev,
                    bt_dev,
                    block_size,
                    valid_dev,
                    (indices_view, lens_view),
                )
                ev.record()
            with torch.cuda.stream(s_b):
                s_b.wait_event(ev)
                comp_view.fill_(-3.25)
            torch.cuda.synchronize()
            if not check_b():
                nfail += 1
                print("  [FAIL streams_overlap] joined A->B not last-writer-wins")
        else:
            with torch.cuda.stream(s_b):
                comp_view.fill_(-3.25)
                ev.record()
            with torch.cuda.stream(s_a):
                s_a.wait_event(ev)
                _run_global_topk(
                    topk_dev,
                    ttr_dev,
                    bt_dev,
                    block_size,
                    valid_dev,
                    (indices_view, lens_view),
                )
            torch.cuda.synchronize()
            if not check_a():
                nfail += 1
                print("  [FAIL streams_overlap] joined B->A not last-writer-wins")
    return nfail


def scenario_q_out_reuse() -> int:
    """q_out buffer reuse across layers + side-stream snapshot join."""
    nfail = 0
    dev = torch.device("cuda:0")
    max_tokens = 64
    padded_heads, q_head_dim = 8, 512
    pool = _make_pool(
        max_tokens, dev, padded_q_heads=padded_heads, q_head_dim=q_head_dim
    )
    for seed in range(3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        for step, num_tokens in enumerate([16, 8, 16, 4, 16]):
            q_out = pool.q_out(num_tokens)
            content = torch.randn(
                num_tokens, padded_heads, q_head_dim, generator=g
            ).to(dev)
            content = content.to(torch.bfloat16)
            q_out.copy_(content)
            torch.cuda.synchronize()
            if not torch.equal(q_out.cpu(), content.cpu()):
                nfail += 1
                print(f"  [FAIL q_out seed={seed} step={step}] content")

    # Side-stream reader snapshot joined before the next layer overwrites.
    for seed in range(2):
        q_out = pool.q_out(8)
        content_a = torch.randn(
            8,
            padded_heads,
            q_head_dim,
            generator=torch.Generator(device="cpu").manual_seed(seed * 3 + 1),
        ).to(dev).to(torch.bfloat16)
        snap = torch.full_like(content_a, 7.0)
        q_out.copy_(content_a)
        main_ev = torch.cuda.Event()
        main_ev.record()
        s1 = torch.cuda.Stream(device=dev)
        with torch.cuda.stream(s1):
            s1.wait_event(main_ev)
            snap.copy_(q_out)
        torch.cuda.synchronize()
        if not torch.equal(snap.cpu(), content_a.cpu()):
            nfail += 1
            print(f"  [FAIL q_out seed={seed}] side-stream snapshot")
        content_b = torch.randn(
            8,
            padded_heads,
            q_head_dim,
            generator=torch.Generator(device="cpu").manual_seed(seed * 3 + 2),
        ).to(dev).to(torch.bfloat16)
        q_out.copy_(content_b)
        torch.cuda.synchronize()
        if not torch.equal(q_out.cpu(), content_b.cpu()):
            nfail += 1
            print(f"  [FAIL q_out seed={seed}] layer B overwrite")
    return nfail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scenario",
        choices=["aliasing", "global_topk", "streams_overlap", "q_out", "all"],
        default="all",
    )
    args = ap.parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (8, 9), (
        "this oracle targets the SM89 DSv4 path"
    )
    import vllm._C_stable_libtorch  # noqa: F401

    from vllm.models.deepseek_v4.compressor import _prefer_two_stage_compressor
    from vllm.utils.import_utils import has_cutedsl

    print(
        "dispatch audit: has_cutedsl=%s prefer_two_stage=%s "
        "(C128 compressor_scratch live only when both are true on CUDA)"
        % (has_cutedsl(), _prefer_two_stage_compressor())
    )

    nfail = 0
    sc = args.scenario
    if sc in ("aliasing", "all"):
        print("== aliasing ==")
        nfail += scenario_aliasing()
    if sc in ("global_topk", "all"):
        print("== global_topk ==")
        nfail += scenario_global_topk()
    if sc in ("streams_overlap", "all"):
        print("== streams_overlap ==")
        nfail += scenario_streams_overlap()
    if sc in ("q_out", "all"):
        print("== q_out ==")
        nfail += scenario_q_out_reuse()
    print(f"total: {'PASS' if nfail == 0 else f'{nfail} FAIL'}")


if __name__ == "__main__":
    main()
