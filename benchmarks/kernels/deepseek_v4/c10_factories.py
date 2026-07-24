# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _combine_topk_swa_indices_triton_baseline,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed

_SPARSE_PREFILL_TOPK_ALIGNMENT = 128
_SOURCE_GUARD = -777

CombineFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def _load_fused_candidate(backend: str) -> CombineFn | None:
    if backend == "hybrid":
        from vllm.models.deepseek_v4.common.ops.cache_utils import (
            combine_topk_swa_indices_sm120,
        )

        return combine_topk_swa_indices_sm120
    if backend == "triton":
        from vllm.models.deepseek_v4.common.ops.cache_utils import (
            combine_topk_swa_indices_fused_triton,
        )

        return combine_topk_swa_indices_fused_triton
    try:
        from vllm.models.deepseek_v4.nvidia.ops.combine_topk_swa_cutedsl import (
            combine_topk_swa_indices_cutedsl,
        )
    except (ImportError, ModuleNotFoundError):
        return None
    return combine_topk_swa_indices_cutedsl


def _split_lengths(total: int, count: int) -> list[int]:
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    num_tokens = int(args["num_tokens"])
    context_tokens = int(args["context_tokens"])
    request_batch = int(args.get("request_batch", 1))
    compress_ratio = int(args.get("compress_ratio", 4))
    topk = int(args.get("topk", 512))
    topk_width = int(args.get("topk_width", max(topk, 1)))
    window_size = int(args.get("window_size", 128))
    context_jitter = int(args.get("context_jitter", 0))
    query_base = int(args.get("query_base", 0))
    offset_inputs = bool(args.get("offset_inputs", False))
    candidate_mode = str(args.get("candidate_mode", "mirror"))
    candidate_backend = str(args.get("candidate_backend", "cutedsl"))
    seed = int(args.get("seed", 0))

    raw_query_lens = args.get("query_lens")
    query_lens = (
        [int(length) for length in raw_query_lens]
        if raw_query_lens is not None
        else _split_lengths(num_tokens, request_batch)
    )
    if num_tokens <= 0 or context_tokens <= 0:
        raise ValueError("C10 token and context counts must be positive")
    if request_batch not in (1, 4):
        raise ValueError("C10 request_batch must be 1 or 4")
    if len(query_lens) != request_batch or sum(query_lens) != num_tokens:
        raise ValueError("C10 query_lens must match request_batch and num_tokens")
    if any(length < 0 for length in query_lens):
        raise ValueError("C10 query lengths must be non-negative")
    if compress_ratio <= 0 or topk < 0 or topk_width < 0 or window_size <= 0:
        raise ValueError("C10 compression, widths, and window must be valid")
    if topk > topk_width:
        raise ValueError("C10 topk cannot exceed the source row width")
    if context_jitter < 0 or query_base < 0:
        raise ValueError("C10 context jitter and query base must be non-negative")
    if candidate_mode not in ("mirror", "fused"):
        raise ValueError("C10 candidate_mode must be mirror or fused")
    if candidate_backend not in ("cutedsl", "hybrid", "triton"):
        raise ValueError("C10 candidate_backend must be cutedsl, hybrid, or triton")

    seq_lens_host = [
        context_tokens - request_index * context_jitter
        for request_index in range(request_batch)
    ]
    if any(
        seq_len < query_len
        for seq_len, query_len in zip(seq_lens_host, query_lens, strict=True)
    ):
        raise ValueError("C10 query length cannot exceed its sequence length")
    gather_lens_host = [
        query_len + min(seq_len - query_len, window_size - 1)
        for seq_len, query_len in zip(seq_lens_host, query_lens, strict=True)
    ]

    n = int(
        args.get(
            "n",
            0
            if compress_ratio == 1
            else max(seq_len // compress_ratio for seq_len in seq_lens_host),
        )
    )
    m = int(args.get("m", n + max(gather_lens_host)))
    if n < 0 or m < n + max(gather_lens_host):
        raise ValueError("C10 M/N do not fit the gathered workspace")

    set_random_seed(seed)
    device = torch.device("cuda")
    query_lens_tensor = torch.tensor(query_lens, dtype=torch.int32, device=device)
    query_start_values = torch.empty(
        request_batch + 1, dtype=torch.int32, device=device
    )
    query_start_values[0] = query_base
    torch.cumsum(query_lens_tensor, dim=0, out=query_start_values[1:])
    query_start_values[1:].add_(query_base)
    seq_lens_values = torch.tensor(seq_lens_host, dtype=torch.int32, device=device)
    gather_lens_values = torch.tensor(
        gather_lens_host, dtype=torch.int32, device=device
    )

    def offset_vector(values: torch.Tensor) -> torch.Tensor:
        if not offset_inputs:
            return values
        storage = torch.full(
            (values.shape[0] + 2,),
            _SOURCE_GUARD,
            dtype=values.dtype,
            device=values.device,
        )
        storage[1:-1].copy_(values)
        return storage[1:-1]

    query_start_loc = offset_vector(query_start_values)
    seq_lens = offset_vector(seq_lens_values)
    gather_lens = offset_vector(gather_lens_values)

    request_indices = torch.arange(
        request_batch, dtype=torch.int32, device=device
    ).repeat_interleave(query_lens_tensor)
    request_starts = torch.tensor(
        [sum(query_lens[:index]) for index in range(request_batch)],
        dtype=torch.int32,
        device=device,
    )
    token_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    local_token_indices = token_indices - request_starts[request_indices]
    positions = (
        seq_lens_values[request_indices]
        - query_lens_tensor[request_indices]
        + local_token_indices
    )
    topk_lens = torch.minimum(
        torch.div(positions + 1, compress_ratio, rounding_mode="floor"),
        torch.full_like(positions, topk),
    )
    swa_lens = torch.minimum(
        positions + 1,
        torch.full_like(positions, window_size),
    )
    if bool((topk_lens > topk_width).any().item()):
        raise ValueError("C10 source width is smaller than a valid topk length")

    topk_storage = torch.full(
        (num_tokens + 2, topk_width + 3),
        _SOURCE_GUARD,
        dtype=torch.int32,
        device=device,
    )
    topk_indices = topk_storage[1:-1, :topk_width]
    source_columns = torch.arange(topk_width, dtype=torch.int32, device=device)
    source_values = (source_columns[None, :] * 67 + token_indices[:, None] * 131) % max(
        n, 1
    )
    topk_indices.copy_(
        torch.where(
            source_columns[None, :] < topk_lens[:, None],
            source_values,
            torch.full_like(source_values, _SOURCE_GUARD),
        )
    )

    combined_width = (
        cdiv(topk + window_size, _SPARSE_PREFILL_TOPK_ALIGNMENT)
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    output_columns = torch.arange(combined_width, dtype=torch.int32, device=device)
    expected = torch.full(
        (num_tokens, combined_width), -1, dtype=torch.int32, device=device
    )
    request_offsets = request_indices[:, None] * m
    if topk_width > 0:
        safe_topk_columns = output_columns.clamp_max(topk_width - 1)
        topk_values = topk_indices[:, safe_topk_columns]
        valid_topk = output_columns[None, :] < topk_lens[:, None]
        expected = torch.where(valid_topk, topk_values + request_offsets, expected)

    swa_offsets = output_columns[None, :] - topk_lens[:, None]
    valid_swa = (swa_offsets >= 0) & (swa_offsets < swa_lens[:, None])
    gather_starts = seq_lens_values - gather_lens_values
    swa_values = (
        request_offsets
        + n
        + swa_offsets
        + positions[:, None]
        - swa_lens[:, None]
        + 1
        - gather_starts[request_indices, None]
    )
    expected = torch.where(valid_swa, swa_values, expected)
    expected_lens = topk_lens + swa_lens

    return {
        "topk_indices": topk_indices,
        "query_start_loc": query_start_loc,
        "seq_lens": seq_lens,
        "gather_lens": gather_lens,
        "window_size": window_size,
        "compress_ratio": compress_ratio,
        "topk": topk,
        "m": m,
        "n": n,
        "expected": expected,
        "expected_lens": expected_lens,
        "candidate_mode": candidate_mode,
        "candidate_backend": candidate_backend,
        "shape": {
            "name": (
                f"t{num_tokens}-ctx{context_tokens}-b{request_batch}"
                f"-c{compress_ratio}-k{topk}-kw{topk_width}"
                f"-w{window_size}-qb{query_base}-off{int(offset_inputs)}"
            ),
            "num_tokens": num_tokens,
            "context_tokens": context_tokens,
            "request_batch": request_batch,
            "query_lens": query_lens,
            "seq_lens": seq_lens_host,
            "gather_lens": gather_lens_host,
            "compress_ratio": compress_ratio,
            "topk": topk,
            "topk_width": topk_width,
            "window_size": window_size,
            "combined_width": combined_width,
            "m": m,
            "n": n,
            "query_base": query_base,
            "offset_inputs": offset_inputs,
            "candidate_mode": candidate_mode,
            "candidate_backend": candidate_backend,
            "chain": "initialize-and-combine-flashmla-sparse-indices",
        },
    }


def build_c10_combine_topk_swa_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    candidate_impl = _load_fused_candidate(inputs["candidate_backend"])
    if inputs["candidate_mode"] == "fused" and candidate_impl is None:
        raise RuntimeError("C10 fused candidate is unavailable")

    baseline_lens = torch.empty_like(inputs["expected_lens"])
    candidate_lens = torch.empty_like(inputs["expected_lens"])

    def run_baseline() -> torch.Tensor:
        nonlocal baseline_lens
        output, baseline_lens = _combine_topk_swa_indices_triton_baseline(
            inputs["topk_indices"],
            inputs["query_start_loc"],
            inputs["seq_lens"],
            inputs["gather_lens"],
            inputs["window_size"],
            inputs["compress_ratio"],
            inputs["topk"],
            inputs["m"],
            inputs["n"],
        )
        return output

    def run_candidate() -> torch.Tensor:
        nonlocal candidate_lens
        implementation = (
            _combine_topk_swa_indices_triton_baseline
            if inputs["candidate_mode"] == "mirror"
            else candidate_impl
        )
        assert implementation is not None
        output, candidate_lens = implementation(
            inputs["topk_indices"],
            inputs["query_start_loc"],
            inputs["seq_lens"],
            inputs["gather_lens"],
            inputs["window_size"],
            inputs["compress_ratio"],
            inputs["topk"],
            inputs["m"],
            inputs["n"],
        )
        return output

    def compare(
        baseline_output: torch.Tensor,
        candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "baseline_indices_exact": torch.equal(baseline_output, inputs["expected"]),
            "candidate_indices_exact": torch.equal(
                candidate_output, inputs["expected"]
            ),
            "baseline_lens_exact": torch.equal(baseline_lens, inputs["expected_lens"]),
            "candidate_lens_exact": torch.equal(
                candidate_lens, inputs["expected_lens"]
            ),
        }
        return {"passed": all(checks.values()), "exact": checks}

    return ChainCase(
        baseline=Provider(
            "torch-full-plus-triton-combine",
            run_baseline,
            {"initialization": "torch.full", "combine": "triton"},
        ),
        candidate=Provider(
            (
                "mirror-full-plus-triton"
                if inputs["candidate_mode"] == "mirror"
                else f"{inputs['candidate_backend']}-fused-initialize-combine"
            ),
            run_candidate,
            {
                "candidate_mode": inputs["candidate_mode"],
                "candidate_backend": inputs["candidate_backend"],
            },
            correctness_comparator=compare,
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
