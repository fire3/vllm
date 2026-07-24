# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import (
    CorrectnessTolerances,
    Provider,
    compare_outputs,
)
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import fp8_mqa_logits_triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_mqa_topk_indices,
)
from vllm.utils.torch_utils import set_random_seed

NUM_HEADS = 64
HEAD_DIM = 128
TOPK_TOKENS = 512


def _make_inputs(args: Mapping[str, Any]):
    num_queries = int(args["num_queries"])
    num_keys = int(args["num_keys"])
    compress_ratio = int(args.get("compress_ratio", 4))
    query_offset = int(args.get("query_offset", 0))
    causal = bool(args.get("causal", True))
    seed = int(args.get("seed", 0))
    if num_queries <= 0 or num_keys <= 0:
        raise ValueError("A03 requires positive query and key lengths")
    if compress_ratio not in (4, 128):
        raise ValueError("A03 compress_ratio must be 4 or 128")

    set_random_seed(seed)
    q = torch.randn(
        (num_queries, NUM_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    k = torch.randn(
        (num_keys, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    k_scale = torch.empty(
        num_keys,
        device="cuda",
        dtype=torch.float32,
    ).uniform_(0.001, 0.01)
    weights = torch.randn(
        (num_queries, NUM_HEADS),
        device="cuda",
        dtype=torch.float32,
    )
    cu_seqlen_ks = torch.zeros(
        num_queries,
        device="cuda",
        dtype=torch.int32,
    )
    if causal:
        cu_seqlen_ke = torch.div(
            query_offset
            + torch.arange(num_queries, device="cuda", dtype=torch.int32)
            + 1,
            compress_ratio,
            rounding_mode="floor",
        ).clamp_(max=num_keys)
    else:
        cu_seqlen_ke = torch.full(
            (num_queries,),
            num_keys,
            device="cuda",
            dtype=torch.int32,
        )
    shape = {
        "name": (
            f"m{num_queries}-n{num_keys}-h{NUM_HEADS}-d{HEAD_DIM}"
            f"-c{compress_ratio}-q{query_offset}"
        ),
        "M": num_queries,
        "N": num_keys,
        "H": NUM_HEADS,
        "D": HEAD_DIM,
        "compress_ratio": compress_ratio,
        "query_offset": query_offset,
        "causal": causal,
    }
    return q, (k, k_scale), weights, cu_seqlen_ks, cu_seqlen_ke, shape


def _make_logits_inputs(args: Mapping[str, Any]):
    if bool(args.get("causal", False)):
        raise ValueError(
            "A03 logits benchmarks require causal=false because clean_logits=false "
            "leaves masked entries undefined"
        )
    return _make_inputs({**args, "causal": False})


def build_a03_logits_case(args: Mapping[str, Any]) -> ChainCase:
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_logits_inputs(args)
    baseline_output = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )
    candidate_output = torch.empty_like(baseline_output)

    def run_baseline() -> torch.Tensor:
        return fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=False,
            out=baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        return fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=True,
            out=candidate_output,
        )

    shape["chain"] = "fp8-mqa-logits"
    return ChainCase(
        baseline=Provider(
            "triton-tf32",
            run_baseline,
            {"mma": "tf32", "preallocated_output": True},
        ),
        candidate=Provider(
            "triton-native-fp8",
            run_candidate,
            {"mma": "fp8", "preallocated_output": True},
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=0.25,
            rtol=0.01,
            max_mean_relative=1e-4,
            min_cosine=0.99999,
            require_allclose=False,
        ),
    )


def _deepgemm_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    from vllm.third_party import deep_gemm

    return deep_gemm.fp8_fp4_mqa_logits(
        (q, None),
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        max_seqlen_k=0,
        logits_dtype=torch.float32,
    )


def build_a03_deepgemm_logits_case(args: Mapping[str, Any]) -> ChainCase:
    from vllm.third_party import deep_gemm

    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_logits_inputs(args)
    baseline_output = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )
    deep_gemm.set_pdl(True)

    def run_baseline() -> torch.Tensor:
        return fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=True,
            out=baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        return _deepgemm_mqa_logits(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )

    shape["chain"] = "fp8-mqa-logits"
    return ChainCase(
        baseline=Provider(
            "triton-native-fp8",
            run_baseline,
            {"mma": "fp8", "preallocated_output": True},
        ),
        candidate=Provider(
            "deepgemm-sm120-fused",
            run_candidate,
            {
                "mma": "fp8",
                "pdl": True,
                "preallocated_output": False,
                "padded_output_stride": True,
            },
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=1e-5,
            rtol=1e-5,
            max_mean_relative=1e-5,
            min_cosine=0.999999,
            require_allclose=True,
        ),
    )


def build_a03_production_logits_case(args: Mapping[str, Any]) -> ChainCase:
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_logits_inputs(args)
    baseline_output = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )

    def run_baseline() -> torch.Tensor:
        return fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=False,
            out=baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        return fp8_fp4_mqa_logits(
            (q, None),
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=False,
        )

    shape["chain"] = "fp8-mqa-logits-production-dispatch"
    return ChainCase(
        baseline=Provider(
            "triton-tf32",
            run_baseline,
            {"mma": "tf32", "preallocated_output": True},
        ),
        candidate=Provider(
            "sm120-production-dispatch",
            run_candidate,
            {
                "preferred_backend": "deepgemm-sm120-fused",
                "fallback_backend": "triton-native-fp8",
                "preallocated_output": False,
            },
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=4e-6,
            rtol=1e-6,
            max_mean_relative=1e-5,
            min_cosine=0.999999,
            require_allclose=True,
        ),
    )


def _run_topk(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    output.fill_(-1)
    topk_tokens = output.shape[1]
    if topk_tokens == 0 or logits.shape[1] == 0:
        return output
    torch.ops._C.top_k_per_row_prefill(
        logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        output,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )
    output.add_(cu_seqlen_ks[:, None])
    valid = (output >= cu_seqlen_ks[:, None]) & (output < cu_seqlen_ke[:, None])
    output.masked_fill_(~valid, -1)
    return output


def _sort_topk_indices(output: torch.Tensor) -> torch.Tensor:
    return output.sort(dim=1).values


def _compare_topk_with_ties(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerances: CorrectnessTolerances,
    reference_logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> dict[str, Any]:
    reference_valid = (reference >= cu_seqlen_ks[:, None]) & (
        reference < cu_seqlen_ke[:, None]
    )
    candidate_valid = (candidate >= cu_seqlen_ks[:, None]) & (
        candidate < cu_seqlen_ke[:, None]
    )
    reference_indices = torch.where(reference_valid, reference, -1).sort(dim=1).values
    candidate_indices = torch.where(candidate_valid, candidate, -1).sort(dim=1).values
    width = reference_indices.shape[1]

    def membership(
        sorted_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.searchsorted(sorted_indices, values)
        in_bounds = positions < width
        positions = positions.clamp_max(width - 1)
        return in_bounds & (torch.gather(sorted_indices, 1, positions) == values)

    reference_valid = reference_indices >= 0
    candidate_valid = candidate_indices >= 0
    reference_in_candidate = membership(candidate_indices, reference_indices)
    candidate_in_reference = membership(reference_indices, candidate_indices)
    reference_only = reference_valid & ~reference_in_candidate
    candidate_only = candidate_valid & ~candidate_in_reference
    mismatch_rows = (reference_only | candidate_only).any(dim=1)
    reference_duplicates = (reference_indices[:, 1:] == reference_indices[:, :-1]) & (
        reference_indices[:, 1:] >= 0
    )
    candidate_duplicates = (candidate_indices[:, 1:] == candidate_indices[:, :-1]) & (
        candidate_indices[:, 1:] >= 0
    )

    def selected_reference_scores(indices: torch.Tensor) -> torch.Tensor:
        valid = indices >= 0
        gathered = torch.gather(
            reference_logits,
            1,
            indices.clamp_min(0).to(torch.int64),
        )
        scores = torch.where(valid, gathered, torch.full_like(gathered, -1e20))
        return scores.sort(dim=1).values

    score_correctness = compare_outputs(
        selected_reference_scores(reference_indices),
        selected_reference_scores(candidate_indices),
        tolerances,
    )
    valid_counts_equal = bool(
        torch.equal(reference_valid.sum(dim=1), candidate_valid.sum(dim=1))
    )
    reference_duplicate_count = int(reference_duplicates.sum().item())
    candidate_duplicate_count = int(candidate_duplicates.sum().item())
    duplicate_counts_equal = bool(
        torch.equal(
            reference_duplicates.sum(dim=1),
            candidate_duplicates.sum(dim=1),
        )
    )
    reference_only_count = int(reference_only.sum().item())
    candidate_only_count = int(candidate_only.sum().item())
    valid_reference_count = int(reference_valid.sum().item())
    common_count = valid_reference_count - reference_only_count
    score_correctness.update(
        {
            "passed": bool(
                score_correctness["passed"]
                and valid_counts_equal
                and duplicate_counts_equal
            ),
            "equivalence": "tie-aware-reference-score-multiset",
            "index_set_exact": reference_only_count == candidate_only_count == 0,
            "index_mismatch_rows": int(mismatch_rows.sum().item()),
            "index_symmetric_difference": (reference_only_count + candidate_only_count),
            "index_overlap_ratio": (
                common_count / valid_reference_count
                if valid_reference_count > 0
                else 1.0
            ),
            "valid_counts_equal": valid_counts_equal,
            "reference_duplicate_count": reference_duplicate_count,
            "candidate_duplicate_count": candidate_duplicate_count,
            "duplicate_counts_equal": duplicate_counts_equal,
        }
    )
    return score_correctness


def build_a03_topk_case(args: Mapping[str, Any]) -> ChainCase:
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_inputs(args)
    baseline_logits = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )
    candidate_logits = torch.empty_like(baseline_logits)
    baseline_output = torch.empty(
        (shape["M"], TOPK_TOKENS),
        device="cuda",
        dtype=torch.int32,
    )
    candidate_output = torch.empty_like(baseline_output)

    def run_baseline() -> torch.Tensor:
        fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=False,
            out=baseline_logits,
        )
        return _run_topk(
            baseline_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=True,
            out=candidate_logits,
        )
        return _run_topk(
            candidate_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_output,
        )

    shape["chain"] = "fp8-mqa-logits-persistent-topk"
    shape["topk"] = TOPK_TOKENS
    return ChainCase(
        baseline=Provider(
            "triton-tf32-topk",
            run_baseline,
            {
                "mma": "tf32",
                "topk": "persistent",
                "topk_order": "unordered-set",
            },
            correctness_transform=_sort_topk_indices,
        ),
        candidate=Provider(
            "triton-native-fp8-topk",
            run_candidate,
            {
                "mma": "fp8",
                "topk": "persistent",
                "topk_order": "unordered-set",
            },
            correctness_transform=_sort_topk_indices,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=0.0,
            rtol=0.0,
            max_mean_relative=0.0,
            require_allclose=True,
        ),
    )


def build_a03_deepgemm_topk_case(args: Mapping[str, Any]) -> ChainCase:
    from vllm.third_party import deep_gemm

    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_inputs(args)
    baseline_logits = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )
    baseline_output = torch.empty(
        (shape["M"], TOPK_TOKENS),
        device="cuda",
        dtype=torch.int32,
    )
    candidate_output = torch.empty_like(baseline_output)
    deep_gemm.set_pdl(True)

    def run_baseline() -> torch.Tensor:
        fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=True,
            out=baseline_logits,
        )
        return _run_topk(
            baseline_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        candidate_logits = _deepgemm_mqa_logits(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
        return _run_topk(
            candidate_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_output,
        )

    shape["chain"] = "fp8-mqa-logits-persistent-topk"
    shape["topk"] = TOPK_TOKENS
    return ChainCase(
        baseline=Provider(
            "triton-native-fp8-topk",
            run_baseline,
            {
                "mma": "fp8",
                "topk": "persistent",
                "topk_order": "unordered-set",
            },
        ),
        candidate=Provider(
            "deepgemm-sm120-fused-topk",
            run_candidate,
            {
                "mma": "fp8",
                "pdl": True,
                "preallocated_logits": False,
                "padded_logits_stride": True,
                "topk": "persistent",
                "topk_order": "unordered-set",
                "topk_correctness": "tie-aware-reference-score-multiset",
            },
            correctness_comparator=lambda reference, candidate, tolerances: (
                _compare_topk_with_ties(
                    reference,
                    candidate,
                    tolerances,
                    baseline_logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                )
            ),
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=4e-6,
            rtol=1e-6,
            require_allclose=True,
        ),
    )


def build_a03_production_topk_case(args: Mapping[str, Any]) -> ChainCase:
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, shape = _make_inputs(args)
    baseline_logits = torch.empty(
        (shape["M"], shape["N"]),
        device="cuda",
        dtype=torch.float32,
    )
    baseline_output = torch.empty(
        (shape["M"], TOPK_TOKENS),
        device="cuda",
        dtype=torch.int32,
    )
    candidate_output = torch.empty_like(baseline_output)

    def run_baseline() -> torch.Tensor:
        fp8_mqa_logits_triton(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            native_fp8=False,
            out=baseline_logits,
        )
        return _run_topk(
            baseline_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            baseline_output,
        )

    def run_candidate() -> torch.Tensor:
        if fp8_fp4_mqa_topk_indices(
            (q, None),
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_output,
        ):
            return candidate_output
        candidate_logits = fp8_fp4_mqa_logits(
            (q, None),
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=False,
        )
        return _run_topk(
            candidate_logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            candidate_output,
        )

    shape["chain"] = "fp8-mqa-logits-persistent-topk-production-dispatch"
    shape["topk"] = TOPK_TOKENS
    return ChainCase(
        baseline=Provider(
            "triton-tf32-topk",
            run_baseline,
            {
                "mma": "tf32",
                "topk": "persistent",
                "topk_order": "unordered-set",
            },
        ),
        candidate=Provider(
            "sm120-production-dispatch-topk",
            run_candidate,
            {
                "preferred_backend": "deepgemm-sm120-fused",
                "fallback_backend": "triton-native-fp8",
                "topk": "persistent",
                "topk_order": "unordered-set",
                "topk_correctness": "tie-aware-reference-score-multiset",
            },
            correctness_comparator=lambda reference, candidate, tolerances: (
                _compare_topk_with_ties(
                    reference,
                    candidate,
                    tolerances,
                    baseline_logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                )
            ),
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(
            atol=4e-6,
            rtol=1e-6,
            require_allclose=True,
        ),
    )
