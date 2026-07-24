# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from benchmarks.kernels.deepseek_v4.compare import compare_ledgers


def make_result(shape: str, candidate_us: float) -> dict[str, Any]:
    raw_samples = [
        {
            "round": round_index,
            "baseline_us": 10.0,
            "candidate_us": candidate_us,
            "candidate_improvement_pct": (10.0 - candidate_us) * 10.0,
        }
        for round_index in range(5)
    ]
    return {
        "shape": {"name": shape},
        "correctness": {"passed": True},
        "cuda_graph": {"enabled": True},
        "raw_samples": raw_samples,
    }


def make_ledger(
    process_index: int,
    candidate_us: float = 9.0,
    coverage_candidate_us: float | None = None,
) -> dict[str, Any]:
    results = [make_result("target-shape", candidate_us)]
    if coverage_candidate_us is not None:
        results.append(make_result("coverage-shape", coverage_candidate_us))
    return {
        "schema_version": 1,
        "operator_id": "A01",
        "candidate": "cutlass-vs-triton",
        "benchmark_config": {
            "rounds": 5,
            "warmup_ms": 500.0,
            "measurement_ms": 2_000.0,
            "min_total_calls": 1_000,
        },
        "environment": {
            "process_uuid": f"process-{process_index}",
            "vllm_source": {"sha": "vllm-sha"},
            "flashinfer_source": {"sha": "flashinfer-sha"},
        },
        "results": results,
    }


def test_compare_ledgers_keeps_candidate_that_passes_fixed_gate() -> None:
    comparison = compare_ledgers(
        [make_ledger(index) for index in range(5)],
        bootstrap_samples=500,
    )
    assert comparison["decision"] == "KEPT"
    assert comparison["shapes"][0]["passed"]
    assert comparison["shapes"][0]["absolute_improvement_us"] == 1.0


def test_compare_ledgers_allows_neutral_non_target_shape() -> None:
    comparison = compare_ledgers(
        [make_ledger(index, coverage_candidate_us=10.0) for index in range(5)],
        target_shapes={"target-shape"},
        bootstrap_samples=500,
    )
    assert comparison["decision"] == "KEPT"
    coverage = next(
        shape
        for shape in comparison["shapes"]
        if shape["shape_key"] == "coverage-shape"
    )
    assert coverage["passed"]
    assert not coverage["target"]


def test_compare_ledgers_rejects_insufficient_processes() -> None:
    comparison = compare_ledgers(
        [make_ledger(index) for index in range(4)],
        bootstrap_samples=500,
    )
    assert comparison["decision"] == "REJECTED"
    assert not comparison["shapes"][0]["checks"]["independent_processes"]


def test_compare_ledgers_rejects_target_shape_regression() -> None:
    comparison = compare_ledgers(
        [make_ledger(index, candidate_us=10.5) for index in range(5)],
        bootstrap_samples=500,
    )
    assert comparison["decision"] == "REJECTED"
    assert not comparison["shapes"][0]["checks"]["relative_improvement"]


def test_compare_ledgers_rejects_non_target_shape_regression() -> None:
    comparison = compare_ledgers(
        [make_ledger(index, coverage_candidate_us=10.5) for index in range(5)],
        target_shapes={"target-shape"},
        bootstrap_samples=500,
    )
    assert comparison["decision"] == "REJECTED"
    coverage = next(
        shape
        for shape in comparison["shapes"]
        if shape["shape_key"] == "coverage-shape"
    )
    assert not coverage["checks"]["no_coverage_shape_regression"]
