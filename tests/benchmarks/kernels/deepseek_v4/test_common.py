# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.deepseek_v4.common import (
    BenchmarkConfig,
    CorrectnessTolerances,
    Provider,
    benchmark_pair,
    compare_outputs,
    paired_bootstrap_ci,
    percentile,
    summarize,
)


def test_statistics_are_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.2) == pytest.approx(1.8)
    assert summarize(values) == {
        "count": 5,
        "p20": pytest.approx(1.8),
        "p50": 3.0,
        "p80": pytest.approx(4.2),
        "mad": 1.0,
        "mean": 3.0,
    }
    first = paired_bootstrap_ci(values, samples=500, seed=7)
    second = paired_bootstrap_ci(values, samples=500, seed=7)
    assert first == second


def test_compare_outputs_supports_fp8_style_aggregate_gate() -> None:
    reference = torch.tensor([100.0, 200.0, 300.0])
    candidate = reference + torch.tensor([0.1, -0.1, 0.1])
    result = compare_outputs(
        reference,
        candidate,
        CorrectnessTolerances(
            atol=0.0,
            rtol=0.0,
            max_mean_relative=0.001,
            min_cosine=0.999,
            require_allclose=False,
        ),
    )
    assert result["passed"]
    assert not result["allclose"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_benchmark_pair_uses_cuda_graph_and_records_abba() -> None:
    value = torch.ones(128, device="cuda")
    baseline_out = torch.empty_like(value)
    candidate_out = torch.empty_like(value)

    def baseline() -> torch.Tensor:
        torch.add(value, 1.0, out=baseline_out)
        return baseline_out

    def candidate() -> torch.Tensor:
        torch.add(value, 1.0, out=candidate_out)
        return candidate_out

    result = benchmark_pair(
        Provider("baseline", baseline),
        Provider("candidate", candidate),
        shape={"name": "smoke"},
        config=BenchmarkConfig(
            rounds=2,
            warmup_replays=1,
            warmup_ms=1.0,
            measurement_ms=2.0,
            min_total_calls=10,
            graph_repeats=2,
            bootstrap_samples=100,
        ),
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
    assert result["cuda_graph"]["enabled"]
    assert result["raw_samples"][0]["order"] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert result["raw_samples"][1]["order"] == [
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_benchmark_pair_applies_correctness_transform_outside_timing() -> None:
    baseline_source = torch.tensor([[3, 1, 2]], device="cuda")
    candidate_source = torch.tensor([[2, 3, 1]], device="cuda")
    baseline_out = torch.empty_like(baseline_source)
    candidate_out = torch.empty_like(candidate_source)
    transform_calls = {"baseline": 0, "candidate": 0}

    def baseline() -> torch.Tensor:
        baseline_out.copy_(baseline_source)
        return baseline_out

    def candidate() -> torch.Tensor:
        candidate_out.copy_(candidate_source)
        return candidate_out

    def transform(label: str, output: torch.Tensor) -> torch.Tensor:
        transform_calls[label] += 1
        return output.sort(dim=-1).values

    result = benchmark_pair(
        Provider(
            "baseline",
            baseline,
            correctness_transform=lambda output: transform("baseline", output),
        ),
        Provider(
            "candidate",
            candidate,
            correctness_transform=lambda output: transform("candidate", output),
        ),
        shape={"name": "unordered-output"},
        config=BenchmarkConfig(
            rounds=2,
            warmup_replays=1,
            warmup_ms=1.0,
            measurement_ms=2.0,
            min_total_calls=10,
            graph_repeats=2,
            bootstrap_samples=100,
        ),
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )

    assert result["correctness"]["passed"]
    assert transform_calls == {"baseline": 1, "candidate": 1}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_benchmark_pair_uses_custom_correctness_comparator_once() -> None:
    baseline_source = torch.tensor([1.0], device="cuda")
    candidate_source = torch.tensor([2.0], device="cuda")
    baseline_out = torch.empty_like(baseline_source)
    candidate_out = torch.empty_like(candidate_source)
    comparator_calls = 0

    def baseline() -> torch.Tensor:
        baseline_out.copy_(baseline_source)
        return baseline_out

    def candidate() -> torch.Tensor:
        candidate_out.copy_(candidate_source)
        return candidate_out

    def comparator(
        reference: torch.Tensor,
        candidate_output: torch.Tensor,
        tolerances: CorrectnessTolerances,
    ) -> dict[str, object]:
        nonlocal comparator_calls
        comparator_calls += 1
        assert float(reference.item()) == 1.0
        assert float(candidate_output.item()) == 2.0
        assert tolerances.atol == 0.0
        return {"passed": True, "custom": True}

    result = benchmark_pair(
        Provider("baseline", baseline),
        Provider("candidate", candidate, correctness_comparator=comparator),
        shape={"name": "custom-correctness"},
        config=BenchmarkConfig(
            rounds=2,
            warmup_replays=1,
            warmup_ms=1.0,
            measurement_ms=2.0,
            min_total_calls=10,
            graph_repeats=2,
            bootstrap_samples=100,
        ),
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )

    assert result["correctness"] == {"passed": True, "custom": True}
    assert comparator_calls == 1
