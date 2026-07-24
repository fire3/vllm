# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kernels.deepseek_v4.common import (
    SCHEMA_VERSION,
    paired_bootstrap_ci,
    summarize,
    write_json_atomic,
)

MIN_RELATIVE_IMPROVEMENT_PCT = 3.0
MIN_ABSOLUTE_IMPROVEMENT_US = 0.5
MAX_COVERAGE_SHAPE_REGRESSION_PCT = 3.0
DEFAULT_MIN_PROCESSES = 5


def _shape_key(shape: Mapping[str, Any]) -> str:
    name = shape.get("name")
    if name:
        return str(name)
    return json.dumps(shape, sort_keys=True, separators=(",", ":"))


def load_ledgers(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load and validate process ledgers."""

    ledgers = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            ledger = json.load(handle)
        if ledger.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema in {path}")
        ledgers.append(ledger)
    if not ledgers:
        raise ValueError("at least one ledger is required")

    operator_ids = {ledger["operator_id"] for ledger in ledgers}
    candidates = {ledger["candidate"] for ledger in ledgers}
    if len(operator_ids) != 1:
        raise ValueError(f"mixed operator IDs: {sorted(operator_ids)}")
    if len(candidates) != 1:
        raise ValueError(f"mixed candidates: {sorted(candidates)}")
    return ledgers


def compare_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    target_shapes: set[str] | None = None,
    min_processes: int = DEFAULT_MIN_PROCESSES,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate independent processes and apply the fixed keep gate."""

    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(
        list
    )
    for ledger in ledgers:
        for result in ledger["results"]:
            grouped[_shape_key(result["shape"])].append((ledger, result))

    if target_shapes is None:
        target_shapes = set(grouped)
    missing_shapes = target_shapes - set(grouped)
    shape_results = []
    all_coverage_pass = not missing_shapes

    for shape_key in sorted(grouped):
        entries = grouped[shape_key]
        process_ids = {ledger["environment"]["process_uuid"] for ledger, _ in entries}
        baseline_us = []
        candidate_us = []
        paired_pct = []
        correctness_passed = True
        cuda_graph_enabled = True
        protocol_compliant = True
        for ledger, result in entries:
            correctness_passed &= bool(result["correctness"]["passed"])
            cuda_graph_enabled &= bool(result["cuda_graph"]["enabled"])
            config = ledger["benchmark_config"]
            protocol_compliant &= (
                int(config["rounds"]) >= 5
                and float(config["warmup_ms"]) >= 500.0
                and float(config["measurement_ms"]) >= 2_000.0
                and int(config["min_total_calls"]) >= 1_000
            )
            for sample in result["raw_samples"]:
                baseline_us.append(float(sample["baseline_us"]))
                candidate_us.append(float(sample["candidate_us"]))
                paired_pct.append(float(sample["candidate_improvement_pct"]))

        baseline_summary = summarize(baseline_us)
        candidate_summary = summarize(candidate_us)
        paired_summary = summarize(paired_pct)
        ci_low, ci_high = paired_bootstrap_ci(
            paired_pct,
            samples=bootstrap_samples,
            seed=seed,
        )
        baseline_p50 = float(baseline_summary["p50"])
        candidate_p50 = float(candidate_summary["p50"])
        absolute_improvement_us = baseline_p50 - candidate_p50
        relative_improvement_pct = absolute_improvement_us / baseline_p50 * 100.0
        regression_pct = -relative_improvement_pct
        target = shape_key in target_shapes

        checks = {
            "correctness": correctness_passed,
            "cuda_graph": cuda_graph_enabled,
            "timing_protocol": protocol_compliant,
            "independent_processes": len(process_ids) >= min_processes,
            "relative_improvement": (
                not target or relative_improvement_pct >= MIN_RELATIVE_IMPROVEMENT_PCT
            ),
            "absolute_improvement": (
                not target or absolute_improvement_us >= MIN_ABSOLUTE_IMPROVEMENT_US
            ),
            "paired_ci_positive": not target or ci_low > 0.0,
            "no_coverage_shape_regression": (
                regression_pct <= MAX_COVERAGE_SHAPE_REGRESSION_PCT
            ),
        }
        shape_passed = all(checks.values())
        all_coverage_pass &= shape_passed
        shape_results.append(
            {
                "shape_key": shape_key,
                "shape": dict(entries[0][1]["shape"]),
                "target": target,
                "process_count": len(process_ids),
                "baseline_us": baseline_summary,
                "candidate_us": candidate_summary,
                "paired_improvement_pct": {
                    **paired_summary,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                },
                "absolute_improvement_us": absolute_improvement_us,
                "relative_improvement_pct": relative_improvement_pct,
                "checks": checks,
                "passed": shape_passed,
            }
        )

    source_shas = {
        (
            ledger["environment"]["vllm_source"]["sha"],
            ledger["environment"]["flashinfer_source"]["sha"],
        )
        for ledger in ledgers
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operator_id": ledgers[0]["operator_id"],
        "candidate": ledgers[0]["candidate"],
        "decision": "KEPT" if all_coverage_pass else "REJECTED",
        "gate": {
            "min_relative_improvement_pct": MIN_RELATIVE_IMPROVEMENT_PCT,
            "min_absolute_improvement_us": MIN_ABSOLUTE_IMPROVEMENT_US,
            "max_coverage_shape_regression_pct": (MAX_COVERAGE_SHAPE_REGRESSION_PCT),
            "min_independent_processes": min_processes,
            "bootstrap_samples": bootstrap_samples,
        },
        "missing_target_shapes": sorted(missing_shapes),
        "source_sha_pairs": [
            {"vllm": vllm_sha, "flashinfer": flashinfer_sha}
            for vllm_sha, flashinfer_sha in sorted(source_shas)
        ],
        "shapes": shape_results,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledgers", type=Path, nargs="+")
    parser.add_argument("--target-shape", action="append", default=[])
    parser.add_argument("--min-processes", type=int, default=DEFAULT_MIN_PROCESSES)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Return a non-zero status when the candidate is rejected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ledgers = load_ledgers(args.ledgers)
    target_shapes = set(args.target_shape) or None
    comparison = compare_ledgers(
        ledgers,
        target_shapes=target_shapes,
        min_processes=args.min_processes,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.output:
        write_json_atomic(args.output, comparison)
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return int(args.check and comparison["decision"] != "KEPT")


if __name__ == "__main__":
    sys.exit(main())
