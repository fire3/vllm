# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.kernels.deepseek_v4.common import (
    BenchmarkConfig,
    CorrectnessTolerances,
    Provider,
    benchmark_pair,
    build_ledger,
    default_ledger_path,
    write_json_atomic,
)


@dataclasses.dataclass(frozen=True)
class ChainCase:
    """A model-faithful pair of complete consumer chains."""

    baseline: Provider
    candidate: Provider
    shape: Mapping[str, Any]
    tolerances: CorrectnessTolerances


ChainFactory = Callable[[Mapping[str, Any]], ChainCase]


def load_factory(specification: str) -> ChainFactory:
    """Load an explicit ``module:function`` chain factory."""

    module_name, separator, function_name = specification.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{specification} is not callable")
    return factory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--factory-args", default="{}")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=500.0)
    parser.add_argument("--measurement-ms", type=float, default=2_000.0)
    parser.add_argument("--min-total-calls", type=int, default=1_000)
    parser.add_argument("--graph-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        default=Path("/home/yyf/flashinfer-sm120-v0613"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    factory_args = json.loads(args.factory_args)
    if not isinstance(factory_args, dict):
        raise TypeError("factory arguments must be a JSON object")
    case = load_factory(args.factory)(factory_args)
    if not all(
        hasattr(case, attribute)
        for attribute in ("baseline", "candidate", "shape", "tolerances")
    ):
        raise TypeError("chain factory must return ChainCase")

    config = BenchmarkConfig(
        rounds=args.rounds,
        warmup_ms=args.warmup_ms,
        measurement_ms=args.measurement_ms,
        min_total_calls=args.min_total_calls,
        graph_repeats=args.graph_repeats,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        nvtx=args.nvtx,
    )
    result = benchmark_pair(
        case.baseline,
        case.candidate,
        shape=case.shape,
        config=config,
        tolerances=case.tolerances,
    )
    ledger = build_ledger(
        operator_id=args.chain_id,
        phase=args.phase,
        candidate=args.candidate_name,
        results=[result],
        config=config,
        repo_root=args.repo_root,
        flashinfer_root=args.flashinfer_root,
        command=sys.argv,
    )
    output = args.output or default_ledger_path(
        args.chain_id,
        args.candidate_name,
        ledger["environment"]["process_uuid"],
    )
    write_json_atomic(output, ledger)
    print(json.dumps({"ledger": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
