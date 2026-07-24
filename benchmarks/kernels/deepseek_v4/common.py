# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from vllm.utils.torch_utils import set_random_seed

SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = Path(
    os.environ.get(
        "DSV4_KERNEL_STATE_DIR",
        "/home/yyf/vllm/.omx/state/dsv4-sm120-kernels",
    )
)


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    """Controls one paired CUDA Graph benchmark process."""

    rounds: int = 5
    warmup_replays: int = 5
    warmup_ms: float = 500.0
    measurement_ms: float = 2_000.0
    min_total_calls: int = 1_000
    graph_repeats: int = 20
    bootstrap_samples: int = 20_000
    seed: int = 0
    nvtx: bool = False

    def __post_init__(self) -> None:
        if self.rounds < 2:
            raise ValueError("rounds must be at least 2")
        if self.warmup_replays < 1:
            raise ValueError("warmup_replays must be positive")
        if self.warmup_ms <= 0.0:
            raise ValueError("warmup_ms must be positive")
        if self.measurement_ms <= 0.0:
            raise ValueError("measurement_ms must be positive")
        if self.min_total_calls < 1:
            raise ValueError("min_total_calls must be positive")
        if self.graph_repeats < 1:
            raise ValueError("graph_repeats must be positive")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")


@dataclasses.dataclass(frozen=True)
class Provider:
    """A provider with static CUDA inputs suitable for graph capture."""

    name: str
    fn: Callable[[], torch.Tensor]
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    correctness_transform: Callable[[torch.Tensor], torch.Tensor] | None = None
    correctness_comparator: (
        Callable[
            [torch.Tensor, torch.Tensor, CorrectnessTolerances],
            dict[str, Any],
        ]
        | None
    ) = None


@dataclasses.dataclass(frozen=True)
class CorrectnessTolerances:
    atol: float
    rtol: float
    max_mean_relative: float | None = None
    min_cosine: float | None = None
    require_allclose: bool = True


@dataclasses.dataclass
class GraphRunner:
    """Captures repeated provider calls and reports per-call GPU latency."""

    provider: Provider
    repeats: int
    stream: torch.cuda.Stream | None = None
    graph: torch.cuda.CUDAGraph | None = None
    output: torch.Tensor | None = None

    def capture(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")

        current = torch.cuda.current_stream()
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.output = self.provider.fn()
        current.wait_stream(self.stream)
        torch.accelerator.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            for _ in range(self.repeats):
                self.output = self.provider.fn()
        torch.accelerator.synchronize()

    def replay(self) -> None:
        if self.graph is None or self.stream is None:
            raise RuntimeError("CUDA Graph has not been captured")
        with torch.cuda.stream(self.stream):
            self.graph.replay()

    def measure_us(self, graph_replays: int = 1) -> float:
        if self.graph is None or self.stream is None:
            raise RuntimeError("CUDA Graph has not been captured")
        if graph_replays < 1:
            raise ValueError("graph_replays must be positive")

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record()
            for _ in range(graph_replays):
                self.graph.replay()
            end.record()
        end.synchronize()
        calls = self.repeats * graph_replays
        return start.elapsed_time(end) * 1_000.0 / calls

    def warmup_for(self, duration_ms: float, min_replays: int) -> int:
        if self.graph is None:
            raise RuntimeError("CUDA Graph has not been captured")
        replay_count = 0
        started = time.perf_counter()
        while (
            replay_count < min_replays
            or (time.perf_counter() - started) * 1_000.0 < duration_ms
        ):
            for _ in range(10):
                self.replay()
            replay_count += 10
            torch.accelerator.synchronize()
        return replay_count


@contextlib.contextmanager
def nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    """Emit an NVTX range without adding a dependency on the nvtx package."""

    if not enabled:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile."""

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: Sequence[float]) -> dict[str, float | int]:
    """Summarize latency or paired-improvement samples."""

    if not values:
        raise ValueError("values must not be empty")
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "p20": percentile(values, 0.2),
        "p50": median,
        "p80": percentile(values, 0.8),
        "mad": statistics.median(deviations),
        "mean": statistics.fmean(values),
    }


def paired_bootstrap_ci(
    paired_improvements_pct: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the median paired relative-latency improvement."""

    if not paired_improvements_pct:
        raise ValueError("paired samples must not be empty")
    rng = random.Random(seed)
    count = len(paired_improvements_pct)
    estimates = []
    for _ in range(samples):
        resample = [paired_improvements_pct[rng.randrange(count)] for _ in range(count)]
        estimates.append(statistics.median(resample))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def compare_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerances: CorrectnessTolerances,
) -> dict[str, Any]:
    """Compare provider outputs and retain useful error diagnostics."""

    if reference.shape != candidate.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }

    reference_f32 = reference.float()
    candidate_f32 = candidate.float()
    finite = bool(
        torch.isfinite(reference_f32).all() and torch.isfinite(candidate_f32).all()
    )
    difference = (candidate_f32 - reference_f32).abs()
    denominator = reference_f32.abs().clamp_min(torch.finfo(torch.float32).eps)
    max_abs = float(difference.max().item())
    max_rel = float((difference / denominator).max().item())
    mean_abs = float(difference.mean().item())
    mean_relative = float(
        (difference.mean() / reference_f32.abs().mean().clamp_min(1e-12)).item()
    )
    reference_flat = reference_f32.flatten().double()
    candidate_flat = candidate_f32.flatten().double()
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_flat,
            candidate_flat,
            dim=0,
            eps=1e-12,
        ).item()
    )
    allclose = torch.allclose(
        candidate_f32,
        reference_f32,
        atol=tolerances.atol,
        rtol=tolerances.rtol,
    )
    passed = finite
    if tolerances.require_allclose:
        passed &= bool(allclose)
    if tolerances.max_mean_relative is not None:
        passed &= mean_relative <= tolerances.max_mean_relative
    if tolerances.min_cosine is not None:
        passed &= cosine >= tolerances.min_cosine
    return {
        "passed": bool(passed),
        "finite": finite,
        "allclose": bool(allclose),
        "atol": tolerances.atol,
        "rtol": tolerances.rtol,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "mean_abs": mean_abs,
        "mean_relative": mean_relative,
        "cosine": cosine,
        "max_mean_relative": tolerances.max_mean_relative,
        "min_cosine": tolerances.min_cosine,
    }


def benchmark_pair(
    baseline: Provider,
    candidate: Provider,
    *,
    shape: Mapping[str, Any],
    config: BenchmarkConfig,
    tolerances: CorrectnessTolerances,
) -> dict[str, Any]:
    """Run correctness plus balanced ABBA timing in one CUDA process."""

    if baseline.name == candidate.name:
        raise ValueError("baseline and candidate providers must differ")
    set_random_seed(config.seed)

    with torch.inference_mode(), nvtx_range("correctness", config.nvtx):
        reference = baseline.fn().detach().clone()
        candidate_output = candidate.fn().detach().clone()
        comparator = candidate.correctness_comparator
        if baseline.correctness_comparator is not None:
            if comparator is not None:
                raise ValueError(
                    "only one provider may define a correctness comparator"
                )
            comparator = baseline.correctness_comparator
        if comparator is None:
            if baseline.correctness_transform is not None:
                reference = baseline.correctness_transform(reference)
            if candidate.correctness_transform is not None:
                candidate_output = candidate.correctness_transform(candidate_output)
        torch.accelerator.synchronize()
    correctness = (
        comparator(reference, candidate_output, tolerances)
        if comparator is not None
        else compare_outputs(reference, candidate_output, tolerances)
    )
    if not correctness["passed"]:
        raise AssertionError(f"provider correctness failed: {correctness}")

    runners = {
        "baseline": GraphRunner(baseline, config.graph_repeats),
        "candidate": GraphRunner(candidate, config.graph_repeats),
    }
    with torch.inference_mode(), nvtx_range("capture", config.nvtx):
        runners["baseline"].capture()
        runners["candidate"].capture()
        warmup_graph_replays = {
            label: runner.warmup_for(config.warmup_ms, config.warmup_replays)
            for label, runner in runners.items()
        }

    calibration_us = {label: runner.measure_us() for label, runner in runners.items()}
    observations_per_provider = config.rounds * 2
    target_observation_ms = config.measurement_ms / observations_per_provider
    slowest_graph_ms = max(calibration_us.values()) * config.graph_repeats / 1_000.0
    duration_replays = math.ceil(target_observation_ms / slowest_graph_ms)
    minimum_call_replays = math.ceil(
        config.min_total_calls / (observations_per_provider * config.graph_repeats)
    )
    graph_replays_per_observation = max(
        1,
        duration_replays,
        minimum_call_replays,
    )

    raw_samples: list[dict[str, Any]] = []
    baseline_round_us: list[float] = []
    candidate_round_us: list[float] = []
    paired_improvements_pct: list[float] = []
    for round_index in range(config.rounds):
        labels = (
            ["baseline", "candidate", "candidate", "baseline"]
            if round_index % 2 == 0
            else ["candidate", "baseline", "baseline", "candidate"]
        )
        observations = []
        with (
            torch.inference_mode(),
            nvtx_range(
                f"round_{round_index}_{''.join(label[0] for label in labels)}",
                config.nvtx,
            ),
        ):
            for sequence_index, label in enumerate(labels):
                latency_us = runners[label].measure_us(graph_replays_per_observation)
                observations.append(
                    {
                        "sequence_index": sequence_index,
                        "provider": label,
                        "latency_us": latency_us,
                        "graph_replays": graph_replays_per_observation,
                        "kernel_calls": (
                            graph_replays_per_observation * config.graph_repeats
                        ),
                    }
                )

        baseline_us = statistics.fmean(
            item["latency_us"]
            for item in observations
            if item["provider"] == "baseline"
        )
        candidate_us = statistics.fmean(
            item["latency_us"]
            for item in observations
            if item["provider"] == "candidate"
        )
        improvement_pct = (baseline_us - candidate_us) / baseline_us * 100.0
        baseline_round_us.append(baseline_us)
        candidate_round_us.append(candidate_us)
        paired_improvements_pct.append(improvement_pct)
        raw_samples.append(
            {
                "round": round_index,
                "order": labels,
                "observations": observations,
                "baseline_us": baseline_us,
                "candidate_us": candidate_us,
                "candidate_improvement_pct": improvement_pct,
            }
        )

    ci_low, ci_high = paired_bootstrap_ci(
        paired_improvements_pct,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    return {
        "shape": dict(shape),
        "cuda_graph": {
            "enabled": True,
            "calls_per_replay": config.graph_repeats,
            "graph_replays_per_observation": graph_replays_per_observation,
            "warmup_graph_replays": warmup_graph_replays,
            "calibration_us": calibration_us,
        },
        "correctness": correctness,
        "providers": {
            "baseline": {"name": baseline.name, "metadata": dict(baseline.metadata)},
            "candidate": {
                "name": candidate.name,
                "metadata": dict(candidate.metadata),
            },
        },
        "raw_samples": raw_samples,
        "summary": {
            "baseline_us": summarize(baseline_round_us),
            "candidate_us": summarize(candidate_round_us),
            "paired_improvement_pct": {
                **summarize(paired_improvements_pct),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            },
        },
    }


def _run(command: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _git_fingerprint(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha": _run(["git", "rev-parse", "HEAD"], path),
        "branch": _run(["git", "branch", "--show-current"], path),
        "dirty": bool(_run(["git", "status", "--porcelain=v1"], path)),
    }


def collect_environment(
    *,
    repo_root: Path,
    flashinfer_root: Path,
) -> dict[str, Any]:
    """Collect exact source, package, GPU, and process identifiers."""

    flashinfer_file = "unavailable"
    try:
        import flashinfer

        flashinfer_file = str(Path(flashinfer.__file__).resolve())
    except ImportError:
        pass

    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "device_count": torch.accelerator.device_count(),
        }
    nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,temperature.gpu,clocks.sm,"
            "clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "captured_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "process_uuid": str(uuid.uuid4()),
        "pid": os.getpid(),
        "vllm_source": _git_fingerprint(repo_root),
        "flashinfer_source": _git_fingerprint(flashinfer_root),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "vllm_version": _package_version("vllm"),
        "flashinfer_version": _package_version("flashinfer-python"),
        "flashinfer_file": flashinfer_file,
        "gpu": gpu,
        "nvidia_smi": nvidia_smi,
    }


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def build_ledger(
    *,
    operator_id: str,
    phase: str,
    candidate: str,
    results: Sequence[Mapping[str, Any]],
    config: BenchmarkConfig,
    repo_root: Path,
    flashinfer_root: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Build a serializable evidence ledger for one benchmark process."""

    payload = {
        "schema_version": SCHEMA_VERSION,
        "operator_id": operator_id,
        "phase": phase,
        "candidate": candidate,
        "status": "MEASURED",
        "environment": collect_environment(
            repo_root=repo_root,
            flashinfer_root=flashinfer_root,
        ),
        "benchmark_config": dataclasses.asdict(config),
        "command": list(command),
        "results": list(results),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON without exposing a partially written ledger."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def default_ledger_path(
    operator_id: str,
    candidate: str,
    process_uuid: str,
) -> Path:
    safe_candidate = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in candidate
    )
    return (
        DEFAULT_STATE_DIR
        / "ledgers"
        / operator_id
        / f"{safe_candidate}-{process_uuid}.json"
    )
