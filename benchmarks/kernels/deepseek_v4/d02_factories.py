# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import _gumbel_sample_kernel, gumbel_sample

_PROCESSED_GUARD = -777.0


@triton.jit
def _reduce_gumbel_argmax_kernel(
    sampled_ptr,
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    NUM_BLOCKS: tl.constexpr,
    REDUCTION_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    block_offsets = tl.arange(0, REDUCTION_SIZE)
    block_mask = block_offsets < NUM_BLOCKS
    local_values = tl.load(
        local_max_ptr + token_idx * local_max_stride + block_offsets,
        mask=block_mask,
        other=float("-inf"),
    )
    nan_offsets = tl.where(local_values != local_values, block_offsets, REDUCTION_SIZE)
    first_nan = tl.min(nan_offsets, axis=0)
    finite_values = tl.where(local_values != local_values, float("-inf"), local_values)
    max_value = tl.max(finite_values, axis=0)
    max_offsets = tl.where(
        block_mask & (finite_values == max_value), block_offsets, REDUCTION_SIZE
    )
    first_max = tl.min(max_offsets, axis=0)
    winner = tl.where(first_nan < NUM_BLOCKS, first_nan, first_max)
    sampled = tl.load(local_argmax_ptr + token_idx * local_argmax_stride + winner)
    tl.store(sampled_ptr + token_idx, sampled)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    num_tokens = int(args.get("num_tokens", 4))
    num_reqs = int(args.get("num_reqs", num_tokens))
    valid_tokens = int(args.get("valid_tokens", num_tokens))
    vocab_size = int(args.get("vocab_size", 129280))
    apply_temperature = bool(args.get("apply_temperature", True))
    use_fp64 = bool(args.get("use_fp64", False))
    output_mode = str(args.get("output_mode", "none"))
    num_output_cols = int(args.get("num_output_cols", 7))
    output_col = int(args.get("output_col", 3))
    candidate_mode = str(args.get("candidate_mode", "mirror"))
    logits_stride_extra = int(args.get("logits_stride_extra", 0))
    seed = int(args.get("seed", 0))

    if num_tokens <= 0 or num_reqs <= 0 or vocab_size <= 0:
        raise ValueError("D02 token, request, and vocabulary counts must be positive")
    if not 0 <= valid_tokens <= num_tokens:
        raise ValueError("D02 valid_tokens must be in [0, num_tokens]")
    if output_mode not in ("none", "scalar", "per-token"):
        raise ValueError("D02 output_mode must be none, scalar, or per-token")
    if candidate_mode not in (
        "mirror",
        "fused",
        "fused512",
        "fused2048",
        "torch-argmax",
        "dispatch",
    ):
        raise ValueError("unsupported D02 candidate mode")
    if logits_stride_extra < 0:
        raise ValueError("D02 logits_stride_extra must be non-negative")

    raw_temperatures = args.get("temperatures", [1.0])
    if isinstance(raw_temperatures, float | int):
        raw_temperatures = [float(raw_temperatures)]
    temperatures = [float(value) for value in raw_temperatures]
    if not temperatures:
        raise ValueError("D02 temperatures must not be empty")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits_storage = torch.randn(
        (num_tokens, vocab_size + logits_stride_extra),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    logits = logits_storage[:, :vocab_size]
    expanded_idx_mapping = torch.full(
        (num_tokens,), -1, dtype=torch.int32, device="cuda"
    )
    if valid_tokens:
        expanded_idx_mapping[:valid_tokens] = (
            torch.arange(valid_tokens, dtype=torch.int32, device="cuda") % num_reqs
        )
    temperature = torch.tensor(
        [temperatures[index % len(temperatures)] for index in range(num_reqs)],
        dtype=torch.float32,
        device="cuda",
    )
    seeds = torch.arange(
        0x12340000,
        0x12340000 + num_reqs,
        dtype=torch.int64,
        device="cuda",
    )
    positions = torch.arange(71, 71 + num_tokens, dtype=torch.int64, device="cuda")

    processed_col: torch.Tensor | None = None
    if output_mode == "scalar":
        if not 0 <= output_col < num_output_cols:
            raise ValueError("D02 scalar output column is out of range")
        processed_col = torch.tensor(output_col, dtype=torch.int32, device="cuda")
    elif output_mode == "per-token":
        processed_col = (
            torch.arange(num_tokens, dtype=torch.int32, device="cuda") % num_output_cols
        )

    if output_mode != "none":
        seen: set[tuple[int, int]] = set()
        mapping_host = expanded_idx_mapping.tolist()
        cols_host = (
            processed_col.tolist()
            if processed_col is not None and processed_col.ndim > 0
            else [output_col] * num_tokens
        )
        for token_idx in range(valid_tokens):
            key = (int(mapping_host[token_idx]), int(cols_host[token_idx]))
            if key in seen:
                raise ValueError("D02 processed-logits writes must not alias")
            seen.add(key)

    expected_processed = None
    if output_mode != "none":
        expected_processed = torch.full(
            (num_reqs, num_output_cols, vocab_size),
            _PROCESSED_GUARD,
            dtype=torch.float32,
            device="cuda",
        )
        for token_idx in range(valid_tokens):
            req_idx = int(expanded_idx_mapping[token_idx].item())
            col = (
                int(processed_col[token_idx].item())
                if processed_col is not None and processed_col.ndim > 0
                else output_col
            )
            row = logits[token_idx]
            temp = float(temperature[req_idx].item())
            if apply_temperature and temp != 0.0:
                row = row / temp
            expected_processed[req_idx, col].copy_(row)

    if candidate_mode == "torch-argmax":
        if valid_tokens != num_tokens or output_mode != "none":
            raise ValueError("D02 torch-argmax requires valid rows and no writeback")
        if any(value != 0.0 for value in temperatures):
            raise ValueError("D02 torch-argmax is valid only for all-greedy inputs")

    return {
        "logits": logits,
        "logits_snapshot": logits.clone(),
        "expanded_idx_mapping": expanded_idx_mapping,
        "mapping_snapshot": expanded_idx_mapping.clone(),
        "temperature": temperature,
        "temperature_snapshot": temperature.clone(),
        "seeds": seeds,
        "seeds_snapshot": seeds.clone(),
        "positions": positions,
        "positions_snapshot": positions.clone(),
        "processed_col": processed_col,
        "processed_col_snapshot": (
            processed_col.clone() if processed_col is not None else None
        ),
        "expected_processed": expected_processed,
        "num_tokens": num_tokens,
        "num_reqs": num_reqs,
        "valid_tokens": valid_tokens,
        "vocab_size": vocab_size,
        "apply_temperature": apply_temperature,
        "use_fp64": use_fp64,
        "output_mode": output_mode,
        "num_output_cols": num_output_cols,
        "candidate_mode": candidate_mode,
        "shape": {
            "name": (
                f"t{num_tokens}-r{num_reqs}-valid{valid_tokens}-v{vocab_size}"
                f"-temp{'_'.join(str(value) for value in temperatures)}"
                f"-apply{int(apply_temperature)}-fp64{int(use_fp64)}"
                f"-out{output_mode}-stride{logits_stride_extra}"
            ),
            "num_tokens": num_tokens,
            "num_reqs": num_reqs,
            "valid_tokens": valid_tokens,
            "vocab_size": vocab_size,
            "temperatures": temperatures,
            "apply_temperature": apply_temperature,
            "use_fp64": use_fp64,
            "output_mode": output_mode,
            "num_output_cols": num_output_cols,
            "logits_stride_extra": logits_stride_extra,
            "candidate_mode": candidate_mode,
            "chain": "gumbel-block-argmax-and-final-reduction",
        },
    }


def _allocate_state(inputs: Mapping[str, Any], block_size: int) -> dict[str, Any]:
    num_blocks = triton.cdiv(inputs["vocab_size"], block_size)
    local_max_dtype = torch.float64 if inputs["use_fp64"] else torch.float32
    processed = (
        torch.full(
            (
                inputs["num_reqs"],
                inputs["num_output_cols"],
                inputs["vocab_size"],
            ),
            _PROCESSED_GUARD,
            dtype=torch.float32,
            device="cuda",
        )
        if inputs["output_mode"] != "none"
        else None
    )
    return {
        "block_size": block_size,
        "num_blocks": num_blocks,
        "local_argmax": torch.empty(
            (inputs["num_tokens"], num_blocks), dtype=torch.int64, device="cuda"
        ),
        "local_max": torch.empty(
            (inputs["num_tokens"], num_blocks), dtype=local_max_dtype, device="cuda"
        ),
        "sampled": torch.empty(
            (inputs["num_tokens"],), dtype=torch.int64, device="cuda"
        ),
        "processed": processed,
    }


def _launch_first_stage(state: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    processed = state["processed"]
    _gumbel_sample_kernel[(inputs["num_tokens"], state["num_blocks"])](
        state["local_argmax"],
        state["local_argmax"].stride(0),
        state["local_max"],
        state["local_max"].stride(0),
        processed,
        processed.stride(0) if processed is not None else 0,
        inputs["processed_col"],
        inputs["logits"],
        inputs["logits"].stride(0),
        inputs["expanded_idx_mapping"],
        inputs["seeds"],
        inputs["positions"],
        inputs["temperature"],
        inputs["vocab_size"],
        BLOCK_SIZE=state["block_size"],
        APPLY_TEMPERATURE=inputs["apply_temperature"],
        USE_FP64=inputs["use_fp64"],
        PER_TOKEN_COL=(
            inputs["processed_col"] is not None and inputs["processed_col"].ndim > 0
        ),
    )


def _launch_frozen_baseline(
    state: Mapping[str, Any], inputs: Mapping[str, Any]
) -> torch.Tensor:
    _launch_first_stage(state, inputs)
    max_block_idx = state["local_max"].argmax(dim=-1, keepdim=True)
    return state["local_argmax"].gather(dim=-1, index=max_block_idx).view(-1)


def _launch_fused_candidate(
    state: Mapping[str, Any], inputs: Mapping[str, Any]
) -> torch.Tensor:
    _launch_first_stage(state, inputs)
    reduction_size = _next_power_of_two(state["num_blocks"])
    _reduce_gumbel_argmax_kernel[(inputs["num_tokens"],)](
        state["sampled"],
        state["local_argmax"],
        state["local_argmax"].stride(0),
        state["local_max"],
        state["local_max"].stride(0),
        NUM_BLOCKS=state["num_blocks"],
        REDUCTION_SIZE=reduction_size,
    )
    return state["sampled"]


def build_d02_gumbel_sample_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    baseline = _allocate_state(inputs, 1024)
    candidate_block_size = {
        "fused512": 512,
        "fused2048": 2048,
    }.get(inputs["candidate_mode"], 1024)
    candidate = _allocate_state(inputs, candidate_block_size)

    def run_baseline() -> torch.Tensor:
        return _launch_frozen_baseline(baseline, inputs)

    def run_candidate() -> torch.Tensor:
        if inputs["candidate_mode"] == "mirror":
            return _launch_frozen_baseline(candidate, inputs)
        if inputs["candidate_mode"] == "torch-argmax":
            return inputs["logits"].argmax(dim=-1)
        if inputs["candidate_mode"] == "dispatch":
            return gumbel_sample(
                inputs["logits"],
                inputs["expanded_idx_mapping"],
                inputs["temperature"],
                inputs["seeds"],
                inputs["positions"],
                inputs["apply_temperature"],
                output_processed_logits=candidate["processed"],
                output_processed_logits_col=inputs["processed_col"],
                use_fp64=inputs["use_fp64"],
            )
        return _launch_fused_candidate(candidate, inputs)

    def compare(
        baseline_output: torch.Tensor,
        candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "sampled_exact": torch.equal(baseline_output, candidate_output),
            "sampled_dtype": candidate_output.dtype == torch.int64,
            "sampled_range": bool(
                (candidate_output >= 0).all()
                and (candidate_output < inputs["vocab_size"]).all()
            ),
            "logits_unchanged": torch.equal(
                inputs["logits"], inputs["logits_snapshot"]
            ),
            "mapping_unchanged": torch.equal(
                inputs["expanded_idx_mapping"], inputs["mapping_snapshot"]
            ),
            "temperature_unchanged": torch.equal(
                inputs["temperature"], inputs["temperature_snapshot"]
            ),
            "seeds_unchanged": torch.equal(inputs["seeds"], inputs["seeds_snapshot"]),
            "positions_unchanged": torch.equal(
                inputs["positions"], inputs["positions_snapshot"]
            ),
        }
        if inputs["processed_col"] is not None:
            checks["processed_col_unchanged"] = torch.equal(
                inputs["processed_col"], inputs["processed_col_snapshot"]
            )
        expected_processed = inputs["expected_processed"]
        if expected_processed is not None:
            checks["baseline_processed_exact"] = torch.equal(
                baseline["processed"], expected_processed
            )
            checks["candidate_processed_exact"] = torch.equal(
                candidate["processed"], expected_processed
            )
        if all(value == 0.0 for value in inputs["shape"]["temperatures"]):
            checks["greedy_reference"] = torch.equal(
                candidate_output, inputs["logits"].argmax(dim=-1)
            )
        return {"passed": all(checks.values()), "exact": checks}

    candidate_name = {
        "mirror": "frozen-triton-mirror",
        "fused": "triton-fused-final-reduction",
        "fused512": "triton-fused-reduction-block512",
        "fused2048": "triton-fused-reduction-block2048",
        "torch-argmax": "torch-argmax-all-greedy",
        "dispatch": "production-fused-final-reduction",
    }[inputs["candidate_mode"]]
    return ChainCase(
        baseline=Provider(
            "triton-block1024-plus-aten-argmax-gather",
            run_baseline,
            {"launches": 3, "candidate_active": False},
        ),
        candidate=Provider(
            candidate_name,
            run_candidate,
            {
                "launches": (
                    3
                    if inputs["candidate_mode"] == "mirror"
                    else (1 if inputs["candidate_mode"] == "torch-argmax" else 2)
                ),
                "candidate_active": inputs["candidate_mode"] != "mirror",
                "first_stage_block_size": candidate_block_size,
            },
            correctness_comparator=compare,
        ),
        shape=inputs["shape"],
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
