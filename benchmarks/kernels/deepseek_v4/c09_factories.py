# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    dequantize_and_gather_k_cache_triton,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
    DequantGatherKCacheKernel,
    dequantize_and_gather_k_cache_cutedsl,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed

HEAD_DIM = 512
HEAD_BYTES = 584
BLOCK_ALIGNMENT = 576
GUARD_VALUE = -1234.0


def _make_lengths(
    base: int,
    jitter: int,
    request_batch: int,
    *,
    minimum: int = 1,
) -> list[int]:
    return [max(minimum, base - index * jitter) for index in range(request_batch)]


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    seq_len = int(args["seq_len"])
    request_batch = int(args.get("request_batch", 1))
    block_size = int(args.get("block_size", 64))
    offset = int(args.get("offset", 0))
    seq_jitter = int(args.get("seq_jitter", 0))
    use_gather_lens = bool(args.get("use_gather_lens", False))
    gather_len = int(args.get("gather_len", seq_len))
    gather_jitter = int(args.get("gather_jitter", 0))
    output_padding = int(args.get("output_padding", 0))
    seed = int(args.get("seed", 0))
    regime = str(args.get("regime", "compressed-full"))

    if seq_len <= 0:
        raise ValueError("C09 seq_len must be positive")
    if request_batch not in (1, 4):
        raise ValueError("C09 request_batch must be 1 or 4")
    if block_size not in (2, 16, 64, 128, 256):
        raise ValueError("C09 block_size must be 2, 16, 64, 128, or 256")
    if offset < 0 or seq_jitter < 0 or gather_jitter < 0 or output_padding < 0:
        raise ValueError("C09 offset, jitters, and output padding must be non-negative")
    if gather_len <= 0:
        raise ValueError("C09 gather_len must be positive")

    seq_lens_list = _make_lengths(seq_len, seq_jitter, request_batch)
    if use_gather_lens:
        gather_lens_list = _make_lengths(
            gather_len,
            gather_jitter,
            request_batch,
        )
        if any(
            current_gather > current_seq
            for current_gather, current_seq in zip(
                gather_lens_list, seq_lens_list, strict=True
            )
        ):
            raise ValueError("C09 gather length cannot exceed sequence length")
    else:
        gather_lens_list = list(seq_lens_list)

    set_random_seed(seed)
    device = torch.device("cuda")
    blocks_per_request = [cdiv(length, block_size) for length in seq_lens_list]
    max_blocks_per_request = max(blocks_per_request)
    num_blocks = sum(blocks_per_request) + 1
    physical_blocks = torch.randperm(num_blocks - 1, device=device)
    block_table = torch.full(
        (request_batch, max_blocks_per_request),
        -1,
        dtype=torch.int32,
        device=device,
    )

    slot_mappings = []
    block_start = 0
    for request_index, (request_len, request_blocks) in enumerate(
        zip(seq_lens_list, blocks_per_request, strict=True)
    ):
        pages = physical_blocks[block_start : block_start + request_blocks]
        block_table[request_index, :request_blocks] = pages
        positions = torch.arange(request_len, dtype=torch.int64, device=device)
        slots = (
            pages[positions // block_size].to(torch.int64) * block_size
            + positions % block_size
        )
        slot_mappings.append(slots)
        block_start += request_blocks

    slot_mapping = torch.cat(slot_mappings)
    k_rows = torch.randn(
        slot_mapping.shape[0],
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    block_stride = cdiv(block_size * HEAD_BYTES, BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
    k_cache_storage = torch.full(
        (num_blocks, block_stride),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )
    quantize_and_insert_k_cache(
        k_rows,
        k_cache_storage,
        slot_mapping,
        block_size=block_size,
    )
    k_cache = torch.as_strided(
        k_cache_storage,
        size=(num_blocks, block_size, HEAD_BYTES),
        stride=(block_stride, HEAD_BYTES, 1),
    )

    seq_lens_tensor = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    gather_lens_tensor = (
        torch.tensor(gather_lens_list, dtype=torch.int32, device=device)
        if use_gather_lens
        else None
    )
    max_gather_tokens = max(gather_lens_list)
    output_rows = offset + max_gather_tokens + output_padding
    return {
        "k_cache": k_cache,
        "seq_lens": seq_lens_tensor,
        "gather_lens": gather_lens_tensor,
        "block_table": block_table,
        "block_size": block_size,
        "offset": offset,
        "output_rows": output_rows,
        "max_gather_tokens": max_gather_tokens,
        "gather_lens_list": gather_lens_list,
        "shape": {
            "name": (
                f"{regime}-s{seq_len}-g{gather_len}-b{request_batch}"
                f"-block{block_size}-off{offset}-sj{seq_jitter}"
                f"-gj{gather_jitter}-lens{int(use_gather_lens)}"
                f"-pad{output_padding}"
            ),
            "regime": regime,
            "seq_len": seq_len,
            "seq_lens": seq_lens_list,
            "gather_len": gather_len,
            "gather_lens": gather_lens_list,
            "request_batch": request_batch,
            "block_size": block_size,
            "block_stride": block_stride,
            "offset": offset,
            "seq_jitter": seq_jitter,
            "gather_jitter": gather_jitter,
            "use_gather_lens": use_gather_lens,
            "output_padding": output_padding,
            "output_capacity": output_rows - offset,
            "total_gather_tokens": sum(gather_lens_list),
        },
    }


def _allocate_output(inputs: Mapping[str, Any]) -> torch.Tensor:
    return torch.full(
        (
            inputs["seq_lens"].shape[0],
            inputs["output_rows"],
            HEAD_DIM,
        ),
        GUARD_VALUE,
        dtype=torch.bfloat16,
        device=inputs["seq_lens"].device,
    )


def _valid_rows_changed(output: torch.Tensor, inputs: Mapping[str, Any]) -> bool:
    offset = inputs["offset"]
    for request_index, gather_len in enumerate(inputs["gather_lens_list"]):
        valid = output[request_index, offset : offset + gather_len]
        if bool(valid.eq(GUARD_VALUE).all().item()):
            return False
    return True


def _inactive_rows_intact(output: torch.Tensor, inputs: Mapping[str, Any]) -> bool:
    offset = inputs["offset"]
    for request_index, gather_len in enumerate(inputs["gather_lens_list"]):
        prefix = output[request_index, :offset]
        suffix = output[request_index, offset + gather_len :]
        if not bool(prefix.eq(GUARD_VALUE).all().item()):
            return False
        if not bool(suffix.eq(GUARD_VALUE).all().item()):
            return False
    return True


def build_c09_dequant_gather_case(args: Mapping[str, Any]) -> ChainCase:
    inputs = _make_inputs(args)
    baseline_mode = str(args.get("baseline_mode", "triton"))
    if baseline_mode not in ("triton", "cutedsl-fixed"):
        raise ValueError("C09 baseline_mode must be triton or cutedsl-fixed")
    baseline = _allocate_output(inputs)
    candidate = _allocate_output(inputs)

    def run_baseline() -> torch.Tensor:
        if baseline_mode == "triton":
            dequantize_and_gather_k_cache_triton(
                baseline,
                inputs["k_cache"],
                inputs["seq_lens"],
                inputs["gather_lens"],
                inputs["block_table"],
                inputs["block_size"],
                inputs["offset"],
            )
        else:
            DequantGatherKCacheKernel.compile(
                block_size=inputs["block_size"],
                has_gather_lens=inputs["gather_lens"] is not None,
                num_worker_ctas=1024,
            )(
                baseline,
                inputs["k_cache"],
                inputs["seq_lens"],
                inputs["gather_lens"],
                inputs["block_table"],
                inputs["offset"],
            )
        return baseline

    def run_candidate() -> torch.Tensor:
        dequantize_and_gather_k_cache_cutedsl(
            candidate,
            inputs["k_cache"],
            inputs["seq_lens"],
            inputs["gather_lens"],
            inputs["block_table"],
            inputs["block_size"],
            inputs["offset"],
            max_gather_tokens=inputs["max_gather_tokens"],
        )
        return candidate

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        checks = {
            "exact_output": torch.equal(baseline, candidate),
            "baseline_valid_rows": _valid_rows_changed(baseline, inputs),
            "candidate_valid_rows": _valid_rows_changed(candidate, inputs),
            "baseline_inactive_rows": _inactive_rows_intact(baseline, inputs),
            "candidate_inactive_rows": _inactive_rows_intact(candidate, inputs),
        }
        return {"passed": all(checks.values()), "exact": checks}

    shape = dict(inputs["shape"])
    shape["baseline_mode"] = baseline_mode
    shape["name"] += f"-base-{baseline_mode}"
    return ChainCase(
        baseline=Provider(
            (
                "triton-128-workers-per-request"
                if baseline_mode == "triton"
                else "cutedsl-fixed-1024-ctas-per-request"
            ),
            run_baseline,
            {
                "baseline_mode": baseline_mode,
                "workers_per_request": 128 if baseline_mode == "triton" else 1024,
            },
        ),
        candidate=Provider(
            "cutedsl-capacity-bucketed-ctas",
            run_candidate,
            {"max_ctas_per_request": 1024, "warps_per_cta": 4, "stages": 4},
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )
