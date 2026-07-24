# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import CorrectnessTolerances, Provider
from benchmarks.kernels.deepseek_v4.run_chain import ChainCase
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.mla.indexer import _prepare_uniform_decode_kernel

TRITON_BLOCK_SIZE = 1024


def _load_candidate() -> Callable[..., Any]:
    from vllm.v1.attention.backends.mla import indexer

    candidate = getattr(indexer, "_launch_uniform_decode_metadata", None)
    if candidate is None:
        raise RuntimeError("C07 candidate dispatch is unavailable")
    return candidate


def _make_inputs(args: Mapping[str, Any]) -> dict[str, Any]:
    context_tokens = int(args["context_tokens"])
    request_batch = int(args.get("request_batch", 1))
    decode_len = int(args.get("decode_len", 8))
    block_size = int(args.get("block_size", 256))
    max_model_len = int(args.get("max_model_len", 131072))
    context_jitter = int(args.get("context_jitter", 0))
    compress_ratio = int(args.get("compress_ratio", 4))
    seed = int(args.get("seed", 0))

    if context_tokens <= 0 or context_tokens > max_model_len:
        raise ValueError("C07 context_tokens must be in (0, max_model_len]")
    if request_batch not in (1, 4):
        raise ValueError("C07 request_batch must be 1 or 4")
    if decode_len not in range(1, 9):
        raise ValueError("C07 decode_len must be in [1, 8]")
    if block_size not in (64, 256):
        raise ValueError("C07 block_size must be 64 or 256")
    if max_model_len <= 0 or max_model_len % block_size:
        raise ValueError("C07 max_model_len must be a positive block multiple")
    if context_jitter < 0:
        raise ValueError("C07 context_jitter must be non-negative")
    if compress_ratio not in (1, 4, 128):
        raise ValueError("C07 compress_ratio must be 1, 4, or 128")

    seq_lens_list = [
        context_tokens - request_index * context_jitter
        for request_index in range(request_batch)
    ]
    if any(seq_len < decode_len for seq_len in seq_lens_list):
        raise ValueError("C07 sequence length must cover every decode token")

    set_random_seed(seed)
    device = torch.device("cuda")
    block_table_width = cdiv(max_model_len, block_size)
    block_table = torch.arange(
        request_batch * block_table_width,
        dtype=torch.int32,
        device=device,
    ).view(request_batch, block_table_width)
    for request_index, seq_len in enumerate(seq_lens_list):
        actual_blocks = cdiv(seq_len, block_size)
        block_table[request_index, actual_blocks:] = -1

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    expected_seq_lens = (
        seq_lens[:, None]
        - decode_len
        + 1
        + torch.arange(decode_len, dtype=torch.int32, device=device)[None, :]
    ).reshape(-1)
    expected_block_table = block_table.repeat_interleave(decode_len, dim=0)
    expected_decode_lens = torch.ones(
        request_batch * decode_len, dtype=torch.int32, device=device
    )
    return {
        "seq_lens": seq_lens,
        "block_table": block_table,
        "expected_seq_lens": expected_seq_lens,
        "expected_compressed_seq_lens": expected_seq_lens // compress_ratio,
        "expected_block_table": expected_block_table,
        "expected_decode_lens": expected_decode_lens,
        "num_decode_tokens": request_batch * decode_len,
        "decode_len": decode_len,
        "compress_ratio": compress_ratio,
        "shape": {
            "name": (
                f"ctx{context_tokens}-b{request_batch}-l{decode_len}"
                f"-block{block_size}-width{block_table_width}-j{context_jitter}"
                f"-c{compress_ratio}"
            ),
            "context_tokens": context_tokens,
            "request_batch": request_batch,
            "decode_len": decode_len,
            "draft_tokens": decode_len - 1,
            "num_decode_tokens": request_batch * decode_len,
            "block_size": block_size,
            "block_table_width": block_table_width,
            "max_model_len": max_model_len,
            "context_jitter": context_jitter,
            "compress_ratio": compress_ratio,
        },
    }


def _allocate_outputs(inputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    device = inputs["seq_lens"].device
    return {
        "seq_lens": torch.empty_like(inputs["expected_seq_lens"]),
        "block_table": torch.empty_like(inputs["expected_block_table"]),
        "decode_lens": torch.empty_like(inputs["expected_decode_lens"]),
        "return_value": torch.empty(1, dtype=torch.int32, device=device),
    }


def _launch_baseline(
    outputs: Mapping[str, torch.Tensor], inputs: Mapping[str, Any]
) -> None:
    _prepare_uniform_decode_kernel[(inputs["num_decode_tokens"],)](
        inputs["seq_lens"],
        outputs["seq_lens"],
        inputs["block_table"],
        inputs["block_table"].stride(0),
        outputs["block_table"],
        outputs["block_table"].stride(0),
        outputs["decode_lens"],
        inputs["decode_len"],
        BLOCK_SIZE=TRITON_BLOCK_SIZE,
        COMPRESS_RATIO=1,
    )


def _build_case(args: Mapping[str, Any], *, fuse_compression: bool) -> ChainCase:
    inputs = _make_inputs(args)
    baseline = _allocate_outputs(inputs)
    candidate = _allocate_outputs(inputs)
    candidate_impl = _load_candidate()

    def run_baseline() -> torch.Tensor:
        _launch_baseline(baseline, inputs)
        if fuse_compression and inputs["compress_ratio"] > 1:
            torch.div(
                baseline["seq_lens"],
                inputs["compress_ratio"],
                rounding_mode="floor",
                out=baseline["seq_lens"],
            )
        return baseline["return_value"]

    def run_candidate() -> torch.Tensor:
        fused_compression = candidate_impl(
            inputs["seq_lens"],
            candidate["seq_lens"],
            inputs["block_table"],
            candidate["block_table"],
            candidate["decode_lens"],
            inputs["num_decode_tokens"],
            inputs["decode_len"],
            inputs["compress_ratio"] if fuse_compression else 1,
        )
        if fuse_compression and inputs["compress_ratio"] > 1 and not fused_compression:
            torch.div(
                candidate["seq_lens"],
                inputs["compress_ratio"],
                rounding_mode="floor",
                out=candidate["seq_lens"],
            )
        return candidate["return_value"]

    def compare(
        _baseline_output: torch.Tensor,
        _candidate_output: torch.Tensor,
        _tolerances: CorrectnessTolerances,
    ) -> dict[str, Any]:
        expected_seq_lens = (
            inputs["expected_compressed_seq_lens"]
            if fuse_compression
            else inputs["expected_seq_lens"]
        )
        checks = {
            "baseline_seq_lens": torch.equal(baseline["seq_lens"], expected_seq_lens),
            "baseline_block_table": torch.equal(
                baseline["block_table"], inputs["expected_block_table"]
            ),
            "baseline_decode_lens": torch.equal(
                baseline["decode_lens"], inputs["expected_decode_lens"]
            ),
            "candidate_seq_lens": torch.equal(candidate["seq_lens"], expected_seq_lens),
            "candidate_block_table": torch.equal(
                candidate["block_table"], inputs["expected_block_table"]
            ),
            "candidate_decode_lens": torch.equal(
                candidate["decode_lens"], inputs["expected_decode_lens"]
            ),
        }
        return {"passed": all(checks.values()), "exact": checks}

    shape = dict(inputs["shape"])
    shape["chain"] = (
        "uniform-decode-metadata-plus-seq-len-compression"
        if fuse_compression
        else "uniform-decode-metadata-and-block-table-expansion"
    )
    shape["name"] += "-chain" if fuse_compression else "-standalone"
    baseline_launches = 1 + int(fuse_compression and inputs["compress_ratio"] > 1)
    return ChainCase(
        baseline=Provider(
            "triton-uniform-decode",
            run_baseline,
            {"launches": baseline_launches, "block_size": TRITON_BLOCK_SIZE},
        ),
        candidate=Provider(
            "sm120-production-dispatch",
            run_candidate,
            {
                "candidate_active": True,
                "launches": 1,
            },
            correctness_comparator=compare,
        ),
        shape=shape,
        tolerances=CorrectnessTolerances(atol=0.0, rtol=0.0),
    )


def build_c07_uniform_decode_case(args: Mapping[str, Any]) -> ChainCase:
    return _build_case(args, fuse_compression=False)


def build_c07_uniform_decode_compression_chain_case(
    args: Mapping[str, Any],
) -> ChainCase:
    return _build_case(args, fuse_compression=True)
