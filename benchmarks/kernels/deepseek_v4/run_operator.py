# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from benchmarks.kernels.deepseek_v4.common import (
    BenchmarkConfig,
    CorrectnessTolerances,
    Provider,
    benchmark_pair,
    build_ledger,
    default_ledger_path,
    write_json_atomic,
)
from vllm.utils.torch_utils import set_random_seed

TOKEN_COUNTS = (1, 4, 7, 8, 16, 32, 64, 128, 256, 1024, 8192)
DEEPSEEK_LINEAR_SHAPES = (
    (576, 7168),
    (2112, 7168),
    (24576, 7168),
    (32768, 512),
    (7168, 16384),
    (7168, 18432),
    (36864, 7168),
    (24576, 1536),
    (12288, 7168),
    (4096, 7168),
    (7168, 2048),
)
A01_PROVIDERS = (
    "triton",
    "cutlass",
    "cutlass-padded",
    "cutlass-dispatch",
    "flashinfer",
)
A02_PROVIDERS = (
    "triton",
    "flashinfer-kernel",
    "flashinfer-input-copy",
    "flashinfer-output-copy",
    "flashinfer-chain",
    "cutlass-loop",
    "cutlass-loop-output-copy",
    "cutlass-batched-direct",
    "cutlass-batched-dispatch",
)
A02_GROUP_COUNTS = (2, 8)
A02_OUT_RANK = 1024
A02_HIDDEN_SIZE = 4096


class ProviderUnavailable(RuntimeError):
    """Raised when an explicitly requested provider cannot run a shape."""


def _shape_name(m: int, n: int, k: int) -> str:
    return f"m{m}-n{n}-k{k}"


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K or MxNxK")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape  # type: ignore[return-value]


def parse_a02_shape(value: str) -> tuple[int, int, int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("A02 shape must be T,G,N,K or TxGxNxK")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape dimensions must be integers") from error
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape  # type: ignore[return-value]


def _default_a01_dispatch(n: int, k: int) -> str:
    from vllm.benchmarks.lib.utils import default_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        create_fp8_quant_key,
    )

    @default_vllm_config()
    def select() -> str:
        kernel = init_fp8_linear_kernel(
            weight_quant_key=create_fp8_quant_key(
                static=True,
                group_shape=GroupShape(128, 128),
            ),
            activation_quant_key=create_fp8_quant_key(
                static=False,
                group_shape=GroupShape(1, 128),
            ),
            input_dtype=torch.bfloat16,
            out_dtype=torch.bfloat16,
            weight_shape=(n, k),
            module_name="dsv4_a01_dispatch_probe",
        )
        return type(kernel).__name__

    return select()


def _build_a01_providers(
    m: int,
    n: int,
    k: int,
    *,
    seed: int,
) -> tuple[dict[str, Provider], dict[str, Any]]:
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        _use_triton_for_sm12x_block_fp8,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _w8a8_triton_block_scaled_mm,
        get_w8a8_block_fp8_configs,
        per_token_group_quant_fp8,
    )
    from vllm.platforms import current_platform
    from vllm.triton_utils import triton
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
    from vllm.utils.flashinfer import has_flashinfer_fp8_blockscale_gemm

    if k % 128:
        raise ProviderUnavailable("A01 requires K divisible by 128")
    set_random_seed(seed)
    device = torch.device("cuda")
    input_bf16 = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight_bf16 = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    activation, activation_scale = per_token_group_quant_fp8(
        input_bf16,
        128,
        column_major_scales=False,
        use_ue8m0=False,
    )
    activation_cutlass, activation_scale_cutlass = per_token_group_quant_fp8(
        input_bf16,
        128,
        column_major_scales=True,
        use_ue8m0=False,
    )
    weight, weight_scale = per_block_cast_to_fp8(
        weight_bf16,
        [128, 128],
        use_ue8m0=False,
    )
    if not torch.equal(activation, activation_cutlass):
        raise AssertionError("row-major and column-major quantization payloads differ")

    triton_output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    configs = get_w8a8_block_fp8_configs(n, k, 128, 128)
    if configs:
        kernel_config = configs[min(configs, key=lambda size: abs(size - m))]
    else:
        kernel_config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    def triton_grid(meta: dict[str, Any]) -> tuple[int]:
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    def run_triton() -> torch.Tensor:
        _w8a8_triton_block_scaled_mm[triton_grid](
            activation,
            weight,
            triton_output,
            activation_scale,
            weight_scale,
            m,
            n,
            k,
            128,
            128,
            activation.stride(0),
            activation.stride(1),
            weight.stride(1),
            weight.stride(0),
            triton_output.stride(0),
            triton_output.stride(1),
            activation_scale.stride(0),
            activation_scale.stride(1),
            weight_scale.stride(1),
            weight_scale.stride(0),
            **kernel_config,
        )
        return triton_output

    cutlass_output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    weight_column_major = weight.T
    weight_scale_column_major = weight_scale.T

    def run_cutlass() -> torch.Tensor:
        torch.ops._C.cutlass_scaled_mm(
            cutlass_output,
            activation_cutlass,
            weight_column_major,
            activation_scale_cutlass,
            weight_scale_column_major,
            None,
        )
        return cutlass_output

    padded_n = ((n + 127) // 128) * 128
    if padded_n == n:
        padded_weight_column_major = weight_column_major
        padded_cutlass_output = cutlass_output
        logical_cutlass_output = cutlass_output
    else:
        padded_weight = torch.zeros(
            (padded_n, k),
            device=device,
            dtype=weight.dtype,
        )
        padded_weight[:n].copy_(weight)
        padded_weight_column_major = padded_weight.T
        padded_cutlass_output = torch.empty(
            (m, padded_n),
            device=device,
            dtype=torch.bfloat16,
        )
        logical_cutlass_output = torch.empty(
            (m, n),
            device=device,
            dtype=torch.bfloat16,
        )

    def run_cutlass_padded() -> torch.Tensor:
        torch.ops._C.cutlass_scaled_mm(
            padded_cutlass_output,
            activation_cutlass,
            padded_weight_column_major,
            activation_scale_cutlass,
            weight_scale_column_major,
            None,
        )
        if padded_n != n:
            logical_cutlass_output.copy_(padded_cutlass_output[:, :n])
        return logical_cutlass_output

    capability = current_platform.get_device_capability()
    is_sm12x = capability is not None and capability.major == 12
    dispatch_uses_triton = is_sm12x and _use_triton_for_sm12x_block_fp8(m, n, k)
    if dispatch_uses_triton:
        run_cutlass_dispatch = run_triton
        selected_backend = "triton"
    elif is_sm12x:
        run_cutlass_dispatch = run_cutlass_padded
        selected_backend = "cutlass-padded"
    else:
        run_cutlass_dispatch = run_cutlass
        selected_backend = "cutlass"

    providers = {
        "triton": Provider(
            "triton",
            run_triton,
            {
                "symbol": "_w8a8_triton_block_scaled_mm",
                "preallocated_output": True,
                "kernel_config": kernel_config,
            },
        ),
        "cutlass": Provider(
            "cutlass",
            run_cutlass,
            {
                "symbol": "torch.ops._C.cutlass_scaled_mm",
                "preallocated_output": True,
                "swap_ab_requested": m <= 64 or m % 4 != 0,
            },
        ),
        "cutlass-padded": Provider(
            "cutlass-padded",
            run_cutlass_padded,
            {
                "symbol": "torch.ops._C.cutlass_scaled_mm",
                "preallocated_output": True,
                "logical_n": n,
                "padded_n": padded_n,
                "includes_output_slice_copy": padded_n != n,
                "swap_ab_requested": m <= 64 or m % 4 != 0,
            },
        ),
        "cutlass-dispatch": Provider(
            "cutlass-dispatch",
            run_cutlass_dispatch,
            {
                "selected_backend": selected_backend,
                "preallocated_output": True,
                "scope": "CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm",
            },
        ),
    }
    availability = {
        "triton": {"available": True},
        "cutlass": {"available": True},
        "cutlass-padded": {"available": True},
        "cutlass-dispatch": {"available": True},
        "flashinfer": {
            "available": bool(has_flashinfer_fp8_blockscale_gemm()),
            "reason": (
                "FlashInfer 0.6.13 fp8_blockscale_gemm_sm90 rejects SM120"
                if not has_flashinfer_fp8_blockscale_gemm()
                else None
            ),
        },
    }
    if availability["flashinfer"]["available"]:
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        flashinfer_output = torch.empty(
            (m, n),
            device=device,
            dtype=torch.bfloat16,
        )

        def run_flashinfer() -> torch.Tensor:
            return fp8_blockscale_gemm_sm90(
                input_bf16,
                weight,
                None,
                weight_scale,
                out=flashinfer_output,
                out_dtype=torch.bfloat16,
            )

        providers["flashinfer"] = Provider(
            "flashinfer",
            run_flashinfer,
            {
                "symbol": "flashinfer.gemm.fp8_blockscale_gemm_sm90",
                "preallocated_output": True,
            },
        )

    del weight_bf16
    torch.accelerator.empty_cache()
    metadata = {
        "name": _shape_name(m, n, k),
        "M": m,
        "N": n,
        "K": k,
        "block_shape": [128, 128],
        "dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "default_dispatch": _default_a01_dispatch(n, k),
        "provider_availability": availability,
    }
    return providers, metadata


def _build_a02_providers(
    num_tokens: int,
    num_groups: int,
    out_rank: int,
    hidden_size: int,
    *,
    seed: int,
) -> tuple[dict[str, Provider], dict[str, Any]]:
    flashinfer_error = None
    try:
        from flashinfer import bmm_fp8
    except ImportError as error:
        bmm_fp8 = None
        flashinfer_error = str(error)

    from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
        deepseek_v4_sm12x_fp8_einsum,
    )
    from vllm.utils.deep_gemm import get_tma_aligned_size

    if hidden_size % 128 or out_rank % 128:
        raise ProviderUnavailable("A02 requires N and K divisible by 128")
    if num_groups not in A02_GROUP_COUNTS:
        raise ProviderUnavailable(
            f"A02 group count must be one of {A02_GROUP_COUNTS}, got {num_groups}"
        )

    set_random_seed(seed)
    device = torch.device("cuda")
    fp8_dtype = torch.float8_e4m3fn
    scale_k_blocks = hidden_size // 128
    scale_out_blocks = out_rank // 128
    aligned_tokens = get_tma_aligned_size(num_tokens, 4)

    a_group_major = torch.randn(
        (num_groups, num_tokens, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)
    a = a_group_major.transpose(0, 1)
    b = torch.randn(
        (num_groups, out_rank, hidden_size),
        device=device,
        dtype=torch.bfloat16,
    ).to(fp8_dtype)

    a_scale_storage = torch.empty(
        num_groups * scale_k_blocks * aligned_tokens,
        device=device,
        dtype=torch.float32,
    )
    a_scale_group_major = a_scale_storage.as_strided(
        (num_groups, num_tokens, scale_k_blocks),
        (scale_k_blocks * aligned_tokens, 1, aligned_tokens),
    )
    a_scale_group_major.uniform_(0.001, 0.01)
    a_scale = a_scale_group_major.transpose(0, 1)
    a_scale_cutlass_view = a_scale_group_major.permute(0, 2, 1)
    a_scale_cutlass = a_scale_cutlass_view.contiguous()

    b_scale = torch.empty(
        (num_groups, scale_out_blocks, scale_k_blocks),
        device=device,
        dtype=torch.float32,
    ).uniform_(0.001, 0.01)
    b_cutlass = b.transpose(1, 2)
    b_scale_cutlass = b_scale.transpose(1, 2).contiguous()

    triton_output = torch.empty(
        (num_tokens, num_groups, out_rank),
        device=device,
        dtype=torch.bfloat16,
    )
    flashinfer_output = torch.empty(
        (num_groups, num_tokens, out_rank),
        device=device,
        dtype=torch.bfloat16,
    )
    reordered_output = torch.empty_like(triton_output)
    copied_a_scale = torch.empty_like(a_scale_cutlass)
    cutlass_loop_output = torch.empty_like(flashinfer_output)
    cutlass_batched_output = torch.empty_like(triton_output)

    def run_triton() -> torch.Tensor:
        deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, triton_output)
        return triton_output

    def run_flashinfer_kernel() -> torch.Tensor:
        bmm_fp8(
            a_group_major,
            b_cutlass,
            a_scale_cutlass,
            b_scale_cutlass,
            torch.bfloat16,
            out=flashinfer_output,
            backend="cutlass",
        )
        return flashinfer_output.transpose(0, 1)

    def run_flashinfer_input_copy() -> torch.Tensor:
        copied_a_scale.copy_(a_scale_cutlass_view)
        bmm_fp8(
            a_group_major,
            b_cutlass,
            copied_a_scale,
            b_scale_cutlass,
            torch.bfloat16,
            out=flashinfer_output,
            backend="cutlass",
        )
        return flashinfer_output.transpose(0, 1)

    def run_flashinfer_output_copy() -> torch.Tensor:
        bmm_fp8(
            a_group_major,
            b_cutlass,
            a_scale_cutlass,
            b_scale_cutlass,
            torch.bfloat16,
            out=flashinfer_output,
            backend="cutlass",
        )
        reordered_output.copy_(flashinfer_output.transpose(0, 1))
        return reordered_output

    def run_flashinfer_chain() -> torch.Tensor:
        cutlass_scale = a_scale_cutlass
        if not a_scale_cutlass_view.is_contiguous():
            copied_a_scale.copy_(a_scale_cutlass_view)
            cutlass_scale = copied_a_scale
        bmm_fp8(
            a_group_major,
            b_cutlass,
            cutlass_scale,
            b_scale_cutlass,
            torch.bfloat16,
            out=flashinfer_output,
            backend="cutlass",
        )
        reordered_output.copy_(flashinfer_output.transpose(0, 1))
        return reordered_output

    def run_cutlass_loop() -> torch.Tensor:
        for group in range(num_groups):
            torch.ops._C.cutlass_scaled_mm(
                cutlass_loop_output[group],
                a_group_major[group],
                b_cutlass[group],
                a_scale_cutlass[group].transpose(0, 1),
                b_scale[group].transpose(0, 1),
                None,
            )
        return cutlass_loop_output.transpose(0, 1)

    def run_cutlass_loop_output_copy() -> torch.Tensor:
        run_cutlass_loop()
        reordered_output.copy_(cutlass_loop_output.transpose(0, 1))
        return reordered_output

    def run_cutlass_batched_direct() -> torch.Tensor:
        torch.ops._C.deepseek_v4_fp8_bmm_sm120(
            cutlass_batched_output,
            a_group_major,
            b_cutlass,
            a_scale_cutlass,
            b_scale,
        )
        return cutlass_batched_output

    run_cutlass_batched_dispatch = (
        run_cutlass_batched_direct if num_tokens >= 256 else run_triton
    )

    shared_metadata = {
        "equation": "bhr,hdr->bhd",
        "preallocated_output": True,
        "scale_granularity_mnk": [1, 128, 128],
    }
    providers = {
        "triton": Provider(
            "triton",
            run_triton,
            {
                **shared_metadata,
                "symbol": "_deepseek_v4_sm12x_fp8_einsum_kernel",
                "output_layout": "TGN",
            },
        ),
        "flashinfer-kernel": Provider(
            "flashinfer-kernel",
            run_flashinfer_kernel,
            {
                **shared_metadata,
                "symbol": "flashinfer.bmm_fp8",
                "input_scale_copy": False,
                "output_reorder": False,
            },
        ),
        "flashinfer-input-copy": Provider(
            "flashinfer-input-copy",
            run_flashinfer_input_copy,
            {
                **shared_metadata,
                "symbol": "flashinfer.bmm_fp8",
                "input_scale_copy": True,
                "output_reorder": False,
            },
        ),
        "flashinfer-output-copy": Provider(
            "flashinfer-output-copy",
            run_flashinfer_output_copy,
            {
                **shared_metadata,
                "symbol": "flashinfer.bmm_fp8",
                "input_scale_copy": False,
                "output_reorder": True,
            },
        ),
        "flashinfer-chain": Provider(
            "flashinfer-chain",
            run_flashinfer_chain,
            {
                **shared_metadata,
                "symbol": "flashinfer.bmm_fp8",
                "input_scale_copy": not a_scale_cutlass_view.is_contiguous(),
                "output_reorder": True,
            },
        ),
        "cutlass-loop": Provider(
            "cutlass-loop",
            run_cutlass_loop,
            {
                **shared_metadata,
                "symbol": "torch.ops._C.cutlass_scaled_mm",
                "gemm_launches": num_groups,
                "input_scale_copy": False,
                "output_reorder": False,
            },
        ),
        "cutlass-loop-output-copy": Provider(
            "cutlass-loop-output-copy",
            run_cutlass_loop_output_copy,
            {
                **shared_metadata,
                "symbol": "torch.ops._C.cutlass_scaled_mm",
                "gemm_launches": num_groups,
                "input_scale_copy": False,
                "output_reorder": True,
            },
        ),
        "cutlass-batched-direct": Provider(
            "cutlass-batched-direct",
            run_cutlass_batched_direct,
            {
                **shared_metadata,
                "symbol": "torch.ops._C.deepseek_v4_fp8_bmm_sm120",
                "gemm_launches": 1,
                "input_scale_copy": False,
                "output_reorder": False,
                "output_layout": "TGN",
            },
        ),
        "cutlass-batched-dispatch": Provider(
            "cutlass-batched-dispatch",
            run_cutlass_batched_dispatch,
            {
                **shared_metadata,
                "selected_backend": (
                    "cutlass-batched-direct" if num_tokens >= 256 else "triton"
                ),
                "target_bucket": "prefill-tokens-ge-256",
                "output_layout": "TGN",
            },
        ),
    }
    flashinfer_providers = {
        "flashinfer-kernel",
        "flashinfer-input-copy",
        "flashinfer-output-copy",
        "flashinfer-chain",
    }
    if bmm_fp8 is None:
        for provider in flashinfer_providers:
            providers.pop(provider)

    provider_availability = {}
    for provider in A02_PROVIDERS:
        if provider in providers:
            provider_availability[provider] = {"available": True}
        else:
            provider_availability[provider] = {
                "available": False,
                "reason": f"FlashInfer import failed: {flashinfer_error}",
            }
    metadata = {
        "name": (f"t{num_tokens}-g{num_groups}-n{out_rank}-k{hidden_size}"),
        "T": num_tokens,
        "G": num_groups,
        "N": out_rank,
        "K": hidden_size,
        "dtype": "float8_e4m3fn",
        "output_dtype": "bfloat16",
        "scale_dtype": "float32",
        "scale_granularity_mnk": [1, 128, 128],
        "tma_aligned_tokens": aligned_tokens,
        "a_scale_cutlass_zero_copy": a_scale_cutlass_view.is_contiguous(),
        "provider_availability": provider_availability,
    }
    return providers, metadata


def _resolve_shapes(args: argparse.Namespace) -> list[tuple[int, ...]]:
    if args.operator == "A01":
        if args.shape:
            return list(dict.fromkeys(args.shape))
        if args.all_shapes:
            return [(m, n, k) for n, k in DEEPSEEK_LINEAR_SHAPES for m in TOKEN_COUNTS]
        raise ValueError("pass --shape at least once or use --all-shapes")
    if args.a02_shape:
        return list(dict.fromkeys(args.a02_shape))
    if args.all_shapes:
        return [
            (tokens, groups, A02_OUT_RANK, A02_HIDDEN_SIZE)
            for groups in A02_GROUP_COUNTS
            for tokens in TOKEN_COUNTS
        ]
    raise ValueError("pass --a02-shape at least once or use --all-shapes")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("A01", "A02"), required=True)
    providers = tuple(dict.fromkeys((*A01_PROVIDERS, *A02_PROVIDERS)))
    parser.add_argument("--baseline", choices=providers, required=True)
    parser.add_argument("--candidate", choices=providers, required=True)
    parser.add_argument("--shape", type=parse_shape, action="append")
    parser.add_argument("--a02-shape", type=parse_a02_shape, action="append")
    parser.add_argument("--all-shapes", action="store_true")
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
    parser.add_argument("--list-shapes", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_shapes:
        if args.operator == "A01":
            for n, k in DEEPSEEK_LINEAR_SHAPES:
                for m in TOKEN_COUNTS:
                    print(_shape_name(m, n, k))
        else:
            for groups in A02_GROUP_COUNTS:
                for tokens in TOKEN_COUNTS:
                    print(f"t{tokens}-g{groups}-n{A02_OUT_RANK}-k{A02_HIDDEN_SIZE}")
        return 0
    if args.baseline == args.candidate:
        raise ValueError("baseline and candidate must differ")
    shapes = _resolve_shapes(args)
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
    tolerances = CorrectnessTolerances(
        atol=1.0,
        rtol=0.05,
        max_mean_relative=0.001,
        min_cosine=0.999,
        require_allclose=False,
    )
    allowed_providers = A01_PROVIDERS if args.operator == "A01" else A02_PROVIDERS
    invalid_providers = [
        provider
        for provider in (args.baseline, args.candidate)
        if provider not in allowed_providers
    ]
    if invalid_providers:
        raise ValueError(
            f"providers {invalid_providers} are not valid for {args.operator}"
        )

    results = []
    for dimensions in shapes:
        if args.operator == "A01":
            providers, shape = _build_a01_providers(*dimensions, seed=args.seed)
        else:
            providers, shape = _build_a02_providers(*dimensions, seed=args.seed)
        unavailable = [
            name for name in (args.baseline, args.candidate) if name not in providers
        ]
        if unavailable:
            availability = shape["provider_availability"]
            raise ProviderUnavailable(
                ", ".join(
                    f"{name}: {availability[name].get('reason', 'unavailable')}"
                    for name in unavailable
                )
            )
        result = benchmark_pair(
            providers[args.baseline],
            providers[args.candidate],
            shape=shape,
            config=config,
            tolerances=tolerances,
        )
        results.append(result)
        del providers
        gc.collect()
        torch.accelerator.empty_cache()

    candidate_name = f"{args.candidate}-vs-{args.baseline}"
    ledger = build_ledger(
        operator_id=args.operator,
        phase="A",
        candidate=candidate_name,
        results=results,
        config=config,
        repo_root=args.repo_root,
        flashinfer_root=args.flashinfer_root,
        command=sys.argv,
    )
    output = args.output or default_ledger_path(
        args.operator,
        candidate_name,
        ledger["environment"]["process_uuid"],
    )
    write_json_atomic(output, ledger)
    summaries = [
        {
            "shape": result["shape"]["name"],
            "baseline_us": result["summary"]["baseline_us"]["p50"],
            "candidate_us": result["summary"]["candidate_us"]["p50"],
            "improvement_pct": result["summary"]["paired_improvement_pct"]["p50"],
        }
        for result in results
    ]
    print(json.dumps({"ledger": str(output), "results": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
