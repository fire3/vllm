# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
from collections.abc import Callable

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.import_utils import has_cutedsl

_DUAL_RMSNORM_MIN_TOKENS = 16
_DUAL_RMSNORM_SERIAL_TOKENS = 8192


@functools.cache
def _get_dual_rmsnorm_cute_impl() -> Callable[..., None] | None:
    if not has_cutedsl():
        return None
    try:
        cute_dsl = importlib.import_module("flashinfer.cute_dsl")
    except ModuleNotFoundError as exc:
        if exc.name in {"flashinfer", "flashinfer.cute_dsl"}:
            return None
        raise
    return getattr(cute_dsl, "dual_rmsnorm_cute", None)


def _can_use_dual_rmsnorm_cute(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    qr_out: torch.Tensor,
    kv_out: torch.Tensor,
) -> bool:
    tensors = (qr, kv, q_weight, kv_weight, qr_out, kv_out)
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and qr.shape[0] >= _DUAL_RMSNORM_MIN_TOKENS
        and qr.shape[1] == 1536
        and kv.shape[1] == 512
        and q_weight.shape == (1536,)
        and kv_weight.shape == (512,)
        and all(tensor.dtype == torch.bfloat16 for tensor in tensors)
        and all(tensor.device == qr.device for tensor in tensors)
        and all(tensor.is_cuda for tensor in tensors)
        and all(tensor.data_ptr() % 16 == 0 for tensor in tensors)
        and qr.stride(-1) == 1
        and kv.stride(-1) == 1
        and qr_out.stride(-1) == 1
        and kv_out.stride(-1) == 1
        and qr.stride(0) % 8 == 0
        and kv.stride(0) % 8 == 0
        and qr_out.stride(0) % 8 == 0
        and kv_out.stride(0) % 8 == 0
        and q_weight.is_contiguous()
        and kv_weight.is_contiguous()
    )


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
    # CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
    # the launch at max-num-batched-tokens >= 65536 with "invalid argument".
    # int64: q_in_stride can be ~24K (128 heads × 192) and overflows int32
    # past num_tokens ~87K under large chunked prefill.
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    # RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
    # `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel, which
    # keep x, rrms, and w all in fp32 and perform a single cast at store.
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0], (
        f"token dim mismatch: qr={qr.shape}, kv={kv.shape}"
    )
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    if _can_use_dual_rmsnorm_cute(
        qr,
        kv,
        q_weight,
        kv_weight,
        qr_out,
        kv_out,
    ):
        dual_rmsnorm_cute = _get_dual_rmsnorm_cute_impl()
        if dual_rmsnorm_cute is not None:
            dual_rmsnorm_cute(
                qr,
                kv,
                q_weight,
                kv_weight,
                qr_out,
                kv_out,
                eps,
                parallel_tasks=num_tokens < _DUAL_RMSNORM_SERIAL_TOKENS,
            )
            return qr_out, kv_out

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        qr,
        qr_out,
        q_weight,
        qr.stride(0),
        qr_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        BLOCK_SIZE=block_size,
    )
    return qr_out, kv_out
