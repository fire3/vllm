# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up DeepSeek V4 mHC TileLang kernels before serving requests.

Ported from lucifer1004/vllm-jasl with the two env-var knobs removed
(`VLLM_ENABLE_DEEPSEEK_V4_MHC_WARMUP`, `VLLM_DEEPSEEK_V4_MHC_WARMUP_TOKEN_SIZES`).
Gating is intrinsic: non-DSv4 models and layers without hc_* attributes
return early, so the warmup is a no-op except where it's needed.
"""

import time
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.tracing import instrument

logger = init_logger(__name__)

_AUTO_WARMUP_MAX_TOKENS = 16_384
_DEFAULT_TOKEN_SIZE_CANDIDATES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
)


def _normalize_token_sizes(
    token_sizes: Iterable[int],
    *,
    max_tokens: int,
) -> list[int]:
    return sorted({size for size in token_sizes if 1 <= size <= max_tokens})


def _select_mhc_warmup_token_sizes(
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int],
) -> list[int]:
    if max_tokens <= 0:
        return []

    max_auto_tokens = min(max_tokens, _AUTO_WARMUP_MAX_TOKENS)
    candidates = list(_DEFAULT_TOKEN_SIZE_CANDIDATES)
    candidates.extend(cudagraph_capture_sizes)
    candidates.append(max_auto_tokens)
    return _normalize_token_sizes(candidates, max_tokens=max_auto_tokens)


def _find_first_mhc_layer(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4DecoderLayer":
            continue
        if all(
            hasattr(module, attr)
            for attr in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
            )
        ):
            return module
    return None


def _find_deepseek_v4_model(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        if module.__class__.__name__ != "DeepseekV4Model":
            continue
        if all(
            hasattr(module, attr)
            for attr in ("hc_head_fn", "hc_head_scale", "hc_head_base")
        ):
            return module
    return None


def _warmup_layer_mhc(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    if hasattr(layer, "hc_pre") and hasattr(layer, "hc_post"):
        _warmup_layer_mhc_prepost(layer, token_sizes)
    else:
        _warmup_layer_mhc_nvidia(layer, token_sizes)


def _warmup_layer_mhc_prepost(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        residual_slice = residual[:size]
        for fn, scale, base in (
            (layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base),
            (layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base),
        ):
            layer_input, post_mix, comb_mix = layer.hc_pre(
                residual_slice,
                fn,
                scale,
                base,
            )
            layer.hc_post(layer_input, residual_slice, post_mix, comb_mix)


def _broadcast_nsplit_sizes(
    max_tokens: int,
    hidden_size: int,
    n_sms: int,
) -> list[int]:
    """Token sizes that hit every ``n_splits`` bucket of the 2D broadcast mHC
    pre kernel for M in [1, max_tokens].

    ``mhc_pre_broadcast_tilelang`` derives ``n_splits`` from the runtime token
    count (``n_sms // ceil(M / 64)`` capped by the hidden-size split budget),
    and ``n_splits`` is a static TileLang compile key. Warming one M per
    bucket (``M = 64 * grid``) covers every value inference can hit for
    chunked prefill / prefix-cache delta sizes up to ``max_tokens``.
    """
    cap = max((hidden_size + 63) // 64 // 4, 1)
    sizes: list[int] = []
    prev = -1
    for grid in range(1, (max_tokens + 63) // 64 + 1):
        n_splits = min(n_sms // grid, cap)
        if n_splits != prev:
            sizes.append(min(grid * 64, max_tokens))
            prev = n_splits
    return sizes


def _warmup_layer_mhc_nvidia(
    layer: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    """Warm the SM89 nvidia mHC TileLang kernels.

    The nvidia ``DeepseekV4DecoderLayer`` has no ``hc_pre``/``hc_post``
    methods; its forward calls the tilelang module functions directly, so the
    warmup mirrors those calls (2D broadcast pre, 3D pre, fused post-pre for
    attn and ffn, standalone post) on one representative layer. TileLang
    kernels use a dynamic token count, so one compile per static config covers
    every chunk size; the broadcast pre additionally compiles per ``n_splits``
    bucket, so the token list is extended to hit each bucket.
    """
    from vllm.model_executor.kernels.mhc.tilelang import (
        mhc_fused_post_pre_tilelang,
        mhc_post_tilelang,
        mhc_pre_broadcast_tilelang,
        mhc_pre_tilelang,
    )

    max_tokens = max(token_sizes)
    hidden_size = int(layer.hidden_size)
    hc_mult = int(layer.hc_mult)
    device = layer.hc_attn_fn.device
    n_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sizes = sorted(
        set(token_sizes)
        | set(_broadcast_nsplit_sizes(max_tokens, hidden_size, n_sms))
    )

    attn_norm_weight = layer.attn_norm.weight.data
    attn_norm_eps = layer.attn_norm.variance_epsilon
    ffn_norm_weight = layer.ffn_norm.weight.data
    ffn_norm_eps = layer.ffn_norm.variance_epsilon
    rms_eps = layer.rms_norm_eps
    hc_eps = layer.hc_eps
    hc_post_alpha = layer.hc_post_alpha
    sinkhorn_iters = layer.hc_sinkhorn_iters

    residual = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    x = torch.zeros(max_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    for size in sizes:
        residual_slice = residual[:size]
        x_slice = x[:size]
        # First-layer standalone pre: 3D and 2D-broadcast variants.
        post_mix, comb_mix, layer_input = mhc_pre_tilelang(
            residual_slice,
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            rms_eps,
            hc_eps,
            hc_eps,
            hc_post_alpha,
            sinkhorn_iters,
            norm_weight=attn_norm_weight,
            norm_eps=attn_norm_eps,
        )
        if layer.hc_attn_fn_broadcast is not None:
            mhc_pre_broadcast_tilelang(
                x_slice,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                rms_eps,
                hc_eps,
                hc_eps,
                hc_post_alpha,
                sinkhorn_iters,
                norm_weight=attn_norm_weight,
                norm_eps=attn_norm_eps,
                fn_broadcast=layer.hc_attn_fn_broadcast,
            )
        # Fused post-pre used by every subsequent layer (attn and ffn).
        for fn, scale, base, norm_weight, norm_eps in (
            (
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                attn_norm_weight,
                attn_norm_eps,
            ),
            (
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                ffn_norm_weight,
                ffn_norm_eps,
            ),
        ):
            mhc_fused_post_pre_tilelang(
                x_slice,
                residual_slice,
                post_mix,
                comb_mix,
                fn,
                scale,
                base,
                rms_eps,
                hc_eps,
                hc_eps,
                hc_post_alpha,
                sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        # Standalone post (aux hidden-state reconstruction).
        mhc_post_tilelang(x_slice, residual_slice, post_mix, comb_mix)


def _warmup_hc_head(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    # Upstream a8887c208 ("[DSV4] aiter mhc support (ROCm)") refactored
    # ``hc_head`` from a free function into the ``HCHeadOp`` CustomOp
    # instance attached to the model as ``hc_head_op``. We call through
    # that instance so the warmup exercises the same dispatched
    # implementation as the inference path.
    hc_head_op = getattr(model, "hc_head_op", None)
    if hc_head_op is None:
        if not (
            hasattr(model, "hc_head_fn")
            and hasattr(model, "hc_head_scale")
            and hasattr(model, "hc_head_base")
        ):
            return
        _warmup_hc_head_nvidia(model, token_sizes)
        return

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )

    for size in token_sizes:
        hc_head_op(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


def _warmup_hc_head_nvidia(
    model: torch.nn.Module,
    token_sizes: list[int],
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang import (
        hc_head_fused_kernel_tilelang,
    )

    max_tokens = max(token_sizes)
    hidden_size = int(model.config.hidden_size)
    hc_mult = int(model.hc_mult)
    device = model.hc_head_fn.device
    hidden_states = torch.zeros(
        max_tokens,
        hc_mult,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    for size in token_sizes:
        hc_head_fused_kernel_tilelang(
            hidden_states[:size],
            model.hc_head_fn,
            model.hc_head_scale,
            model.hc_head_base,
            model.rms_norm_eps,
            model.hc_eps,
        )


@instrument(span_name="DeepSeek V4 mHC warmup")
def deepseek_v4_mhc_warmup(
    model: torch.nn.Module,
    *,
    max_tokens: int,
    cudagraph_capture_sizes: list[int] | None = None,
) -> None:
    # Cheap model-type gate before walking ``model.modules()``. The class
    # walk below is O(num_layers) and shows up in startup time on very
    # large checkpoints; bail out for any model that is not DeepSeek V4.
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None) if config is not None else None
    if model_type is not None and model_type != "deepseek_v4":
        return

    layer = _find_first_mhc_layer(model)
    if layer is None:
        return

    device = layer.hc_attn_fn.device
    if device.type != "cuda":
        return

    deepseek_model = _find_deepseek_v4_model(model)
    token_sizes = _select_mhc_warmup_token_sizes(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=cudagraph_capture_sizes or [],
    )
    if not token_sizes:
        return

    started = time.perf_counter()
    logger.info(
        "Warming up DeepSeek V4 mHC TileLang kernels for token sizes: %s",
        token_sizes,
    )
    with torch.inference_mode():
        _warmup_layer_mhc(layer, token_sizes)
        if deepseek_model is not None:
            _warmup_hc_head(deepseek_model, token_sizes)
        torch.accelerator.synchronize()
    logger.info(
        "DeepSeek V4 mHC TileLang warmup finished in %.2f seconds.",
        time.perf_counter() - started,
    )
