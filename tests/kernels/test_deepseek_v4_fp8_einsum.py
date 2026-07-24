# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.ops.fp8_einsum as fp8_einsum_ops
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj
from vllm.platforms.interface import DeviceCapability

requires_sm12x = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="DeepSeek V4 CUTLASS FP8 BMM requires SM12x",
)
requires_sm89 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9),
    reason="DeepSeek V4 SM89 fallback requires SM89",
)

HIDDEN_SIZE = 4096
OUT_RANK = 1024
NUM_HEADS = 64
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64


def make_inputs(
    num_tokens: int,
    num_groups: int,
    *,
    weight_groups: int | None = None,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    weight_groups = weight_groups or num_groups
    a_group_major = torch.randn(
        (num_groups, num_tokens, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    a = a_group_major.transpose(0, 1)
    a_scale_cutlass = torch.empty(
        (num_groups, HIDDEN_SIZE // 128, num_tokens),
        device=device,
        dtype=torch.float32,
    ).uniform_(0.001, 0.01)
    a_scale = a_scale_cutlass.permute(2, 0, 1)
    b = torch.randn(
        (weight_groups, OUT_RANK, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    b_scale = torch.empty(
        (weight_groups, OUT_RANK // 128, HIDDEN_SIZE // 128),
        device=device,
        dtype=torch.float32,
    ).uniform_(0.001, 0.01)
    return a, a_scale, b, b_scale


def make_cos_sin_cache(max_position: int) -> torch.Tensor:
    half_dim = ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(half_dim, device="cuda", dtype=torch.float32) / half_dim)
    )
    frequencies = torch.outer(
        torch.arange(max_position, device="cuda", dtype=torch.float32),
        inv_freq,
    )
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


@pytest.mark.parametrize(
    "num_tokens,num_groups,expected",
    [(128, 8, False), (256, 2, True), (256, 8, True)],
)
def test_sm120_cutlass_dispatch_boundary(monkeypatch, num_tokens, num_groups, expected):
    monkeypatch.setattr(
        fp8_einsum_ops.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(12, 0),
    )
    monkeypatch.setattr(
        fp8_einsum_ops,
        "_deepseek_v4_sm120_cutlass_compiled",
        lambda _: True,
    )
    assert (
        fp8_einsum_ops.use_deepseek_v4_sm120_cutlass_fp8_einsum(
            num_tokens,
            num_groups,
            OUT_RANK,
            HIDDEN_SIZE,
        )
        is expected
    )


def test_sm120_cutlass_dispatch_rejects_sm89(monkeypatch):
    monkeypatch.setattr(
        fp8_einsum_ops.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(8, 9),
    )
    monkeypatch.setattr(
        fp8_einsum_ops,
        "_deepseek_v4_sm120_cutlass_compiled",
        lambda _: True,
    )
    assert not fp8_einsum_ops.use_deepseek_v4_sm120_cutlass_fp8_einsum(
        256,
        8,
        OUT_RANK,
        HIDDEN_SIZE,
    )


def test_sm120_cutlass_dispatch_rejects_uncompiled_kernel(monkeypatch):
    monkeypatch.setattr(
        fp8_einsum_ops.current_platform,
        "get_device_capability",
        lambda: DeviceCapability(12, 0),
    )
    monkeypatch.setattr(
        fp8_einsum_ops,
        "_deepseek_v4_sm120_cutlass_compiled",
        lambda _: False,
    )
    assert not fp8_einsum_ops.use_deepseek_v4_sm120_cutlass_fp8_einsum(
        256,
        8,
        OUT_RANK,
        HIDDEN_SIZE,
    )


@pytest.mark.parametrize(
    "num_tokens,num_groups",
    [(256, 2), (256, 8), (257, 8)],
)
@requires_sm12x
@torch.inference_mode()
def test_cutlass_batched_direct_matches_triton(num_tokens, num_groups):
    a, a_scale, b, b_scale = make_inputs(num_tokens, num_groups)
    reference = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty_like(reference)

    fp8_einsum_ops.deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, reference)
    torch.ops._C.deepseek_v4_fp8_bmm_sm120(
        output,
        a.transpose(0, 1),
        b.transpose(1, 2),
        a_scale.transpose(0, 1).permute(0, 2, 1),
        b_scale,
    )

    assert output.is_contiguous()
    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("scale_dims", [2, 3])
@requires_sm12x
@torch.inference_mode()
def test_cutlass_dispatch_preserves_tp_group_slice(monkeypatch, scale_dims):
    num_tokens, num_groups, weight_groups = 256, 8, 16
    a, a_scale, b, b_scale = make_inputs(
        num_tokens,
        num_groups,
        weight_groups=weight_groups,
    )
    monkeypatch.setattr(
        fp8_einsum_ops,
        "get_tensor_model_parallel_rank",
        lambda: 1,
    )
    output = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = torch.empty_like(output)
    wrapper_scale = b_scale.flatten(0, 1) if scale_dims == 2 else b_scale

    fp8_einsum_ops.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b.flatten(0, 1),
        wrapper_scale,
        output,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )
    fp8_einsum_ops.deepseek_v4_sm12x_fp8_einsum(
        a,
        a_scale,
        b[num_groups:],
        b_scale[num_groups:],
        reference,
    )

    torch.testing.assert_close(output, reference, atol=1e-3, rtol=1e-2)


@requires_sm12x
@torch.inference_mode()
def test_cutlass_batched_direct_cuda_graph():
    num_tokens, num_groups = 257, 2
    a, a_scale, b, b_scale = make_inputs(num_tokens, num_groups)
    output = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = torch.empty_like(output)
    a_group_major = a.transpose(0, 1)
    a_scale_cutlass = a_scale.transpose(0, 1).permute(0, 2, 1)
    b_cutlass = b.transpose(1, 2)

    torch.ops._C.deepseek_v4_fp8_bmm_sm120(
        reference,
        a_group_major,
        b_cutlass,
        a_scale_cutlass,
        b_scale,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.deepseek_v4_fp8_bmm_sm120(
            output,
            a_group_major,
            b_cutlass,
            a_scale_cutlass,
            b_scale,
        )

    output_ptr = output.data_ptr()
    for _ in range(100):
        graph.replay()
    torch.accelerator.synchronize()
    assert output.data_ptr() == output_ptr
    assert torch.equal(output, reference)


@requires_sm12x
@torch.inference_mode()
def test_o_proj_uses_compact_cutlass_chain():
    num_tokens, num_groups = 256, 8
    heads_per_group = NUM_HEADS // num_groups
    max_position = 4096
    o = torch.randn(
        (num_tokens, NUM_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.randint(
        max_position,
        (num_tokens,),
        device="cuda",
        dtype=torch.long,
    )
    cos_sin_cache = make_cos_sin_cache(max_position)
    _, _, weight, weight_scale = make_inputs(num_tokens, num_groups)
    wo_a = torch.nn.Module()
    wo_a.register_buffer("weight", weight.flatten(0, 1))
    wo_a.register_buffer("weight_scale", weight_scale.flatten(0, 1))

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=num_groups,
        heads_per_group=heads_per_group,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        compact_scales=True,
    )
    reference = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    fp8_einsum_ops.deepseek_v4_sm12x_fp8_einsum(
        o_fp8,
        o_scale,
        weight,
        weight_scale,
        reference,
    )

    output = deep_gemm_fp8_o_proj(
        o,
        positions,
        cos_sin_cache,
        wo_a,
        torch.nn.Identity(),
        n_groups=num_groups,
        heads_per_group=heads_per_group,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        o_lora_rank=OUT_RANK,
        einsum_recipe=(1, 128, 128),
        tma_aligned_scales=False,
    )

    assert output.shape == (num_tokens, num_groups * OUT_RANK)
    torch.testing.assert_close(
        output.view_as(reference),
        reference,
        atol=1e-3,
        rtol=1e-2,
    )


@requires_sm12x
@torch.inference_mode()
def test_default_padded_scale_falls_back_to_triton():
    num_tokens, num_groups = 257, 8
    heads_per_group = NUM_HEADS // num_groups
    max_position = 4096
    o = torch.randn(
        (num_tokens, NUM_HEADS, HEAD_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.randint(
        max_position,
        (num_tokens,),
        device="cuda",
        dtype=torch.long,
    )
    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        make_cos_sin_cache(max_position),
        n_groups=num_groups,
        heads_per_group=heads_per_group,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
    )
    _, _, weight, weight_scale = make_inputs(num_tokens, num_groups)
    output = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = torch.empty_like(output)

    assert not o_scale.transpose(0, 1).permute(0, 2, 1).is_contiguous()
    fp8_einsum_ops.deepseek_v4_fp8_einsum(
        o_fp8,
        o_scale,
        weight.flatten(0, 1),
        weight_scale.flatten(0, 1),
        output,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )
    fp8_einsum_ops.deepseek_v4_sm12x_fp8_einsum(
        o_fp8,
        o_scale,
        weight,
        weight_scale,
        reference,
    )

    assert torch.equal(output, reference)


@requires_sm89
@torch.inference_mode()
def test_sm89_wrapper_uses_triton_fallback():
    num_tokens, num_groups = 256, 2
    a, a_scale, b, b_scale = make_inputs(num_tokens, num_groups)
    output = torch.empty(
        (num_tokens, num_groups, OUT_RANK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    reference = torch.empty_like(output)

    fp8_einsum_ops.deepseek_v4_fp8_einsum(
        a,
        a_scale,
        b.flatten(0, 1),
        b_scale.flatten(0, 1),
        output,
        "bhr,hdr->bhd",
        [1, 128, 128],
    )
    fp8_einsum_ops.deepseek_v4_sm12x_fp8_einsum(
        a,
        a_scale,
        b,
        b_scale,
        reference,
    )

    assert torch.equal(output, reference)
