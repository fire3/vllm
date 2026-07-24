# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/sgl-project/sglang/pull/2575
import itertools
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quant_utils import (
    native_per_token_group_quant_fp8,
    native_w8a8_block_matmul,
)
from tests.kernels.utils import fp8_ulp_distance
from vllm.config import VllmConfig
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFp8BlockScaledMMKernel,
    _use_triton_for_sm12x_block_fp8,
    cutlass_scaled_mm,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    w8a8_triton_block_scaled_mm,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_gemm_nt,
    get_tma_aligned_size,
    per_block_cast_to_fp8,
    should_use_deepgemm_for_fp8_linear,
)
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    has_flashinfer_fp8_blockscale_gemm,
)
from vllm.utils.import_utils import has_deep_gemm
from vllm.utils.torch_utils import set_default_torch_dtype

if current_platform.get_device_capability() < (9, 0):
    pytest.skip("FP8 Triton requires CUDA 9.0 or higher", allow_module_level=True)

vllm_config = VllmConfig()

# Test configurations
DTYPES = [torch.bfloat16]  # [torch.half, torch.bfloat16, torch.float32]
# Quantization test configs
NUM_TOKENS = [7, 2050]
D = [512, 4096, 5120, 13824]
GROUP_SIZE = [64, 128, 512]
COLUMN_MAJOR_SCALES = [True, False]
TMA_ALIGNED_SCALES = [True, False]
# Matmul test configs
M = [1, 7, 8, 83, 4096]
N = [128, 512, 576, 7168, 13824]
K = [256, 3884, 4096, 13824, 16384]
# Deepseek-V3's intermediate size 18432, so N is 18432*2/8=4608 at TP8
# and its hidden size is 7168.
BLOCK_SIZE = [[128, 128]]
OUT_DTYPES = [torch.bfloat16]  # [torch.float32, torch.half, torch.bfloat16]
SEEDS = [0]

# Skip all tests if CUDA is not available
pytest.importorskip("torch.cuda")


@pytest.fixture(autouse=True)
def setup_cuda():
    torch.set_default_device("cuda")


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "num_tokens,d,dtype,group_size,column_major_scales,tma_aligned_scales,seed",
    itertools.product(
        NUM_TOKENS,
        D,
        DTYPES,
        GROUP_SIZE,
        COLUMN_MAJOR_SCALES,
        TMA_ALIGNED_SCALES,
        SEEDS,
    ),
)
@torch.inference_mode()
def test_per_token_group_quant_fp8(
    num_tokens, d, dtype, group_size, column_major_scales, tma_aligned_scales, seed
):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype)

    ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
    out, scale = per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major_scales,
        tma_aligned_scales=tma_aligned_scales,
    )

    if current_platform.is_rocm():
        # On gfx950 the Triton and PyTorch FP8 kernels can round in opposite
        # directions when an element lands at the midpoint between two adjacent
        # e4m3fn values (1-ULP tie-breaking). Verify: (1) no element is more
        # than 1 FP8 ULP away, and (2) fewer than 0.05% of elements have any
        # mismatch. Observed worst case across all parameter combos: 0.049%,
        # max ULP = 1.
        ulp = fp8_ulp_distance(out, ref_out)
        assert (ulp <= 1).all(), (
            f"FP8 mismatch > 1 ULP: {int((ulp > 1).sum())} elements"
        )
        assert float((ulp > 0).float().mean()) < 5e-4, (
            f"Too many 1-ULP mismatches: {int((ulp > 0).sum())}/{ulp.numel()}"
        )
    else:
        assert torch.allclose(
            out.to(torch.float32), ref_out.to(torch.float32), rtol=0.15
        )
    assert torch.allclose(scale, ref_scale)

    if column_major_scales:
        assert scale.stride()[-2] == 1
        if tma_aligned_scales:
            assert scale.stride()[-1] == get_tma_aligned_size(num_tokens, 4)


@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(current_platform.fp8_dtype())
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(current_platform.fp8_dtype())

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@torch.inference_mode()
def test_w8a8_block_fp8_matmul_e8m0_scales():
    # DeepSeek-V4-style checkpoints store block scales in exponent-only
    # E8M0, which Triton cannot bind directly; the kernel upcasts them to
    # fp32 before launch. Regression test for
    # https://github.com/vllm-project/vllm/issues/47818.
    M, N, K = 83, 512, 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    torch.manual_seed(0)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_size[0] - 1) // block_size[0]
    k_tiles = (K + block_size[1] - 1) // block_size[1]

    # Power-of-two scales round-trip E8M0 <-> fp32 exactly.
    As = torch.exp2(torch.randint(-8, 0, (M, k_tiles)).to(torch.float32))
    Bs = torch.exp2(torch.randint(-8, 0, (n_tiles, k_tiles)).to(torch.float32))

    ref_out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(
        A_fp8,
        B_fp8,
        As.to(torch.float8_e8m0fnu),
        Bs.to(torch.float8_e8m0fnu),
        block_size,
        out_dtype,
    )
    assert torch.equal(out, ref_out)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUTLASS only supported on CUDA platform."
)
@pytest.mark.skipif(
    current_platform.get_device_capability().major == 12,
    reason="SM12x requires the linear kernel's load-time N padding.",
)
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_matmul():
    # Test simple case where weight.shape % 128 != 0,
    # like in DSV3 kv_a_proj_with_mqa
    M = 32
    N = 576
    K = 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    seed = 0

    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

    A_fp8, As = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=False
    )
    # CUTLASS uses column-major format for scales
    A_fp8_cutlass, As_cutlass = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True
    )

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = cutlass_scaled_mm(A_fp8_cutlass, B_fp8, As_cutlass, Bs, block_size, out_dtype)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    current_platform.get_device_capability().major != 12,
    reason="SM12x-specific CUTLASS output-block padding regression.",
)
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_sm12x_pads_output_block(
    default_vllm_config,
):
    m, n, k = 32, 576, 7168
    block_size = [128, 128]
    torch.manual_seed(0)
    input_bf16 = torch.randn((m, k), dtype=torch.bfloat16)
    weight_bf16 = torch.randn((n, k), dtype=torch.bfloat16)
    bias = torch.randn((n,), dtype=torch.bfloat16)
    weight, weight_scale = per_block_cast_to_fp8(weight_bf16, block_size)
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=create_fp8_quant_key(
            static=True,
            group_shape=GroupShape(*block_size),
        ),
        activation_quant_key=create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, block_size[1]),
        ),
        weight_shape=(n, k),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    kernel = CutlassFp8BlockScaledMMKernel(config)
    activation, activation_scale = kernel.quant_fp8(input_bf16)
    reference = w8a8_triton_block_scaled_mm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        block_size,
        torch.bfloat16,
    )
    reference += bias
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(weight, requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(weight_scale, requires_grad=False),
    )

    kernel.process_weights_after_loading(layer)
    output = kernel.apply_weights(layer, input_bf16, bias=bias)

    current_stream = torch.cuda.current_stream()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(current_stream)
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            kernel.apply_weights(layer, input_bf16, bias=bias)
    current_stream.wait_stream(capture_stream)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = kernel.apply_weights(layer, input_bf16, bias=bias)
    graph.replay()
    torch.accelerator.synchronize()

    assert layer.weight.shape == (640, k)
    assert output.shape == (m, n)
    assert torch.equal(graph_output, output)
    relative_difference = (output.float() - reference.float()).abs().mean()
    relative_difference /= reference.float().abs().mean()
    assert relative_difference < 0.001


@pytest.mark.parametrize(
    ("m", "n", "k", "expected"),
    [
        (32, 24576, 7168, True),
        (64, 24576, 1536, True),
        (64, 32768, 512, True),
        (64, 36864, 7168, True),
        (1, 24576, 7168, False),
        (128, 24576, 7168, False),
        (64, 12288, 7168, False),
    ],
)
def test_sm12x_block_fp8_dispatch_policy(m, n, k, expected):
    assert _use_triton_for_sm12x_block_fp8(m, n, k) is expected


@pytest.mark.skipif(
    current_platform.get_device_capability().major != 12,
    reason="SM12x-specific merged linear output-block padding regression.",
)
@pytest.mark.parametrize("skip_bias_add", [False, True])
@torch.inference_mode()
def test_w8a8_block_fp8_cutlass_sm12x_merged_linear(
    default_vllm_config,
    dist_init,
    monkeypatch,
    skip_bias_add,
):
    m, n, k = 7, 576, 128
    block_size = [128, 128]
    default_vllm_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type="deepseek_v4")
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=block_size,
    )
    monkeypatch.setattr(
        "vllm.envs.VLLM_DISABLED_KERNELS",
        ["DeepGemmFp8BlockScaledMMKernel"],
    )
    with set_default_torch_dtype(torch.bfloat16):
        layer = MergedColumnParallelLinear(
            input_size=k,
            output_sizes=[256, 320],
            bias=True,
            skip_bias_add=skip_bias_add,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            disable_tp=True,
        )
    assert isinstance(
        layer.quant_method.fp8_linear,
        CutlassFp8BlockScaledMMKernel,
    )

    torch.manual_seed(1)
    input_bf16 = torch.randn((m, k), dtype=torch.bfloat16)
    weight_bf16 = torch.randn((n, k), dtype=torch.bfloat16)
    bias = torch.randn((n,), dtype=torch.bfloat16)
    weight, weight_scale = per_block_cast_to_fp8(weight_bf16, block_size)
    layer.weight.copy_(weight)
    layer.weight_scale_inv.copy_(weight_scale)
    layer.bias.copy_(bias)
    layer.quant_method.process_weights_after_loading(layer)

    activation, activation_scale = layer.quant_method.fp8_linear.quant_fp8(input_bf16)
    reference = w8a8_triton_block_scaled_mm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        block_size,
        torch.bfloat16,
    )
    if not skip_bias_add:
        reference += bias

    output, output_bias = layer(input_bf16)

    assert layer.output_partition_sizes == [256, 320]
    assert layer.weight.shape == (640, k)
    assert output.shape == (m, n)
    assert output.is_contiguous()
    if skip_bias_add:
        assert torch.equal(output_bias, bias)
    else:
        assert output_bias is None
    relative_difference = (output.float() - reference.float()).abs().mean()
    relative_difference /= reference.float().abs().mean()
    assert relative_difference < 0.001


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@pytest.mark.skipif(not has_deep_gemm(), reason="DeepGemm kernels not available.")
@torch.inference_mode()
def test_w8a8_block_fp8_deep_gemm_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max

    # only aligned sizes are supported by deepgemm
    if not should_use_deepgemm_for_fp8_linear(
        output_dtype=out_dtype, weight_shape=B_fp32.shape, supports_deep_gemm=True
    ):
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    A_fp8, As_fp8 = per_token_group_quant_fp8(
        A_fp32, block_size[1], column_major_scales=True, tma_aligned_scales=True
    )
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_fp32, block_size=block_size)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = torch.zeros((M, N), device="cuda", dtype=out_dtype)

    assert As_fp8.shape == (M, (K + 127) // 128), (
        f"{As_fp8.shape} != {(M, (K + 127) // 128)}"
    )

    fp8_gemm_nt((A_fp8, As_fp8), (B_fp8, Bs_fp8), out)

    rel_diff = torch.mean(
        torch.abs(out.to(torch.float32) - ref_out.to(torch.float32))
    ) / torch.mean(torch.abs(ref_out.to(torch.float32)))
    assert rel_diff < 0.001


@pytest.mark.skipif(
    current_platform.is_fp8_fnuz(),
    reason="This platform supports e4m3fnuz, not e4m3fn.",
)
@pytest.mark.parametrize(
    "M,N,K,block_size,out_dtype,seed",
    itertools.product(M, N, K, BLOCK_SIZE, OUT_DTYPES, SEEDS),
)
@torch.inference_mode()
def test_w8a8_block_fp8_flashinfer_matmul(M, N, K, block_size, out_dtype, seed):
    if not has_flashinfer_fp8_blockscale_gemm():
        pytest.skip(
            "FlashInfer block GEMM not available (requires SM90+ and FlashInfer)"
        )
    # only aligned sizes
    if K % 128 != 0 or N % 64 != 0:
        pytest.skip(f"Skipping test; invalid size {M}, {N}, {K}")

    torch.manual_seed(seed)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max = fp8_info.max

    A_bf16 = (torch.rand(M, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max
    B_bf16 = (torch.rand(N, K, dtype=torch.bfloat16) - 0.5) * 2 * fp8_max

    A_fp8, As_fp8 = per_token_group_quant_fp8(A_bf16, block_size[1], use_ue8m0=False)
    B_fp8, Bs_fp8 = per_block_cast_to_fp8(B_bf16, block_size, use_ue8m0=False)

    As = As_fp8.to(torch.float32)
    Bs = Bs_fp8.to(torch.float32)

    ref_out = native_w8a8_block_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    out = flashinfer_fp8_blockscale_gemm(
        input=A_bf16,
        weight=B_fp8,
        input_scale=None,
        weight_scale=Bs,
        out_dtype=out_dtype,
    )

    rel_diff = torch.mean(
        torch.abs(out.to(torch.bfloat16) - ref_out.to(torch.bfloat16))
    ) / torch.mean(torch.abs(ref_out.to(torch.bfloat16)))
    assert rel_diff < 0.001
