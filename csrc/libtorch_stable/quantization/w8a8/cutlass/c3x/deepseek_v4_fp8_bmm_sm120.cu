#include <torch/csrc/stable/tensor.h>

#include "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"

namespace vllm {

void deepseek_v4_fp8_bmm_sm120_impl(torch::stable::Tensor& out,
                                    torch::stable::Tensor const& a,
                                    torch::stable::Tensor const& b,
                                    torch::stable::Tensor const& a_scales,
                                    torch::stable::Tensor const& b_scales) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(a.dim() == 3 && b.dim() == 3 && out.dim() == 3);
  STD_TORCH_CHECK(a_scales.dim() == 3 && b_scales.dim() == 3);
  STD_TORCH_CHECK(a.scalar_type() == ScalarType::Float8_e4m3fn &&
                  b.scalar_type() == ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(out.scalar_type() == ScalarType::BFloat16);
  STD_TORCH_CHECK(a_scales.scalar_type() == ScalarType::Float &&
                  b_scales.scalar_type() == ScalarType::Float);

  int32_t groups = a.size(0);
  int32_t m = a.size(1);
  int32_t k = a.size(2);
  int32_t n = b.size(2);
  STD_TORCH_CHECK(b.size(0) == groups && b.size(1) == k);
  STD_TORCH_CHECK(out.size(0) == m && out.size(1) == groups &&
                  out.size(2) == n);
  STD_TORCH_CHECK(k % 128 == 0 && n % 128 == 0);
  STD_TORCH_CHECK(a.is_contiguous() && out.is_contiguous());
  STD_TORCH_CHECK(b.stride(0) == k * n && b.stride(1) == 1 && b.stride(2) == k);
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  STD_TORCH_CHECK(a_scales.size(0) == groups && a_scales.size(1) == k / 128 &&
                  a_scales.size(2) == m);
  STD_TORCH_CHECK(b_scales.size(0) == groups && b_scales.size(1) == n / 128 &&
                  b_scales.size(2) == k / 128);

  using Gemm =
      typename sm120_blockwise_fp8_config_pingpong<cutlass::bfloat16_t>::Gemm;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using ScaleConfig = typename Gemm::ScaleConfig;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, groups));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, groups));
  StrideC stride_c =
      cute::make_stride(int64_t(groups * n), cute::_1{}, int64_t(n));
  StrideD stride_d = stride_c;
  LayoutSFA layout_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, groups));
  LayoutSFB layout_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, groups));

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.ptr_A =
      static_cast<typename Gemm::ElementAB const*>(a.data_ptr());
  mainloop_args.dA = stride_a;
  mainloop_args.ptr_B =
      static_cast<typename Gemm::ElementAB const*>(b.data_ptr());
  mainloop_args.dB = stride_b;
  mainloop_args.ptr_SFA = static_cast<float const*>(a_scales.data_ptr());
  mainloop_args.layout_SFA = layout_sfa;
  mainloop_args.ptr_SFB = static_cast<float const*>(b_scales.data_ptr());
  mainloop_args.layout_SFB = layout_sfb;

  auto out_ptr = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, out_ptr, stride_c, out_ptr, stride_d};
  auto problem_shape = cute::make_shape(m, n, k, groups);
  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), problem_shape, mainloop_args,
                                       epilogue_args);
}

}  // namespace vllm
