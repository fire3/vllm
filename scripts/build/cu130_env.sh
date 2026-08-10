#!/usr/bin/env bash
# Source this file to configure your shell for building a CUDA 13.0 vLLM wheel
# that only carries SM 8.9 (NVIDIA Ada, e.g. RTX 40-series) as PTX.
#
#   source scripts/build/cu130_env.sh

# nvcc lives under /usr/local (here /usr/local/cuda -> /usr/local/cuda-13.0).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# vLLM uses this to name the wheel variant and to sanity-check the toolchain.
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-13.0}"

# PyTorch + vLLM read TORCH_CUDA_ARCH_LIST to build the nvcc -gencode flags.
#   8.9     -> SASS for sm_89 only (smallest; runs only on Ada / SM 8.9)
#   8.9+PTX -> same SASS plus PTX, so newer GPUs can JIT from it (default)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"

# Optional: give the wheel a distinguishable version, e.g.
#   export VLLM_VERSION_OVERRIDE=0.26.0+cu130sm89
