#!/usr/bin/env bash
# Source this file to configure your shell for building a CUDA 13.0 vLLM wheel
# that only carries SM 8.9 (NVIDIA Ada, e.g. RTX 40-series) as PTX.
#
#   source scripts/build/sm89_env.sh

# nvcc lives under /usr/local (here /usr/local/cuda -> /usr/local/cuda-13.0).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# vLLM uses this to name the wheel variant and to sanity-check the toolchain.
export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION:-13.0}"

# PyTorch + vLLM read TORCH_CUDA_ARCH_LIST to build the nvcc -gencode flags.
#   8.9     -> SASS for sm_89 only (smallest; runs only on Ada / SM 8.9)
#   8.9+PTX -> same SASS plus PTX, so newer GPUs can JIT from it (default)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9+PTX}"

# Wheel version: default to the public version (derived from git tags, with the
# same next-version rule setuptools-scm uses) plus the local marker "+sm89", so
# every wheel built through these scripts is identifiable as an SM89 build, e.g.
#   0.27.1.dev28+g7a1d4cf79  ->  vllm-0.27.1+sm89-cp312-...
# Override with VLLM_VERSION_OVERRIDE to keep a custom version, e.g.
#   export VLLM_VERSION_OVERRIDE=0.26.0+cu130sm89
if [[ -z "${VLLM_VERSION_OVERRIDE:-}" ]]; then
  last_tag="$(git -C "${REPO_ROOT}" describe --tags --abbrev=0 2>/dev/null || true)"
  if [[ -z "${last_tag}" ]]; then
    last_tag="v0.27.0"
  fi
  dist="$(git -C "${REPO_ROOT}" rev-list --count "${last_tag}..HEAD" 2>/dev/null || echo 0)"
  public="${last_tag#v}"
  if [[ "${dist}" != "0" && "${public}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    # Beyond the last release tag, bump the final component like
    # setuptools-scm's next-version guess (v0.27.0 -> 0.27.1.devN -> 0.27.1).
    public="${public%.*}.$(( ${public##*.} + 1 ))"
  fi
  export VLLM_VERSION_OVERRIDE="${public}+sm89"
fi
