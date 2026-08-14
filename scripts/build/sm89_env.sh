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

# Wheel version: default to the setuptools-scm public version plus the local
# marker "+sm89", so every wheel built through these scripts is identifiable
# as an SM89 build, e.g.
#   0.27.1.dev28+g7a1d4cf79  ->  vllm-0.27.1+sm89-cp312-...
# Override with VLLM_VERSION_OVERRIDE to keep a custom version, e.g.
#   export VLLM_VERSION_OVERRIDE=0.26.0+cu130sm89
if [[ -z "${VLLM_VERSION_OVERRIDE:-}" ]]; then
  if python3 -c 'import setuptools_scm' >/dev/null 2>&1; then
    scm_public="$(
      python3 - <<'PY'
from setuptools_scm import get_version
v = get_version()
for sep in (".dev", ".post"):
    v = v.split(sep, 1)[0]
print(v)
PY
    )" || scm_public="0.27.0"
  else
    scm_public="$(
      git -C "${REPO_ROOT}" describe --tags --abbrev=0 2>/dev/null \
        | sed 's/^v//' || echo 0.27.0
    )"
  fi
  export VLLM_VERSION_OVERRIDE="${scm_public}+sm89"
fi
