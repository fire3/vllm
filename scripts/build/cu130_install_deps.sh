#!/usr/bin/env bash
# Install the cu130 PyTorch build and the build toolchain into the *active*
# conda environment. Run this inside the conda env you will build in.
#
#   conda activate vllm-dev
#   bash scripts/build/cu130_install_deps.sh            # torch + build deps
#   bash scripts/build/cu130_install_deps.sh --runtime  # also sync runtime deps
#
# Inside China, set USE_CN_MIRROR=1 to download deps from China mirrors
# (Aliyun PyTorch wheels, Tsinghua PyPI) for faster installs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CONDA_PREFIX:?run inside your conda env, e.g. \"conda activate vllm-dev\"}/bin/python"
INSTALL_RUNTIME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime) INSTALL_RUNTIME=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--runtime]" >&2
      echo "  --runtime  also sync the runtime deps pinned in requirements/cuda.txt" >&2
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Set USE_CN_MIRROR=1 to download deps from China mirrors (Aliyun PyTorch
# wheels, Tsinghua PyPI) for faster installs inside China.
USE_CN_MIRROR="${USE_CN_MIRROR:-0}"
if [[ "${USE_CN_MIRROR}" == "1" ]]; then
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirrors.aliyun.com/pytorch-wheels/cu130}"
  PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
else
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
  PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
fi

# Sanity check: nvcc must be reachable.
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version

# Keep the torch pin in sync with requirements/ automatically: the cu130
# wheels are not on PyPI, so torch is installed from the dedicated index only.
TORCH_VERSION="${TORCH_VERSION:-}"
if [[ -z "${TORCH_VERSION}" ]]; then
  TORCH_VERSION="$(grep -m1 '^torch==' "${REPO_ROOT}/requirements/build/cuda.txt" | cut -d= -f3)"
fi
if [[ -z "${TORCH_VERSION}" ]]; then
  echo "ERROR: cannot determine torch version from requirements/build/cuda.txt" >&2
  exit 1
fi

uv pip install --python "${PYTHON}" \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Install the remaining build requirements from PyPI, but strip the torch
# family first: torch is already installed from the cu130 index above, and a
# plain PyPI install would replace it with the CPU wheel. Mirrors the
# filtering done in scripts/build/docker/Dockerfile.dev.
sed -E '/^(torch|torchaudio|torchvision|torchcodec)([[:space:]]*[<>=]|$)/d' \
  "${REPO_ROOT}/requirements/build/cuda.txt" > "${TMP}/build-cuda.txt"
uv pip install --python "${PYTHON}" \
  --index-url "${PYPI_INDEX_URL}" \
  -r "${TMP}/build-cuda.txt"

if [[ "${INSTALL_RUNTIME}" == "1" ]]; then
  # Sync runtime deps (flashinfer, tilelang, ...) to the pins in
  # requirements/cuda.txt. Torch-family lines are stripped for the same reason
  # as above; --extra-index-url entries inside the file (e.g. flashinfer.ai)
  # are preserved.
  cp "${REPO_ROOT}/requirements/common.txt" "${REPO_ROOT}/requirements/cuda.txt" "${TMP}/"
  sed -E '/^(torch|torchaudio|torchvision|torchcodec)([[:space:]]*[<>=]|$)/d' \
    "${TMP}/cuda.txt" > "${TMP}/cuda.filtered" && mv "${TMP}/cuda.filtered" "${TMP}/cuda.txt"
  uv pip install --python "${PYTHON}" \
    --index-url "${PYPI_INDEX_URL}" \
    --extra-index-url "${TORCH_INDEX_URL}" \
    -r "${TMP}/cuda.txt"
fi

echo "Dependencies installed. Next: bash scripts/build/cu130_build_wheel.sh"
