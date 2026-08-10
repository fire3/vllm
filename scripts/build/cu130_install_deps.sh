#!/usr/bin/env bash
# Install the cu130 PyTorch build and the build toolchain into the *active*
# conda environment. Run this inside the conda env you will build in.
#
#   conda activate vllm-dev
#   bash scripts/build/cu130_install_deps.sh
#
# Inside China, set USE_CN_MIRROR=1 to download deps from China mirrors
# (Aliyun PyTorch wheels, Tsinghua PyPI) for faster installs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CONDA_PREFIX:?run inside your conda env, e.g. \"conda activate vllm-dev\"}/bin/python"

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

# PyTorch cu130 build. It is NOT on PyPI, so install it from the dedicated
# index (only the cu130 index is used for this command).
uv pip install --python "${PYTHON}" \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==2.11.0"

# Remaining build-only requirements (cmake, ninja, setuptools-rust, ...).
# These are NOT on the PyTorch cu130 index, so install them from the PyPI
# mirror. torch 2.11.0+cu130 already satisfies the ==2.11.0 pin, so uv keeps
# the cu130 build instead of fetching the PyPI CPU one.
uv pip install --python "${PYTHON}" \
  --index-url "${PYPI_INDEX_URL}" \
  -r "${REPO_ROOT}/requirements/build/cuda.txt"

echo "Dependencies installed. Next: bash scripts/build/cu130_build_wheel.sh"
