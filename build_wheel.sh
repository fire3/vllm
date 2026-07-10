#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:/home/yyf/.cargo/bin:$PATH"
export VLLM_TARGET_DEVICE=cuda
export VLLM_MAIN_CUDA_VERSION=13.0
BASE_VERSION=$(.venv/bin/python setup.py --version)
export VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE:-${BASE_VERSION}.cu130}
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export MAX_JOBS=${MAX_JOBS:-16}
export NVCC_THREADS=${NVCC_THREADS:-2}

echo "=== START $(date +%T) | nvcc $(nvcc --version | tail -1) | arch=$TORCH_CUDA_ARCH_LIST ==="
.venv/bin/python -m build --wheel --no-isolation --outdir dist-sm89
echo "=== END $(date +%T) ==="
ls -lh dist-sm89/*.whl
echo "WHEEL_BUILD_OK"
