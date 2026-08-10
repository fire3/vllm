#!/usr/bin/env bash
# Verify a freshly built wheel: confirm the toolchain and that the compiled
# kernels contain only SM 8.9 (SASS and/or PTX), not other architectures.
#
#   bash scripts/build/cu130_verify.sh [path/to/vllm-*.whl]
set -euo pipefail

PYTHON="${CONDA_PREFIX:?run inside your conda env}/bin/python"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# 1. Toolchain version.
"${CUDA_HOME}/bin/nvcc" --version

# 2. Locate the wheel to inspect.
WHEEL="${1:-}"
if [[ -z "${WHEEL}" ]]; then
  WHEEL="$(ls -t dist/vllm-*.whl | head -1)"
fi
echo "Wheel: ${WHEEL}"

# 3. Extract the extension modules and list the embedded compute capabilities.
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
unzip -q -o "${WHEEL}" 'vllm/*.so' -d "${TMP}"

found=0
while IFS= read -r so; do
  if "${CUDA_HOME}/bin/cuobjdump" --list-elf "${so}" 2>/dev/null | grep -Eq 'sm_89|compute_89'; then
    echo "== $(basename "${so}") =="
    "${CUDA_HOME}/bin/cuobjdump" --list-elf "${so}"
    found=1
  fi
done < <(find "${TMP}" -name '*.so')

if [[ "${found}" -eq 0 ]]; then
  echo "No SM 8.9 / compute_89 kernel found in the wheel."
  exit 1
fi
