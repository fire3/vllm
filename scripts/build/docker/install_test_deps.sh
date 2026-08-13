#!/usr/bin/env bash
# 在容器里安装 vLLM 开发/测试依赖（pytest、pre-commit、mypy 等）。
#
# 与官方 requirements/dev.txt 的区别：会剔除 test/cuda.txt 里 torch 系的硬 pin，
# 保留镜像当前安装的 torch 版本（scripts/build 的 cu130 流程默认是 2.13.0）。
#
#   /usr/local/bin/vllm-install-test-deps
set -euo pipefail

cd /workspace/vllm
PYTHON="/opt/venv/bin/python"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.org/simple}"

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

cp -a requirements "${TMP}/requirements"
sed -E '/^(torch|torchaudio|torchvision|torchcodec)([[:space:]]*[<>=]|$)/d' \
  "${TMP}/requirements/test/cuda.txt" > "${TMP}/requirements/test/cuda.txt.new"
mv "${TMP}/requirements/test/cuda.txt.new" "${TMP}/requirements/test/cuda.txt"

uv pip install --python "${PYTHON}" \
  --index-url "${UV_DEFAULT_INDEX}" \
  --extra-index-url "${TORCH_INDEX_URL}" \
  -r "${TMP}/requirements/dev.txt"

uv pip install --python "${PYTHON}" -e tests/vllm_test_utils

echo
echo "开发/测试依赖安装完成。当前 torch 版本："
uv pip show --python "${PYTHON}" torch | grep -E '^(Name|Version):'
