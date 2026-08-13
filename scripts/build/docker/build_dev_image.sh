#!/usr/bin/env bash
# 构建 vLLM DSV4 SM89 开发镜像。
#
#   bash scripts/build/docker/build_dev_image.sh [--no-cache] [--pull] [--progress plain]
#
# 配置见同目录 dev.env（也可用同名环境变量覆盖）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/scripts/build/docker"

# shellcheck source=dev.env
. "${DOCKER_DIR}/dev.env"

# 根据 CUDA_VERSION 推导 PyTorch 索引（13.0.3 -> cu130）
CUDA_TAG="cu$(printf '%s' "${CUDA_VERSION}" | cut -d. -f1,2 | tr -d '.')"
if [[ -z "${TORCH_INDEX_URL}" ]]; then
  if [[ "${USE_CN_MIRROR}" == "1" ]]; then
    TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/${CUDA_TAG}"
  else
    TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
  fi
fi
if [[ -z "${UV_DEFAULT_INDEX}" ]]; then
  if [[ "${USE_CN_MIRROR}" == "1" ]]; then
    UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  else
    UV_DEFAULT_INDEX="https://pypi.org/simple"
  fi
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-cache|--pull) EXTRA_ARGS+=("$1") ;;
    --progress) EXTRA_ARGS+=("--progress" "$2"); shift ;;
    -h|--help)
      echo "用法: $0 [--no-cache] [--pull] [--progress plain]" >&2
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
  shift
done

BUILD_ARGS=(
  --build-arg "CUDA_VERSION=${CUDA_VERSION}"
  --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
  --build-arg "TORCH_VERSION=${TORCH_VERSION}"
  --build-arg "VLLM_MAIN_CUDA_VERSION=${VLLM_MAIN_CUDA_VERSION}"
  --build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  --build-arg "PROTOC_VERSION=${PROTOC_VERSION}"
  --build-arg "UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}"
  --build-arg "TORCH_INDEX_URL=${TORCH_INDEX_URL}"
  --build-arg "UV_PYTHON_INSTALL_MIRROR=${UV_PYTHON_INSTALL_MIRROR}"
  --build-arg "RUSTUP_DIST_SERVER=${RUSTUP_DIST_SERVER}"
  --build-arg "PROTOC_GH_MIRROR=${PROTOC_GH_MIRROR}"
  --build-arg "INSTALL_TEST_DEPS=${INSTALL_TEST_DEPS}"
)

echo "== 构建参数 =="
printf '  CUDA=%s  Ubuntu=%s  Python=%s  Torch=%s  Arch=%s\n' \
  "${CUDA_VERSION}" "${UBUNTU_VERSION}" "${PYTHON_VERSION}" "${TORCH_VERSION}" "${TORCH_CUDA_ARCH_LIST}"
printf '  PyTorch index: %s\n  PyPI index: %s\n' "${TORCH_INDEX_URL}" "${UV_DEFAULT_INDEX}"
echo "== 开始构建镜像: ${IMAGE_TAG} =="

docker build "${EXTRA_ARGS[@]}" \
  -f "${DOCKER_DIR}/Dockerfile.dev" \
  -t "${IMAGE_TAG}" \
  "${BUILD_ARGS[@]}" \
  "${REPO_ROOT}"

echo
echo "构建完成: ${IMAGE_TAG}"
echo "下一步: bash scripts/build/docker/run_dev.sh up"
