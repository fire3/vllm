#!/usr/bin/env bash
# vLLM DSV4 SM89 开发容器管理入口。
#
#   bash scripts/build/docker/run_dev.sh <命令> [参数...]
#
# 命令:
#   build              构建镜像（等价于 build_dev_image.sh）
#   up                 创建并启动容器（不存在则创建），随后进入容器
#   exec [CMD...]      在运行中的容器里执行命令（默认 bash）
#   run [CMD...]       一次性运行一个容器（不保留）
#   gpu                容器内 nvidia-smi 验证
#   deps [ARGS...]     同步 .deps（bash scripts/build/prepare_deps.sh --gpu ...）
#   wheel [--editable] 构建 wheel（scripts/build/cu130_build_wheel.sh ...）
#   verify [WHEEL]     校验 wheel 里的 SM89 内核
#   serve [ARGS...]    vllm serve ...
#   test [ARGS...]     python -m pytest ...
#   install-test-deps  安装 pytest 等测试依赖（保留当前 torch 版本）
#   stop | rm | logs | status
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DOCKER_DIR="${REPO_ROOT}/scripts/build/docker"

# shellcheck source=dev.env
. "${DOCKER_DIR}/dev.env"

DOCKER="${DOCKER:-docker}"

# 容器公共参数（up 会额外加 --name / -d）
COMMON_FLAGS=(
  --gpus all
  --ipc host
  --shm-size "${SHM_SIZE}"
  -v "${REPO_ROOT}:${WORKDIR_MOUNT}"
  -v "${DEPS_VOLUME}:${WORKDIR_MOUNT}/.deps"
  -v "${UV_VOLUME}:/opt/uv"
  -v "${HOME_VOLUME}:/home/dev"
  -p "${VLLM_PORT}:8000"
  -e "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  -e "VLLM_MAIN_CUDA_VERSION=${VLLM_MAIN_CUDA_VERSION}"
  -e "VLLM_ENABLE_CUDA_COMPATIBILITY=${VLLM_ENABLE_CUDA_COMPATIBILITY}"
  -e "CONDA_PREFIX=/opt/venv"
  -e "HF_HOME=/home/dev/.cache/huggingface"
  -e "VLLM_CACHE_ROOT=/home/dev/.cache/vllm"
  -e "TRITON_CACHE_DIR=/home/dev/.cache/triton"
  -e "CCACHE_DIR=/home/dev/.cache/ccache"
  -e "UV_CACHE_DIR=/opt/uv/cache"
)
if [[ -n "${VLLM_VERSION_OVERRIDE:-}" ]]; then
  COMMON_FLAGS+=(-e "VLLM_VERSION_OVERRIDE=${VLLM_VERSION_OVERRIDE}")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  COMMON_FLAGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${VLLM_GIT_MIRROR:-}" ]]; then
  COMMON_FLAGS+=(-e "VLLM_GIT_MIRROR=${VLLM_GIT_MIRROR}")
fi
if [[ "${MOUNT_SSH}" == "1" && -d "${HOME}/.ssh" ]]; then
  COMMON_FLAGS+=(-v "${HOME}/.ssh:/home/dev/.ssh:ro")
fi

container_exists() {
  ${DOCKER} inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [[ "$(${DOCKER} inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" == "true" ]]
}

ensure_running() {
  if ! container_running; then
    echo "容器 ${CONTAINER_NAME} 未在运行，先启动它 ..."
    cmd_up --no-attach
  fi
}

# 在运行中的容器里执行命令（脚本化场景，不强制分配 TTY）
exec_in() {
  local tty=()
  if [[ -t 0 ]]; then
    tty=(-it)
  fi
  ${DOCKER} exec "${tty[@]}" -w "${WORKDIR_MOUNT}" "${CONTAINER_NAME}" bash -lc "$1"
}

# 交互式进入/执行
cmd_exec() {
  local tty=()
  if [[ -t 0 ]]; then
    tty=(-it)
  fi
  if [[ $# -eq 0 ]]; then
    ${DOCKER} exec "${tty[@]}" -w "${WORKDIR_MOUNT}" "${CONTAINER_NAME}" bash -l
  else
    ${DOCKER} exec "${tty[@]}" -w "${WORKDIR_MOUNT}" "${CONTAINER_NAME}" bash -lc "$*"
  fi
}

cmd_run() {
  local tty=()
  if [[ -t 0 ]]; then
    tty=(-it)
  fi
  if [[ $# -eq 0 ]]; then
    ${DOCKER} run "${tty[@]}" --rm "${COMMON_FLAGS[@]}" \
      -w "${WORKDIR_MOUNT}" "${IMAGE_TAG}" bash -l
  else
    ${DOCKER} run "${tty[@]}" --rm "${COMMON_FLAGS[@]}" \
      -w "${WORKDIR_MOUNT}" "${IMAGE_TAG}" bash -lc "$*"
  fi
}

cmd_up() {
  local attach=1
  if [[ "${1:-}" == "--no-attach" ]]; then
    attach=0
  fi
  if ! container_exists; then
    echo "== 创建并启动容器 ${CONTAINER_NAME} =="
    ${DOCKER} run -d --name "${CONTAINER_NAME}" "${COMMON_FLAGS[@]}" \
      "${IMAGE_TAG}" bash -lc 'sleep infinity'
  else
    echo "== 容器已存在，启动它 =="
    ${DOCKER} start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "== GPU 验证 =="
  ${DOCKER} exec "${CONTAINER_NAME}" nvidia-smi | head -12
  if [[ "${attach}" == "1" ]]; then
    cmd_exec
  fi
}

cmd_deps() {
  ensure_running
  exec_in "bash scripts/build/prepare_deps.sh --gpu $*"
}

cmd_wheel() {
  ensure_running
  exec_in "source scripts/build/cu130_env.sh && bash scripts/build/cu130_build_wheel.sh $*"
}

cmd_verify() {
  ensure_running
  exec_in "source scripts/build/cu130_env.sh && bash scripts/build/cu130_verify.sh $*"
}

cmd_serve() {
  ensure_running
  if ! ${DOCKER} exec "${CONTAINER_NAME}" bash -lc 'command -v vllm >/dev/null' 2>/dev/null; then
    echo "容器内还没有可执行的 vllm，先执行:"
    echo "  bash scripts/build/docker/run_dev.sh wheel --editable"
    exit 1
  fi
  exec_in "vllm serve $*"
}

cmd_test() {
  ensure_running
  exec_in "python -m pytest $*"
}

cmd_install_test_deps() {
  ensure_running
  exec_in "/usr/local/bin/vllm-install-test-deps"
}

usage() {
  sed -n '2,18p' "${BASH_SOURCE[0]}"
}

case "${1:-help}" in
  build) shift; exec "${DOCKER_DIR}/build_dev_image.sh" "$@" ;;
  up) cmd_up ;;
  exec) shift; cmd_exec "$@" ;;
  run) shift; cmd_run "$@" ;;
  gpu) ensure_running; ${DOCKER} exec "${CONTAINER_NAME}" nvidia-smi ;;
  deps) shift; cmd_deps "$@" ;;
  wheel) shift; cmd_wheel "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  serve) shift; cmd_serve "$@" ;;
  test) shift; cmd_test "$@" ;;
  install-test-deps) cmd_install_test_deps ;;
  stop) ${DOCKER} stop "${CONTAINER_NAME}" ;;
  rm) ${DOCKER} rm -f "${CONTAINER_NAME}" ;;
  logs) shift; ${DOCKER} logs "$@" "${CONTAINER_NAME}" ;;
  status) ${DOCKER} ps -a --filter "name=${CONTAINER_NAME}" ;;
  help|-h|--help) usage; exit 0 ;;
  *) echo "未知命令: ${1:-}" >&2; usage; exit 2 ;;
esac
