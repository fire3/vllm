#!/usr/bin/env bash
# Build the vLLM wheel against the cu130 PyTorch already installed in the
# active conda env, targeting only SM 8.9 via PTX.
#
#   bash scripts/build/sm89_build_wheel.sh [--editable]
#
# Pass --editable to additionally install the freshly compiled vLLM into the
# active conda env in editable mode (the wheel is still built as well).
#
# Build output is printed live and saved to $BUILD_LOG. Set VERBOSE=0 to
# suppress verbose CMake output.
set -euo pipefail

EDITABLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --editable)
      EDITABLE=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--editable]" >&2
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--editable]" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${CONDA_PREFIX:?run inside your conda env, e.g. \"conda activate vllm-dev\"}/bin/python"
DIST_DIR="${DIST_DIR:-${REPO_ROOT}/dist}"
BUILD_LOG="${BUILD_LOG:-${DIST_DIR}/build_cu130.log}"

# Environment defaults (CUDA_HOME, PATH, TORCH_CUDA_ARCH_LIST,
# VLLM_MAIN_CUDA_VERSION, and the default VLLM_VERSION_OVERRIDE with the
# "+sm89" wheel marker).
source scripts/build/sm89_env.sh

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
# VERBOSE=1 makes setup.py pass -DCMAKE_VERBOSE_MAKEFILE=ON to CMake, so the
# build emits every compile command into the log.
export VERBOSE="${VERBOSE:-1}"

# vLLM 用 FetchContent 把约 10 个 git 仓库拉到 $ROOT/.deps。ExternalProject 的
# git clone 步骤靠 stamp 判定是否重下：只有
#   <name>-populate-gitclone-lastrun.txt 比 <name>-populate-gitinfo.txt 新
# 才跳过 clone，否则每次 configure 都会把已有 <name>-src 目录 rm -rf 后重新
# git clone（完全不看 src 目录是否完好）。一旦 gitinfo.txt 比 lastrun 新
# （例如上次 clone 后 submodule 更新失败、脚本没走到写 lastrun），此后每次
# 构建都会重新下载整个仓库。FETCHCONTENT_UPDATES_DISCONNECTED 只跳过 update
# step，管不到 download step，所以这里默认 FETCHCONTENT_FULLY_DISCONNECTED=ON：
# 完全绕过 population（不跑 clone/update），直接使用 .deps/<name>-src。
# 前提是 .deps 已预取齐全（缺失会报错）；需要联网首次下载时设
# FETCHCONTENT_FULLY_DISCONNECTED=0 回退到"跳过 update"模式。
FETCHCONTENT_FULLY_DISCONNECTED="${FETCHCONTENT_FULLY_DISCONNECTED:-1}"
if [[ "${FETCHCONTENT_FULLY_DISCONNECTED}" == "1" ]]; then
  export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
else
  FETCHCONTENT_UPDATES_DISCONNECTED="${FETCHCONTENT_UPDATES_DISCONNECTED:-1}"
  if [[ "${FETCHCONTENT_UPDATES_DISCONNECTED}" == "1" ]]; then
    export CMAKE_ARGS="${CMAKE_ARGS:+${CMAKE_ARGS} }-DFETCHCONTENT_UPDATES_DISCONNECTED=ON"
  fi
fi

# The build backend runs in this env (--no-build-isolation below), so make
# sure its dependencies are present. All of these come from install_deps.sh.
if ! "${PYTHON}" -c "import setuptools_rust, setuptools_scm, wheel, ninja, cmake" 2>/dev/null; then
  echo "Build dependencies are missing. Run first:"
  echo "  bash scripts/build/sm89_install_deps.sh"
  exit 1
fi

# Refuse to build against a non-cu130 torch: the wheel would link the wrong
# PyTorch ABI and fail at runtime.
TORCH_BUILD="$("${PYTHON}" -c "import torch; print(torch.__version__)" 2>/dev/null || true)"
case "${TORCH_BUILD}" in
  *+cu130) ;;
  *)
    echo "The active env does not have the cu130 torch build (got: ${TORCH_BUILD:-torch not importable})."
    echo "Run first: bash scripts/build/sm89_install_deps.sh"
    exit 1
    ;;
esac

mkdir -p "${DIST_DIR}"

# --no-build-isolation lets the build reuse the torch already installed in the
# conda env; an isolated build would pull the CPU torch from PyPI instead.
# Tee keeps stdout live while also persisting the full log for inspection.
# CMake additionally writes its own configure diagnostics into
# build/temp.linux-x86_64-cpython-312/CMakeFiles/CMakeConfigureLog.yaml.
"${PYTHON}" -m pip wheel . \
  -v \
  --no-build-isolation \
  --no-deps \
  -w "${DIST_DIR}" 2>&1 | tee "${BUILD_LOG}"

# Editable install: compile the extensions in place into the source tree and
# link the conda env to this checkout, so Python changes take effect without
# reinstalling. The CMake build tree is reused, so this is mostly an
# incremental rebuild plus a cmake --install into the source tree. Like the
# wheel build, --no-build-isolation/--no-deps keep the cu130 torch and other
# deps already installed in the conda env untouched.
if [[ "${EDITABLE}" == "1" ]]; then
  echo "Installing vLLM in editable mode..."
  "${PYTHON}" -m pip install -e . \
    --no-build-isolation \
    --no-deps 2>&1 | tee -a "${BUILD_LOG}"
  echo "Editable install complete."
fi

echo "Build log: ${BUILD_LOG}"
echo "Built wheel:"
ls -1t "${DIST_DIR}"/vllm-*.whl | head -1
