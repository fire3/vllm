#!/usr/bin/env bash
# 在 Ubuntu 22.04/24.04 服务器上准备 Docker + GPU 开发环境：
#   - Docker Engine (docker-ce) + compose 插件 + buildx
#   - NVIDIA Container Toolkit（让容器通过 --gpus all 访问 GPU）
#   - 可选 registry mirror（国内加速）
#
#   sudo bash scripts/build/docker/setup_host.sh [选项]
#
# 幂等：已安装的组件会跳过。
set -euo pipefail

VERIFY_ONLY=0
SKIP_DOCKER=0
SKIP_NVIDIA=0
GPU_IMAGE="nvidia/cuda:13.0.3-base-ubuntu24.04"
REGISTRY_MIRRORS=()

usage() {
  cat <<'EOF'
用法: sudo bash scripts/build/docker/setup_host.sh [选项]

选项:
  --verify-only         只验证 Docker + GPU 容器，不安装任何东西
  --skip-docker         跳过 Docker 安装（仅安装/配置 NVIDIA toolkit）
  --skip-nvidia         跳过 NVIDIA Container Toolkit 安装
  --registry-mirror URL 追加一个 Docker registry 镜像源（可重复使用）
  --gpu-image IMAGE     验证用的镜像（默认 nvidia/cuda:13.0.3-base-ubuntu24.04）
  -h, --help            显示帮助
EOF
}

if [[ ${EUID} -ne 0 ]]; then
  echo "需要 root 权限，自动通过 sudo 重新执行 ..."
  exec sudo -H bash "$0" "$@"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verify-only) VERIFY_ONLY=1 ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    --skip-nvidia) SKIP_NVIDIA=1 ;;
    --registry-mirror) REGISTRY_MIRRORS+=("$2"); shift ;;
    --gpu-image) GPU_IMAGE="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

verify_docker_gpu() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker 未安装，无法验证。" >&2
    return 1
  fi
  echo "== 验证 host nvidia-smi =="
  nvidia-smi | head -12
  echo "== 验证 docker --gpus all =="
  docker run --rm --gpus all "${GPU_IMAGE}" nvidia-smi | head -12
  echo "== GPU 容器访问正常 =="
}

install_docker() {
  echo "== 安装 Docker Engine =="
  apt-get update -y
  apt-get install -y --no-install-recommends ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y --no-install-recommends \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
}

install_nvidia_toolkit() {
  echo "== 安装 NVIDIA Container Toolkit =="
  apt-get update -y
  apt-get install -y --no-install-recommends curl gnupg
  install -m 0755 -d /usr/share/keyrings
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -y
  apt-get install -y --no-install-recommends nvidia-container-toolkit
}

configure_registry_mirrors() {
  local file="/etc/docker/daemon.json"
  if [[ ${#REGISTRY_MIRRORS[@]} -eq 0 ]]; then
    return
  fi
  echo "== 配置 Docker registry mirrors =="
  if [[ -f "${file}" ]]; then
    if command -v jq >/dev/null 2>&1; then
      local mirror_json tmp
      mirror_json="$(printf '%s\n' "${REGISTRY_MIRRORS[@]}" | jq -R -s -c 'split("\n") | map(select(length > 0))')"
      tmp="$(mktemp)"
      jq --argjson ms "${mirror_json}" '. + { "registry-mirrors": $ms }' "${file}" > "${tmp}"
      mv "${tmp}" "${file}"
    else
      echo "  [!] ${file} 已存在且没有 jq，跳过自动合并；请手动加上 registry-mirrors。" >&2
    fi
  else
    {
      echo "{"
      echo "  \"registry-mirrors\": ["
      local i=0
      for m in "${REGISTRY_MIRRORS[@]}"; do
        if (( i > 0 )); then echo ","; fi
        printf '    "%s"' "${m}"
        i=$((i + 1))
      done
      echo
      echo "  ]"
      echo "}"
    } > "${file}"
  fi
  if command -v docker >/dev/null 2>&1; then
    systemctl restart docker
  fi
}

# ---- 系统与驱动检查 ----
if [[ -r /etc/os-release ]]; then
  # shellcheck source=/dev/null
  . /etc/os-release
else
  echo "无法识别系统 (/etc/os-release 不存在)。" >&2
  exit 1
fi
case "${ID:-}" in
  ubuntu|debian) ;;
  *) echo "警告: 未在 Ubuntu/Debian 上验证过（当前 ${ID:-unknown}），Docker 仓库地址可能需要调整。" >&2 ;;
esac
CODENAME="${VERSION_CODENAME:-}"
ARCH="$(dpkg --print-architecture 2>/dev/null || uname -m)"

echo "== 系统信息 =="
echo "  OS: ${PRETTY_NAME:-unknown}   Arch: ${ARCH}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo >&2
  echo "错误: 宿主机上没有 nvidia-smi，NVIDIA 驱动未安装/未生效。" >&2
  echo "  请先安装驱动，例如: sudo ubuntu-drivers install（或手动安装对应版本）。" >&2
  echo "  CUDA 13 容器要求驱动足够新（nvidia-smi 里 CUDA Version >= 13.0）。" >&2
  exit 1
fi
DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
DRIVER_CUDA="$(nvidia-smi | awk -F: '/CUDA Version/ {print $2}' | head -1 | tr -cd '0-9.')"
echo "  GPU 驱动: ${DRIVER_VERSION}   驱动支持的最高 CUDA: ${DRIVER_CUDA:-unknown}"
if [[ -n "${DRIVER_CUDA}" && "${DRIVER_CUDA%%.*}" -lt 13 ]]; then
  echo "  [!] 驱动支持的最高 CUDA 为 ${DRIVER_CUDA}，跑 CUDA 13 容器可能报" >&2
  echo "      'CUDA driver version is insufficient'，建议升级驱动。" >&2
fi

if [[ "${VERIFY_ONLY}" == "1" ]]; then
  verify_docker_gpu
  exit 0
fi

# ---- 安装 Docker ----
if [[ "${SKIP_DOCKER}" == "1" ]]; then
  echo "跳过 Docker 安装（--skip-docker）"
elif command -v docker >/dev/null 2>&1; then
  echo "Docker 已安装: $(docker --version)"
  if ! docker compose version >/dev/null 2>&1; then
    echo "缺少 compose 插件，补装 ..."
    apt-get update -y
    apt-get install -y --no-install-recommends docker-compose-plugin
  fi
else
  install_docker
fi

configure_registry_mirrors

# ---- 安装 NVIDIA Container Toolkit ----
if [[ "${SKIP_NVIDIA}" == "1" ]]; then
  echo "跳过 NVIDIA Container Toolkit 安装（--skip-nvidia）"
elif command -v nvidia-ctk >/dev/null 2>&1; then
  echo "nvidia-ctk 已安装: $(nvidia-ctk --version | head -1)"
else
  install_nvidia_toolkit
fi
if [[ "${SKIP_NVIDIA}" != "1" ]]; then
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi

# ---- docker 组 ----
TARGET_USER="${SUDO_USER:-root}"
if [[ "${TARGET_USER}" != "root" ]]; then
  usermod -aG docker "${TARGET_USER}" || true
  echo "已将 ${TARGET_USER} 加入 docker 组（重新登录后生效）。"
fi

verify_docker_gpu

cat <<EOF

== 宿主机准备完成 ==
下一步:
  1. 把仓库复制到服务器（git clone ...）
  2. bash scripts/build/docker/build_dev_image.sh
  3. bash scripts/build/docker/run_dev.sh up
详见 scripts/build/docker/README.md
EOF
