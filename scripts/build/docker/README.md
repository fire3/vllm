# Docker 开发环境（Ubuntu 22.04 宿主机 + Ubuntu 24.04 容器）

这套脚本把本仓库（DSV4 SM89 / cu130 fork）的构建、调试环境整个搬进 Docker：

- 宿主机只装 Docker + NVIDIA Container Toolkit，不装 conda / CUDA toolkit；
- 容器内预装 CUDA 13.0 工具链、Python 3.12 + uv venv、cu130 版 PyTorch、vLLM
  构建/运行依赖（FlashInfer、tilelang 等）、Rust、protoc、pre-commit；
- 仓库源码通过 bind mount 挂载进容器，改动即时生效，无需反复重建镜像；
- 默认 `TORCH_CUDA_ARCH_LIST=8.9+PTX`，与 `scripts/build` 的 cu130 流程一致。

> 说明：你说的 "docker 24.04" 我按 **Ubuntu 24.04 容器镜像**理解（Docker Engine
> 本身没有 24.04 这个版本号）。宿主机仍是 Ubuntu 22.04。想用 Ubuntu 22.04 容器，
> 把 `dev.env` 里的 `UBUNTU_VERSION` 改成 `22.04` 重新构建即可。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `setup_host.sh` | 服务器一次性准备：装 Docker Engine + compose + NVIDIA Container Toolkit，验证 GPU |
| `build_dev_image.sh` | 构建开发镜像（`docker build` 封装，读 `dev.env`） |
| `Dockerfile.dev` | 开发镜像定义（Ubuntu 24.04 + CUDA 13.0 devel） |
| `run_dev.sh` | 容器生命周期 + 开发命令入口（build/up/exec/deps/wheel/serve/test ...） |
| `dev.env` | 全部可调参数（版本、镜像源、容器名、端口等），可用同名环境变量覆盖 |
| `entrypoint.sh` | 容器入口，初始化缓存目录 |
| `install_test_deps.sh` | 容器内补装 pytest 等测试依赖（保留当前 torch 版本） |

## 前置条件

- 服务器：Ubuntu 22.04/24.04，x86_64；
- NVIDIA GPU，宿主机已装驱动（`nvidia-smi` 能跑）。CUDA 13 容器要求驱动较新
  （`nvidia-smi` 显示的 CUDA Version ≥ 13.0，约对应 R580+ 驱动）；
- 服务器能访问 Docker Hub / PyPI / PyTorch 索引（国内可开 `USE_CN_MIRROR=1`，见下文）。

## 快速开始

在服务器上执行（仓库先 clone 到任意目录）：

```bash
# 1. 宿主机准备（一次即可）：Docker + GPU 容器支持
sudo bash scripts/build/docker/setup_host.sh

# 2. 构建开发镜像（首次较久，torch + FlashInfer 等约 10~20 分钟，看网络）
bash scripts/build/docker/build_dev_image.sh

# 3. 创建并启动容器，自动进入容器 shell
bash scripts/build/docker/run_dev.sh up
```

进入容器后（工作目录 `/workspace/vllm`），按现有流程构建 SM89 wheel：

```bash
# 4. 预取 CMake FetchContent 依赖（.deps 是命名卷，首次构建必须执行）
bash scripts/build/prepare_deps.sh --gpu

# 5. 构建 wheel，并同时以 editable 模式安装到 /opt/venv（改 Python 代码即时生效）
bash scripts/build/cu130_build_wheel.sh --editable

# 6. 校验 wheel 里的内核确实只带 SM89
bash scripts/build/cu130_verify.sh
```

之后调试：

```bash
# 起 OpenAI 兼容服务
bash scripts/build/docker/run_dev.sh serve --model deepseek-ai/DeepSeek-V4 --max-model-len 8192

# 跑测试（先装测试依赖）
bash scripts/build/docker/run_dev.sh install-test-deps
bash scripts/build/docker/run_dev.sh test tests/test_xxx.py -v
```

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `run_dev.sh up` | 创建/启动容器并进入（自动跑 `nvidia-smi` 验证 GPU） |
| `run_dev.sh exec [CMD]` | 在运行中的容器里执行命令，默认 bash |
| `run_dev.sh run [CMD]` | 一次性跑一个容器（`--rm`，不保留） |
| `run_dev.sh gpu` | 容器内 `nvidia-smi` |
| `run_dev.sh deps` | 同步 `.deps`（切分支/版本后重跑） |
| `run_dev.sh wheel --editable` | 构建 wheel + editable 安装 |
| `run_dev.sh verify` | 校验 wheel 的 SM89 内核 |
| `run_dev.sh serve [args]` | 容器内 `vllm serve ...` |
| `run_dev.sh test [args]` | 容器内 `python -m pytest ...` |
| `run_dev.sh install-test-deps` | 补装 pytest 等测试依赖 |
| `run_dev.sh logs / stop / rm / status` | 容器运维 |

## 配置项（dev.env）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CUDA_VERSION` | `13.0.3` | nvidia/cuda 镜像版本 |
| `UBUNTU_VERSION` | `24.04` | 容器基础系统版本 |
| `PYTHON_VERSION` | `3.12` | uv 管理的 Python 版本 |
| `TORCH_VERSION` | `2.13.0` | cu130 PyTorch，与 `scripts/build` 及仓库 requirements 一致 |
| `TORCH_CUDA_ARCH_LIST` | `8.9+PTX` | 编译目标架构（Ada，PTX 可 JIT 到更新的卡） |
| `VLLM_MAIN_CUDA_VERSION` | `13.0` | wheel 变体名 |
| `IMAGE_TAG` | `vllm-dsv4-dev:cu130-ubuntu24.04-sm89` | 镜像名 |
| `USE_CN_MIRROR` | `0` | 1 = 阿里云 PyTorch + 清华 PyPI |
| `CONTAINER_NAME` | `vllm-dsv4-dev` | 容器名 |
| `VLLM_PORT` | `8000` | 宿主机映射到容器 8000 的端口 |
| `MOUNT_SSH` | `1` | 把 `~/.ssh` 只读挂进容器 |
| `SHM_SIZE` | `8g` | `/dev/shm` 大小（大数据加载用） |

覆盖方式：改 `dev.env`，或在执行前 `export` 同名变量，例如：

```bash
export TORCH_VERSION=2.13.0
bash scripts/build/docker/build_dev_image.sh
```

## 国内网络加速

```bash
# 构建镜像用国内源
export USE_CN_MIRROR=1
bash scripts/build/docker/build_dev_image.sh

# Docker Hub 拉取加速（宿主机，可选）
sudo bash scripts/build/docker/setup_host.sh --registry-mirror https://docker.1ms.run

# git clone 加速（prepare_deps 拉取 .deps 时生效）
export VLLM_GIT_MIRROR=https://ghfast.top/https://github.com/
bash scripts/build/docker/run_dev.sh deps
```

其他可选加速项（`dev.env`）：

- `UV_PYTHON_INSTALL_MIRROR`：uv 托管 Python 下载镜像（如
  `https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download`）；
- `RUSTUP_DIST_SERVER`：rustup 镜像（如 `https://rsproxy.cn`）；
- `PROTOC_GH_MIRROR`：protoc GitHub 下载镜像（如 `https://ghfast.top/https://github.com`）。

## torch 版本说明

`scripts/build/cu130_install_deps.sh` 会自动从仓库 `requirements/build/cuda.txt`
读取 torch pin，镜像默认 `TORCH_VERSION=2.13.0`，与仓库 requirements 保持一致。
如想改用其他版本（例如临时测试别的 torch），构建镜像时覆盖即可：

```bash
export TORCH_VERSION=2.11.0
bash scripts/build/docker/build_dev_image.sh
```

镜像在装构建/运行/测试依赖时会剔除 requirements 里 torch 系的硬 pin，因此无论选
哪个版本，`install-test-deps` / `INSTALL_TEST_DEPS=1` 都不会把 torch 悄悄升回去。

## 常见问题

- **容器里 `nvidia-smi` 报错 / 看不到 GPU**：先跑 `setup_host.sh` 完成
  NVIDIA Container Toolkit 安装；确认宿主机驱动够新（CUDA ≥ 13）。
- **`run_dev.sh wheel` 报 `.deps` 缺失**：先执行 `run_dev.sh deps`。构建脚本默认
  `FETCHCONTENT_FULLY_DISCONNECTED=1`，不会自动联网下载依赖。
- **切换了分支/版本后构建报错**：`.deps` 是命名卷，按新分支重新
  `run_dev.sh deps`（内部是 `prepare_deps.sh --gpu`，会按当前 CMake 配置重新拉取）。
- **想清掉所有容器数据**：`run_dev.sh stop`、`run_dev.sh rm`，再
  `docker volume rm vllm-dsv4-deps vllm-dsv4-uv vllm-dsv4-home`（会删除缓存和依赖，谨慎）。
- **改 C++/CUDA 内核后只改 Python 不生效**：需要重跑
  `run_dev.sh wheel --editable`（复用 CMake 增量构建）。
- **驱动较老**：给容器加 `export VLLM_ENABLE_CUDA_COMPATIBILITY=1` 再
  `run_dev.sh rm && run_dev.sh up`（仅数据中心级 GPU 生效）。
