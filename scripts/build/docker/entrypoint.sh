#!/usr/bin/env bash
# vLLM 开发容器入口：准备缓存目录并执行传入命令。
set -euo pipefail

mkdir -p \
  /home/dev/.cache/huggingface \
  /home/dev/.cache/vllm \
  /home/dev/.cache/triton \
  /home/dev/.cache/ccache \
  /home/dev/.ssh

git config --global --add safe.directory /workspace/vllm 2>/dev/null || true

exec "$@"
