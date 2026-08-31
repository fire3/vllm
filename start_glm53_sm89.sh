#!/usr/bin/env bash
# Start GLM-5.3-Flash on SM89 (8x L40S) with the Triton sparse MLA backend.
#
# Requires the vllm-glm5.3 conda env (Python 3.12, torch 2.13+cu130) and the
# glm53-flash branch with the SM89 port (TRITON_MLA_SPARSE_SM89 backend +
# Triton indexer MQA-logits fallback).
#
# Usage:
#   ./start_glm53_sm89.sh            # serve /data1/GLM-5.3-Flash on :8091
#   MODEL=/path/to/model PORT=9999 MAX_MODEL_LEN=131072 GPU_MEM_UTILIZATION=0.95 \
#     ./start_glm53_sm89.sh
set -e

cd "$(dirname "$0")"

export LD_LIBRARY_PATH=/home/user/anaconda3/envs/vllm-glm5.3/lib:${LD_LIBRARY_PATH:-}
# NCCL P2P init is flaky on this box (spins in cudaStreamSynchronize during
# communicator setup); disable P2P so ranks come up reliably.
export NCCL_P2P_DISABLE=1
# The CUDA-graph memory estimate (default since v0.21) reserves an extra
# ~0.5 GiB/GPU that 256K-context GLM-5.3-Flash cannot spare; disable it so the
# KV cache gets the headroom.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
# Size the sparse-indexer K-gather workspace to ~max_model_len tokens instead
# of the 40x heuristic: saves ~1.3 GiB/GPU of fixed memory at 256K context.
export VLLM_SPARSE_INDEXER_WORKSPACE_TOKENS="${VLLM_SPARSE_INDEXER_WORKSPACE_TOKENS:-262144}"
# Bound the per-sub-chunk indexer logits buffer (M*N*4 bytes) so long-context
# prefills keep their transient peak inside the 44.39 GiB physical limit.
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
# Reduce allocator fragmentation under the 256K prefill's transient peak.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL="${MODEL:-/data1/GLM-5.3-Flash}"
PORT="${PORT:-8091}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTILIZATION="${GPU_MEM_UTILIZATION:-0.98}"

exec /home/user/anaconda3/envs/vllm-glm5.3/bin/vllm serve "$MODEL" \
  --served-model-name glm-5.3-flash \
  --tensor-parallel-size 8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "$GPU_MEM_UTILIZATION" \
  --max-num-seqs 32 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_MLA_SPARSE_SM89 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT"
