#!/usr/bin/env bash
# ncu/nsys profile runner for the DSv4 tiled fused decode kernel.
#
# Requirements: GPU free (the shared gserver must not be running the e2e
# vLLM serve), conda env vllm-dsv4-sm89.
#
# Usage:
#   bash benchmarks/kernels/profile_sparse_mla_decode_ncu.sh [ncu|nsys|both]
set -euo pipefail

cd "$(dirname "$0")/../.."
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate vllm-dsv4-sm89
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
NCU=/usr/local/cuda/bin/ncu
NSYS=/usr/local/cuda/bin/nsys

MODE="${1:-ncu}"
BENCH="python benchmarks/kernels/profile_sparse_mla_decode.py"

# fused-direct: one Triton kernel + 5 sink ops per iteration. Warmup 3 iters,
# then profile the 4th kernel launch. Keep total GPU time short for replay.
ARGS="--mode fused-direct --batch 8 --heads 8 --swa-topk 128 --extra-topk 128 \
--pages 1024 --iters 4 --warmup 3 --check"

if [ "$MODE" = "ncu" ] || [ "$MODE" = "both" ]; then
  echo "==> ncu: fused kernel, SpeedOfLight + compute + memory + scheduler + occupancy"
  $NCU --target-processes all \
    --kernel-name "_tiled_sparse_prefill_kernel" \
    --launch-skip 3 --launch-count 1 \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    --section WarpStateStats \
    --section Occupancy \
    --section LaunchStats \
    --print-summary per-kernel \
    $BENCH $ARGS
fi

if [ "$MODE" = "nsys" ] || [ "$MODE" = "both" ]; then
  echo "==> nsys: public decode path kernel timeline"
  rm -f /tmp/nsys_decode_public.nsys-rep
  $NSYS profile --force-overwrite true -o /tmp/nsys_decode_public \
    --cuda-memory-usage false \
    $BENCH --mode fused --iters 30 --warmup 5 --check
  $NSYS stats --force-overwrite true --report cuda_gpu_kern_sum \
    --format table /tmp/nsys_decode_public.nsys-rep
fi
