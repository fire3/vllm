#!/usr/bin/env bash
# ncu on the fused sparse-MLA decode kernel, run as root (the shared gserver
# sets RmProfilingAdminOnly=1, so perf counters need admin).
#
# Run on gserver:
#   sudo bash benchmarks/kernels/profile_sparse_mla_decode_ncu_sudo.sh
#
# Writes the report to benchmarks/kernels/ncu_fused_kernel.txt.
set -euo pipefail

cd /home/user/vllm
export HOME=/home/user  # reuse the user's Triton kernel cache
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate vllm-dsv4-sm89
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
NCU=/usr/local/cuda/bin/ncu
OUT=/home/user/vllm/benchmarks/kernels/ncu_fused_kernel.txt

BENCH="python benchmarks/kernels/profile_sparse_mla_decode.py \
  --mode fused-direct --batch 8 --heads 8 --swa-topk 128 --extra-topk 128 \
  --pages 1024 --iters 4 --warmup 3 --check"

echo "==> ncu fused kernel (BLOCK_H=8/BLOCK_K=32/warps=8) $(date)" | tee "$OUT"
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
  $BENCH 2>&1 | tee -a "$OUT"

chmod 644 "$OUT" 2>/dev/null || true
echo "==> done: $OUT"
