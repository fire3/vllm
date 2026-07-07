# SM89 DSv4-Flash flashinfer MLA 适配实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 SM89（4×RTX 4090 48GB）上用 flashinfer sparse MLA CUDA kernel（移植自
SM120 实现）运行 DeepSeek-V4-Flash-DSpark 推理，并以 SM120 本地基准数据为参照
完成验证与瓶颈优化。

**Architecture:** 三个代码域——① flashinfer fork（`/home/yyf/flashinfer`，
v0.6.14 基础上按方案 A 在 `arch/` 原语层加 SM89 分支，kernel 主体共用）；
② vLLM 分支 `sm89-dsv4-flashinfer`（只改 capability gating、backend 选择、
非 MLA 算子 fallback）；③ benchmark 脚本（`benchmarks/sparse_mla_sm89_port/`，
随分支提交）。SM89 数值逻辑先通过编译开关在本地 SM120 卡上验证，再交叉编译
sm_89 部署远程。

**Tech Stack:** CUDA (PTX inline asm: mma.sync/cp.async/mbarrier)、flashinfer
0.6.14 JIT、Triton（对照 kernel）、vLLM main、uv + Python 3.12。

**Spec:** `docs/superpowers/specs/2026-07-07-sm89-dsv4-flashinfer-design.md`

**关键背景事实**（执行者无需重新发现）：
- SM120 sparse MLA kernel 源码：flashinfer wheel 内
  `flashinfer/data/include/flashinfer/attention/sparse_mla_sm120/`（fork 仓库中
  为 `include/flashinfer/attention/sparse_mla_sm120/`），编译单元在
  `csrc/sparse_mla_sm120*.cu`，JIT 入口 `flashinfer/jit/mla.py::gen_sparse_mla_sm120_module`
  （`supported_major_versions=[12]`），Python 分发层
  `flashinfer/mla/_sparse_mla_sm120.py`。
- 用户侧 API：`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4(query,
  swa_kv_cache, workspace_buffer, sparse_indices, compressed_kv_cache=None,
  sparse_topk_lens=None, seq_lens=None, out=None, bmm1_scale=1.0, bmm2_scale=1.0,
  sinks=None, kv_layout="HND", cum_seq_lens_q=None, max_q_len=None,
  enable_pdl=None, swa_topk_lens=None, extra_sparse_indices=None,
  extra_sparse_topk_lens=None)`，按设备架构自动分发（SM100→trtllm-gen cubin，
  SM120→本模块 JIT kernel）。
- vLLM 侧 DSv4 SM120 调用点：`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
  （`DeepseekV4FlashInferSM120Attention._forward_decode/_forward_prefill`），
  capability gate 在同文件 `supports_compute_capability`（`major in [10, 12]`）
  和 `vllm/models/deepseek_v4/nvidia/model.py:779,788`
  （`device_capability.major == 12`）。
- Triton 对照 kernel（旧分支）：
  `sm89-deepseek-v4-flash:vllm/v1/attention/backends/mla/sparse_mla_kernels.py:1597`
  `accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(q[T,H,512],
  k_cache(uint8 fp8_ds_mla), slot_ids[T,C], lens[T], block_size, scale,
  max_score[T,H]f32, denom[T,H]f32, acc[T,H,512]f32, candidate_offset,
  head_block_size)`——chunk 累加式 online-softmax API，输出需
  `acc/denom` 归一化。
- SM89 不可用指令（kernel 源码已核实）：
  `mma.sync.aligned.kind::mxf8f6f4.block_scale...ue8m0`（arch/mma_sm120.cuh:62-68）、
  `mbarrier.arrive.expect_tx`（arch/barrier.cuh:63-71）、
  `setmaxnreg`（prefill_kernel.cuh:127,162,735,831）。
  可用：`mma.sync.m16n8k32.f32.e4m3.e4m3.f32`、`mma.m16n8k16 bf16`、
  `cp.async`、`ldmatrix`、`mbarrier`（不含 expect_tx）、
  `cp.async.mbarrier.arrive`（SM80+）。
- Ada 动态 smem 上限：101376 B（99KB）/block（opt-in）。decode dsv4 kernel
  动态 smem 布局注释在 `csrc/sparse_mla_sm120_decode_dsv4.cu:37-56`
  （含 58KB 双缓冲 KV）。
- DSv4-Flash serve 参考配置：`deepseek_v4_vllm_source/run_deepseek_v4_flash.sh`
  与 `tests/evals/gsm8k/configs/moe-refactor/DeepSeek-V4-Flash-deepgemm-mega-moe.yaml`
  （`--kv-cache-dtype fp8 --block-size 256 --tokenizer-mode deepseek_v4`，
  MoE backend 可用 `--moe-backend` 切换）。DSpark：
  `--speculative_config.method=dspark`（`vllm/config/speculative.py:60`）。
- MXFP4 MoE backend 枚举：`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
  `Mxfp4MoeBackend.{DEEPGEMM_MXFP4, MARLIN, BATCHED_MARLIN}`。
- fp8_ds_mla 打包布局参考（token 576B）：448B fp8 NoPE + 4B×8 scale +
  64×bf16 RoPE；vLLM 构造/校验助手见
  `tests/v1/attention/test_sparse_mla_backends.py`。

---

## Phase 0：SM120 测试环境

### Task 1: 建立 SM120 基准 env（uv + flashinfer 0.6.14 + vLLM main 源码）

**Files:**
- Create: `.venv-sm120/`（不提交；不要动已有 `.venv`，它是 sm89 旧分支 wheel 环境）

- [ ] **Step 1: 创建 venv 并安装 vLLM（precompiled 优先）**

```bash
cd /home/yyf/vllm
uv venv --python 3.12 .venv-sm120
VLLM_USE_PRECOMPILED=1 uv pip install -p .venv-sm120/bin/python -e . --torch-backend=auto \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

预期：安装成功。若 precompiled wheel 与本地 commit 不匹配报错，退化为完整源码
编译：`uv pip install -p .venv-sm120/bin/python -e . --torch-backend=auto`
（本机约 1-2 小时，可 `run_in_background`）。

- [ ] **Step 2: 安装 flashinfer 0.6.14 与基准依赖**

```bash
uv pip install -p .venv-sm120/bin/python flashinfer-python==0.6.14 flashinfer-cubin==0.6.13 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注：`flashinfer-cubin` 若有 0.6.14 则同版本；没有则 0.6.13（cubin 只影响
SM100 路径，本项目不用）。

- [ ] **Step 3: 冒烟验证 sparse_mla_sm120 JIT 编译并在本卡运行**

```bash
.venv-sm120/bin/python - <<'EOF'
import torch
from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
T, H, D, SWA = 4, 64, 512, 128
q = torch.randn(T, 1, H, D, device="cuda", dtype=torch.bfloat16)
swa_kv = torch.randn(64, 64, 1, D, device="cuda", dtype=torch.bfloat16)  # NHD
idx = torch.arange(SWA, device="cuda", dtype=torch.int32).expand(T, SWA).contiguous()
lens = torch.full((T,), SWA, device="cuda", dtype=torch.int32)
ws = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
out = trtllm_batch_decode_sparse_mla_dsv4(
    query=q, swa_kv_cache=swa_kv, workspace_buffer=ws, sparse_indices=idx,
    seq_lens=lens, bmm1_scale=D ** -0.5, kv_layout="NHD")
print("OK", out.shape, out.dtype)
EOF
```

预期：首次运行触发 JIT 编译（数分钟），输出 `OK torch.Size([4, 1, 64, 512]) torch.bfloat16`
（若形状约定不同，以实际 API 校验错误信息修正调用——这一步同时是 API 形状
契约的实测确认，后续 bench 代码以此为准）。

- [ ] **Step 4: 记录 env 信息到 bench 目录**

```bash
mkdir -p benchmarks/sparse_mla_sm89_port
.venv-sm120/bin/python -c "import torch, flashinfer, vllm; print(torch.__version__, flashinfer.__version__, vllm.__version__)" \
  | tee benchmarks/sparse_mla_sm89_port/ENV_SM120.txt
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv >> benchmarks/sparse_mla_sm89_port/ENV_SM120.txt
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/sparse_mla_sm89_port/ENV_SM120.txt
git commit -m "bench(sm89-port): record SM120 benchmark environment"
```

---

## Phase 1：SM120 基准测试

### Task 2: 提取 Triton 对照 kernel 为独立模块

**Files:**
- Create: `benchmarks/sparse_mla_sm89_port/triton_sparse_mla_ref.py`

- [ ] **Step 1: 从旧分支导出 kernel 文件**

```bash
git show sm89-deepseek-v4-flash:vllm/v1/attention/backends/mla/sparse_mla_kernels.py \
  > benchmarks/sparse_mla_sm89_port/triton_sparse_mla_ref.py
```

- [ ] **Step 2: 验证可导入**

```bash
.venv-sm120/bin/python -c "
import sys; sys.path.insert(0, 'benchmarks/sparse_mla_sm89_port')
import triton_sparse_mla_ref as m
print(callable(m.accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead))"
```

预期：`True`。若因 vllm 内部 import 失败（旧分支符号在 main 中移动），
删除该文件中未被这两个函数依赖的 import 与函数，只保留
`accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead`、
对应 prefill/chunk kernel 及其 `@triton.jit` 依赖、`next_power_of_2` 等
纯 triton/torch 助手，重试直至可导入。

- [ ] **Step 3: Commit**

```bash
git add benchmarks/sparse_mla_sm89_port/triton_sparse_mla_ref.py
git commit -m "bench(sm89-port): vendor Triton sparse MLA reference kernels from sm89 branch"
```

### Task 3: MLA decode 基准（flashinfer vs Triton）

**Files:**
- Create: `benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py`
- Create: `benchmarks/sparse_mla_sm89_port/common.py`

- [ ] **Step 1: 写公共输入构造（fp8_ds_mla cache + 索引）**

`common.py`——复用 vLLM 主树测试助手构造合法 fp8_ds_mla cache，避免手写布局：

```python
# SPDX-License-Identifier: Apache-2.0
"""Shared input builders for sparse MLA benchmarks (DSv4-Flash shapes)."""
import torch

# DeepSeek-V4-Flash dims (verify once against HF config, see bench README)
KV_LORA_RANK = 448
QK_ROPE_DIM = 64
HEAD_DIM = KV_LORA_RANK + QK_ROPE_DIM  # 512
INDEX_TOPK = 2048
SWA_WIDTH = 128


def make_fp8_ds_mla_cache(num_slots: int, device="cuda", seed=0):
    """Build a valid packed fp8_ds_mla cache via vLLM's cache op."""
    torch.manual_seed(seed)
    from vllm import _custom_ops as ops
    kv_c = torch.randn(num_slots, KV_LORA_RANK, device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(num_slots, QK_ROPE_DIM, device=device, dtype=torch.bfloat16)
    entry = 576  # 448 fp8 + 8*4 scale + 64*2 rope bytes
    cache = torch.zeros(num_slots, entry, device=device, dtype=torch.uint8)
    scale = torch.tensor(1.0, device=device)
    slot_mapping = torch.arange(num_slots, device=device, dtype=torch.long)
    ops.concat_and_cache_mla(kv_c, k_pe, cache.view(num_slots, 1, entry),
                             slot_mapping, "fp8_ds_mla", scale)
    return cache, kv_c, k_pe


def make_topk_indices(num_tokens, num_slots, topk, device="cuda", seed=1):
    torch.manual_seed(seed)
    idx = torch.stack([
        torch.randperm(num_slots, device=device)[:topk] for _ in range(num_tokens)
    ]).to(torch.int32)
    return idx


def bench_cuda(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True); end = torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms
```

注意：`ops.concat_and_cache_mla` 的确切签名以 main 的
`tests/v1/attention/test_sparse_mla_backends.py` 中用法为准，执行时先读该
测试文件校正（该文件已有 fp8_ds_mla cache 构造代码，直接照抄其调用形式）。

- [ ] **Step 2: 写 decode bench 主体**

`bench_sparse_mla_decode.py`：

```python
# SPDX-License-Identifier: Apache-2.0
"""Decode-path sparse MLA: flashinfer sparse_mla_sm120 vs Triton reference.

Sweeps num_tokens in {1,8,32,64,128,256} x q_len {1, 3(dspark)} at
num_heads in {16(TP=4? verify), 64} topk=INDEX_TOPK, fp8_ds_mla KV.
"""
import sys, json, argparse, torch
sys.path.insert(0, "benchmarks/sparse_mla_sm89_port")
from common import (HEAD_DIM, INDEX_TOPK, SWA_WIDTH, make_fp8_ds_mla_cache,
                    make_topk_indices, bench_cuda)
import triton_sparse_mla_ref as tri


def run_flashinfer(q, swa_cache, comp_cache, swa_idx, comp_idx, topk_lens, ws, scale):
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
    return trtllm_batch_decode_sparse_mla_dsv4(
        query=q, swa_kv_cache=swa_cache, workspace_buffer=ws,
        sparse_indices=swa_idx, compressed_kv_cache=comp_cache,
        bmm1_scale=scale, kv_layout="NHD",
        swa_topk_lens=None, extra_sparse_indices=comp_idx,
        extra_sparse_topk_lens=topk_lens)


def run_triton(q, cache, slot_ids, lens, scale):
    T, H = q.shape[0], q.shape[1]
    ms = torch.full((T, H), float("-inf"), device=q.device, dtype=torch.float32)
    dn = torch.zeros((T, H), device=q.device, dtype=torch.float32)
    acc = torch.zeros((T, H, HEAD_DIM), device=q.device, dtype=torch.float32)
    tri.accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead(
        q, cache, slot_ids, lens, 64, scale, ms, dn, acc)
    return acc / dn.unsqueeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-heads", type=int, default=64)
    p.add_argument("--out", default="benchmarks/sparse_mla_sm89_port/results_decode.json")
    args = p.parse_args()
    H, results = args.num_heads, []
    num_slots = 65536
    cache, _, _ = make_fp8_ds_mla_cache(num_slots)
    ws = torch.zeros(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    scale = HEAD_DIM ** -0.5
    for T in (1, 8, 32, 64, 128, 256):
        q_bf16 = torch.randn(T, H, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        idx = make_topk_indices(T, num_slots, INDEX_TOPK)
        lens = torch.full((T,), INDEX_TOPK, device="cuda", dtype=torch.int32)
        t_tri = bench_cuda(lambda: run_triton(q_bf16, cache, idx, lens, scale))
        # flashinfer: SWA-128 + compressed columns; here approximate with the
        # unified topk as extra_sparse and first 128 as SWA — exact tensor
        # prep mirrors DeepseekV4FlashInferSM120Attention._forward_decode
        # (vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:700-765);
        # adjust per that code path when first run fails shape checks.
        q_fi = q_bf16.unsqueeze(1)
        swa_cache = cache.view(num_slots, 1, -1).unsqueeze(0)  # refine per API check
        t_fi = bench_cuda(lambda: run_flashinfer(
            q_fi, swa_cache, swa_cache, idx[:, :SWA_WIDTH].contiguous(),
            idx.view(T, 1, -1), lens, ws, scale))
        results.append({"T": T, "H": H, "topk": INDEX_TOPK,
                        "triton_ms": t_tri, "flashinfer_ms": t_fi,
                        "speedup": t_tri / t_fi})
        print(results[-1])
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
```

关键提示：flashinfer 侧张量准备（SWA/compressed 双池、NHD 布局、uint8 视图）
必须与 `flashinfer_sparse.py:700-765` 的真实调用一致——先运行让 API 的
`_check_dsv4_sparse_mla_inputs` 报形状错误，按错误对齐（该校验信息完整）。
数值上先用 Step 3 的正确性对拍确认两条路径算的是同一注意力，再看时间。

- [ ] **Step 3: 正确性对拍（不是纯计时）**

在 bench 中先跑 T=4 的小 case，对比 flashinfer 输出与 Triton 归一化输出：

```python
out_fi = run_flashinfer(...)[:, 0]  # [T,H,512] bf16
out_tri = run_triton(...)           # [T,H,512] f32
diff = (out_fi.float() - out_tri).abs().max() / out_tri.abs().max()
assert diff < 3e-2, f"mismatch {diff}"   # fp8 KV 双路径容差
```

预期：相对误差 < 3e-2（两条路径都从同一 fp8 cache 反量化）。失败则先修
张量准备再计时。

- [ ] **Step 4: 运行并记录**

```bash
.venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --num-heads 64 | tee benchmarks/sparse_mla_sm89_port/results_decode_h64.log
.venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --num-heads 16 --out benchmarks/sparse_mla_sm89_port/results_decode_h16.json \
  | tee benchmarks/sparse_mla_sm89_port/results_decode_h16.log
```

预期：产出两组 JSON。head 数取 DSv4-Flash HF config 的
`num_attention_heads`（执行时
`.venv-sm120/bin/python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Flash', trust_remote_code=True); print(c.num_attention_heads)"`
确认，并按 TP=4 折算修正 `--num-heads` 取值）。

- [ ] **Step 5: Commit**

```bash
git add benchmarks/sparse_mla_sm89_port/
git commit -m "bench(sm89-port): SM120 decode sparse MLA flashinfer vs Triton"
```

### Task 4: MLA prefill 基准（flashinfer orchestrator vs Triton chunk 路径）

**Files:**
- Create: `benchmarks/sparse_mla_sm89_port/bench_sparse_mla_prefill.py`

- [ ] **Step 1: 写 prefill bench**

结构与 decode bench 相同，两点差异：
1. flashinfer 侧走 varlen：`cum_seq_lens_q`（`[B+1]` int32）+ `max_q_len`，
   token 数扫 `{256, 1024, 4096, 8192}`（单请求 chunked prefill 形态，
   num_tokens > 64 自动路由 prefill orchestrator——见
   `_sparse_mla_sm120.py::_DECODE_MAX_TOKENS = 64`）。
2. Triton 侧用 `triton_sparse_mla_ref.py` 中 prefill 使用的同一 chunk 累加
   kernel 按 4096-candidate chunk 循环调用（旧分支
   `nvidia/flashmla.py` 中 prefill 对该 kernel 的调用方式为准，
   `git show sm89-deepseek-v4-flash:vllm/models/deepseek_v4/nvidia/flashmla.py`
   中搜 `accumulate_fp8ds` 找到循环形态后照搬）。

对拍与计时逻辑复用 `common.bench_cuda`，正确性容差同 decode。

- [ ] **Step 2: 运行并记录**

```bash
.venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_prefill.py \
  | tee benchmarks/sparse_mla_sm89_port/results_prefill.log
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/sparse_mla_sm89_port/
git commit -m "bench(sm89-port): SM120 prefill sparse MLA flashinfer vs Triton"
```

### Task 5: MXFP4 MoE 基准（DeepGEMM vs Marlin）

**Files:**
- Create: `benchmarks/sparse_mla_sm89_port/bench_mxfp4_moe.py`

- [ ] **Step 1: 确认 DeepGEMM 在 SM120 可用性**

```bash
.venv-sm120/bin/python -c "
import vllm.utils.deep_gemm as dg
print('has_deep_gemm:', dg.has_deep_gemm())
import torch; print('cc:', torch.cuda.get_device_capability())"
```

若 `has_deep_gemm()` 为 False 或 DeepGEMM 不支持 SM120（导入/运行时报不支持
架构），安装 `uv pip install -p .venv-sm120/bin/python deep-gemm` 重试；
仍不可用则本 Task 只产出 Marlin 单边数据，在结果文件记录
`deepgemm_unavailable_reason` 后跳到 Step 4（spec 风险预案）。

- [ ] **Step 2: 写 MoE bench**

以 vLLM 现成 `benchmarks/kernels/benchmark_moe.py` 的计时框架为骨架，构造
DSv4-Flash 专家维度（执行时从 HF config 读取
`n_routed_experts / moe_intermediate_size / hidden_size / num_experts_per_tok`
并写死进脚本头部常量），token 数扫 `{1, 16, 64, 256, 1024, 4096}`，两个
被测对象通过 oracle 的两个 backend 分别构造 fused-MoE 层：
`Mxfp4MoeBackend.DEEPGEMM_MXFP4` 与 `Mxfp4MoeBackend.MARLIN`
（`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`；权重打包/scale
布局差异由 oracle 各自的 process 路径处理，参考同文件
`_pack_deepgemm_mxfp4_scales` 与 MARLIN 分支）。输出 JSON：
`{tokens, deepgemm_ms, marlin_ms, ratio}`。

- [ ] **Step 3: 运行并记录**

```bash
.venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_mxfp4_moe.py \
  | tee benchmarks/sparse_mla_sm89_port/results_mxfp4.log
```

- [ ] **Step 4: 汇总基准报告**

写 `benchmarks/sparse_mla_sm89_port/README.md`：环境、方法、三组结果表、
结论（flashinfer 相对 Triton 的 decode/prefill 加速比；Marlin 相对 DeepGEMM
的代价即 SM89 用 Marlin 的预期损失）。

- [ ] **Step 5: Commit**

```bash
git add benchmarks/sparse_mla_sm89_port/
git commit -m "bench(sm89-port): SM120 MXFP4 DeepGEMM vs Marlin + benchmark report"
```

---

## Phase 2：flashinfer fork SM89 移植

### Task 6: 建立 flashinfer fork 与 SM120 基线

**Files:**
- Create: `/home/yyf/flashinfer/`（独立 git 仓库，分支 `sparse-mla-sm89`）

- [ ] **Step 1: clone + 切 tag + 建分支**

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git /home/yyf/flashinfer
cd /home/yyf/flashinfer && git checkout v0.6.14 -b sparse-mla-sm89
git submodule update --init --recursive
```

（网络不通 GitHub 时用镜像 `https://gitclone.com/github.com/flashinfer-ai/flashinfer.git`。）

- [ ] **Step 2: 源码安装进 bench env（替换 wheel 版）**

```bash
cd /home/yyf/vllm
uv pip uninstall -p .venv-sm120/bin/python flashinfer-python
uv pip install -p .venv-sm120/bin/python -e /home/yyf/flashinfer --no-build-isolation \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- [ ] **Step 3: SM120 基线回归——fork 未改动时测试须过**

```bash
ls /home/yyf/flashinfer/tests | grep -i "sparse_mla\|mla"   # 找到官方测试文件名
.venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<找到的sparse_mla测试文件> -x -q
```

预期：PASS（记录用时与通过数，此为后续每次改动的回归命令）。同时重跑
Task 3 decode bench 确认源码安装性能与 wheel 一致（±5%）。

### Task 7: JIT 门与 SM89 强制原语开关

**Files:**
- Modify: `/home/yyf/flashinfer/flashinfer/jit/mla.py`（`gen_sparse_mla_sm120_module`）
- Modify: `/home/yyf/flashinfer/flashinfer/mla/_sparse_mla_sm120.py`（capability gate）

- [ ] **Step 1: 写失败测试（Python 层 gate 接受 (8,9)）**

`/home/yyf/flashinfer/tests/attention/test_sparse_mla_sm89_gate.py`：

```python
import pytest
from flashinfer.jit.mla import gen_sparse_mla_sm120_module


def test_jit_module_includes_sm89_arch():
    spec = gen_sparse_mla_sm120_module()
    flags = " ".join(spec.extra_cuda_cflags)
    assert "compute_89" in flags or "sm_89" in flags, flags
```

- [ ] **Step 2: 运行确认失败**

```bash
.venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/attention/test_sparse_mla_sm89_gate.py -v
```

预期：FAIL（当前只有 12x gencode）。

- [ ] **Step 3: 改 JIT 门**

`jit/mla.py::gen_sparse_mla_sm120_module`：
`supported_major_versions=[12]` → `supported_major_versions=[8, 12]`。
若 `get_nvcc_flags_list` 对 major=8 生成的是 `sm_80/86/89` 全套，确认其实现
（`flashinfer/compilation_context.py` 一带）后收敛到只加 `-gencode
arch=compute_89,code=sm_89`（sm_80/86 无 FP8 MMA，显式排除）。
`_sparse_mla_sm120.py` 中所有 `@supported_compute_capability([120, 121])`
（以实际 grep 结果为准）追加 `89`。

- [ ] **Step 4: 加 SM89 强制原语编译开关**

同文件 `gen_jit_spec(...)` 的 `extra_cuda_cflags` 支持从环境变量注入：

```python
import os
extra = []
if os.environ.get("FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS") == "1":
    extra.append("-DSPARSE_MLA_FORCE_SM89_PRIMS=1")
```

并把 `extra` 并进 flags、同时改 JIT 模块名（如 `"sparse_mla_sm120" + ("_sm89prims" if ... else "")`）
避免缓存冲突。CUDA 侧统一用宏：

```c
// include/flashinfer/attention/sparse_mla_sm120/arch/common.cuh 顶部
#if defined(SPARSE_MLA_FORCE_SM89_PRIMS) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900)
#define SPARSE_MLA_USE_SM89_PRIMS 1
#else
#define SPARSE_MLA_USE_SM89_PRIMS 0
#endif
```

- [ ] **Step 5: 测试过 + 回归 + Commit**

```bash
.venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/attention/test_sparse_mla_sm89_gate.py -v  # PASS
.venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<sparse_mla官方测试> -x -q                 # 仍PASS
cd /home/yyf/flashinfer && git add -A && git commit -m "jit: add sm89 target and SM89-prims force flag for sparse MLA"
```

### Task 8: arch 原语——块缩放 MMA → 普通 FP8 MMA + 软件 scale

**Files:**
- Modify: `/home/yyf/flashinfer/include/flashinfer/attention/sparse_mla_sm120/arch/mma_sm120.cuh`
- Modify: 调用点（`grep -rn "mma_fp8_block_scaled" include/flashinfer/attention/sparse_mla_sm120/`）

- [ ] **Step 1: 通读现状**

先完整读 `mma_sm120.cuh` 与所有 `mma_fp8_block_scaled_m16n8k32` 调用点
（预计在 `common/scale_mma.cuh`、`common/xv_rope_mma.cuh`），弄清 scale
操作数来源（ue8m0 scale 寄存器如何装配、每 32-K 块一个 scale）。

- [ ] **Step 2: 实现 SM89 版本（同签名）**

在 `mma_sm120.cuh` 的 block-scaled 函数内加分支（保持外部接口不变）：

```c
__device__ __forceinline__ MmaFp8Result mma_fp8_block_scaled_m16n8k32(
    /* 原参数不变: a_frag, b_frag, c_frag, ue8m0 scale byte(s) ... */) {
#if SPARSE_MLA_USE_SM89_PRIMS
  // 1) plain FP8 MMA（SM89 原生）
  MmaFp8Result d = mma_fp8_m16n8k32(a_frag, b_frag, /*c=*/zero_frag());
  // 2) ue8m0 scale = 2^(e-127)，纯指数幂：用 exp2f/位构造 float 后累加
  float s = __uint_as_float((static_cast<uint32_t>(scale_ue8m0)) << 23);
  d.d0 = fmaf(d.d0, s, c_frag.d0);
  d.d1 = fmaf(d.d1, s, c_frag.d1);
  d.d2 = fmaf(d.d2, s, c_frag.d2);
  d.d3 = fmaf(d.d3, s, c_frag.d3);
  return d;
#else
  /* 原 mma.sync.kind::mxf8f6f4.block_scale...ue8m0 inline asm 不动 */
#endif
}
```

要点：`ue8m0` 是 8-bit 纯指数（bias 127），`(uint32)e << 23` 位拼即得
float 2^(e-127)，零乘法开销；`scale_vec::1X` 语义 = 每次 MMA 的整个 K=32
块共用一个 scale，因此在累加前对 `d` 统一乘即可等价。若实际代码中 A/B 两侧
各有 scale（读源码确认），两 scale 相乘后同样处理。fragment 字段名以实际
`MmaFp8Result` 定义为准。

- [ ] **Step 3: 单元验证（本地 SM120 跑 SM89 原语）**

flashinfer tests 下新增
`test_sparse_mla_sm89_prims.py`——用官方 sparse MLA 测试同样的入口跑一遍，
但设置 `FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1`：

```python
import os, subprocess, sys

def test_sm89_prims_match_reference():
    env = dict(os.environ, FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS="1")
    r = subprocess.run([sys.executable, "-m", "pytest",
                        "tests/<sparse_mla官方测试>", "-x", "-q"],
                       env=env, cwd="/home/yyf/flashinfer")
    assert r.returncode == 0
```

```bash
FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 \
  .venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<sparse_mla官方测试> -x -q
```

预期：PASS（数值与 stock 路径一致，因为都是 e4m3×e4m3→f32 + 同一 scale，
只是 scale 应用位置从 MMA 指令内移到累加器）。

- [ ] **Step 4: Commit**

```bash
cd /home/yyf/flashinfer && git add -A && git commit -m "arch: software-scaled plain FP8 MMA path for SM89 (replaces mxf8f6f4 block_scale)"
```

### Task 9: arch 原语——mbarrier expect_tx → cp.async 完成跟踪

**Files:**
- Modify: `/home/yyf/flashinfer/include/flashinfer/attention/sparse_mla_sm120/arch/barrier.cuh`
- Modify: `/home/yyf/flashinfer/include/flashinfer/attention/sparse_mla_sm120/arch/cp_async.cuh`
- 调用点：`grep -rn "arrive_expect_tx\|mbarrier_arrive_expect_tx" include/flashinfer/attention/sparse_mla_sm120/`

- [ ] **Step 1: 通读生产者-消费者结构**

读 decode/prefill kernel 中 expect_tx 的用法：谁 arrive、谁 wait、tx_bytes
怎么算。SM89 替换策略（保持 mbarrier 骨架，最小改动）：

```c
// barrier.cuh
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
#if SPARSE_MLA_USE_SM89_PRIMS
  // SM89: no transaction accounting. The producer issues all its cp.asyncs,
  // then makes the LAST cp.async of the tile arrive on the mbarrier via
  // cp.async.mbarrier.arrive (SM80+); tx_bytes is ignored.
  (void)mbar; (void)tx_bytes;
#else
  /* 原 expect_tx asm 不动 */
#endif
}

// cp_async.cuh 新增
__device__ __forceinline__ void cp_async_mbarrier_arrive(uint64_t* mbar) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n" ::"r"(addr));
}
```

生产者侧调用点改造规则：原来"arrive_expect_tx(mbar, bytes); 发起 N 个
cp.async"的序列，SM89 分支改为"发起 N 个 cp.async;
cp_async_mbarrier_arrive(mbar)"——每个发起线程各 arrive 一次，则 mbarrier
初始化的 arrive count 需在 SM89 分支按参与线程数设置（读 kernel 中
`mbarrier_init` 的 count 参数并加对应 `#if` 分支）。消费者 `wait_parity`
逻辑两архitecture一致，无需改。

- [ ] **Step 2: 强制原语全量回归（SM120 卡）**

```bash
FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 \
  .venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<sparse_mla官方测试> -x -q
```

预期：PASS。若挂死（barrier count 不匹配的典型表现），用
`compute-sanitizer --tool synccheck` 定位：

```bash
FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 compute-sanitizer --tool synccheck \
  .venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<sparse_mla官方测试> -x -q -k <最小case>
```

- [ ] **Step 3: Commit**

```bash
cd /home/yyf/flashinfer && git add -A && git commit -m "arch: cp.async.mbarrier.arrive completion tracking for SM89 (replaces expect_tx)"
```

### Task 10: setmaxnreg 空操作 + smem 预算核对

**Files:**
- Modify: `/home/yyf/flashinfer/include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh:127,162,735,831`
- Modify: `/home/yyf/flashinfer/csrc/sparse_mla_sm120_decode_dsv4.cu`（仅当 smem 超限）

- [ ] **Step 1: setmaxnreg 守护**

四处 `asm volatile("setmaxnreg...")` 包上：

```c
#if !SPARSE_MLA_USE_SM89_PRIMS
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(32));
#endif
```

（inc 同理。SM89 无此指令，去掉后 warp-spec 生产者组寄存器不缩减，
可能降低占用率——先求正确，占用率问题留给远程 profile 阶段。）

- [ ] **Step 2: smem 预算核对**

```bash
grep -n "DYN_SMEM_BYTES" -A 8 /home/yyf/flashinfer/csrc/sparse_mla_sm120_decode_dsv4.cu
# 人工累加常量表达式，或加一行 printf 编译期 static_assert：
# static_assert(DYN_SMEM_BYTES <= 99 * 1024, "exceeds Ada smem");
```

判定规则：`DYN_SMEM_BYTES + 静态 smem > 101376` 时，SM89 分支
（`#if SPARSE_MLA_USE_SM89_PRIMS`）将 `DSV4_KV_BUF_COUNT` 从 2 降为 1
（单缓冲，去掉 KV 预取重叠）或将 `DSV4_BI` 64→32，取先满足预算者；
decode_dsv3_2 与 prefill kernel 同样核对。改动后必跑 Step 3 回归。

- [ ] **Step 3: 强制原语全量回归 + decode bench 复测**

```bash
FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 \
  .venv-sm120/bin/python -m pytest /home/yyf/flashinfer/tests/<sparse_mla官方测试> -x -q
FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --out benchmarks/sparse_mla_sm89_port/results_decode_sm89prims_on_sm120.json
```

预期：测试 PASS；bench 数值给出"SM89 风格原语在 SM120 上"的性能
（预估相对 stock 有 0-20% 损失，作为 SM89 真机性能的上界参考，写进
bench README）。

- [ ] **Step 4: Commit（flashinfer + vllm bench 结果各自仓库）**

```bash
cd /home/yyf/flashinfer && git add -A && git commit -m "arch: guard setmaxnreg and fit smem budget for SM89"
cd /home/yyf/vllm && git add benchmarks/sparse_mla_sm89_port/ && git commit -m "bench(sm89-port): SM89-prims-on-SM120 upper-bound numbers"
```

### Task 11: sm_89 交叉编译验证

- [ ] **Step 1: 直接用 nvcc 编所有编译单元到 sm_89**

```bash
cd /home/yyf/flashinfer
for f in csrc/sparse_mla_sm120*.cu; do
  nvcc -std=c++17 -arch=compute_89 -code=sm_89 -c "$f" -o /tmp/$(basename $f).o \
    -Iinclude -Icsrc $(.venv-sm120/bin/python -c "import torch.utils.cpp_extension as c; print(' '.join('-I'+p for p in c.include_paths()))") \
    2>&1 | tee /tmp/sm89_compile_$(basename $f).log
done
```

预期：全部编译通过，无 `not supported on sm_89` 类错误。任何残留的
SM90+/SM120a 指令都会在这里现形——逐个回到对应 arch 文件补
`SPARSE_MLA_USE_SM89_PRIMS` 分支。注意 `__CUDA_ARCH__ < 900` 已含
在宏定义里，sm_89 编译自动选 SM89 原语，无需 FORCE 环境变量。

- [ ] **Step 2: Python 层确认 (8,9) 分发**

```bash
.venv-sm120/bin/python -c "
from flashinfer.utils import supported_compute_capability  # 若gate实现不同按实调整
import flashinfer.mla._sparse_mla_sm120 as m
print('sm89 accepted:', m)  # 结合Task 7的gate测试断言 89 在允许列表"
```

- [ ] **Step 3: Commit + 记录 fork commit id**

```bash
cd /home/yyf/flashinfer && git add -A && git commit -m "build: verify sm_89 cross-compilation of sparse MLA module"
git rev-parse HEAD   # 记录，写入 vLLM 分支 docs（Task 12 Step 4）
```

---

## Phase 3：vLLM 分支 SM89 接线

### Task 12: capability gate 扩展到 (8,9)

**Files:**
- Modify: `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py:85-135`
- Modify: `vllm/models/deepseek_v4/nvidia/model.py:779,788`
- Modify: `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`（`FlashInferMLASparseSM120Backend` 的 capability 校验，grep `major == 12`）
- Modify: `vllm/platforms/cuda.py:129-133`（`_get_backend_priorities` 的 else 分支）
- Test: `tests/v1/attention/test_flashinfer_sparse_mla_sm89_selection.py`

- [ ] **Step 1: 写失败测试**

仿照现有 `tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py`：

```python
# SPDX-License-Identifier: Apache-2.0
"""SM89 selects the FlashInfer sparse MLA path (SM120 kernels ported to Ada)."""
from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm89_capability_accepted(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    with set_current_vllm_config(_fake_vllm_config("deepseek_v4")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 9),
            attn_type="decoder",
        )
    assert invalid_reasons == []


def test_sm86_capability_rejected(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    with set_current_vllm_config(_fake_vllm_config("deepseek_v4")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(8, 6),
            attn_type="decoder",
        )
    assert invalid_reasons != []
```

（`validate_configuration` 的参数列表照抄 sm120 测试同名调用；若本测试中的
kv_cache_dtype/block_size 组合被其他校验拒绝，对齐 sm120 测试所用组合。）

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv-sm120/bin/python -m pytest tests/v1/attention/test_flashinfer_sparse_mla_sm89_selection.py -v
```

预期：`test_sm89_capability_accepted` FAIL（(8,9) 被拒）。

- [ ] **Step 3: 最小实现**

统一的判定助手（放 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`，
供各处 import）：

```python
def _is_flashinfer_sparse_jit_capability(capability: DeviceCapability) -> bool:
    """SM120/121 natively; SM89 via the ported JIT kernels (exact 8.9 only)."""
    return capability.major == 12 or (capability.major, capability.minor) == (8, 9)
```

逐点替换：
- `flashinfer_sparse.py` `supports_compute_capability`：
  `capability.major in [10, 12]` → `capability.major == 10 or
  _is_flashinfer_sparse_jit_capability(capability)`；
  `supports_combination` 与 `get_kv_cache_shape` 中 `major == 12` 分支同样
  改为调用该助手。
- `model.py:779,788` `device_capability.major == 12` → 助手调用。
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` 中
  `FlashInferMLASparseSM120Backend` 相关 `major == 12` 校验同改。
- `vllm/platforms/cuda.py` `_get_backend_priorities` else 分支（major==8 落
  这里）追加 `AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120`（DSv4 专用
  选择在 `_select_dsv4_attn_cls`，此处补的是通用回退表）。

- [ ] **Step 4: 测试过 + lint + Commit**

```bash
.venv-sm120/bin/python -m pytest tests/v1/attention/test_flashinfer_sparse_mla_sm89_selection.py tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py -v
pre-commit run --files $(git diff --name-only HEAD)
git add -A && git commit -m "sm89: extend flashinfer sparse MLA capability gates to Ada (8,9)

记录配套 flashinfer fork commit: <Task 11 Step 3 的 rev-parse 输出>"
```

### Task 13: 非 MLA 算子 SM89 审计与 fallback 接线

**Files:**
- Audit: `vllm/models/deepseek_v4/nvidia/ops/{fused_indexer_q_cutedsl,dequant_gather_k_cutedsl,sparse_attn_compress_cutedsl,o_proj,prepare_megamoe}.py`
- Modify: 各 op 的分发点（audit 后确定）
- Test: `tests/v1/attention/test_dsv4_sm89_op_dispatch.py`

- [ ] **Step 1: 审计——列出每个 op 在 (8,9) 下的行为**

对五个文件逐一：找到其 capability/`has_flashinfer_cutedsl` 分发条件，回答
"(8,9) 时选什么实现？该实现 Ada 能跑吗？"。对照旧分支处理方式：

```bash
git show d4fbbdc8d --stat   # 旧分支 "Disable CuTe-DSL on Ada" 改了哪些文件
git show d4fbbdc8d          # 具体怎么改
```

产出审计清单写入 commit message。已知参考结论（执行时验证）：CuTe-DSL
kernel 在 Ada 不可用 → 需回退到 `vllm/models/deepseek_v4/common/ops/` 的
Triton 实现（`fused_indexer_q.py`、`fused_inv_rope_fp8_quant.py` 等，main
已有）；o_proj FP8 einsum 与 mHC（hc_mult 相关 GEMM）在 main 的默认实现若为
纯 torch/Triton 则无需改。

- [ ] **Step 2: 写分发测试**

```python
# tests/v1/attention/test_dsv4_sm89_op_dispatch.py
"""(8,9) 下 DSv4 各算子分发不选 CuTe-DSL/Blackwell-only 实现。"""
# 对审计出的每个分发函数，monkeypatch current_platform.get_device_capability
# 返回 DeviceCapability(8, 9)，断言返回的可调用对象是 Triton/torch 实现
# （按 Step 1 审计出的函数名逐一写断言，形如：）
#   fn = select_indexer_q_impl(...)
#   assert "cutedsl" not in fn.__module__
```

（断言目标在 Step 1 审计后填成具体函数名——这是本 Task 内的顺序依赖，
不是留白。）

- [ ] **Step 3: 接线修复 → 测试过 → Commit**

```bash
.venv-sm120/bin/python -m pytest tests/v1/attention/test_dsv4_sm89_op_dispatch.py -v
pre-commit run --files $(git diff --name-only HEAD)
git add -A && git commit -m "sm89: route DSv4 aux ops to portable Triton impls on Ada"
```

### Task 14: MXFP4 MoE oracle 在 (8,9) 选 Marlin

**Files:**
- Audit/Modify: `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`
- Test: `tests/quantization/test_mxfp4_sm89_backend.py`

- [ ] **Step 1: 写测试**

```python
# tests/quantization/test_mxfp4_sm89_backend.py
"""MXFP4 MoE backend resolution on SM89 must land on MARLIN."""
# monkeypatch capability → DeviceCapability(8, 9)，调用 oracle 的 backend
# 解析入口（同文件中 map_mxfp4_backend / select 逻辑，读源码定位），断言
# 结果in {MARLIN, BATCHED_MARLIN} 且不含 DEEPGEMM_MXFP4。
```

- [ ] **Step 2: 跑测试**

预期：大概率直接 PASS（DeepGEMM 的 `is_supported_config` 会拒 (8,9)）——
则本 Task 只提交测试固化行为；若 FAIL 则在 oracle 的 capability 过滤处修复。

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "sm89: pin MXFP4 MoE backend to Marlin on Ada (test)"
```

### Task 15: SM120 全量回归 + sm_89 wheel 交叉编译

- [ ] **Step 1: SM120 回归（分支改动不得破坏 SM120）**

```bash
.venv-sm120/bin/python -m pytest tests/v1/attention/test_flashinfer_sparse_mla_sm120_api.py \
  tests/v1/attention/test_sparse_mla_backends.py -v
.venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --out /tmp/regress_decode.json   # 与 Task 3 结果对比 ±5%
```

- [ ] **Step 2: 交叉编译 sm_89 wheel（供远程直接安装）**

```bash
cd /home/yyf/vllm
TORCH_CUDA_ARCH_LIST="8.9" uv build --wheel -o dist-sm89/ 2>&1 | tail -20
ls -lh dist-sm89/
```

预期：产出 `vllm-*.whl`（编译约 1-2 小时，`run_in_background`）。同时打包
flashinfer fork：远程用 `pip install -e /path/to/flashinfer`（JIT 在远程
首跑时按 sm_89 编译，无需本地预编）。

- [ ] **Step 3: Commit + 更新分支 README**

在 `benchmarks/sparse_mla_sm89_port/README.md` 追加"远程部署工件"节：
wheel 路径、flashinfer fork commit、安装命令。

```bash
git add -A && git commit -m "sm89: SM120 regression green; build sm_89 wheel for remote deploy"
```

### Task 16: 提醒用户提供远程 SM89 服务器（阻塞点）

- [ ] **Step 1: 汇报并请求登陆方式**

向用户输出：Phase 0-3 完成摘要（SM120 基准结论、fork commit、wheel 位置、
SM120 回归结果），并明确请求远程 4×4090 48GB 服务器的登陆方式（SSH host/
user/密钥或密码，建议通过 ssh-manager MCP 配置）。**此任务后停止，等待用户。**

---

## Phase 4：远程 SM89 验证与优化（用户提供服务器后执行）

### Task 17: 远程环境 + SM89 kernel 正确性与性能

- [ ] **Step 1: 远程 env**

```bash
# 远程（经 ssh-manager 或用户给的方式登陆）：
# 1. rsync 本地 dist-sm89/*.whl 与 /home/yyf/flashinfer 到远程
# 2. uv venv --python 3.12 ~/venv-sm89 && uv pip install <wheel> --torch-backend=auto
# 3. uv pip install -e ~/flashinfer --no-build-isolation
# 4. rsync benchmarks/sparse_mla_sm89_port/ 到远程
```

- [ ] **Step 2: kernel 正确性（首次在真 SM89 上）**

```bash
python -m pytest ~/flashinfer/tests/<sparse_mla官方测试> -x -q          # JIT 编 sm_89
python -m pytest ~/flashinfer/tests/attention/test_sparse_mla_sm89_gate.py -v
# vLLM 侧（wheel 已装）：DSpark 非因果 sparse MLA 测试此前只在 SM90/100 跑，
# 现在 (8,9) 门打开后应在 SM89 实跑而非 skip：
python -m pytest tests/v1/attention/test_dspark_noncausal_sparse_mla.py -v
python -m pytest tests/v1/attention/test_flashinfer_sparse_mla_sm89_selection.py -v
```

预期：PASS。失败优先级：非法指令 → Task 11 漏网；数值错 → Task 8 scale
语义；挂死 → Task 9 barrier count；dspark 测试仍 skip → 其 skip 条件里的
capability 列表需同步加 (8,9)。

- [ ] **Step 3: SM89 上 flashinfer vs Triton 性能对比（用户要求的核心数据）**

```bash
python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --out results_decode_sm89.json
python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_prefill.py \
  --out results_prefill_sm89.json
```

结果回传本地并入 `benchmarks/sparse_mla_sm89_port/README.md`，commit。

### Task 18: 远程 serve + profile + 瓶颈优化

- [ ] **Step 1: vllm serve（flashinfer backend）**

```bash
# 远程：
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 --block-size 256 \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --speculative_config.method=dspark \
  --speculative_config.num_speculative_tokens=2
```

（模型下载/磁盘、`--attention-backend` 是否需显式指定、gpu-memory-utilization
按远程实况调；出词 sanity + DSpark 接受率看 serve 日志 metrics。
4×4090 48GB 魔改卡若 NCCL P2P 异常，先试 `NCCL_P2P_DISABLE=1`，并对照旧
sm89 分支 README 记录的 NCCL 配置。）

- [ ] **Step 2: profile**

```bash
# 远程 serve 前 export VLLM_TORCH_PROFILER_DIR=~/vllm_profile
# 用 benchmarks/benchmark_serving.py 打负载，start/stop profile：
python benchmarks/benchmark_serving.py --backend vllm --model deepseek-ai/DeepSeek-V4-Flash \
  --num-prompts 32 --profile
```

trace 拉回本地用 `.venv-sm120/bin/python -m torch_tb_profiler` 或直接解析
JSON，按算子聚合 GPU 时间，分 prefill/decode 各列 Top-10。

- [ ] **Step 3: 瓶颈优化（数据驱动，单独立项）**

按 profile 结果确定优化对象（候选按预期概率排序：sparse MLA kernel 占比 →
调 smem/占用率/split 策略；indexer Triton kernel → 调 block size；Marlin
MoE → 已知代价，对照 Task 5 数据判断是否合理）。每项优化以"修改 → kernel
bench 复测 → e2e profile 复测"闭环，提交到各自仓库。此步的具体改动由
profile 数据决定，完成 Step 2 后与用户对齐优先级再动手。

---

## 执行注意事项

- 每个 Task 的 flashinfer 改动都要跑两条回归：stock SM120 路径（不带
  FORCE 环境变量）与 SM89-prims 路径（带），都必须 PASS 才能 commit。
- vLLM 仓库遵守 AGENTS.md：uv、pre-commit、88 列、Google docstring。
- flashinfer fork 遵守其自身代码风格（clang-format 配置在其仓库根）。
- 所有 bench 结果 JSON/log 提交进 `benchmarks/sparse_mla_sm89_port/`，
  报告更新在同目录 README.md。
- Task 16 是硬阻塞点：完成后必须停下等用户提供远程服务器，不得跳过。
