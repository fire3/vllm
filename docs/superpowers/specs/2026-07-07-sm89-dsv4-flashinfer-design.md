# DeepSeek-V4-Flash-DSpark SM89 适配设计（flashinfer MLA 路径）

日期：2026-07-07
分支：`sm89-dsv4-flashinfer`（基于 main `93e2ab711`）
状态：已由用户确认

## 目标

在 SM89（Ada Lovelace，远程 4×RTX 4090 48GB）上运行 DeepSeek-V4-Flash-DSpark
推理，sparse MLA attention 使用 flashinfer 的 CUDA kernel（移植自 SM120 实现），
替代旧 `sm89-deepseek-v4-flash` 分支的 Triton 路径。MoE 保持 MXFP4 Marlin。

成功标准：
1. SM120 本地基准报告：flashinfer sparse MLA vs Triton sparse MLA、
   DeepGEMM MXFP4 vs Marlin。
2. flashinfer fork 的 sparse MLA kernel 在 SM89 上数值正确（对照 torch/FP32
   参考与 Triton kernel 输出），且 SM120 路径零回归。
3. 远程 SM89 上 `vllm serve` DSv4-Flash-DSpark（TP=4、
   `--kv-cache-dtype fp8_ds_mla`、flashinfer backend）正常出词，DSpark
   投机解码接受率正常。
4. vllm profile 定位 prefill/decode 瓶颈算子并完成针对性优化。

## 背景事实（探索确认）

- main 已合并 DSpark（#46995/#47093/#47429）与 SM120 支持（#43477/#46506）。
- SM120 的 "flashinfer MLA" = flashinfer **0.6.14** 的 `_sparse_mla_sm120`
  模块：JIT 编译的 warp-specialized CUDA 源码 kernel（decode dsv4/dsv3_2
  独立 kernel + prefill orchestrator + merge kernel），非 trtllm-gen cubin，
  源码位于 `include/flashinfer/attention/sparse_mla_sm120/`。
- vLLM main 的 `flashinfer_mla_sparse_sm120.py` 传递 `kv_scale_format`
  参数，该参数 0.6.13 不存在——main 实际需要 flashinfer ≥0.6.14，
  requirements 中 0.6.13 的 pin 已过时。
- SM100 走 trtllm-gen cubin（`trtllm_batch_decode_sparse_mla_dsv4`），
  不可移植；SM89 移植基于 SM120 的 JIT 源码路径。
- kernel arch 原语已隔离在 `arch/` 子目录：`mma_sm120.cuh`、`barrier.cuh`、
  `cp_async.cuh`、`ldmatrix_sm120.cuh`。
- 旧 `sm89-deepseek-v4-flash` 分支与 main 分叉于 2026-06-15（main 领先
  861 commit），只作参考，不 rebase。

## 1. 分支与仓库布局

- vLLM：新分支 `sm89-dsv4-flashinfer`，从 main 起步。
- flashinfer：clone 至 `/home/yyf/flashinfer`，基于 v0.6.14 tag，分支
  `sparse-mla-sm89`，pip 源码安装进测试 env。vLLM 分支 docs 记录配套
  flashinfer commit。
- 旧分支 Triton kernel 通过 git worktree 单独 checkout 供 benchmark import。

## 2. 阶段一：SM120 基准测试（本地 RTX PRO 6000）

- 新建独立 env（uv，torch + flashinfer 0.6.14 + vllm main 源码 build）。
- MLA 对比：flashinfer `sparse_mla_sm120` decode/prefill kernel vs 旧分支
  Triton sparse MLA kernel
  （`accumulate_fp8ds_global_slots_sparse_mla_attention_chunk_multihead` 系）。
  Shape 取 DeepSeek-V4-Flash 真实配置（TP=4 折算 head 数、index_topk、
  fp8_ds_mla KV 布局）；decode 扫 batch 1–256 含 DSpark 投机 q_len>1；
  prefill 扫 chunk 长度。输出延时 + 有效带宽。
- MXFP4 对比：DeepGEMM MXFP4 grouped GEMM vs Marlin MoE，V4-Flash 专家
  维度，扫 token 数。结果仅用于量化 SM89 用 Marlin 的代价（SM89 不移植
  DeepGEMM，已确认）。
- 产出：benchmark 报告，作为移植性能基线与验收参照。

## 3. 阶段二：flashinfer fork SM89 移植（方案 A：单源码树 + arch 层替换）

`arch/` 原语加 `__CUDA_ARCH__` 分支（或平行 sm89 头文件实现同一接口），
kernel 主体（decode_dsv4_kernel.cuh / prefill_kernel.cuh / orchestrator）
不动或最小改动：

| SM120 用法 | SM89 替换 |
|---|---|
| `mma.sync.kind::mxf8f6f4.block_scale...ue8m0`（SM120a 专属） | 普通 `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`（SM89 原生）+ 累加器软件乘 pow2 scale |
| `mbarrier.arrive.expect_tx`（SM90+） | `cp.async.mbarrier.arrive`（SM80+）保持 mbarrier 结构；备选 commit_group/wait_group |
| `setmaxnreg`（SM90+，prefill 用） | 空操作，接受占用率损失 |
| JIT `supported_major_versions=[12]` | `[8,12]`，Python 层精确 gate 至 `(8,9)` |

- 共享内存核对：decode kernel 动态 smem（58KB 双缓冲 KV + Q/softmax 等）
  若超 Ada 99KB/block 上限，SM89 降单缓冲 KV 或缩小 BI=64 tile。
- 验证技巧：SM89 替换路径加编译开关，先在 SM120 上强制走 SM89 风格原语
  （普通 MMA + 软件 scale + cp.async 完成跟踪）编译运行，本地卡验证数值
  正确后再交叉编译 sm_89，减少远程调试。

## 4. 阶段三：vLLM 分支 SM89 接线

- `DeepseekV4FlashInferMLASparseBackend.supports_compute_capability`：
  `[10,12]` → 增加 `(8,9)`；`supports_combination`、`get_kv_cache_shape`
  的 major==12 分支同步扩展。
- `_select_dsv4_attn_cls` 与 v1 `flashinfer_mla_sparse_sm120.py`：`(8,9)`
  复用 `DeepseekV4FlashInferSM120Attention` 路径（类名中性化或加 alias）。
- `_get_backend_priorities` major==8 加入 DSv4 sparse 后端。
- 非 MLA 算子审计：`nvidia/ops/` 三个 CuTe-DSL kernel（fused_indexer_q、
  dequant_gather_k、sparse_attn_compress）、o_proj FP8 einsum、mHC pre/post
  GEMM——逐个确认 main 在 SM120 的实现是否 Ada 兼容；不兼容的接
  `common/ops/` 已有 Triton 实现或从旧分支移植（参考旧分支 `d4fbbdc8d`）。
- MXFP4 MoE：确认 oracle 在 (8,9) 选 MARLIN（旧分支已验证 Marlin 路径）。
- DSpark：main 实现随 attention backend 透传；验证 SWA-128 tile 与非因果
  投机窗口在新 backend 下正确。

## 5. 阶段四/五：验证与远程部署

- 本地验证：flashinfer fork 在 SM120 全量回归（kernel 测试 + serve
  smoke）；sm_89 交叉编译通过。
- **完成后提醒用户提供远程 4×4090 48GB 服务器**，随后依次：
  1. SM89 kernel 正确性 + flashinfer vs Triton 性能对比；
  2. `vllm serve` DSv4-Flash-DSpark（TP=4、fp8_ds_mla、flashinfer backend）；
  3. vllm profile（torch profiler trace）定位 prefill/decode 瓶颈算子；
  4. 针对性优化。
- vLLM 分支 build：优先本地交叉编译 sm_89 wheel
  （`TORCH_CUDA_ARCH_LIST=8.9`）直接部署，避免远程长时间编译。

## 6. 测试策略

- kernel 正确性：随机输入对照 torch/FP32 参考与 Triton kernel 输出，
  覆盖 fp8_ds_mla 布局、varlen、topk 填充（-1 无效索引）、SWA 窗口边界、
  DSpark q_len>1。
- SM120 回归：fork 前后 kernel 测试与 e2e 输出一致。
- E2E：serve 出词 sanity + DSpark 接受率 + profile。

## 7. 风险与应对

- smem 超 Ada 上限 → 单缓冲/缩 tile，接受性能损失。
- Ada FP8 MMA FP32 累加吞吐减半（相对 FP16 累加）→ decode 访存受限影响
  可控；必要时评估 FP16 累加两段求和。
- 移植后仍慢于 Triton → backend 可切换，以 SM89 实测数据定 serve 默认。
- DeepGEMM 在 SM120 不可用 → benchmark 退化为 Marlin 单边数据并记录原因。
- 4×4090 48GB 魔改卡 P2P/NCCL 兼容性 → 沿用旧分支 serve 的 NCCL 配置经验。
