# DeepSeek-V4-Flash 在 SM89（Ada / RTX 4090）上的支持——设计与开发文档

> **目标代码库**：vLLM `v0.26.0`（`/home/fire3/SRC/vllm`）
> **交付物**：把 DeepSeek-V4-Flash 的推理从 SM90/SM100/SM120 扩展到 **SM89（Ada：RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada）** 的完整开发方案。本文档对仓库内改动自洽完整，无需再查阅其他代码仓库。

---

## 0. 目标与结论

SM89 支持由两层构成：

1. **外部依赖：FlashInfer 0.6.14「SM89 sparse-MLA」版 wheel**。这是真正的内核层移植（把 sparse-MLA 的 JIT 编译器从只认 SM120 开放到 capability 8.9，并修复 per-MMA 的 UE8M0 scale 精度问题）。它不属于 vLLM 仓库，运行时由 `has_flashinfer_sparse_mla_sm89()` 探测。
2. **仓库内改动：约 11 个 Python 文件**。让 SM89 复用既有的 SM120 FlashInfer sparse-MLA 后端，同时把推理路径里所有 DeepGEMM / CuTe-DSL / SM90+ 专属算子强制落到可移植的 Triton/torch 实现上。

核心结论：

- 不需要改 C++ / CUDA 源码，也不需要改动注意力索引器的 scheduler-metadata 逻辑（它们已经按硬件能力门控好）。
- 需要新增的是 3 个**自包含**的 Triton 内核模块（`fp8_einsum.py`、`sm12x_mqa.py`、`sm12x_deep_gemm_fallbacks.py`），以及若干既有文件里的门控与接线改动。
- 性能上限由硬件决定：Ada 有 FP8 张量核，但没有 FP4 张量核、没有硬件 microscaling MMA，因此 FP4 专家 MoE 只能走 Marlin WNA16 反量化（慢）；预期单并发 decode 约 82 tok/s（4×RTX 4090）。

---

## 1. 背景与硬件约束

> 本章写给两类读者：**了解 DeepSeek V4 模型结构但不熟悉 vLLM 代码**的人，以及**熟悉 vLLM 通用代码但没接触过稀疏注意力 / DeepGEMM**的人。先讲清三个基础概念（GPU 架构、算子/内核、vLLM 的硬件探测），再逐个介绍推理路径上的关键算子，最后给出硬件约束结论。

### 1.1 三个必须知道的基础概念

**① GPU 架构与 compute capability（SM 编号）**

NVIDIA 每代 GPU 有不同的「计算能力」（compute capability，记作 `(major, minor)`，常写成 SM 编号，如 SM89）。张量核（Tensor Core，GPU 上做矩阵乘法的专用单元）能算哪些精度，由架构决定：

| capability | 架构代号 | 代表卡 | 张量核支持 |
|---|---|---|---|
| (8, 9) | Ada | RTX 4090 / L40 / L40S / L4 / RTX 6000 Ada | FP16 / BF16 / FP8 |
| (9, 0) | Hopper | H100 / H200 | FP16 / BF16 / FP8（新增 TMA 硬件） |
| (10, x) | Blackwell 数据中心 | B200 / GB200 | +FP4、microscaling（UE8M0） |
| (12, x) | Blackwell 消费级 | RTX 50 系 | +FP4、microscaling |

vLLM 用 `current_platform.get_device_capability()`（`vllm/platforms/interface.py:420`）拿到本机 capability，再用三个辅助函数做判定（`interface.py:457-493`）：

- `is_device_capability((8, 9))`：**精确匹配**——capability 恰好等于 `(8, 9)` 才返回 `True`；
- `is_device_capability_family(120)`：**按 major 匹配**——任何 `12.x` 都算；
- `has_device_capability((8, 9))`：**大于等于**比较。

**② 算子 / 内核（kernel）/ 降级路径**

- **算子（operator/op）**：模型里的一个计算单元，比如「矩阵乘」「注意力」。
- **内核（kernel）**：算子在 GPU 上的具体实现。同一个算子可以有多个内核，分别针对不同硬件优化。vLLM 里的内核主要来自：DeepGEMM（DeepSeek 自研 GEMM 库）、FlashInfer（注意力库）、CUTLASS / CuTe-DSL（NVIDIA 模板库）、Triton（Python 编写的可移植内核）、以及 PyTorch 原生算子（`torch.matmul` 等）。
- **降级路径（fallback）**：当最优内核在当前硬件上不可用时，改用一个**可移植但通常较慢**的实现。vLLM 的惯例是「最优内核可用就用最优，不可用就自动降级」，而「可用」的判断通常是一个 capability / 包探测函数。

**③ vLLM 的硬件能力探测函数（本方案反复用到）**

| 函数 | 位置 | 语义 |
|---|---|---|
| `has_deep_gemm()` | `vllm/utils/import_utils.py` | DeepGEMM 包是否安装 |
| `is_deep_gemm_supported()` | `vllm/utils/deep_gemm.py:93` | DeepGEMM 包已装 **且** 当前架构支持（`support_deep_gemm()` 只认 SM90 / SM100 / SM120，`vllm/platforms/cuda.py:665`） |
| `has_cutedsl()` | `vllm/utils/import_utils.py:547` | `cutlass` 包是否安装（**只看包、不看硬件**——这正是 SM89 上需要加门控的原因） |
| `has_flashinfer_sparse_mla_sm120()` | `vllm/utils/flashinfer.py:216` | FlashInfer 是否提供 sparse-MLA 解码 API |
| `current_platform.support_deep_gemm()` | `vllm/platforms/cuda.py:665` | 架构是否在 SM90 / SM100 / SM120 家族 |

关键点：**SM89（Ada）不在 DeepGEMM 的支持列表里**，所以 `is_deep_gemm_supported()` 在 Ada 上天然返回 `False`；但 `has_cutedsl()` 只查包，装了 `cutlass` 就会返回 `True`——如果不加门控，SM89 会误入 CuTe-DSL 内核。这正是方案 5.2 节要修的那一处。

### 1.2 DeepSeek-V4-Flash 推理路径上的关键算子

#### 1.2.1 Lightning Indexer（DeepSeek 稀疏注意力 DSA）

**它是什么**：普通注意力里每个 query 要和全部历史 token 算相关度；DeepSeek V4 用**稀疏注意力**——先让一个轻量索引器（Lightning Indexer）算出每个 query 与所有历史 token 的粗略相关度（FP8 MQA logits），再取 **top-k**，只保留最相关的 k 个 token 的索引。之后的正式注意力（sparse MLA）只在这 k 个 token 上做，大幅降低计算量。由于所有 query head 共享同一份 KV（`num_kv_heads=1`），所以叫 MQA（Multi-Query Attention）。

**vLLM 代码位置**：
- 算子层：`vllm/model_executor/layers/sparse_attn_indexer.py` 的 `SparseAttnIndexer`（CustomOp，`__init__` 在 718 行）。它的构造器在 CUDA 且无 DeepGEMM 时**直接抛错**（750-754 行）：“Sparse Attention Indexer CUDA op requires DeepGEMM support”——这就是 SM89 上必须先放宽构造器的原因（方案 5.5 节）。
- 调度/元数据层：`vllm/v1/attention/backends/mla/indexer.py`。`DeepseekV4IndexerBackend`（164 行）把每个请求的 KV 块表、序列长度整理成内核要的 metadata；745 行只在 `has_deep_gemm()` 为真时才调用 `get_paged_mqa_logits_metadata`，SM89 上天然跳过，无需改动。

**上游最优实现**：DeepGEMM 的 `fp8_fp4_mqa_logits`（prefill）/ `fp8_fp4_paged_mqa_logits`（decode，paged 指 KV 按块存储），直接在 FP8 张量核上算 logits，并用 `top_k_per_row` 取 top-k。它们只支持 SM90+（SM100+ 变体还要 TMA / FP4）。

**SM89 处理**：新增 Triton 直出 top-k 路径（`sm12x_mqa.py` + `sm12x_deep_gemm_fallbacks.py`）：分块算 logits + `torch.topk` 合并，**不物化完整 logits 矩阵**（长上下文下完整 logits 是 `[seq, seq]` 的 FP32 矩阵，会占几个 GiB）。

#### 1.2.2 本地 MLA（sparse MLA）

**它是什么**：MLA（Multi-head Latent Attention）是 DeepSeek 的注意力机制：不直接存 KV，而是把 KV 压缩成低秩 latent，用时再解压。**sparse MLA** 是 FlashInfer 为 DeepSeek V4 定制的解码内核：输入 top-k 索引，只从 KV cache 里读被选中的块做注意力。

**vLLM 代码位置**：
- 后端：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`，`FlashInferMLASparseSM120Backend`（141 行）的 `supports_compute_capability` 只认 `capability.major == 12`（169-171 行）。
- 实现：`vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` 的 `FlashInferMLASparseSM120Impl`（32 行），构造器用 `has_flashinfer_sparse_mla_sm120()` 检查 API（93-99 行），并要求 `fp8_ds_mla` 打包 KV cache 布局（67-71 行）。
- DSv4 专用层：`vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`，`DeepseekV4FlashInferMLASparseBackend`（63 行）的 `supports_compute_capability` 认 `major in [10, 12]`（100-101 行）；`DeepseekV4FlashInferSM120Attention`（528 行）是 SM120 专用注意力层。
- 注意力类选择：`vllm/models/deepseek_v4/nvidia/model.py::_select_dsv4_attn_cls`（760 行），`major == 12` 才路由到 `DeepseekV4FlashInferSM120Attention`，否则走 FlashMLA。

**上游绑定**：FlashInfer 的 sparse-MLA **JIT 编译器默认只对 SM120 开放**（这是外部依赖里的硬件门控，不在 vLLM 仓库内）。所以上游 vLLM 只在 SM120 上启用这套后端。

**SM89 处理**：整套后端/实现/注意力层复用不动，只把三处 capability 检查（`flashinfer_mla_sparse.py:170`、`flashinfer_sparse.py:101`、`_select_dsv4_attn_cls`）从 `major == 12` 放宽为「12 或 (8,9)」，并新增 `has_flashinfer_sparse_mla_sm89()` 探测（方案 5.1 节）确认真实设备在 FlashInfer 内部被解析为 sparse 后端。

#### 1.2.3 o_proj

**它是什么**：注意力层最后要把各 head 的输出合并回 hidden 空间。DeepSeek V4 的 o_proj 是「inverse RoPE → FP8 量化 → einsum → wo_b 线性层」四步。einsum 形状 `bhr,hdr->bhd`：`b` 是 batch、`h` 是 head group、`r` 是 rope 维度、`d` 是 head dim——即把「每个 group 的 rope 部分」与「每个 group 的投影权重」做矩阵乘。

**vLLM 代码位置**：`vllm/models/deepseek_v4/nvidia/ops/o_proj.py`：
- `compute_fp8_einsum_recipe()`（13 行）：按 capability 选 recipe——`cap.major <= 9` 用 `(1, 128, 128)`（FP32 block scale，MN 粒度 128）+ `tma_aligned_scales=False`；`>= 10` 用 `(1, 1, 128)`（INT32 packed UE8M0 scale）+ `tma_aligned_scales=True`。
- `deep_gemm_fp8_o_proj()`（28 行）：先调 `fused_inv_rope_fp8_quant`（Triton 内核，`vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`）把 o 量化成 FP8，然后调 **DeepGEMM 的 `fp8_einsum`**（从 `vllm.utils.deep_gemm` import）。

**上游绑定**：`fp8_einsum` 是 DeepGEMM 内核，需要 TMA（SM90+ 的硬件单元，Ada 没有）和 FP8 张量核；SM100+ 上还用 UE8M0 microscaling scale。

**SM89 处理**：新增 Triton FP8 einsum（`fp8_einsum.py`）。SM89 上 Triton 不能把 FP8 operand 直接喂给 `tl.dot`（Ada 的 Triton FP8 dot 支持不完整），所以先转 bf16、再用 TF32 精度 dot；scale 用 FP32 block scale（`tma_aligned_scales=False`）。

#### 1.2.4 FP4 专家 MoE + mHC（Hyper Connection）

**它是什么**：MoE（Mixture of Experts）每一层有几百个「专家」（小 FFN），router 给每个 token 挑 top-k 个专家计算。DeepSeek V4 的专家权重是 **FP4（4-bit）量化**的。mHC（Hyper Connection）是 V4 特有的结构：把输入直接「超连接」到 FFN 输出附近，需要额外的 pre/post GEMM。

**vLLM 代码位置**：
- 专家后端选择：`vllm/model_executor/layers/fused_moe/oracle/mxfp4.py::select_deepseek_v4_mxfp4_moe_backend`（566 行）——MXFP4 MoE 后端选择器；Marlin 实现在 `marlin_moe`（211 行）/ `marlin_utils_fp4`（741 行，`prepare_moe_mxfp4_layer_for_marlin`）。权重量化方法在 `vllm/models/deepseek_v4/quant_config.py`（`Mxfp4MoEMethod`，152 行）。
- mega_moe（DeepGEMM 的融合 MoE）：`vllm/models/deepseek_v4/nvidia/model.py` 的 `use_mega_moe`（524 行）与 `_init_mega_moe_experts`（605 行），运行时调 `deep_gemm.fp8_fp4_mega_moe`（498 行）。
- mHC 的 pre/post GEMM：DeepGEMM 提供 `tf32_hc_prenorm_gemm`（`vllm/utils/deep_gemm.py` 的 `_tf32_hc_prenorm_gemm_impl`）。

**上游绑定**：FP4 专家权重在 SM100+ 上用原生 FP4 MMA（Blackwell 张量核支持 FP4）；mega_moe 是 DeepGEMM 的融合内核，只支持 SM90+；mHC 的 pre/post GEMM 用 DeepGEMM / TileLang 的 TF32 内核。

**SM89 处理**：Ada 没有 FP4 张量核，FP4 权重只能反量化回 FP16 走 **Marlin WNA16**（Marlin 是 vLLM 内置的权重 GEMM 内核，SM89 的 cutlass dispatch 早已支持）。mHC 的 pre/post GEMM 用新增 Triton TF32 内核（`sm12x_mqa.py::tf32_hc_prenorm_gemm_triton`）。

#### 1.2.5 辅助算子

**它是什么**：大算子周围还有几个小算子：
- **Indexer 的 rope+quant**：给 query 加 RoPE 位置编码并量化成 FP8（`vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`）。
- **KV 反量化与 gather**：按块表从 KV cache 里取出需要的 K 并反量化（`vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`）。
- **KV 压缩**：把新算出的 KV 压缩成 MLA 的低秩 latent 存进 cache（`vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py`）。

**上游绑定**：SM100+ 上用 CuTe-DSL 实现（用到 SM90+ 指令如 TMA / wgmma）。这些算子都有既有的 Triton/torch 降级实现。

**SM89 处理**：把 `has_cutedsl()` 在 SM89 上改成返回 `False`（`vllm/utils/import_utils.py:547`），所有 `… and has_cutedsl()` 调用点自动走既有降级，无需逐个改。该改动只影响 SM89，其他架构照旧。

### 1.3 上游实现与硬件绑定一览

| 子系统 | 上游目标硬件 | 绑定的实现 | 在 SM89 上的处理 |
|---|---|---|---|
| Sparse MLA 本地注意力 | SM90/SM100 | FlashInfer sparse MLA（SM120 JIT） | 复用同一套，靠 FlashInfer SM89 版开放到 8.9 |
| Lightning Indexer（FP8 MQA logits） | SM100+ | DeepGEMM `fp8_fp4_paged_mqa_logits` | 新增 Triton 直出 top-k（不物化完整 logits） |
| o_proj FP8 einsum | SM100+ | DeepGEMM `fp8_einsum`（需要 TMA / FP8 张量核） | 新增 Triton FP8 einsum（SM89 上 FP8→bf16 再算） |
| mHC pre/post GEMM | SM100+ | DeepGEMM / TileLang | Triton TF32 降级 |
| Indexer rope+quant / KV dequant / compress | SM100+ | CuTe-DSL（SM90+ 指令） | 关闭 `has_cutedsl()` → 走既有 Triton/torch 降级 |
| FP4 专家 MoE | SM100+ | DeepGEMM / FlashInfer-CUTLASS FP4 | Marlin WNA16（FP4→FP16 反量化） |

### 1.4 硬件约束

**为什么 Ada 是这些算子的「下限」**：

- Ada **有 FP8 张量核**（e4m3/e5m2），所以 FP8 类算子（MQA logits、einsum、sparse MLA 的 FP8 KV cache）都能算，只是 Triton 的 FP8 `tl.dot` 支持不完整，需要 bf16 / TF32 变通。
- Ada **没有 FP4 张量核**，也没有硬件 **microscaling（E8M0 / UE8M0）MMA**——E8M0 是 SM100+ 才有的 8-bit 指数格式，用作 block scale。
- 因此 **FP4 专家 MoE 无法使用原生 FP4 MMA**，只能反量化回 FP16 走 Marlin——**这是性能瓶颈，不是实现缺陷**。预期单并发 decode 约 82 tok/s（4×RTX 4090，见第 0 章）。

---

## 2. 总体设计

改动沿两条轴展开：

**轴 A —— 让 SM89 能使用 sparse-MLA 本地注意力。**
由于内核本身已有 SM89 版（外部依赖），仓库内只需：

1. 新增探测函数 `has_flashinfer_sparse_mla_sm89()`；
2. 让 FlashInfer 后端/实现的 capability 检查接受 `(8, 9)`；
3. 把 SM89 的注意力类选择路由到既有的 `DeepseekV4FlashInferSM120Attention`。

**轴 B —— 把 DeepGEMM / CuTe-DSL 专属算子从 SM89 上“摘下”。**
不降级这些算子会在 Ada 上失败或不正确。仓库内通过两个“单点开关”加三个“新降级模块”完成：

- `has_cutedsl()` 在 `(8, 9)` 上返回 `False`：一处改动即可让所有 CuTe-DSL 调用点（indexer rope+quant、KV dequant、compressor、inverse-rope-quant、qk-rmsnorm）自动走既有可移植降级。
- `is_mqa_backend_available()` / `_use_sm12x_mqa_fallback()`：让 Lightning Indexer 在非 DeepGEMM 时走新的 Triton 直出 top-k 路径。
- o_proj 的 FP8 einsum 走新增的 `fp8_einsum.py` Triton 内核。

设计原则：

- **最小侵入**：能复用既有降级就不新增；新增内核模块保持自包含，只依赖 vLLM 既有工具函数（`vllm.platforms`、`vllm.triton_utils`、`vllm.utils.deep_gemm`、`vllm.distributed`）。
- **不影响其他硬件**：所有门控都以 capability 为条件，SM90/SM100 的路径保持不变。

---

## 3. 现状审视：当前代码已具备 / 需新增

### 3.1 当前代码已具备（改方案中只需复用，不用动）

| 能力 | 位置 |
|---|---|
| DeepSeek V4 完整模型管线 | `vllm/models/deepseek_v4/` |
| FlashInfer sparse-MLA 的 SM120 后端与实现 | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`、`flashinfer_mla_sparse_sm120.py` |
| DeepSeek V4 专用 FlashInfer 后端与 SM120 注意力层 | `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` |
| capability 门控基建 | `vllm/platforms`（`is_device_capability_family`、`is_device_capability` 等） |
| SM120 sparse-MLA 探测 | `vllm/utils/flashinfer.py::has_flashinfer_sparse_mla_sm120` |
| Marlin WNA16（含 SM89 cutlass dispatch）与 Mxfp4→Marlin 能力门控 | `csrc/.../marlin*`、`vllm/model_executor/layers/quantization/`、`fused_moe/oracle/mxfp4.py` |
| 索引器 scheduler-metadata 的 DeepGEMM 门控 | `vllm/v1/attention/backends/mla/indexer.py`（用 `has_deep_gemm()`，SM89 上天然为 `False`） |

### 3.2 当前代码缺什么（本方案要补）

1. SM89 的 FlashInfer sparse-MLA 探测（`has_flashinfer_sparse_mla_sm89`）。
2. `has_cutedsl()` 在 SM89 上禁用（否则会误选 CuTe-DSL）。
3. Lightning Indexer 的非 DeepGEMM MQA 路径（当前 `SparseAttnIndexer` 构造器在非 DeepGEMM 上直接 `raise`）。
4. o_proj 的非 DeepGEMM FP8 einsum（当前 `o_proj.py` 直接调用 DeepGEMM `fp8_einsum`）。
5. 各 FlashInfer 后端/实现/注意力层对 `(8, 9)` 的 capability 接受。

---

## 4. 改动清单总览

| 文件 | 类型 | 内容 |
|---|---|---|
| `vllm/utils/flashinfer.py` | 改 | 新增 `has_flashinfer_sparse_mla_sm89()` |
| `vllm/utils/import_utils.py` | 改 | `has_cutedsl()` 在 `(8,9)` 返回 `False` |
| `vllm/models/deepseek_v4/compressor.py` | 改 | head=512 压缩分支加 `has_cutedsl()` 门控（该分支当前不查 `has_cutedsl()`，SM89 上会误入 cutedsl） |
| `vllm/utils/deep_gemm.py` | 改 | 新增 `_use_sm12x_mqa_fallback()`、`is_mqa_backend_available()`、`fp8_fp4_mqa_topk_indices()`、`fp8_fp4_paged_mqa_topk_indices()` |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 改 | 构造器放宽 + prefill/decode 接入直出 top-k |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | 改 | 改走 Triton FP8 einsum 降级路径 |
| `vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py` | **新增** | SM89/SM12x Triton FP8 einsum（o_proj 降级） |
| `vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py` | **新增** | Triton MQA / paged-MQA / mHC prenorm 内核 |
| `vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py` | **新增** | 直出 top-k、torch/triton MQA topk、mHC TF32 降级 |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py` | 改 | `FlashInferMLASparseSM120Backend` 接受 `(8,9)` |
| `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | 改 | `FlashInferMLASparseSM120Impl` 接受 `(8,9)` |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | 改 | 后端与 SM120 注意力层接受 `(8,9)` |
| `vllm/models/deepseek_v4/nvidia/model.py` | 改 | `_select_dsv4_attn_cls` 把 `(8,9)` 路由到 SM120 注意力 |
| `tests/`、`benchmarks/sparse_mla_sm89_port/` | **新增** | 见第 6 节 |

---
## 5. 分步实现

实现顺序：**5.0 环境 → 5.1 → 5.2 → 5.3+5.4 → 5.5 → 5.6 → 5.7 → 5.8 → 第 6 节验证**。

### 5.0 环境与前置依赖

三个新内核模块只依赖 `torch`、`vllm.platforms`、`vllm.triton_utils`、`vllm.utils.deep_gemm`、`vllm.distributed`、`fp8_utils`——这些在 v0.26.0 里都已存在，可直接使用。但 FlashInfer 的 SM89 内核在仓库外，必须先装好：

```bash
# 安装 FlashInfer 0.6.14 的 SM89 sparse-MLA 版（必需，否则后端拒绝启动）
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install torch==2.11.0 flashinfer-cubin==0.6.13 --torch-backend=cu130
uv pip install /path/to/flashinfer_python-0.6.14*sm89*.whl
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

面向 SM89 编译：

```bash
export VLLM_TARGET_DEVICE=cuda
export CUDA_HOME=/usr/local/cuda-13.0
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
```

> 依赖关系：`has_flashinfer_sparse_mla_sm89()` 探测的正是「该 SM89 版 FlashInfer」；没有它，第 5.7 节的后端会直接抛错。官方 0.6.14 不含 SM89 补丁，运行时需 `FLASHINFER_DISABLE_VERSION_CHECK=1`。

### 5.1 `vllm/utils/flashinfer.py`：新增探测函数

在 `has_flashinfer_sparse_mla_sm120()` 之后新增。逻辑＝先确认 SM120 接口存在，再确认真实设备在 FlashInfer 内部被解析为 sparse 后端：

```python
@functools.cache
def has_flashinfer_sparse_mla_sm89() -> bool:
    """Return whether the installed FlashInfer enables sparse MLA on SM89."""
    if not has_flashinfer_sparse_mla_sm120():
        return False
    try:
        from flashinfer.mla._core import _resolve_dsv4_sparse_mla_backend

        device = torch.device(
            "cuda", torch.accelerator.current_device_index()
        )
        return _resolve_dsv4_sparse_mla_backend(device) == "sparse"
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return False
```

### 5.2 `vllm/utils/import_utils.py`：在 SM89 上关闭 CuTe-DSL

改动目标：只要 GPU 是 SM89，就认为 CuTe-DSL 不可用（即使装了 cutlass），让所有 `… and has_cutedsl()` 调用点自动走既有 Triton/torch 降级。

把现有实现：

```python
def has_cutedsl() -> bool:
    """Whether the optional `cutelass` package is available."""
    return _has_module("cutlass")
```

改为：

```python
def has_cutedsl() -> bool:
    """Whether the optional `cutlass` package is available."""
    if not _has_module("cutlass"):
        return False

    # CuTe-DSL kernels used by DeepSeek V4 target SM90+ instructions. Ada SM89
    # must use the existing Triton/torch fallbacks instead.
    from vllm.platforms import current_platform

    return not (
        current_platform.is_cuda()
        and current_platform.is_device_capability((8, 9))
    )
```

这一处改动覆盖了**真正以 `has_cutedsl()` 为门控**的全部调用点：`fused_indexer_q.py`（353、410 行，两条都有 `else` Triton 降级）、`cache_utils.py`（403 行，`dequantize_and_gather_k_cache`，Triton 降级在同函数内）、`sparse_attn_indexer.py`（57 行，DCP merge 的 fail-closed 断言，见下）。这些无需逐个改。

> ⚠️ **必须额外处理：`compressor.py` 不在 `has_cutedsl()` 门控体系内。**
> `vllm/models/deepseek_v4/compressor.py:387` 的分支是 `if current_platform.is_cuda() and self.head_dim == 512:`——**不查 `has_cutedsl()`**，直接 `from .nvidia.ops.sparse_attn_compress_cutedsl import compress_norm_rope_store_cutedsl`。DSv4-Flash 的 `head_dim == 512`（`DeepseekV4FlashInferMLASparseBackend.get_supported_head_sizes` 返回 `[512]`），所以 SM89 上此分支必然命中：装了 `cutlass` 会执行 SM90+ 指令（运行失败/错误），没装则 `ImportError`。必须把该条件改为：
>
> ```python
> if (
>     current_platform.is_cuda()
>     and self.head_dim == 512
>     and has_cutedsl()   # SM89 上为 False → 落到下方 triton 分支
> ):
>     ...
> ```
>
> 并验证 `compress_norm_rope_store_triton`（`compressor.py:412` 的 `else` 分支）对 `head_dim=512` 的支持——该 Triton 内核当前注释说用于 indexer（head=128）/AMD，需确认其 scale 布局与 cutedsl 版在 head=512 时一致，必要时为 SM89 新增 head=512 Triton 压缩路径。

> ⚠️ **另外两个文件不在 `has_cutedsl()` 体系内，无需（也无法）通过本开关覆盖：**
> - `fused_inv_rope_fp8_quant.py`：不查 `has_cutedsl()`，走 `tma_aligned_scales` 参数（`vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:160,198`）。SM89 由 5.6 节的 `compute_fp8_einsum_recipe()` 传 `tma_aligned_scales=False`，已覆盖；其调用的 `get_tma_aligned_size` 是纯 Python 对齐函数（`vllm/utils/deep_gemm.py:653`），不依赖 DeepGEMM，SM89 上不会抛错。
> - `fused_qk_rmsnorm.py`：纯 Triton 内核，本就没有 CuTe-DSL 分支，不受影响。

> 验证：SM89 上 `vllm.utils.import_utils.has_cutedsl()` 应为 `False`。

### 5.3 `vllm/utils/deep_gemm.py`：新增 MQA 降级开关与直出 top-k 入口

在 `is_deep_gemm_supported()` 附近新增判定与可用性函数：

```python
def _use_sm12x_mqa_fallback() -> bool:
    """Use portable MQA/HC kernels where DeepGEMM is unsupported."""
    return current_platform.is_device_capability_family(120) or (
        current_platform.is_cuda()
        and current_platform.is_device_capability((8, 9))
    )


def is_mqa_backend_available() -> bool:
    """Whether MQA kernels are available through DeepGEMM or a fallback."""
    return has_deep_gemm() or _use_sm12x_mqa_fallback()
```

再新增两个「尝试直出 top-k」的入口函数，实际实现委托给 5.4 的新模块。它们只在 SM89/SM12x 且拷贝 FP8 路径（`q[1] is None`）时生效，其余情况返回 `False` 让上层走原路径：

```python
def fp8_fp4_mqa_topk_indices(
    q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
) -> bool:
    """Write SM89/SM12x FP8 MQA top-k indices without full logits."""
    if not (
        current_platform.is_cuda()
        and _use_sm12x_mqa_fallback()
        and q[1] is None
    ):
        return False
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks.fp8_fp4_mqa_topk_indices(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
    )


def fp8_fp4_paged_mqa_topk_indices(
    q, kv_cache, weights, context_lens, block_tables,
    max_model_len, topk_indices,
) -> bool:
    """Write SM89/SM12x FP8 paged-MQA top-k indices without full logits."""
    if not (
        current_platform.is_cuda()
        and _use_sm12x_mqa_fallback()
        and q[1] is None
    ):
        return False
    from vllm.models.deepseek_v4.nvidia.ops import sm12x_deep_gemm_fallbacks

    return sm12x_deep_gemm_fallbacks.fp8_fp4_paged_mqa_topk_indices(
        q, kv_cache, weights, context_lens, block_tables,
        max_model_len, topk_indices,
    )
```

注意：`fp8_fp4_mqa_logits` / `get_paged_mqa_logits_metadata` / `fp8_fp4_paged_mqa_logits` 这些 DeepGEMM 函数保持原样，SM89 路径不经过它们；上面两个新入口是并行的“先尝试”路径。

### 5.4 新增三个自包含内核模块

三个文件都放在 `vllm/models/deepseek_v4/nvidia/ops/` 下，且**只 import 0.26 已有的模块**，可独立编译/单测。下面 5.4.1–5.4.3 给出每个模块的职责、对外接口与**完整参考实现**（放在第 5.4.4 代码段中，可直接落盘）。

- `fp8_einsum.py`：为 `o_proj` 提供 SM89/SM12x 的 FP8 einsum，替代 DeepGEMM `fp8_einsum`。核心是 Triton 内核：SM12x 用原生 FP8 张量核 `tl.dot`，SM89 把 FP8 operand 先转 bf16、再用 TF32 `tl.dot`（Ada 的 Triton 无法直接把 FP8 operand 喂给 `tl.dot`）；scale 布局在 SM89/SM12x 用 FP32 block scale（`tma_aligned_scales=False`）。
- `sm12x_mqa.py`：纯 Triton 的内核层。提供单序列 MQA logits、paged-MQA logits（含从打包 uint8 KV 布局拆 FP8 值与 fp32 scale）、以及 mHC prenorm 的 TF32 GEMM。
- `sm12x_deep_gemm_fallbacks.py`：在 `sm12x_mqa.py` 之上做更上层的降级封装。提供“不物化完整 logits、分块算 logits + `torch.topk` 合并、直接写 top-k 索引”的 `fp8_fp4_mqa_topk_indices` / `fp8_fp4_paged_mqa_topk_indices`，以及单序列非 paged 的降级和 mHC prenorm 降级。

对外接口契约：

```text
# sm12x_mqa.py
fp8_mqa_logits_triton(q, k, scale, weights, cu_seqlen_ks, cu_seqlen_ke,
                      max_seq_len, num_heads, head_dim) -> Tensor[fp32]
fp8_paged_mqa_logits_triton(q, kv_cache, weights, context_lens, block_tables,
                            max_model_len, token_start, token_count) -> Tensor[fp32]
tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split) -> Tensor

# sm12x_deep_gemm_fallbacks.py
fp8_fp4_mqa_topk_indices(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke,
                         topk_indices) -> bool
fp8_fp4_paged_mqa_topk_indices(q, kv_cache, weights, context_lens,
                               block_tables, max_model_len,
                               topk_indices) -> bool

# fp8_einsum.py
deepseek_v4_fp8_einsum_config(capability_major) -> (einsum_recipe, tma_aligned_scales)
deepseek_v4_fp8_einsum(o_fp8, o_scale, wo_a_w, wo_a_scale, out, "bhr,hdr->bhd",
                       einsum_recipe) -> None
use_deepseek_v4_sm120_cutlass_fp8_einsum(num_tokens, num_groups, out_dim) -> bool
```

（三个模块的完整实现见 5.4.4。）

### 5.4.4 完整参考实现

以下三份代码分别对应 `fp8_einsum.py`、`sm12x_mqa.py`、`sm12x_deep_gemm_fallbacks.py`，可直接保存为独立文件。它们自包含，仅依赖 vLLM 既有工具，无需其他改动即可 `import`。

---

#### 文件 1：`vllm/models/deepseek_v4/nvidia/ops/fp8_einsum.py`

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM89/SM12x Triton FP8 einsum kernels for DeepSeek V4."""

import torch

from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import fp8_einsum


@triton.jit
def _deepseek_v4_sm12x_fp8_einsum_kernel(
    a_ptr,
    a_scale_ptr,
    b_ptr,
    b_scale_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_groups: tl.constexpr,
    out_rank: tl.constexpr,
    hidden_size: tl.constexpr,
    a_stride_token: tl.constexpr,
    a_stride_group: tl.constexpr,
    a_stride_hidden: tl.constexpr,
    a_scale_stride_token: tl.constexpr,
    a_scale_stride_group: tl.constexpr,
    a_scale_stride_hidden: tl.constexpr,
    b_stride_group: tl.constexpr,
    b_stride_out: tl.constexpr,
    b_stride_hidden: tl.constexpr,
    b_scale_stride_group: tl.constexpr,
    b_scale_stride_out: tl.constexpr,
    b_scale_stride_hidden: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_group: tl.constexpr,
    out_stride_rank: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    UPCAST_FP8: tl.constexpr,
) -> None:
    token_block = tl.program_id(0)
    out_block = tl.program_id(1)
    group = tl.program_id(2)

    token_offsets = token_block * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
    accum = tl.zeros((BLOCK_TOKENS, BLOCK_OUT), dtype=tl.float32)

    for hidden_start in range(0, hidden_size, BLOCK_HIDDEN):
        hidden = hidden_start + hidden_offsets
        a = tl.load(
            a_ptr
            + token_offsets[:, None] * a_stride_token
            + group * a_stride_group
            + hidden[None, :] * a_stride_hidden,
            mask=(token_offsets[:, None] < num_tokens)
            & (hidden[None, :] < hidden_size),
            other=0.0,
        )
        b = tl.load(
            b_ptr
            + group * b_stride_group
            + out_offsets[None, :] * b_stride_out
            + hidden[:, None] * b_stride_hidden,
            mask=(out_offsets[None, :] < out_rank) & (hidden[:, None] < hidden_size),
            other=0.0,
        )
        if UPCAST_FP8:
            # SM89/Ada: Triton may not lower an FP8-operand tl.dot on this arch,
            # so upcast to bf16 before the MMA. SM12x keeps the native FP8 dot.
            a = a.to(tl.bfloat16)
            b = b.to(tl.bfloat16)
        raw = tl.dot(a, b, out_dtype=tl.float32)
        hidden_scale_block = hidden_start // BLOCK_HIDDEN
        a_scale = tl.load(
            a_scale_ptr
            + token_offsets * a_scale_stride_token
            + group * a_scale_stride_group
            + hidden_scale_block * a_scale_stride_hidden,
            mask=token_offsets < num_tokens,
            other=0.0,
        )
        b_scale = tl.load(
            b_scale_ptr
            + group * b_scale_stride_group
            + (out_offsets // 128) * b_scale_stride_out
            + hidden_scale_block * b_scale_stride_hidden,
            mask=out_offsets < out_rank,
            other=0.0,
        )
        accum += raw * a_scale[:, None] * b_scale[None, :]

    tl.store(
        out_ptr
        + token_offsets[:, None] * out_stride_token
        + group * out_stride_group
        + out_offsets[None, :] * out_stride_rank,
        accum,
        mask=(token_offsets[:, None] < num_tokens) & (out_offsets[None, :] < out_rank),
    )


def deepseek_v4_sm12x_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Compute ``bhr,hdr->bhd`` with FP32 block scales on SM89/SM12x.

    ``a`` is the transposed output of ``fused_inv_rope_fp8_quant`` with shape
    ``[tokens, groups, hidden]``. ``b`` is ``wo_a`` reshaped to
    ``[groups, out_rank, hidden]``.
    """
    num_tokens, num_groups, hidden_size = a.shape
    b_groups, out_rank, b_hidden_size = b.shape
    assert b_groups == num_groups
    assert b_hidden_size == hidden_size
    assert out.shape == (num_tokens, num_groups, out_rank)
    assert hidden_size % 128 == 0
    assert out_rank % 128 == 0
    assert a.dtype == torch.float8_e4m3fn
    assert b.dtype == torch.float8_e4m3fn
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if a_scale.dtype == e8m0_dtype:
        a_scale = _upcast_e8m0_to_fp32(a_scale)
    if b_scale.dtype == e8m0_dtype:
        b_scale = _upcast_e8m0_to_fp32(b_scale)
    assert a_scale.dtype == torch.float32
    assert b_scale.dtype == torch.float32

    if num_tokens == 0:
        return

    block_tokens = 16
    block_out = 128
    block_hidden = 128
    # SM12x has native FP8 tensor-core dot; SM89/Ada upcasts FP8->bf16 first.
    upcast_fp8 = not current_platform.is_device_capability_family(120)
    grid = (
        triton.cdiv(num_tokens, block_tokens),
        triton.cdiv(out_rank, block_out),
        num_groups,
    )
    _deepseek_v4_sm12x_fp8_einsum_kernel[grid](
        a,
        a_scale,
        b,
        b_scale,
        out,
        num_tokens,
        num_groups,
        out_rank,
        hidden_size,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        a_scale.stride(0),
        a_scale.stride(1),
        a_scale.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_TOKENS=block_tokens,
        BLOCK_OUT=block_out,
        BLOCK_HIDDEN=block_hidden,
        UPCAST_FP8=upcast_fp8,
        num_warps=4,
        num_stages=3,
    )


def deepseek_v4_fp8_einsum_config(
    capability_major: int,
) -> tuple[tuple[int, int, int], bool]:
    if capability_major == 10:
        return (1, 1, 128), True
    return (1, 128, 128), False


def _deepseek_v4_sm120_cutlass_compiled(capability: int) -> bool:
    try:
        return torch.ops._C.deepseek_v4_fp8_bmm_sm120_supported(capability)
    except AttributeError:
        return False


def use_deepseek_v4_sm120_cutlass_fp8_einsum(
    num_tokens: int,
    num_groups: int,
    out_rank: int,
    hidden_size: int,
) -> bool:
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major == 12
        and _deepseek_v4_sm120_cutlass_compiled(capability.to_int())
        and num_tokens >= 256
        and num_groups in (2, 8)
        and out_rank == 1024
        and hidden_size == 4096
    )


def _use_deepseek_v4_sm12x_triton_fp8_einsum(
    equation: str,
    recipe: list[int],
    b_scale: torch.Tensor,
) -> bool:
    capability = current_platform.get_device_capability()
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    # SM12x (major 12) and SM89/Ada (8.9) both use the Triton FP8 einsum here:
    # they lack the SM90/SM100-only DeepGEMM fp8_einsum kernel.
    is_supported_arch = capability is not None and (
        capability.major == 12 or (capability.major, capability.minor) == (8, 9)
    )
    return (
        is_supported_arch
        and equation == "bhr,hdr->bhd"
        and tuple(recipe) == (1, 128, 128)
        and b_scale.dtype in (torch.float32, e8m0_dtype)
    )


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    if equation == "bhr,hdr->bhd" and b.dim() == 2:
        num_groups = out.shape[1]
        out_rank = out.shape[2]
        hidden_size = a.shape[2]
        if b.shape[0] % out_rank != 0:
            raise RuntimeError(
                "DeepSeek V4 fp8 einsum weight rows must be divisible by "
                f"out_rank={out_rank}, got {b.shape[0]}"
            )
        b_groups = b.shape[0] // out_rank
        group_start = 0
        if b_groups != num_groups:
            if b_groups % num_groups != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum weight groups must match the "
                    "TP-local output groups or be an integer multiple of "
                    f"them, got weight_groups={b_groups}, "
                    f"output_groups={num_groups}"
                )
            group_partitions = b_groups // num_groups
            group_start = (
                get_tensor_model_parallel_rank() % group_partitions
            ) * num_groups
        b = b.view(b_groups, out_rank, hidden_size)
        if group_start != 0 or b_groups != num_groups:
            b = b.narrow(0, group_start, num_groups)

        if b_scale.dim() == 2:
            scale_mn = recipe[1]
            scale_k_pack = 4 if b_scale.dtype == torch.int32 else 1
            scale_k = recipe[2] * scale_k_pack
            scale_out_blocks = (out_rank + scale_mn - 1) // scale_mn
            scale_hidden_blocks = (hidden_size + scale_k - 1) // scale_k
            if b_scale.shape[0] % scale_out_blocks != 0:
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale rows must be divisible by "
                    f"scale_out_blocks={scale_out_blocks}, "
                    f"got {b_scale.shape[0]}"
                )
            scale_groups = b_scale.shape[0] // scale_out_blocks
            if scale_groups not in (num_groups, b_groups):
                raise RuntimeError(
                    "DeepSeek V4 fp8 einsum scale groups must match the "
                    "TP-local output groups or weight groups, got "
                    f"scale_groups={scale_groups}, output_groups={num_groups}, "
                    f"weight_groups={b_groups}"
                )
            b_scale = b_scale.view(
                scale_groups,
                scale_out_blocks,
                scale_hidden_blocks,
            )
            if scale_groups == b_groups and scale_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)
        elif b_scale.dim() == 3 and b_scale.shape[0] == b_groups:
            if b_groups != num_groups:
                b_scale = b_scale.narrow(0, group_start, num_groups)

        use_sm120_cutlass = (
            tuple(recipe) == (1, 128, 128)
            and a_scale.dtype == torch.float32
            and b_scale.dtype == torch.float32
            and use_deepseek_v4_sm120_cutlass_fp8_einsum(
                a.shape[0],
                num_groups,
                out_rank,
                hidden_size,
            )
        )
        if use_sm120_cutlass:
            a_group_major = a.transpose(0, 1)
            a_scale_cutlass = a_scale.transpose(0, 1).permute(0, 2, 1)
            b_cutlass = b.transpose(1, 2)
            if (
                a_group_major.is_contiguous()
                and a_scale_cutlass.is_contiguous()
                and b_scale.is_contiguous()
                and out.is_contiguous()
            ):
                torch.ops._C.deepseek_v4_fp8_bmm_sm120(
                    out,
                    a_group_major,
                    b_cutlass,
                    a_scale_cutlass,
                    b_scale,
                )
                return

        if _use_deepseek_v4_sm12x_triton_fp8_einsum(equation, recipe, b_scale):
            deepseek_v4_sm12x_fp8_einsum(a, a_scale, b, b_scale, out)
            return

    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))
```

---

#### 文件 2：`vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py`

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fallback kernels used by the local DeepSeek V4 path."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions")
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


@triton.jit
def _fp8_mqa_logits_kernel(
    q_ptr,
    k_ptr,
    scale_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    num_q: tl.constexpr,
    seq_len_kv: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_q
    valid_n = offs_n < seq_len_kv
    seq_start = tl.load(cu_seqlen_ks_ptr + offs_m, mask=valid_m, other=0)
    seq_end = tl.load(cu_seqlen_ke_ptr + offs_m, mask=valid_m, other=0)
    seq_mask = (offs_n[None, :] >= seq_start[:, None]) & (
        offs_n[None, :] < seq_end[:, None]
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + offs_m[:, None] * stride_qm
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            )
            k = tl.load(
                k_ptr + offs_n[:, None] * stride_kn + d[None, :] * stride_kd,
                mask=valid_n[:, None] & (d[None, :] < head_dim),
                other=0.0,
            )
            if NATIVE_FP8:
                scores += tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            else:
                scores += tl.dot(
                    q.to(tl.float32),
                    tl.trans(k.to(tl.float32)),
                    input_precision="tf32",
                )
        scale = tl.load(scale_ptr + offs_n, mask=valid_n, other=0.0)
        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(seq_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    native_fp8: bool | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    k_fp8, scale = kv
    num_q, num_heads, head_dim = q.shape
    seq_len_kv = k_fp8.shape[0]
    if out is None:
        logits = torch.empty(
            (num_q, seq_len_kv),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        if (
            out.shape != (num_q, seq_len_kv)
            or out.dtype != torch.float32
            or out.device != q.device
        ):
            raise ValueError(
                "FP8 MQA logits output must have shape "
                f"{(num_q, seq_len_kv)}, dtype float32, and device {q.device}"
            )
        logits = out
    if num_q == 0 or seq_len_kv == 0:
        return logits

    if native_fp8 is None:
        native_fp8 = current_platform.is_device_capability_family(120)
    block_m = _fp8_mqa_logits_block_m(num_q, seq_len_kv)
    grid = (triton.cdiv(num_q, block_m), triton.cdiv(seq_len_kv, 128))
    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        num_q,
        seq_len_kv,
        num_heads,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_fp8.stride(0),
        k_fp8.stride(1),
        weights.stride(0),
        weights.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_D=64,
        NATIVE_FP8=native_fp8,
        num_warps=4,
    )
    return logits


def _fp8_mqa_logits_block_m(num_q: int, seq_len_kv: int) -> int:
    if seq_len_kv <= 16 * 1024:
        return 16
    return 64


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows: tl.constexpr,
    logits_width: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    ).to(tl.int64)

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset[None, :] * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[:, :, None] * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=context_mask[:, :, None] & (d[None, None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(context_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows: tl.constexpr,
    logits_width: tl.constexpr,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel optimised for long ``token_count``.

    Each Triton program handles one logical row (``batch * next_n + q_pos``)
    across a ``BLOCK_N``-wide window of token positions. Q is loaded once per
    head tile and reused for every K element in the window, which preserves
    L2 / register locality and avoids the M-axis padding waste of the
    generic 2D-tiled kernel at long contexts (mt-bench c=1 MTP=2 num_rows=3
    with token_count=131072 launches 12k programs of 128 logits each rather
    than 8k programs of 64 logits with 25 % M-axis waste).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    ).to(tl.int64)

    scale = tl.load(
        scale_ptr + block_idx * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_pos * stride_qn
                + heads[:, None] * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[None, :] * stride_kvb
                + block_offset[None, :] * stride_kvs
                + d[:, None] * stride_kvd,
                mask=context_mask[None, :] & (d[:, None] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    """Rowwise paged-MQA logits wrapper.

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` inside ``_fp8_paged_mqa_logits_rowwise_kernel`` lands on
    tensor-core friendly tile shapes. DSv4-Flash (head_dim=128,
    num_heads=64) satisfies both and is the only model that exercises this
    path today; the generic 2D kernel below remains the fallback for
    misaligned shapes.
    """
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = torch.empty(
        (num_rows, token_count),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # Aligned head shapes (DSv4-Flash and any future MQA model with
    # ``head_dim % 64 == 0`` and ``num_heads % 4 == 0``) get the rowwise
    # kernel, which keeps long-context decode (>100K tokens) on a per-row
    # grid that re-uses Q across the full token window. The generic 2D
    # kernel below still handles misaligned shapes and remains the canonical
    # reference for the rowwise variant.
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = torch.empty(
        (num_rows, token_count),
        device=q.device,
        dtype=torch.float32,
    )
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    # Adaptive BLOCK_M: the kernel masks off positions >= num_rows, so a fixed
    # BLOCK_M=4 wastes ~75% of M-axis work in the common single-stream decode
    # case (num_rows=1). Pick the smallest power-of-2 tile that still covers
    # num_rows so we keep one grid-program for typical decode while still
    # benefiting from larger tiles when batch / MTP push num_rows higher.
    if num_rows <= 1:
        block_m = 1
    elif num_rows <= 2:
        block_m = 2
    elif num_rows <= 4:
        block_m = 4
    else:
        block_m = 8
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(token_count, 64))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


@triton.jit
def _tf32_hc_prenorm_gemm_kernel(
    x_ptr,
    fn_ptr,
    out_ptr,
    sqrsum_ptr,
    M,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_fnn: tl.constexpr,
    stride_fnk: tl.constexpr,
    stride_outs: tl.constexpr,
    stride_outm: tl.constexpr,
    stride_outn: tl.constexpr,
    stride_sqs: tl.constexpr,
    stride_sqm: tl.constexpr,
    NUM_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    split_k = tl.cdiv(K, NUM_SPLIT)
    split_begin = pid_s * split_k
    split_end = tl.minimum(split_begin + split_k, K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k0 in tl.range(0, split_k, BLOCK_K):
        k = split_begin + k0 + offs_k
        k_mask = k < split_end
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        fn = tl.load(
            fn_ptr + offs_n[None, :] * stride_fnn + k[:, None] * stride_fnk,
            mask=(offs_n[None, :] < N) & k_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(x, fn, input_precision="tf32", out_dtype=tl.float32)
        sq += tl.sum(x * x, axis=1)

    tl.store(
        out_ptr
        + pid_s * stride_outs
        + offs_m[:, None] * stride_outm
        + offs_n[None, :] * stride_outn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )

    if pid_n == 0:
        tl.store(
            sqrsum_ptr + pid_s * stride_sqs + offs_m * stride_sqm,
            sq,
            mask=offs_m < M,
        )


def tf32_hc_prenorm_gemm_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> None:
    assert x.dim() == 2
    assert fn.dim() == 2
    assert out.dim() == 3
    assert sqrsum.dim() == 2

    m, k = x.shape
    n = fn.shape[0]
    assert fn.shape[1] == k
    assert out.shape == (num_split, m, n)
    assert sqrsum.shape == (num_split, m)

    if m == 0:
        return

    block_m = 16
    block_n = triton.next_power_of_2(n)
    block_n = min(max(block_n, 16), 32)
    block_k = 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n), num_split)
    _tf32_hc_prenorm_gemm_kernel[grid](
        x,
        fn,
        out,
        sqrsum,
        m,
        k,
        n,
        x.stride(0),
        x.stride(1),
        fn.stride(0),
        fn.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        sqrsum.stride(0),
        sqrsum.stride(1),
        num_split,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
```

---

#### 文件 3：`vllm/models/deepseek_v4/nvidia/ops/sm12x_deep_gemm_fallbacks.py`

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM12x fallback implementations for DeepGEMM-only interfaces."""

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SM120_MQA_LOGITS_MAX_SCORE_BYTES = 64 * 1024 * 1024
_SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES = 512 * 1024 * 1024
_SM120_MQA_TRITON_CHUNKED_TOPK_CHUNK_SIZE = 32768
_SM120_PAGED_MQA_TOPK_CHUNK_SIZE = 8192


def _use_sm12x_fallback() -> bool:
    """SM12x (Blackwell client) or SM89 (Ada): both run these portable Triton/
    torch fallbacks in place of the DeepGEMM-only kernels."""
    cp = current_platform
    return cp.is_device_capability_family(120) or (
        cp.is_cuda() and cp.is_device_capability((8, 9))
    )


def _top_k_per_row_prefill_op():
    try:
        from vllm import _custom_ops as _custom_ops  # noqa: F401

        return torch.ops._C.top_k_per_row_prefill
    except (AttributeError, ImportError, RuntimeError):
        return None


def _fp8_mqa_logits_head_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    num_heads: int,
) -> int:
    # The SM120 torch path is used on long prefill paths where materializing
    # [head_chunk, M, N] scores can otherwise allocate multiple GiB. Keep the
    # transient score tensor bounded, while still using larger head chunks for
    # short prompts where they are faster.
    score_elems_per_head = max(1, seq_len * seq_len_kv)
    max_heads = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_head * 4)
    return max(1, min(8, num_heads, max_heads))


def _fp8_mqa_logits_k_chunk_size(
    seq_len: int,
    seq_len_kv: int,
    head_chunk_size: int,
) -> int:
    score_elems_per_key = max(1, seq_len * head_chunk_size)
    max_keys = _SM120_MQA_LOGITS_MAX_SCORE_BYTES // (score_elems_per_key * 4)
    return max(1, min(seq_len_kv, max_keys))


def _fp8_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA logits torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    logits = torch.zeros(
        (seq_len, seq_len_kv), device=q_values.device, dtype=torch.float32
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)

    for head_start in range(0, num_heads, head_chunk_size):
        head_end = min(head_start + head_chunk_size, num_heads)
        q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
        q_chunk = q_chunk.transpose(0, 1).contiguous()
        head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
        k_chunk_size = _fp8_mqa_logits_k_chunk_size(
            seq_len, seq_len_kv, head_end - head_start
        )
        for k_start in range(0, seq_len_kv, k_chunk_size):
            k_end = min(k_start + k_chunk_size, seq_len_kv)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            logits[:, k_start:k_end].add_(
                scores[0] if scores.shape[0] == 1 else scores.sum(dim=0)
            )

    if clean_logits:
        offsets = torch.arange(seq_len_kv, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        logits = logits.masked_fill(~valid, float("-inf"))

    return logits


def _fp8_mqa_logits_topk_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_tokens: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 MQA top-k torch path only supports FP8 Q")

    k_values, k_scales = kv
    k_f32 = k_values.to(torch.float32)
    k_f32.mul_(k_scales.reshape(-1, 1).to(torch.float32))
    k_t = k_f32.transpose(0, 1).contiguous()

    seq_len, num_heads, _ = q_values.shape
    seq_len_kv = k_f32.shape[0]
    if out is None:
        out = torch.empty(
            (seq_len, topk_tokens), device=q_values.device, dtype=torch.int32
        )
    else:
        assert out.shape == (seq_len, topk_tokens)
        assert out.dtype == torch.int32
    out.fill_(-1)

    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    head_chunk_size = _fp8_mqa_logits_head_chunk_size(seq_len, seq_len_kv, num_heads)
    k_chunk_size = _fp8_mqa_logits_k_chunk_size(seq_len, seq_len_kv, head_chunk_size)
    max_chunk_topk = min(topk_tokens, k_chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, k_chunk_size):
        k_end = min(k_start + k_chunk_size, seq_len_kv)
        chunk_logits = torch.zeros(
            (seq_len, k_end - k_start),
            device=q_values.device,
            dtype=torch.float32,
        )
        for head_start in range(0, num_heads, head_chunk_size):
            head_end = min(head_start + head_chunk_size, num_heads)
            q_chunk = q_values[:, head_start:head_end, :].to(torch.float32)
            q_chunk = q_chunk.transpose(0, 1).contiguous()
            head_weights = weights[:, head_start:head_end].transpose(0, 1).unsqueeze(-1)
            scores = torch.matmul(q_chunk, k_t[:, k_start:k_end])
            scores.relu_()
            scores.mul_(head_weights)
            chunk_logits.add_(scores[0] if scores.shape[0] == 1 else scores.sum(dim=0))

        offsets = torch.arange(k_start, k_end, device=q_values.device)
        valid = (offsets[None, :] >= cu_seqlen_ks[:, None]) & (
            offsets[None, :] < cu_seqlen_ke[:, None]
        )
        chunk_logits.masked_fill_(~valid, float("-inf"))

        chunk_topk = min(topk_tokens, k_end - k_start)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return out


def _fp8_mqa_logits_topk_triton(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    k_values, _ = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    logits_bytes = q_values.shape[0] * k_values.shape[0] * torch.float32.itemsize
    if logits_bytes > _SM120_MQA_TRITON_TOPK_MAX_LOGITS_BYTES:
        return False

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    logits = fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    topk_tokens = out.shape[1]
    out.fill_(-1)
    if topk_tokens == 0 or logits.shape[1] == 0:
        return True

    topk_op = _top_k_per_row_prefill_op()
    if topk_op is not None:
        topk_op(
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            out,
            logits.shape[0],
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )
        out.add_(cu_seqlen_ks[:, None])
        valid = (out >= cu_seqlen_ks[:, None]) & (out < cu_seqlen_ke[:, None])
        out.masked_fill_(~valid, -1)
    else:
        select_k = min(topk_tokens, logits.shape[1])
        selected = out[:, :select_k]
        values, indices = torch.topk(logits, select_k, dim=1)
        selected.copy_(indices.to(torch.int32))
        selected.masked_fill_(~torch.isfinite(values), -1)
    return True


def _fp8_mqa_logits_topk_triton_chunked(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    q_values, q_scale = q
    k_values, k_scales = kv
    if not (q_scale is None and q_values.dim() == 3 and k_values.dim() == 2):
        return False

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_mqa_logits_triton,
    )

    seq_len = q_values.shape[0]
    seq_len_kv = k_values.shape[0]
    topk_tokens = out.shape[1]
    out.fill_(-1)
    if seq_len == 0 or seq_len_kv == 0 or topk_tokens == 0:
        return True

    chunk_size = max(1, _SM120_MQA_TRITON_CHUNKED_TOPK_CHUNK_SIZE)
    best_values = torch.full(
        (seq_len, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (seq_len, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (seq_len, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (seq_len, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    for k_start in range(0, seq_len_kv, chunk_size):
        k_end = min(k_start + chunk_size, seq_len_kv)
        local_width = k_end - k_start
        local_ks = torch.clamp(cu_seqlen_ks - k_start, min=0, max=local_width)
        local_ke = torch.clamp(cu_seqlen_ke - k_start, min=0, max=local_width)
        chunk_logits = fp8_mqa_logits_triton(
            q_values,
            (k_values[k_start:k_end], k_scales[k_start:k_end]),
            weights,
            local_ks,
            local_ke,
        )
        chunk_topk = min(topk_tokens, local_width)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(k_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(out)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=out)
        best_values, next_best_values = next_best_values, best_values
        out.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def fp8_fp4_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 MQA top-k indices without materializing full logits."""
    if not (current_platform.is_cuda() and _use_sm12x_fallback() and q[1] is None):
        return False
    if _fp8_mqa_logits_topk_triton(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True
    if _fp8_mqa_logits_topk_triton_chunked(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    ):
        return True
    _fp8_mqa_logits_topk_torch(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices.shape[1],
        out=topk_indices,
    )
    return True


def _fp8_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    q_values, q_scale = q
    if clean_logits and q_scale is None and q_values.dim() == 3 and kv[0].dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_mqa_logits_triton,
        )

        return fp8_mqa_logits_triton(q_values, kv, weights, cu_seqlen_ks, cu_seqlen_ke)
    return _fp8_mqa_logits_torch(
        q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits
    )


def _fp8_paged_mqa_logits_torch(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("SM120 paged MQA torch path only supports FP8 Q")

    batch_size, next_n, num_heads, head_dim = q_values.shape
    head_dim_with_scale = kv_cache.shape[-1]
    assert head_dim_with_scale > head_dim
    assert weights.shape == (batch_size * next_n, num_heads)
    assert context_lens.shape == (batch_size, next_n)

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        _view_packed_fp8_paged_mqa_kv_cache,
    )

    kv_values, kv_scales = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_kv, _, _ = kv_values.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )

    q_f32 = q_values.float()
    score_bytes = _SM120_MQA_LOGITS_MAX_SCORE_BYTES
    max_tokens_per_chunk = max(1, score_bytes // max(1, num_heads * 4))
    token_offsets_cache: dict[int, torch.Tensor] = {}

    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens[batch_idx, next_idx].item())
            if context_len <= 0:
                continue

            q_row = q_f32[batch_idx, next_idx]
            row_weights = weights[row]
            for token_start in range(0, context_len, max_tokens_per_chunk):
                token_end = min(context_len, token_start + max_tokens_per_chunk)
                chunk_len = token_end - token_start
                token_offsets = token_offsets_cache.get(chunk_len)
                if token_offsets is None or token_offsets.device != q_values.device:
                    token_offsets = torch.arange(
                        chunk_len, device=q_values.device, dtype=torch.long
                    )
                    token_offsets_cache[chunk_len] = token_offsets
                token_ids = token_start + token_offsets
                logical_blocks = token_ids // block_kv
                token_in_block = token_ids - logical_blocks * block_kv
                physical_blocks = block_tables[batch_idx, logical_blocks]
                kv_chunk = kv_values[physical_blocks, token_in_block, 0].float()
                scale_chunk = kv_scales[physical_blocks, token_in_block, 0].squeeze(-1)
                kv_chunk.mul_(scale_chunk[:, None])
                scores = torch.matmul(q_row, kv_chunk.T)
                scores.relu_()
                scores.mul_(row_weights[:, None])
                logits[row, token_start:token_end] = scores.sum(dim=0)

    return logits


def _fp8_paged_mqa_logits_sm12x(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    q_values, q_scale = q
    if (
        q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            fp8_paged_mqa_logits_triton,
        )

        return fp8_paged_mqa_logits_triton(
            q_values, kv_cache, weights, context_lens, block_tables, max_model_len
        )
    logger.warning_once(
        "SM12x paged-MQA falling back to the torch reference path "
        "(q_scale=%s, q.dim=%s, kv_cache.dtype=%s, kv_cache.shape[-1]=%s, "
        "q_values.shape[-1]=%s). This path is intended for correctness checks "
        "and is not graph-compatible; expect a large per-step latency.",
        "set" if q_scale is not None else "None",
        q_values.dim(),
        kv_cache.dtype,
        kv_cache.shape[-1] if kv_cache.dim() else None,
        q_values.shape[-1],
    )
    return _fp8_paged_mqa_logits_torch(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )


def fp8_fp4_paged_mqa_topk_indices(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    topk_indices: torch.Tensor,
) -> bool:
    """Write SM120 FP8 paged MQA top-k indices without full logits."""
    q_values, q_scale = q
    if not (
        current_platform.is_cuda()
        and _use_sm12x_fallback()
        and q_scale is None
        and q_values.dim() == 4
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[-1] == q_values.shape[-1] + 4
    ):
        return False

    num_rows = q_values.shape[0] * q_values.shape[1]
    topk_tokens = topk_indices.shape[1]
    assert topk_indices.shape == (num_rows, topk_tokens)
    assert topk_indices.dtype == torch.int32
    topk_indices.fill_(-1)
    if num_rows == 0 or topk_tokens == 0 or max_model_len == 0:
        return True

    best_values = torch.full(
        (num_rows, topk_tokens),
        float("-inf"),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_size = max(1, _SM120_PAGED_MQA_TOPK_CHUNK_SIZE)
    max_chunk_topk = min(topk_tokens, chunk_size)
    chunk_values_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    chunk_indices_buf = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int64,
    )
    chunk_indices_i32 = torch.empty(
        (num_rows, max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    candidate_values = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.float32,
    )
    candidate_indices = torch.empty(
        (num_rows, topk_tokens + max_chunk_topk),
        device=q_values.device,
        dtype=torch.int32,
    )
    next_best_values = torch.empty_like(best_values)
    selected = torch.empty(
        (num_rows, topk_tokens),
        device=q_values.device,
        dtype=torch.int64,
    )

    from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
        fp8_paged_mqa_logits_triton,
    )

    for token_start in range(0, max_model_len, chunk_size):
        token_count = min(chunk_size, max_model_len - token_start)
        chunk_logits = fp8_paged_mqa_logits_triton(
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )
        chunk_topk = min(topk_tokens, token_count)
        chunk_values = chunk_values_buf[:, :chunk_topk]
        chunk_indices = chunk_indices_buf[:, :chunk_topk]
        torch.topk(chunk_logits, chunk_topk, dim=1, out=(chunk_values, chunk_indices))
        chunk_indices_out = chunk_indices_i32[:, :chunk_topk]
        chunk_indices_out.copy_(chunk_indices)
        chunk_indices_out.add_(token_start)

        candidate_cols = topk_tokens + chunk_topk
        candidate_values_view = candidate_values[:, :candidate_cols]
        candidate_indices_view = candidate_indices[:, :candidate_cols]
        candidate_values_view[:, :topk_tokens].copy_(best_values)
        candidate_values_view[:, topk_tokens:candidate_cols].copy_(chunk_values)
        candidate_indices_view[:, :topk_tokens].copy_(topk_indices)
        candidate_indices_view[:, topk_tokens:candidate_cols].copy_(chunk_indices_out)
        torch.topk(
            candidate_values_view,
            topk_tokens,
            dim=1,
            out=(next_best_values, selected),
        )
        torch.gather(candidate_indices_view, 1, selected, out=topk_indices)
        best_values, next_best_values = next_best_values, best_values
        topk_indices.masked_fill_(~torch.isfinite(best_values), -1)

    return True


def _tf32_hc_prenorm_gemm_torch(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    """Portable SM12x HyperConnection prenorm GEMM fallback.

    DeepGEMM's split ABI only requires that downstream consumers recover the
    full result by summing over the split dimension. Keep the implementation
    simple by writing the full product to split zero and clearing the rest.
    """
    del num_split
    product = x.float() @ fn.float().T
    norm = x.float().square().sum(dim=-1)

    if out.dim() == 3:
        out.zero_()
        sqrsum.zero_()
        out[0].copy_(product)
        sqrsum[0].copy_(norm)
    else:
        out.copy_(product)
        sqrsum.copy_(norm)
    return out


def _tf32_hc_prenorm_gemm_sm12x(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    num_split: int,
) -> torch.Tensor:
    if out.dim() == 3 and sqrsum.dim() == 2:
        from vllm.models.deepseek_v4.nvidia.ops.sm12x_mqa import (
            tf32_hc_prenorm_gemm_triton,
        )

        tf32_hc_prenorm_gemm_triton(x, fn, out, sqrsum, num_split)
        return out

    return _tf32_hc_prenorm_gemm_torch(x, fn, out, sqrsum, num_split)
```

---
### 5.5 `vllm/model_executor/layers/sparse_attn_indexer.py`：接入直出 top-k

#### (a) import

在 `vllm.utils.deep_gemm` 的导入列表里追加：

```python
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,          # 已有
    fp8_fp4_mqa_topk_indices,    # 新增
    fp8_fp4_paged_mqa_logits,    # 已有
    fp8_fp4_paged_mqa_topk_indices,  # 新增
    has_deep_gemm,
    is_mqa_backend_available,    # 新增
)
```

#### (b) 构造器：放宽 DeepGEMM 依赖

当前（v0.26.0）在 `SparseAttnIndexer.__init__` 里：

```python
if current_platform.is_cuda() and not has_deep_gemm():
    raise RuntimeError(
        "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
        "the current vLLM environment."
    )
```

改为（非 DeepGEMM 时只要有 SM89/SM12x 降级路径即可）：

```python
if current_platform.is_cuda() and not is_mqa_backend_available():
    raise RuntimeError(
        "Sparse Attention Indexer CUDA op requires DeepGEMM or an "
        "SM12x/SM89 MQA fallback in the current vLLM environment."
    )
```

#### (c) prefill 路径：先尝试直出 top-k

在 prefill 的 `for chunk` 循环里、`fp8_fp4_mqa_logits(...) + ops.top_k_per_row_prefill(...)` 之前插入：

```python
if (
    dcp_world_size == 1
    and not current_platform.is_xpu()
    and fp8_fp4_mqa_topk_indices(
        (q_slice_cast, q_scale_slice),
        (k_quant_cast, k_scale_cast),
        weights[chunk.token_start : chunk.token_end],
        cu_seqlen_ks,
        cu_seqlen_ke,
        topk_indices,
    )
):
    continue  # 已直出 top-k，等价于再生产 logits 再 topk
```

> 变量名取自 v0.26.0 prefill 分支现状（`q_slice_cast/q_scale_slice`、`k_quant_cast/k_scale_cast`、`chunk.token_start/token_end`、`cu_seqlen_ks/ke`、`topk_indices`）。若你的版本此处变量名不同，按实际替换即可。

#### (d) decode 路径：先尝试直出 top-k（受内存阈值门控）

在 decode 分支里、`fp8_fp4_paged_mqa_logits(...)` 之前计算日志内存并尝试：

```python
topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
logits_bytes = num_padded_tokens * max_model_len * torch.float32.itemsize
max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
used_direct_topk = False
if (
    dcp_world_size == 1
    and decode_metadata.global_seq_lens is None
    and not current_platform.is_xpu()
    and logits_bytes > max_logits_bytes
):
    used_direct_topk = fp8_fp4_paged_mqa_topk_indices(
        (padded_q_quant_cast, padded_q_scale),
        kv_cache,
        weights[:num_padded_tokens],
        seq_lens,
        decode_metadata.block_table,
        max_model_len,
        topk_indices,
    )

if not used_direct_topk:
    # 原有：logits = fp8_fp4_paged_mqa_logits(...) + 下方 topk 逻辑
    ...
```

> 门槛 `logits_bytes > max_logits_bytes`（默认 512MB）避免长上下文时为全 logits 分配过大内存；短上下文仍走原 logits+topk 路径（精度一致、更快）。

### 5.6 `vllm/models/deepseek_v4/nvidia/ops/o_proj.py`：接入 FP8 einsum 降级

当前 `o_proj.py` 直接 `from vllm.utils.deep_gemm import fp8_einsum`，在 SM89 上会失败。改为：

```python
from vllm.models.deepseek_v4.nvidia.ops.fp8_einsum import (
    deepseek_v4_fp8_einsum,
    deepseek_v4_fp8_einsum_config,
)
```

- `compute_fp8_einsum_recipe()` 返回 `deepseek_v4_fp8_einsum_config(cap.major)`：SM89/SM12x 得到 `((1, 128, 128), False)`（FP32 block scale、非 TMA 对齐）。
- `deep_gemm_fp8_o_proj(...)` 内部把 `fp8_einsum(...)` 替换为 `deepseek_v4_fp8_einsum(a, a_scale, b, b_scale, out, "bhr,hdr->bhd", list(einsum_recipe))`。
- `wo_a` 的 scale 来源要兼容两种：`getattr(wo_a, "weight_scale_inv", None) if not None else wo_a.weight_scale`。

主意的两处细节：

- `fused_inv_rope_fp8_quant` 的 `tma_aligned_scales` / `compact_scales` 两个开关按 `compute_fp8_einsum_recipe()` 结果传入（SM89 走 `tma_aligned_scales=False`、`compact_scales=False`）。
- SM120 专用 CUTLASS 分支（`use_deepseek_v4_sm120_cutlass_fp8_einsum`）只在 SM12 且 C++ op 存在时启用；SM89 永不触发，可放心。

### 5.7 FlashInfer 后端接受 `(8,9)`

四处改动，模式一致：「把 `major == 12` 扩展为 `major == 12 or (8,9)`，并把 `has_flashinfer_sparse_mla_sm120()` 探测换成按设备取 sm89/sm120」。

#### (a) `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`

- `FlashInferMLASparseSM120Backend.supports_compute_capability`：

```python
return capability.major == 12 or (
    capability.major == 8 and capability.minor == 9
)
```

- `supports_combination(...)` 里把 `has_flashinfer_sparse_mla_sm120()` 的地方改成：

```python
is_sm89 = (device_capability.major, device_capability.minor) == (8, 9)
has_sparse_mla = (
    has_flashinfer_sparse_mla_sm89()
    if is_sm89
    else has_flashinfer_sparse_mla_sm120()
)
if not has_sparse_mla:
    return ("FLASHINFER_MLA_SPARSE_SM120 requires a FlashInfer sparse MLA "
            "build compatible with the current GPU")
```

#### (b) `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`

`FlashInferMLASparseSM120Impl.__init__` 里：

```python
capability = current_platform.get_device_capability()
is_sm89 = capability is not None and (
    capability.major, capability.minor
) == (8, 9)
has_sparse_mla = (
    has_flashinfer_sparse_mla_sm89()
    if is_sm89
    else has_flashinfer_sparse_mla_sm120()
)
if not has_sparse_mla:
    raise RuntimeError(
        "FLASHINFER_MLA_SPARSE_SM120 requires a FlashInfer sparse MLA "
        "build compatible with the current GPU."
    )
```

#### (c) `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`

- `DeepseekV4FlashInferMLASparseBackend.supports_compute_capability`：

```python
return capability.major in [10, 12] or capability == DeviceCapability(8, 9)
```

- `supports_combination(...)`：在 `major == 12` 分支旁新增 `(8, 9)` 分支——`kv_cache_dtype` 白名单与 SM12 相同（`fp8` / `fp8_e4m3` / `fp8_ds_mla`），用 `has_flashinfer_sparse_mla_sm89()` 检查；并在兜底返回 `"requires SM10x, SM12x or SM89"`。
- `DeepseekV4FlashInferSM120Attention.__init__`：把 `has_flashinfer_sparse_mla_sm120()` 检查同样改成按设备选 sm89/sm120。
- `get_kv_cache_shape`：SM89 落回默认 `(num_blocks, block_size, head_size)` 分支（`num_kv_heads == 1`），`fp8_ds_mla` 打包布局由 `supports_combination` 保证。

#### (d) 注册表 / warmup

`FLASHINFER_MLA_SPARSE_DSV4` 已在 `vllm/v1/attention/backends/registry.py` 注册，capability 判定都在 `supports_compute_capability` 内，**无需改注册表**。

### 5.8 `vllm/models/deepseek_v4/nvidia/model.py`：路由 `(8,9)` 到 SM120 注意力

当前 `_select_dsv4_attn_cls` 用 `device_capability.major == 12` 判断。抽一个公共判定并加入 `(8,9)`：

```python
def _is_flashinfer_sparse_jit_capability(capability: DeviceCapability) -> bool:
    """SM12 native; SM89 via the SM89 sparse-MLA JIT build."""
    return capability.major == 12 or (
        capability.major, capability.minor
    ) == (8, 9)
```

然后把 `_select_dsv4_attn_cls` 里所有 `device_capability.major == 12` 替换为 `_is_flashinfer_sparse_jit_capability(device_capability)`。这样在 SM89 上，无论是显式 `--attention-backend FLASHINFER_MLA_SPARSE_DSV4` 还是默认后端，都会选到 `DeepseekV4FlashInferSM120Attention`。

`compressor.py`、`common/ops/*` 里既有的 `is_device_capability_family(120)` 门控**不要改**——它们在 SM120 上选 CuTe-DSL；SM89 上因为 `has_cutedsl()` 为 `False` 已自动走降级。
## 6. 测试与验证

### 6.1 算子级自检（无需起模型）

```python
import torch
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm89
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.deep_gemm import is_mqa_backend_available

assert current_platform.get_device_capability() == (8, 9)
assert has_flashinfer_sparse_mla_sm89() is True
assert has_cutedsl() is False
assert is_mqa_backend_available() is True
```

### 6.2 单模块测试

- `fp8_einsum.py`：与 DeepGEMM `fp8_einsum`（SM100 参考）或 fp32 手算对比，覆盖 `num_groups` 与 TP 分片。
- `sm12x_mqa.py` / `sm12x_deep_gemm_fallbacks.py`：构造随机 FP8 query/KV + 已知 scale，校验直出 top-k 与“logits + torch.topk”结果一致；覆盖边界 `topk_tokens == 0`、单 token、长上下文分块合并。

### 6.3 端到端启动

```bash
vllm serve /path/to/DeepSeek-V4-Flash \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4
```

预期日志标志：`Using 'MARLIN' Mxfp4 MoE backend`、`Using SM89/SM12x … MQA top-k path`、`Application startup complete.`。

### 6.4 正确性 / 回归

- decode 精度：FlashInfer SM89 版修复了 per-MMA UE8M0 scale，务必用长上下文 + GSM/数学评测与合并提交前的基准确认误差不放大。
- 其他硬件回归：SM90/SM100 路径不受影响（`_use_sm12x_mqa_fallback`、`has_cutedsl` 都以 capability 为条件）。跑一遍既有 SM100 测试套件确认未回归。

---

## 7. 风险与注意事项

1. **外部 FlashInfer 是硬前提**：没有 SM89 版 FlashInfer，`has_flashinfer_sparse_mla_sm89()` 恒为 `False`，后端直接拒启。该部分无法用仓库代码替代。
2. **性能预期**：decode 被 Marlin FP4 反量化限制在 ~82 tok/s（4×RTX 4090、单并发）。这是 Ada 无 FP4 / 无 microscaling 张量核的硬件上限，不是实现缺陷。
3. **SM80/A800 不在本方案范围**：本文只支持 `(8, 9)`。如需 8.0，需额外在 `_use_sm12x_mqa_fallback` 与各 capability 判定里加入 `(8, 0)`，并重新验证——作为实验性扩展，不建议默认开启。
4. **SM120 专用 CUTLASS op 依赖**：`fp8_einsum.py` 里 `use_deepseek_v4_sm120_cutlass_fp8_einsum` 分支用到 `torch.ops._C.deepseek_v4_fp8_bmm_sm120*`，仅当 capability 为 12 时才会走到，且用 `try/except AttributeError` 保护；SM89 路径不会触达，属安全分支。
5. **版本内类名/上下文**：接入 `sparse_attn_indexer.py` 时以你当前版本的实际变量名为准（5.5 节的变量名是 v0.26.0 现状）；新内核模块自包含，无需改类名。
6. **prefill**：SM89 prefill 复用 SM120 后端同一套 `_forward_prefill`，不需要新内核；“c128a global prefill metadata”式优化是 SM120 专属演进，不要为 SM89 开启。
7. **DCP（decode context parallel）在 SM89 上不受支持**：5.5 节的 prefill/decode 直出 top-k 都以 `dcp_world_size == 1` 为前置条件；若 `dcp_world_size > 1`，SM89 会落回原路径——而原路径在 SM89 上不可行：`fp8_fp4_mqa_logits` 是 DeepGEMM 函数（DeepGEMM 不可用时 `vllm.utils.deep_gemm` 的占位实现直接抛 `RuntimeError`），且 DCP 合并路径 `_assert_cutedsl_dcp_merge_supported`（`vllm/model_executor/layers/sparse_attn_indexer.py:57`）在 `has_cutedsl()` 为 `False` 时直接 `raise`。即 **SM89 + DCP 组合会 fail-closed 拒启**（抛错而非错误结果），属预期行为，文档标注为不支持。

---

## 8. 破坏性评估与升级友好性

### 8.1 是否破坏现有路径？

结论：**对 SM90 / SM100 / SM120 及非 DeepSeek V4 模型无破坏**；对 SM89 是「从不可用到可用」的纯增量。理由：

| 改动 | 影响面 | 破坏性判断 |
|---|---|---|
| `has_cutedsl()` 在 `(8,9)` 返回 `False`（5.2） | 只影响 SM89；SM90/100/120 上 `is_device_capability((8, 9))` 为 `False`，行为不变 | 无。唯一副作用是 SM89 上 DCP merge 断言提前 fail-closed（见第 7 节第 7 条） |
| `_use_sm12x_mqa_fallback()` / `is_mqa_backend_available()`（5.3） | SM12x 家族原本 DeepGEMM 可用，`has_deep_gemm()` 已为 `True`，新函数不改变其结果；SM90 两个判定都是 `False` | 无 |
| `SparseAttnIndexer.__init__` 放宽（5.5b） | 非 CUDA 平台、SM90/100/120 仍满足 `has_deep_gemm()`，分支不变 | 无 |
| prefill/decode 直出 top-k（5.5c/d） | 仅在 `fp8_fp4_mqa_topk_indices(...)` 返回 `True` 时接管，其余情况走原路径 | 无（SM90/100 上该函数因 `q[1] is None` 或 capability 判定返回 `False`） |
| `o_proj.py` 改走 `deepseek_v4_fp8_einsum`（5.6） | SM90/100 上 `_use_deepseek_v4_sm12x_triton_fp8_einsum` 返回 `False`，最终仍调 DeepGEMM `fp8_einsum` | 无 |
| FlashInfer 后端接受 `(8,9)`（5.7） | 仅把 capability 白名单从 `major == 12` 扩到「12 或 (8,9)」，SM90/100 判定不变 | 无 |
| `_select_dsv4_attn_cls` 路由（5.8） | SM12 仍走 `DeepseekV4FlashInferSM120Attention`；`(8,9)` 是新增分支 | 无 |

关键设计保证：**所有门控都以 capability 为条件**，且新分支都在「DeepGEMM / cutedsl 本来就不支持」的硬件上才激活——即改动只发生在原本就会失败或降级的路径上，不存在「把原本正常的快路径改慢」的情况。

### 8.2 升级 vLLM 的友好性

结论：**中等偏友好**。新增 3 个自包含内核模块升级零成本，但 8 处既有文件改动会与上游演进产生小冲突面，需按文件逐一审视。

**升级零成本的部分（新增文件，随补丁整体携带）**：
- `fp8_einsum.py`、`sm12x_mqa.py`、`sm12x_deep_gemm_fallbacks.py`：只依赖 vLLM 既有工具（`vllm.platforms`、`vllm.triton_utils`、`vllm.utils.deep_gemm`、`vllm.distributed`）的**稳定公共 API**，不依赖任何内部私有符号。上游重构这些模块时，只要 API 不变即可原样保留。

**升级需人工审视的部分（既有文件改动，冲突面按风险排序）**：

| 文件 | 升级风险 | 说明 |
|---|---|---|
| `vllm/utils/deep_gemm.py` | 高 | 上游对 DeepGEMM 封装改动频繁；新增函数若与上游同名会冲突。建议给新函数加 `sm89`/`sm12x` 前缀降低撞名概率 |
| `vllm/utils/import_utils.py` | 高 | 全仓库共用工具模块，上游常改；`has_cutedsl()` 语义变更需在升级后复查是否仍成立 |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | 中 | DSv4 专用，但上游随模型演进改动多；capability 判定是单点，冲突易修 |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 中 | prefill/decode 分支变量名随版本漂移（5.5 已注明），升级时按实际变量名重新接线 |
| `vllm/utils/flashinfer.py` | 低-中 | `has_flashinfer_sparse_mla_sm89` 是新增函数，无撞名；但依赖 FlashInfer 内部 `_resolve_dsv4_sparse_mla_backend`，FlashInfer 升级可能改名（已有 `try/except` 兜底为 `False`） |
| `vllm/v1/attention/backends/mla/*`、`nvidia/model.py`、`o_proj.py` | 低 | capability 判定单点、改动模式一致，冲突易修 |

**升级建议**：
1. 以**独立补丁/commit 管理**这 11 个文件的改动，与上游 merge 时逐文件 rebase，不要 squash 成一个大 patch 混入。
2. 升级后跑一遍 6.1 的算子自检 + 6.3 端到端，确认三个探测值（`has_flashinfer_sparse_mla_sm89` / `has_cutedsl` / `is_mqa_backend_available`）仍符合预期——它们是全部门控的地基。
3. 上游若把 SM89 支持合入主干，本补丁可整体删除；在删除前用 `git log` 保留本补丁的提交记录以便追溯。
4. 外部 FlashInfer SM89 wheel 是升级链路上最脆弱的一环：它不在 vLLM 版本管理内，FlashInfer 大版本升级需重新验证 `_resolve_dsv4_sparse_mla_backend` 的存在性与返回语义。

---

## 附录 A：实现顺序检查清单

- [ ] 5.0 环境：SM89 版 FlashInfer + CUDA 13 + torch cu130 + `TORCH_CUDA_ARCH_LIST=8.9+PTX` + `FLASHINFER_DISABLE_VERSION_CHECK=1`
- [ ] 5.1 `utils/flashinfer.py`：`has_flashinfer_sparse_mla_sm89()`
- [ ] 5.2 `utils/import_utils.py`：`has_cutedsl()` 在 `(8,9)` 返回 `False`
- [ ] 5.2b `compressor.py`：head=512 分支加 `has_cutedsl()` 门控（5.2 节的 ⚠️ 补充项，**必须做**）
- [ ] 5.3 `utils/deep_gemm.py`：`_use_sm12x_mqa_fallback` / `is_mqa_backend_available` / `fp8_fp4_mqa_topk_indices` / `fp8_fp4_paged_mqa_topk_indices`
- [ ] 5.4 落盘三个新模块：`fp8_einsum.py`、`sm12x_mqa.py`、`sm12x_deep_gemm_fallbacks.py`
- [ ] 5.5 `sparse_attn_indexer.py`：构造放宽 + prefill/decode 直出 top-k
- [ ] 5.6 `o_proj.py`：走 Triton FP8 einsum
- [ ] 5.7 FlashInfer 后端 4 处接受 `(8,9)`
- [ ] 5.8 `model.py`：`_is_flashinfer_sparse_jit_capability` 路由
- [ ] 6.1 算子自检 + 6.2 单模块测试 + 6.3 端到端 + 6.4 精度回归
