# DeepSeek V4 on SM89 当前实现记录与 FlashInfer 依赖分析

## 1. 文档目的

本文档记录当前工作区里已经完成的 `DeepSeek V4 / SM89 (Ada)` 适配修改，并对照
[`DSV4_SM89_port_design.md`](file:///home/fire3/SRC/vllm/DSV4_SM89_port_design.md)
说明：

- 哪些设计点已经落地
- 哪些仍然是后续项
- 当前方案是否仍然依赖外部 `FlashInfer` 修改
- 如果不改 `FlashInfer`，是否存在替代路线

本文档描述的是**当前实现状态**，不是最终完成态。

## 2. 当前修改概览

本轮已经修改了以下文件：

| 文件 | 目的 | 当前状态 |
|---|---|---|
| `vllm/utils/flashinfer.py` | 新增 `has_flashinfer_sparse_mla_sm89()` 探测 | 已完成 |
| `vllm/utils/import_utils.py` | 在 SM89 上禁用 `has_cutedsl()` | 已完成 |
| `vllm/models/deepseek_v4/compressor.py` | 让 `head_dim == 512` 的 CUDA compressor 在 SM89 上回落到 Triton 路径 | 已完成 |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | 放宽 DeepSeek V4 FlashInfer sparse backend 到 SM89 | 已完成 |
| `vllm/models/deepseek_v4/nvidia/model.py` | 让 DSv4 在 SM89 上路由到 FlashInfer sparse attention 类 | 已完成 |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | 为无 DeepGEMM 的 CUDA 路径增加 Triton grouped matmul fallback | 已完成 |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 放宽构造器，并给 SM89 接入现有 torch MQA fallback | 已完成 |

## 3. 与设计文档的对照

### 3.1 已按设计落地的部分

#### A. FlashInfer SM89 探测

设计文档要求增加 `has_flashinfer_sparse_mla_sm89()`，当前已经实现。

实现逻辑：

1. 先复用 `has_flashinfer_sparse_mla_sm120()`，确认 sparse MLA decode API 在安装包中存在。
2. 再调用 `flashinfer.mla._core._resolve_dsv4_sparse_mla_backend(...)`，确认当前设备在 FlashInfer 内部被解析为 `"sparse"`。

这一步的意义是：即使 vLLM 本地门控放开，如果外部 `FlashInfer` wheel 本身不支持
SM89 sparse MLA，运行时仍会 fail closed。

#### B. CuTe-DSL 在 SM89 上禁用

设计文档明确指出，`has_cutedsl()` 只看包是否存在，不看硬件能力，这会导致 Ada 误入
SM90+/SM100+ 的 CuTe-DSL 路径。

当前已经在 `vllm/utils/import_utils.py` 中做了收口：

- 没装 `cutlass` 时仍返回 `False`
- 装了 `cutlass` 但设备是 `SM89` 时也返回 `False`

这样可以让现有的 Triton / torch fallback 自然接管大部分路径。

#### C. `compressor.py` 的 `head_dim == 512` 误选修复

设计文档指出 `compressor.py` 的 CUDA `head_dim == 512` 分支没有检查 `has_cutedsl()`，
SM89 上会直接走 `sparse_attn_compress_cutedsl`。

当前已经把这段分支改为：

- 仅在 `current_platform.is_cuda() and self.head_dim == 512 and has_cutedsl()` 时走 CuTe-DSL
- 否则自然落到已有的 `compress_norm_rope_store_triton`

额外核对结果：

- `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` 中的
  `compress_norm_rope_store_triton(...)` 本身就支持 `head_dim == 512`
- 因此当前这一步不需要再新补一个 512 专用 Triton compressor

#### D. DeepSeek V4 FlashInfer sparse 路由放宽到 SM89

当前已经在以下位置放宽 capability：

- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
- `vllm/models/deepseek_v4/nvidia/model.py`

效果是：

- `FLASHINFER_MLA_SPARSE_DSV4` 后端现在接受 `(8, 9)`
- DSv4 在 SM89 上会选择 `DeepseekV4FlashInferSM120Attention`
- 运行时仍由 `has_flashinfer_sparse_mla_sm89()` 做最终探测

注意：

这里复用的是 **DSv4 专用 FlashInfer sparse attention 层**，不是泛化的
`vllm/v1/attention/backends/mla/flashinfer_mla_sparse*.py`。

### 3.2 当前实现与设计文档不完全一致的部分

#### A. `SparseAttnIndexer` 当前已经是 Triton fallback，但还不是设计里的最终版 top-k fallback

设计文档给出的目标是：

- 新增 `sm12x_mqa.py`
- 新增 `sm12x_deep_gemm_fallbacks.py`
- 走“不物化完整 logits”的分块 top-k 直出路径

当前还没有做到这一步。

当前实现是：

- 放宽 `SparseAttnIndexer` 构造器，不再要求 CUDA 上必须有 DeepGEMM
- 在 SM89 且 `FP8 Q` 路径下，prefill / decode 分别回退到新的 Triton 路径：
  - `vllm/v1/attention/ops/triton_fp8_mqa_logits.py::fp8_mqa_logits_triton`
  - `vllm/v1/attention/ops/triton_fp8_mqa_logits.py::fp8_paged_mqa_logits_triton`

这条路径的优点是：

- 代码侵入小
- 已经明显优于此前的 torch 参考 fallback
- 能先把功能链路打通

缺点也很明确：

- 仍然会显式生成 logits
- 长上下文下显存和吞吐都会明显差于文档里的 Triton 直出 top-k 方案

因此，这一块目前属于**阶段性的 Triton fallback**，不是最终实现。

#### B. `o_proj` 当前已经不是 torch 回退，但还不是设计里的 Triton FP8 einsum

设计文档要求新增 `fp8_einsum.py`，在 SM89 上提供 Triton FP8 einsum。

当前还没有新增该模块。

当前实现是：

- 在 `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` 中，如果
  `current_platform.is_cuda()` 且 `not is_deep_gemm_supported()`
- 就回退到：
  - `_fused_inverse_rope_gptj(...)`
  - `_get_cached_wo_a_bf16_t(...)`
  - Triton `_grouped_bf16_matmul_kernel`
  - `wo_b(...)`

这条路径的性质：

- 语义正确
- inverse-RoPE 和 `wo_a` 主乘法都已经不再依赖 `torch.einsum`
- 性能上仍会弱于目标中的 Triton FP8 einsum / 原生 FP8 路径

所以它属于**已完成一轮 Triton 优化的过渡实现**。

#### C. 还没有动 `vllm/utils/deep_gemm.py`

设计文档中计划在 `deep_gemm.py` 里新增：

- `_use_sm12x_mqa_fallback()`
- `is_mqa_backend_available()`
- `fp8_fp4_mqa_topk_indices()`
- `fp8_fp4_paged_mqa_topk_indices()`

当前这些都还没有加入。

我这次选择的是更小步的切入方式：

- 直接在 `SparseAttnIndexer` 层接入 fallback
- 不先扩展 `deep_gemm.py` 的 API 面

这样可以减少一次性改动范围，但后续如果要上正式的 Triton 分块 top-k，
把 fallback 下沉回 `deep_gemm.py` 依然是更干净的方向。

#### D. 还没有改通用 `flashinfer_mla_sparse.py` / `flashinfer_mla_sparse_sm120.py`

设计文档计划同时放宽：

- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`

当前没有改这两个文件，原因是：

1. 当前适配目标是 **DeepSeek V4 专用 sparse MLA 路径**
2. DSv4 实际走的是 `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py`
   中的专用 backend / attention
3. 因此先改 DSv4 专用路径就足够让模型路由到 SM89

这不代表设计文档错了，而是说明当前实现选择了更小范围的最小改动集。

#### E. FlashInfer sparse warmup / autotune 的 SM120-only 边角问题已修正

当前已经改为：

- 对通用 `FLASHINFER_MLA_SPARSE_SM120` 继续维持 SM120 判定
- 对 `FLASHINFER_MLA_SPARSE_DSV4` 允许在 `SM89` 上通过
  `has_flashinfer_sparse_mla_sm89()` 进入 autotune / warmup
- 同时把日志文案从 “SM120 sparse MLA” 改成了更准确的通用描述

这意味着：

- SM89 上如果外部 `FlashInfer` 确实支持 DSv4 sparse MLA，就不再被 warmup/autotune 逻辑硬性排除

## 4. 当前方案的能力边界

### 4.1 现在已经能覆盖的东西

从代码结构上看，当前这批改动已经把下面几个“硬阻塞点”清掉了：

1. **DSv4 sparse attention 的后端路由**
2. **SM89 上 CuTe-DSL 的误选**
3. **无 DeepGEMM 时 indexer 构造器直接报错**
4. **无 DeepGEMM 时 o_proj 完全无路可走**

也就是说，当前代码已经具备了“在存在 SM89 版 FlashInfer 的前提下继续往前跑”的基础。

### 4.2 当前还明显缺失的东西

离设计文档的完成态，还有这些关键缺口：

1. **SM89 高性能 indexer fallback**
   - 现在是 Triton dense-logits fallback
   - 目标应该是不物化完整 logits 的 Triton 分块 top-k

2. **SM89 高性能 o_proj fallback**
   - 现在是 inverse-RoPE + Triton bf16 grouped matmul
   - 目标应该是 Triton FP8 einsum

3. **完整测试**
   - 还没补单测 / 集成验证
   - 还没做模型级 smoke test

4. **更完整的 FlashInfer warmup/autotune 验证**
   - 代码边角已经修正，但还缺实机验证

## 5. 当前实现为什么仍然依赖 FlashInfer

结论先说：

**如果继续沿用“复用现有 DSv4 sparse MLA decode 后端”这条路线，那么 `FlashInfer` 仍然需要外部修改。**

原因不是 vLLM 本地门控没有放开，而是**真正执行 sparse MLA decode 的内核在 `FlashInfer` 里**。

证据链如下：

1. DSv4 专用 attention 最终调用的是 `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`
   以及相关 sparse MLA API。
2. 当前 vLLM 本地所做的只是：
   - 允许 SM89 路由进来
   - 在运行时探测这个 API 是否真的能在 SM89 上解析为 `"sparse"`
3. 如果 `FlashInfer` wheel 仍然只支持 SM120，这个探测会失败，vLLM 会拒绝进入该后端。

换句话说：

- **vLLM 本地改动负责“接线”和“降级”**
- **FlashInfer 外部改动负责“真正的 SM89 sparse MLA kernel 可用性”**

这两层缺一不可。

## 6. FlashInfer 是否“必须修改”

这个问题要分成两种含义来看。

### 6.1 如果目标是不大改 vLLM 架构，答案是：基本必须

如果目标是：

- 继续复用当前 `DeepseekV4FlashInferSM120Attention`
- 继续复用 DSv4 sparse MLA 的 FlashInfer decode API
- 尽量不在 vLLM 里自建新的 sparse decode backend

那么答案就是：

**是，FlashInfer 基本必须提供 SM89 sparse MLA 支持。**

因为当前这条 decode 主路径的核心 kernel 不在 vLLM 仓库里。

### 6.2 如果允许大改 vLLM 架构，答案是：理论上不必须

如果接受以下代价：

- 在 vLLM 内部新写或新接一整条非 FlashInfer 的 sparse MLA decode 实现
- 补齐 metadata、KV 布局、top-k 索引、gather、attention kernel、输出投影配套
- 接受更长开发周期与更差初期性能

那么**理论上**可以不依赖 FlashInfer 的 SM89 修改。

但这已经不是“当前方案的小扩展”，而是接近“另起一条后端”。

## 7. 不修改 FlashInfer 的替代方案分析

### 方案 A：改走 FlashMLA

结论：**当前不可行。**

理由：

1. `vllm/vllm/v1/attention/ops/flashmla.py` 明确写死：
   - Dense 只支持 Hopper
   - Sparse 只支持 Hopper 和 Blackwell DC
2. `SM89` 不在其支持范围内

所以这条路不是简单切 backend 可以解决的。

### 方案 B：完全不用 FlashInfer，自己在 vLLM 里做 DSv4 sparse decode backend

结论：**理论可行，工程量最大。**

这条路线需要至少补这些部分：

1. 稀疏 top-k token 的物理索引布局
2. 从压缩 KV / SWA cache 中 gather 所需 token
3. sparse MLA decode kernel
4. 与当前 DSv4 metadata builder 的对齐
5. 与 cudagraph / warmup / workspace 体系的接线

优点：

- 不再受 FlashInfer 的 SM89 发布节奏影响

缺点：

- 开发量非常大
- 初版大概率性能不如 FlashInfer
- 测试与维护成本高

### 方案 C：退回到更通用的 dense / 非 sparse attention

结论：**功能上可能绕得过去，但大概率不适合作为 DSv4 的实用方案。**

原因：

1. DSv4 当前的推理管线是围绕 sparse MLA 组织的
2. indexer、compressed KV、SWA metadata、top-k 索引都已经深度嵌入
3. 强行退回 dense attention，不只是换一个 kernel，而是要重写数据通路
4. 长上下文吞吐和显存都会明显恶化

这更像一个“研究型 fallback”，不是一个工程上划算的主线选择。

### 方案 D：短期保持当前功能版 fallback，只把 `FlashInfer` 替换成更慢的 PyTorch/Triton decode

结论：**理论可行，但仍然需要新增一条 decode 实现，且收益不高。**

本质上它仍然落在“方案 B”的范畴，只是性能目标更低。

问题在于：

- 当前真正缺的不是 indexer 或 o_proj 的 fallback
- 真正外部绑定最强的部分是 **sparse MLA decode**

如果不使用 FlashInfer，就必须自己补这块。

## 8. 推荐结论

### 8.1 对当前路线的判断

当前最合理的主线仍然是：

1. **保留对外部 FlashInfer SM89 sparse MLA 的依赖**
2. 在 vLLM 内继续补齐本地 fallback：
   - indexer Triton 分块 top-k
   - o_proj Triton FP8 einsum
   - 补充 warmup/autotune 的实机验证

这是改动最小、风险最可控、最贴近现有架构的一条路。

### 8.2 如果外部 FlashInfer 暂时改不了

那就需要在两种次优路线里二选一：

1. **功能优先路线**
   - 在 vLLM 内补一条非 FlashInfer sparse decode fallback
   - 先追求可运行
   - 接受性能明显下降

2. **等待外部依赖路线**
   - 继续完成本地 fallback 和测试
   - 等待或自行维护 FlashInfer SM89 版

从工程投入与收益比看，第二条通常更划算。

## 9. 当前未完成事项清单

建议按优先级继续推进：

1. 为 `SparseAttnIndexer` 实现真正的 SM89 Triton 分块 top-k fallback
2. 为 `o_proj` 实现 Triton FP8 einsum，替换当前 bf16 grouped matmul fallback
3. 补最小 smoke test 和模型级验证记录
4. 在具备环境后验证：
   - 无 DeepGEMM 的 SM89 启动路径
   - 有 SM89 版 FlashInfer 时的 sparse decode 实际可用性

## 10. 验证状态

当前只完成了静态代码层面的检查：

- `git diff --check` 已通过

当前还没有完成运行验证，原因是工作区里缺少可直接使用的 Python 环境：

- 没有项目 `.venv`
- 机器上也没有可直接调用的 `uv`

因此，本文档中的“已完成”表示**代码接线已完成**，不表示**实机验证已完成**。
