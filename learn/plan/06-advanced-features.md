# 阶段六：高级特性

> 学习目标：理解 vLLM 的高级功能——量化推理、LoRA 微调适配、投机解码加速、多模态支持、结构化输出及 KV Cache 卸载等。

---

## 6.1 量化（Quantization）

### 核心概念

- **模型量化**：将 FP16/BF16 权重和激活值映射到更低精度（INT4/INT8/FP8），减少显存占用和计算量
- **量化方案**：AWQ、GPTQ、FP8、INT8 SmoothQuant、INT4 等
- **权重量化 vs 激活量化**：weight-only 与 weight+activation 的区别
- **量化 Kernel**：针对低精度数据类型的专用 GPU 计算内核

### 代码路径

| 文件/目录 | 说明 |
|-----------|------|
| `model_executor/layers/quantization/` | 量化层实现（各方案子目录） |
| `csrc/quantization/` | 量化 CUDA Kernels |
| `vllm/_custom_ops.py` ~118KB | 量化操作的 Python 封装 |
| `model_executor/layers/linear.py` ~71KB | 量化感知的线性层 |
| `vllm/quantization/` | 量化工具（旧架构） |

### 阅读要点

1. **量化层架构**：
   - 每个量化方案（AWQ/GPTQ/FP8 等）是一个子目录
   - 统一接口：`QuantizeMethod` 或类似抽象
   - `weight_loader` 如何加载量化后的权重

2. **AWQ vs GPTQ vs FP8**：
   - AWQ：activation-aware 的权重量化，按通道缩放
   - GPTQ：基于 Hessian 矩阵的最优化量化
   - FP8：原生 FP8 训练/推理格式
   - 各方案的精度-速度权衡

3. **量化 Kernel**：
   - 低精度矩阵乘法（W4A16, W8A16, FP8 等）
   - 反量化（dequantize）的融合：边反量化边计算

### 学习目标

- [ ] 理解 AWQ/GPTQ/FP8 三种量化方案的核心原理
- [ ] 知道量化如何与线性层集成（`weight_loader` 路径）
- [ ] 理解权重量化与激活量化的区别
- [ ] 能根据模型和硬件推荐合适的量化方案

### 思考题

1. 为什么 AWQ 需要 calibration dataset？GPTQ 需要吗？
2. FP8 量化相比 INT8 的优势和劣势是什么？
3. 量化模型在 Prefill 和 Decode 阶段的速度提升分别来自哪里？

---

## 6.2 LoRA 适配器

### 核心概念

- **LoRA（Low-Rank Adaptation）**：在冻结原模型权重的基础上，插入可训练的低秩矩阵
- **多 LoRA 管理**：单个模型服务可以挂载多个 LoRA 适配器，动态切换
- **LoRA Kernel**：高效合并 LoRA 权重与原始权重的计算

### 代码路径

| 文件 | 说明 |
|------|------|
| `lora/model_manager.py` ~53KB | LoRA 模型管理 |
| `lora/lora_model.py` ~12KB | LoRA 模型包装 |
| `lora/lora_weights.py` ~9KB | LoRA 权重管理 |
| `lora/worker_manager.py` ~13KB | Worker 上的 LoRA 管理 |
| `lora/request.py` ~2KB | LoRA 请求 |
| `lora/resolver.py` ~3KB | LoRA 解析器 |
| `lora/utils.py` ~15KB | LoRA 工具 |
| `lora/punica_wrapper/` | Punica Kernel 封装 |
| `lora/ops/` | LoRA 操作 |

### 阅读要点

1. **`model_manager.py`**：
   - 多 LoRA 适配器的注册与切换
   - 显存管理：如何缓存热点 LoRA 权重，卸载冷门 LoRA
   - LoRA 与 Base 模型的权重融合策略

2. **LoRA 与推理管线集成**：
   - Model Runner 如何感知当前请求使用的 LoRA
   - `lora_model_runner_mixin.py`：Worker 侧的 LoRA 集成

### 学习目标

- [ ] 理解 LoRA 在推理服务中的工作原理
- [ ] 知道 vLLM 如何同时服务多个 LoRA 适配器
- [ ] 理解 LoRA 权重的显存管理策略

---

## 6.3 投机解码（Speculative Decoding）

### 核心概念

- **投机解码**：用一个小模型（Draft Model）快速生成候选 token，再用大模型（Target Model）验证
- **拒绝采样（Rejection Sampling）**：保证投机解码的输出分布与直接采样完全一致
- **多种投机策略**：Medusa, MLP Speculator, Eagle 等

### 代码路径

| 文件/目录 | 说明 |
|-----------|------|
| `v1/spec_decode/` | 投机解码实现 |
| `v1/sample/rejection_sampler.py` ~36KB | 拒绝采样器 |
| `spec_decode/` | 旧架构下的投机解码 |
| `models/mlp_speculator.py` ~6KB | MLP Speculator 模型 |
| `models/medusa.py` ~8KB | Medusa 模型 |
| `models/eagle.py` / `eagle2_5_vl.py` / `eagle3.py` | Eagle 系列 |

### 阅读要点

1. **投机解码流程**：
   - Draft Model 生成 N 个候选 token（N = 投机窗口大小）
   - Target Model 并行计算所有候选位置的 logits
   - Rejection Sampler 决定接受哪些 token，拒绝哪些
   - 如果拒绝，Target Model 从正确分布重新采样

2. **拒绝采样的数学保证**：
   - 为什么拒绝采样后的分布与直接采样相同
   - 接受率的计算

3. **不同投机策略的对比**：
   - Medusa：添加多个预测头
   - MLP Speculator：轻量级 MLP 预测
   - Eagle：更复杂的自动回归预测

### 学习目标

- [ ] 理解投机解码的原理和工作流程
- [ ] 知道拒绝采样的数学保证
- [ ] 能比较不同投机策略的优缺点

---

## 6.4 多模态

### 核心概念

- **多模态输入**：模型可以同时接受文本、图像、音频、视频输入
- **模态编码器**：每种模态有独立的编码器（CLIP for 图像、Whisper for 音频等）
- **投影层（Projection）**：将模态编码器的输出映射到语言模型的 embedding 空间
- **多模态处理 Pipeline**：输入解析 → 编码 → 投影 → 与文本拼接

### 代码路径

| 文件 | 说明 |
|------|------|
| `multimodal/inputs.py` ~32KB | 多模态输入定义 |
| `multimodal/parse.py` ~24KB | 输入解析 |
| `multimodal/processing/` | 预处理 pipeline |
| `multimodal/registry.py` ~13KB | 多模态模型注册 |
| `multimodal/cache.py` ~23KB | 多模态缓存 |
| `multimodal/audio.py` ~14KB | 音频处理 |
| `multimodal/video.py` ~75KB | 视频处理 |
| `multimodal/image.py` ~2KB | 图像处理 |
| `multimodal/hasher.py` | 数据哈希 |
| `multimodal/media/` | 媒体处理 |
| `multimodal/encoder_budget.py` ~8KB | 编码器预算管理 |

### 阅读要点

1. **多模态输入处理**：
   - `inputs.py`：多模态数据的结构定义（URL, base64, 本地文件）
   - `parse.py`：从不同来源解析多模态数据

2. **Processing Pipeline**：
   - 数据预处理：图像/音频/视频的解码、resize、归一化
   - 缓存策略：避免重复编码相同数据

3. **模型集成**：
   - 多模态模型如何定义（如 `Qwen2VLForConditionalGeneration`）
   - 编码器（Vision Encoder, Audio Encoder）的语言模型集成

### 学习目标

- [ ] 理解 vLLM 多模态输入的处理流程
- [ ] 知道图像、音频、视频的编码路径
- [ ] 理解投影层在多模态模型中的作用

---

## 6.5 结构化输出

### 核心概念

- **结构化输出**：约束模型输出符合特定格式（JSON Schema, Regex, Grammar）
- **工具调用（Function Calling）**：让模型在对话中调用预定义的函数
- **推理（Reasoning）**：支持 R1 风格的思考过程

### 代码路径

| 文件/目录 | 说明 |
|-----------|------|
| `v1/structured_output/` | 结构化输出 |
| `reasoning/` | 推理能力 |
| `tool_parsers/` | 工具调用解析 |
| `v1/sample/thinking_budget_state.py` ~25KB | 思考预算 |

### 阅读要点

1. **结构化输出**：
   - 如何约束 token 采样（masking logits）
   - JSON Schema 到 CFG（Context-Free Grammar）的转换

2. **工具调用**：
   - `tool_parsers/` 支持哪些工具格式
   - 工具调用与普通对话的切换

### 学习目标

- [ ] 理解结构化输出的 token 约束机制
- [ ] 知道工具调用的工作流程

---

## 6.6 KV Cache 卸载

### 核心概念

- **KV Cache 卸载**：将部分 KV Cache 从 GPU 显存移到 CPU 内存，释放 GPU 空间
- **卸载时机**：长上下文场景下，GPU 显存不足以容纳所有 KV Cache
- **再加载**：后续解码需要时再从 CPU 加载到 GPU

### 代码路径

| 文件/目录 | 说明 |
|-----------|------|
| `v1/kv_offload/` | KV Cache 卸载 |
| `v1/simple_kv_offload/` | 简化版卸载方案 |

### 阅读要点

1. **卸载策略**：
   - 哪些层的 KV Cache 优先卸载（早期层？后期层？）
   - 批量传输 vs 逐块传输
   - PCIe 带宽对卸载性能的影响

2. **卸载与调度的交互**：
   - Scheduler 如何感知卸载状态
   - 卸载后的延迟预测

### 学习目标

- [ ] 理解 KV Cache 卸载的动机和场景
- [ ] 知道卸载的基本实现策略

---

## 章节总结

### 高级特性全景

```
vLLM 核心推理引擎
    │
    ├── 量化层 ──── AWQ / GPTQ / FP8 / INT8 / INT4
    │                ↓
    │          更小显存 + 更快计算
    │
    ├── LoRA ─────── 多适配器同时服务
    │                ↓
    │          低成本模型定制
    │
    ├── 投机解码 ─── Draft Model → Target Model 验证
    │                ↓
    │          加速 Decode 阶段 2-3x
    │
    ├── 多模态 ───── 图像 / 音频 / 视频 编码 + 投影
    │                ↓
    │          单一模型处理多种输入
    │
    ├── 结构化输出 ─ JSON / Regex 约束采样
    │                ↓
    │          可靠的可编程输出
    │
    └── KV Cache 卸载 → CPU 内存扩展
                     ↓
                超长上下文支持
```

### 进一步阅读

- 继续阶段七：工具链与工程 → `07-toolchain-engineering.md`
- AWQ 论文 / GPTQ 论文 / LoRA 论文
- 投机解码论文（SpecInfer, Medusa, Eagle）

---

*对应 LEARNING_PLAN.md 第 8 章 | 基于 vLLM 主分支*
