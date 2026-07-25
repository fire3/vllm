# 阶段三：模型层

> 学习目标：理解 vLLM 如何表示和加载各种大语言模型，掌握模型接口体系、神经网络层库、模型注册与加载机制，并能读懂具体模型的实现代码。

---

## 3.1 模型架构模式

### 核心概念

- **模型接口（Interfaces）**：一组 Mixin 类，标记模型支持的并行策略（PP）、LoRA、量化等能力
- **模型注册**：自动发现并注册支持的模型架构
- **配置映射**：从 HuggingFace `config.json` 到 vLLM 内部配置的转换

### 代码路径

| 文件 | 说明 |
|------|------|
| `model_executor/models/interfaces.py` ~55KB | 模型接口 Mixin 体系 |
| `model_executor/models/interfaces_base.py` ~8KB | 基础接口 |
| `model_executor/models/registry.py` ~60KB | 模型注册中心 |
| `model_executor/models/config.py` ~38KB | 模型配置映射 |
| `model_executor/models/__init__.py` ~1KB | 模型包入口 |
| `model_executor/models/module_mapping.py` ~1KB | 模块名映射 |

### 阅读要点

1. **`interfaces.py`**：
   - 理解 Mixin 的设计模式：为什么用 Mixin 而非基类
   - 关键接口：`SupportsPP`（流水线并行）、`SupportsLoRA`（LoRA 支持）、`SupportsQuant`（量化）
   - `is_pp_enabled()` 等辅助方法如何影响模型执行

2. **`registry.py`**：
   - `ModelRegistry` 的数据结构：`Dict[arch_name, model_class]`
   - `@ModelRegistry.register` 装饰器的工作原理
   - `resolve_model_cls()` 如何根据 config 查找模型类
   - 如何通过 `--model` 参数匹配到正确的模型实现

3. **`config.py`**：
   - 从 HuggingFace `PretrainedConfig` 到 vLLM `ModelConfig` 的字段映射
   - 特殊配置项的处理（`trust_remote_code`, `rope_scaling` 等）

### 学习目标

- [ ] 理解 vLLM 的模型接口 Mixin 设计模式
- [ ] 知道模型注册与自动发现的工作机制
- [ ] 能读懂 `config.py` 中的配置映射逻辑

### 思考题

1. 新增一个模型架构（如 MyNewModel）需要修改哪些文件？注册流程是什么？
2. `SupportsPP` 接口具体影响了什么行为？
3. 如果 `config.json` 中的 `architectures` 字段为 `["MyModelForCausalLM"]`，vLLM 如何定位到对应的实现类？

---

## 3.2 神经网络层

### 核心概念

- **`model_executor/layers/`** 是 vLLM 的神经网络层库，类似 PyTorch `nn.Module` 的扩展
- 每个核心算子都有 vLLM 的高性能实现，支持量化、并行、多种 dtype
- 线性层是最复杂的组件：支持多种量化格式 + 张量并行切分

### 代码路径

| 文件/目录 | 行数 | 说明 |
|-----------|------|------|
| `layers/linear.py` | ~71KB | 线性层：`Linear`, `QKVLinear`, `RowParallelLinear` 等 |
| `layers/activation.py` | ~30KB | 激活函数：SwiGLU, GeGLU, QuickGELU 等 |
| `layers/layernorm.py` | ~10KB | LayerNorm / RMSNorm |
| `layers/attention/` | — | 注意力层封装 |
| `layers/vocab_parallel_embedding.py` | ~22KB | 并行 Embedding |
| `layers/logits_processor.py` | ~9KB | Logits 处理器 |
| `layers/fused_moe/` | — | 融合 MoE 层 |
| `layers/quantization/` | — | 量化层 |
| `layers/mamba/` | — | Mamba / SSM 层 |
| `layers/rotary_embedding/` | — | RoPE 位置编码 |
| `layers/pooler/` | — | Pooling 层 |
| `layers/fusion/` | — | 算子融合 |
| `layers/hpc/` | — | 高性能计算 |
| `layers/mla.py` | ~8KB | Multi-head Latent Attention |
| `layers/batch_invariant.py` | ~31KB | Batch 不变性（CUDA Graph 相关） |

### 阅读要点

1. **`linear.py`**（**重点）**：
   - 线性层家族：`Linear`, `QKVLinear`, `MergedColumnParallelLinear`, `RowParallelLinear`
   - 张量并行下的权重切分策略：按列切分（ColumnParallel）vs 按行切分（RowParallel）
   - 量化感知的权重加载：先加载权重再 apply 量化
   - `weight_loader` 机制：不同分区加载权重的自定义逻辑
   - `process_weights_after_loading`：后处理钩子

2. **`activation.py`**：
   - SwiGLU 的实现：`silu(x) * gate(x)`
   - 激活函数的注册和使用模式

3. **`fused_moe/`**：
   - MoE 的前向计算：`router → top-k gating → experts forward → combine`
   - 融合实现如何减少 kernel launch 开销

4. **`mla.py`**：
   - DeepSeek-V2 的 Multi-head Latent Attention
   - KV 压缩的精髓：将 KV 压缩为低维 latent 表示

### 学习目标

- [ ] 理解线性层的张量并行切分策略
- [ ] 知道 `weight_loader` 的工作机制
- [ ] 理解 MoE 层的计算流程
- [ ] 理解 MLA 的核心思想

---

## 3.3 具体模型实现

### 核心概念

- 每个模型文件定义一个完整的 Transformer 或变体架构
- 模型必须实现 `load_weights()` 方法，支持分布式权重加载
- 模型通过 `XXXModel` 和 `XXXForCausalLM` 类分层组织

### 代码路径

| 文件 | 行数 | 说明 |
|------|------|------|
| `models/llama.py` | ~19KB | **标准 Transformer，最佳学习起点** |
| `models/qwen2.py` | ~18KB | 类似 LLaMA 的结构 |
| `models/deepseek_v2.py` | ~75KB | MLA + MoE 复杂架构 |
| `models/qwen2_vl.py` | ~61KB | 视觉语言模型 |
| `models/mamba.py` | ~9KB | SSM 架构 |
| `models/gemma2.py` | ~14KB | Google Gemma |
| `models/phi3v.py` | ~26KB | 多模态 Phi-3 |
| 全部文件 | — | 100+ 模型文件 |

### 阅读要点

1. **通用模型结构**（以 `llama.py` 为例）：
   - `LlamaModel`（主干网络）：`embed_tokens → layers → norm`
   - `LlamaForCausalLM`（语言模型头）：`LlamaModel → lm_head`
   - 每个 Transformer Layer 的结构：`self_attn → mlp → post_attention_norm`
   - `load_weights()` 的实现模式

2. **权重加载模式**：
   - `load_weights()` 的标准实现：遍历 state dict → 匹配参数名 → 调用 `weight_loader`
   - 分布式加载：使用 `ShardedStateLoader` 只加载当前 rank 所需的权重切片

3. **模型并行支持**：
   - 张量并行：Attention 的 QKV 拆分、MLP 的 gate/up/down 拆分
   - 流水线并行：层的分段划分

4. **从 LLaMA 到变体的演进**：
   - Qwen2：RoPE 频率不同、激活函数不同
   - DeepSeek-V2：MLA 替换 MHA、MoE 替换 FFN
   - Mamba：SSM 替换 Attention

### 学习目标

- [ ] 读懂 `llama.py` 的完整实现
- [ ] 能比较不同模型架构的差异点（Attention 机制、FFN 结构、Norm 位置等）
- [ ] 理解 `load_weights()` 的通用实现模式
- [ ] 知道如何为支持的模型添加新变体

### 思考题

1. LLaMA 的 RMSNorm 是在 attention/FFN 之前还是之后？与其他模型（如 GPT-2）的区别是什么？
2. DeepSeek-V2 的 MLA 相比标准 MHA 节省了多少 KV Cache 显存？
3. 如果一个模型使用 GQA（Grouped Query Attention），在代码中如何体现？

---

## 3.4 模型加载

### 核心概念

- **模型加载器（Model Loader）**：负责从磁盘加载权重并分配到各 GPU
- **权重参数管理**：`WeightsLoader` 处理权重的加载、分片、量化
- **模型卸载（Offloading）**：将不活跃的模型权重暂时移到 CPU

### 代码路径

| 文件 | 说明 |
|------|------|
| `model_executor/model_loader/` | 模型加载器目录 |
| `model_executor/parameter.py` ~22KB | 权重参数管理 |
| `model_executor/offloader/` | 模型卸载 |
| `model_executor/warmup/` | GPU 预热 |
| `model_executor/utils.py` | 工具函数 |
| `model_executor/models/utils.py` | 模型工具（~38KB，含权重加载辅助） |

### 阅读要点

1. **模型加载器**：
   - 支持哪些加载器（huggingface, aws s3, modelscope 等）
   - 加载流程：下载 → 读取 safetensors → 分片分配 → 加载到 GPU
   - CPU 内存管理：如何控制 CPU 内存峰值

2. **`parameter.py`**：
   - `ModelParallelParameter` vs `ReplicatedParameter`：何时复制何时分片
   - `ShardedStateLoader`：分布式权重的分片加载
   - `WeightsLoader` 的职责和接口

### 学习目标

- [ ] 理解模型加载的完整流程（从磁盘到 GPU）
- [ ] 知道 `ShardedStateLoader` 的工作原理
- [ ] 理解模型卸载（offloading）的触发条件

---

## 章节总结

### 模型实现层次结构

```
ModelConfig (config.json → vLLM 配置)
    │
    ▼
ModelRegistry.resolve() ──→ 定位模型类
    │
    ▼
ModelLoader.load_model()
    │
    ├── 下载 / 读取权重文件
    ├── 创建模型实例 (e.g., LlamaForCausalLM)
    ├── 调用 load_weights() 分配权重
    │   ├── ShardedStateLoader (分布式)
    │   └── weight_loader per parameter (量化感知)
    │
    ▼
模型就绪 ──→ Worker.execute_model() 调用
```

### 进一步阅读

- 继续阶段四：服务与入口层 → `04-serving-layer.md`
- HuggingFace Transformers 文档（理解 `config.json`）
- 各模型论文（LLaMA, DeepSeek-V2, Mamba 等）

---

*对应 LEARNING_PLAN.md 第 5 章 | 基于 vLLM 主分支*
