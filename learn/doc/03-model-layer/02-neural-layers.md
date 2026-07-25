# 神经网络层

> `vllm/model_executor/layers/` 是 vLLM 的神经网络层库，可视为 PyTorch `nn.Module` 的推理优化扩展。每个核心算子都针对 LLM 推理场景做了专门优化——支持张量并行切分、多种量化格式和多 dtype。

---

## 1. 层库结构总览

```
layers/
├── linear.py                  — 线性层体系（Column/Row/QKV Parallel 等）
├── activation.py              — 激活函数（SwiGLU, GeGLU 等）
├── layernorm.py               — LayerNorm / RMSNorm
├── attention/                 — 注意力层封装
├── vocab_parallel_embedding.py — 并行 Embedding 和 LM Head
├── logits_processor.py        — Logits 处理器
├── fused_moe/                 — 融合 MoE 层
├── quantization/              — 量化层（AWQ, GPTQ, FP8 等）
├── mamba/                     — Mamba / SSM 层
├── rotary_embedding/          — RoPE 位置编码
├── pooler/                    — Pooling 层（embedding 模型用）
├── fusion/                    — 算子融合
├── mla.py                     — Multi-head Latent Attention
├── mhc.py                     — Multi-head Cross-attention
├── lightning_attn.py          — Lightning Attention（线性注意力）
└── ...
```

---

## 2. 线性层体系（核心）

### 文件位置

`vllm/model_executor/layers/linear.py` —— 约 71KB

线性层是 Transformer 中最基本的计算单元（QKV 投影、FFN 门控、输出投影等）。vLLM 的线性层体系围绕**张量并行**和**量化**两个维度展开。

### 类层次

```
LinearMethodBase (abstract)           — 量化策略接口
    ├── UnquantizedLinearMethod       — 无量化
    ├── Fp8LinearMethod               — FP8 量化
    ├── AWQLinearMethod               — AWQ 量化
    ├── GPTQLinearMethod (Marlin)     — GPTQ 量化
    └── ... (更多量化方法)

ParallelLinear (nn.Module)            — 并行线性基类
    ├── ColumnParallelLinear          — 按列切分权重
    ├── RowParallelLinear             — 按行切分权重
    ├── MergedColumnParallelLinear    — 融合多列切分（如 gate+up 投影）
    └── QKVParallelLinear             — Q/K/V 三组投影（head 维度切分）
```

### 张量并行下的权重切分

```
ColumnParallelLinear:
    Weight: [out_features, in_features]
                     ↓
    TP=2:   ┌──────────────────┐
            │  rank0: [1/2, *] │
            │  rank1: [1/2, *] │
            └──────────────────┘
    结果：每个 rank 计算部分输出，不需要通信

RowParallelLinear:
    Weight: [out_features, in_features]
                     ↓
    TP=2:   ┌──────────────────┐
            │  rank0: [*, 1/2] │
            │  rank1: [*, 1/2] │
            └──────────────────┘
    结果：每个 rank 计算部分和，需要 all-reduce 汇总

QKVParallelLinear:
    Q: [num_heads * head_dim, hidden]
    K: [num_kv_heads * head_dim, hidden]
    V: [num_kv_heads * head_dim, hidden]
    TP 下：Q 的 head 数被平分，K/V 的 kv_head 数也被平分
```

### 量化策略接口

`LinearMethodBase` 定义了量化线性层的契约：

```python
class LinearMethodBase(QuantizeMethodBase):
    def create_weights(self, layer, ...) -> None:
        """创建权重参数（含量化参数如 scale, zero_point）"""
        pass
    
    def apply(self, layer, x, ...) -> Tensor:
        """前向传播（含反量化+计算融合）"""
        pass
    
    def process_weights_after_loading(self, layer) -> None:
        """权重加载后的后处理（如重排）"""
        pass
```

每个量化方案（AWQ、GPTQ、FP8 等）实现此接口，线性层在初始化时注入对应的 `linear_method` 实例：

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, ..., linear_method: LinearMethodBase):
        self.linear_method = linear_method
        linear_method.create_weights(self, ...)
    
    def forward(self, x):
        return self.linear_method.apply(self, x)
```

---

## 3. 激活函数

### 文件位置

`vllm/model_executor/layers/activation.py` —— 约 30KB

vLLM 实现了 LLM 推理中常用的融合激活函数：

| 类 | 对应函数 | 说明 |
|----|---------|------|
| `SiluAndMul` | `silu(x) * y` | SwiGLU 的核心算子，将 gate 投影输出拆分为 `x` 和 `y` |
| `GeluAndMul` | `gelu(x) * y` | GeGLU 的核心算子 |
| `QuickGeluAndMul` | `quick_gelu(x) * y` | QuickGeLU 变体 |
| `NewGeluAndMul` | `new_gelu(x) * y` | 另一种 GeLU 近似 |

这些激活函数都是**融合实现**——单个 CUDA kernel 完成激活计算和逐元素乘法，避免中间张量读写。

---

## 4. 注意力层

### 文件位置

`vllm/model_executor/layers/attention/`

### 注意力层的模型侧封装

虽然 v1 架构通过 `AttentionBackend` 抽象了注意力计算，但模型层（如 `llama.py`）中仍然需要一个 `nn.Module` 来调用注意力后端：

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, ...):
        # QKV 投影（使用 QKVParallelLinear）
        self.qkv_proj = QKVParallelLinear(...)
        # 输出投影
        self.o_proj = RowParallelLinear(...)
    
    def forward(self, x, kv_cache, attn_metadata):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(...)
        # 调用 AttentionBackend 的实际注意力计算
        output = attention_backend.forward(q, k, v, kv_cache, attn_metadata)
        return self.o_proj(output)
```

---

## 5. 融合 MoE

### 文件位置

`vllm/model_executor/layers/fused_moe/` 目录

MoE（Mixture of Experts）层将 FFN 替换为多个"专家"子网络，由路由（Router）决定每个 token 使用哪些专家。

### 计算流程

```
输入 x
    │
    ├── Router: x → logits → softmax → top-k 选择
    │
    ├── 选中的专家 FFN 计算（gate_proj + up_proj + down_proj）
    │   └── 融合实现：一次 kernel launch 完成多个专家的计算
    │
    └── Combine：按权重加权求和
```

### 融合实现

融合 MoE 将"路由 → 专家计算 → 加权组合"融合为少量 CUDA kernel，大幅减少调度开销。支持：

- Top-1 / Top-2 路由
- 张量并行下的专家切分
- FP8 / INT8 量化专家

---

## 6. Embedding 与 LM Head

### 文件位置

`vllm/model_executor/layers/vocab_parallel_embedding.py` —— 约 22KB

### 类结构

| 类 | 用途 |
|----|------|
| `VocabParallelEmbedding` | Embedding 层，词表按 TP 维度切分 |
| `ParallelLMHead` | LM 预测头（unembedding），词表并行 |
| `UnquantizedEmbeddingMethod` | 无量化时的 embedding 策略 |

### 词表并行

当词表大小很大（>100K）时，embedding 矩阵无法放入单个 GPU 显存。`VocabParallelEmbedding` 将词表切分到各 TP rank：

- 前向：每个 rank 只查询自己负责的那部分词表
- LM Head：每个 rank 计算部分 logits → all-gather 合并

---

## 7. 其他关键层

| 层 | 文件 | 用途 |
|----|------|------|
| `RMSNorm` | `layers/layernorm.py` | 大多数现代 LLM 使用的 LayerNorm 变体 |
| `MambaMixer` | `layers/mamba/` | SSM 状态空间模型层 |
| `MLAAttention` | `layers/mla.py` | DeepSeek-V2 的 Multi-head Latent Attention |
| `RoPE` | `layers/rotary_embedding/` | 旋转位置编码 |
| `LogitsProcessor` | `layers/logits_processor.py` | 并行化的 logits 后处理 |

---

> **代码参考**：
> - `vllm/model_executor/layers/linear.py` — 线性层体系
> - `vllm/model_executor/layers/activation.py` — 融合激活函数
> - `vllm/model_executor/layers/attention/` — 注意力层
> - `vllm/model_executor/layers/fused_moe/` — 融合 MoE
> - `vllm/model_executor/layers/vocab_parallel_embedding.py` — 并行 Embedding
> - `vllm/model_executor/layers/layernorm.py` — 归一化层
> - `vllm/model_executor/layers/mla.py` — MLA
