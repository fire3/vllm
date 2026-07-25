# 模型实现分析

> `vllm/model_executor/models/` 目录包含 100+ 个模型文件。本章以 LLaMA 为起点逐步深化到复杂架构（MoE、MLA），分析模型实现的通用模式。

---

## 1. 模型文件清单

`model_executor/models/` 目录（约 100+ 个 `.py` 文件），按架构类型可大致分类：

| 类型 | 代表文件 | 说明 |
|------|---------|------|
| **标准 Dense** | `llama.py`, `qwen2.py`, `mistral.py`, `gemma.py` | 基本 Transformer 架构 |
| **MoE** | `mixtral.py`, `deepseek_v2.py`, `qwen2_moe.py` | 专家混合架构 |
| **多模态 (Vision)** | `llava.py`, `qwen2_vl.py`, `qwen3_vl.py`, `internvl.py` | 文本+图像 |
| **多模态 (Audio)** | `whisper.py`, `qwen2_audio.py`, `parakeet.py` | 文本+音频 |
| **SSM** | `mamba.py`, `mamba2.py` | 状态空间模型 |
| **Diffusion** | `diffusion_gemma.py` | 扩散模型 |
| **Spec Decode** | `medusa.py`, `mlp_speculator.py`, `eagle.py` | 投机解码草案模型 |
| **其他** | `bert.py`, `roberta.py` | Encoder-only 模型 |

---

## 2. 模型实现模板（以 LLaMA 为例）

### 文件位置

`vllm/model_executor/models/llama.py` —— 约 19KB

### 类层次

```
LlamaMLP(nn.Module)                         — MLP 块
LlamaAttention(nn.Module)                   — 注意力层
LlamaDecoderLayer(nn.Module)                — 单层 Transformer
LlamaModel(nn.Module)                       — 主干网络
LlamaForCausalLM(nn.Module, Mixin...)       — 完整模型
```

### LlamaMLP

```python
class LlamaMLP(nn.Module):
    def __init__(self, ...):
        # gate_proj + up_proj 融合为一个 MergedColumnParallelLinear
        self.gate_up_proj = MergedColumnParallelLinear(...)
        # down_proj：RowParallelLinear
        self.down_proj = RowParallelLinear(...)
    
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)
```

融合 `gate_proj` 和 `up_proj` 为 `MergedColumnParallelLinear` 是常见的优化——两个投影共享相同输入，融合为一个更大的矩阵乘法，提升 GPU 利用率。

### LlamaAttention

```python
class LlamaAttention(nn.Module):
    def __init__(self, ...):
        self.qkv_proj = QKVParallelLinear(...)
        self.o_proj = RowParallelLinear(...)
        # 使用 vLLM 的 Attention 后端
        self.attn = Attention(...)
    
    def forward(self, x, kv_cache, attn_metadata):
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(...)
        output = self.attn(q, k, v, kv_cache, attn_metadata)
        return self.o_proj(output)
```

`QKVParallelLinear` 将 Q、K、V 三个投影融合为一个，同时处理 TP 下的 head 切分。

### LlamaDecoderLayer

```python
class LlamaDecoderLayer(nn.Module):
    def __init__(self, ...):
        self.input_layernorm = RMSNorm(...)
        self.self_attn = LlamaAttention(...)
        self.post_attention_layernorm = RMSNorm(...)
        self.mlp = LlamaMLP(...)
    
    def forward(self, x, kv_cache, attn_metadata):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, kv_cache, attn_metadata)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x
```

`Pre-norm` 结构：LayerNorm 在子层**之前**而非之后。这已成为现代 LLM 的标准选择。

### LlamaForCausalLM

```python
@support_torch_compile
class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsQuant):
    def __init__(self, ...):
        self.model = LlamaModel(...)
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(...)
    
    def forward(self, input_ids, positions, kv_caches, attn_metadata, ...):
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
        return hidden_states
    
    def compute_logits(self, hidden_states) -> Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits
    
    def load_weights(self, weights):
        # 使用 AutoWeightsLoader 自动加载权重
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
```

### 权重加载模式

```python
def load_weights(self, weights):
    # 标准实现模式
    loader = AutoWeightsLoader(self)
    loader.load_weights(
        weights,
        # 可选的参数分组
        params_group=group_params,
    )
```

`AutoWeightsLoader` 自动匹配 HuggingFace 的 state dict 键名到 vLLM 的参数对象，处理 TP 切分和量化。

---

## 3. 从 LLaMA 到其他架构的演进

### Qwen2（与 LLaMA 的差异）

```python
# Qwen2 与 LLaMA 结构几乎一致，主要差异：
# 1. RoPE 频率计算不同（扩展的频段）
# 2. 默认使用 GQA（Grouped Query Attention）
class Qwen2Attention(nn.Module):
    def __init__(self, ...):
        # num_kv_heads < num_heads（GQA）
        self.qkv_proj = QKVParallelLinear(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,  # GQA
        )
```

### DeepSeek-V2（MLA + MoE）

`deepseek_v2.py`（约 75KB）是两个重大创新的组合：

1. **MLA（Multi-head Latent Attention）**：
   - 将 KV 压缩为低维 latent 表示，减少 KV Cache 显存
   - 实现位于 `layers/mla.py`

2. **MoE（Mixture of Experts）**：
   - 使用 `fused_moe` 层替代标准 FFN
   - 路由策略：top-2 路由

### Mamba（SSM）

`mamba.py`（约 9KB）完全替换了 Attention 层：

```python
class MambaMixer(nn.Module):
    # 使用状态空间模型替代注意力
    # 核心：selective scan 算法
    # 没有 KV Cache（无注意力缓存）
```

---

## 4. 模型注册与 MLPSpeculator

### 注册流程

每个模型文件不需要手动注册到 `registry.py`。注册是通过在 `_TEXT_GENERATION_MODELS` 字典中添加条目完成的。但模型文件本身需要定义那些在注册表中引用的类。

### 特殊模型类型

投机解码的草案模型（如 `MLPSpeculator`、`Medusa`、`Eagle`）也实现了完整的 `nn.Module` 接口，但被设计为与主模型协同工作：

```python
class MLPSpeculator(nn.Module):
    # 预测接下来 N 个 token 的表示
    # 主模型验证这些预测
```

---

> **代码参考**：
> - `vllm/model_executor/models/llama.py` — LLaMA 实现（最佳学习起点）
> - `vllm/model_executor/models/qwen2.py` — Qwen2（GQA 示例）
> - `vllm/model_executor/models/deepseek_v2.py` — DeepSeek-V2（MLA + MoE）
> - `vllm/model_executor/models/mamba.py` — Mamba（SSM）
> - `vllm/model_executor/models/qwen2_vl.py` — 多模态示例
> - `vllm/model_executor/models/registry.py` — 注册表
