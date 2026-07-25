# 模型接口与注册

> vLLM 支持 100+ 种模型架构。模型接口（Mixin）和注册表（Registry）构成了这套模型支持体系的骨架——前者声明模型能力，后者实现模型发现。

---

## 1. 模型接口 Mixin 体系

### 文件位置

`vllm/model_executor/models/interfaces.py` —— 约 55KB

### 设计动机

不同模型的能力各不相同：

- 有的支持多模态输入（`SupportsMultiModal`）
- 有的支持 LoRA 微调（`SupportsLoRA`）
- 有的支持流水线并行切分（`SupportsPP`）
- 有的支持量化（`SupportsQuant`）
- 有的用作 Eagle 投机解码的草稿模型（`SupportsEagle`）

vLLM 使用 **Protocol（协议类型）** 而非继承来表达这些能力。这是 Python 中"鸭子类型"的正式化：

```python
@runtime_checkable
class SupportsMultiModal(Protocol):
    supports_multimodal: ClassVar[Literal[True]] = True
    
    def get_placeholder_str(self) -> str: ...
    def embed_multimodal(self, ...) -> MultiModalEmbeddings: ...
    def configure_mm_token_handling(self, ...): ...
```

### Protocol 的优势

```python
# 不需要继承，满足接口即可
class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsQuant):
    ...

# runtime_checkable 允许 isinstance 检查
if isinstance(model, SupportsMultiModal):
    mm_embeds = model.embed_multimodal(mm_inputs)
```

### 关键 Mixin 列表

| Mixin | 标记字段 | 用途 |
|-------|---------|------|
| `SupportsMultiModal` | `supports_multimodal` | 支持多模态输入 |
| `SupportsPP` | 无（通过 `get_hidden_states()` 检测） | 支持流水线并行切分 |
| `SupportsLoRA` | 无 | 支持 LoRA 适配器 |
| `SupportsQuant` | 无 | 支持量化推理 |
| `SupportsEagle` / `SupportsEagle3` | 无 | 用作 Eagle 投机解码草案模型 |
| `SupportsAudio` | 无 | 支持音频输入 |
| `HasInnerState` | 无 | 模型内部维护状态（如 SSM） |

---

## 2. 模型注册表

### 文件位置

`vllm/model_executor/models/registry.py` —— 约 60KB

### 数据结构

核心是一个字典，将 HuggingFace `config.json` 中的 `architectures` 字段名映射到 vLLM 的实现类：

```python
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "MistralForCausalLM": ("mistral", "MistralForCausalLM"),
    # ... 100+ 条目
}
```

每个条目是一个 `(module_subdir, class_name)` 元组，指向 `model_executor/models/{module_subdir}.py` 中的类。

### 模型发现流程

```
模型加载时
    │
    ├── 1. 从 config.json 读取 "architectures": ["LlamaForCausalLM"]
    │
    ├── 2. 在 _TEXT_GENERATION_MODELS 中查找
    │      → ("llama", "LlamaForCausalLM")
    │
    ├── 3. 动态导入：from .llama import LlamaForCausalLM
    │
    └── 4. 实例化模型
```

### 模型类型判断

注册表模块导出了一组判断函数，用于在运行时确定模型类别：

| 函数 | 用途 |
|------|------|
| `is_text_generation_model(model_cls)` | 是否为文本生成模型 |
| `is_pooling_model(model_cls)` | 是否为 embedding/pooling 模型 |
| `has_inner_state(model_cls)` | 是否维护内部状态（如 Mamba） |
| `supports_multimodal(model_cls)` | 是否支持多模态 |
| `supports_pp(model_cls)` | 是否支持流水线并行 |
| `is_hybrid(model_cls)` | 是否为混合架构（MoE + Dense） |

---

## 3. 模型配置映射

### 文件位置

`vllm/model_executor/models/config.py` —— 约 38KB

### 作用

每个模型架构可能有独特的配置校验逻辑。`VerifyAndUpdateConfig` 基类提供两个钩子：

```python
class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config) -> None:
        """加载前校验和修改配置"""
        pass
    
    @staticmethod
    def verify_and_update_model_config(model_config) -> None:
        """修改 ModelConfig"""
        pass
```

### 实现示例

```python
# DeepSeek-V3.2：强制 KV Cache 数据类型为 auto
class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config):
        if vllm_config.model_config.kv_cache_dtype != "auto":
            logger.warning("DeepSeekV3.2 forces kv_cache_dtype=auto")
            vllm_config.model_config.kv_cache_dtype = "auto"
```

```python
# Gemma-3：根据 use_bidirectional_attention 设置 causal
class Gemma3TextModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config):
        model_config.is_causal = not model_config.use_bidirectional_attention
```

---

## 4. 模块映射

### 文件位置

`vllm/model_executor/models/module_mapping.py` —— 约 1KB

这个模块定义了 HuggingFace 模块名到 vLLM 模块名的映射，用于权重加载时的名称匹配：

```python
# 示例映射条目
"model.layers.{i}.self_attn.q_proj": "model.layers.{i}.self_attn.qkv_proj"
```

---

> **代码参考**：
> - `vllm/model_executor/models/interfaces.py` — Mixin 协议体系
> - `vllm/model_executor/models/registry.py` — 模型注册表
> - `vllm/model_executor/models/config.py` — 配置映射
> - `vllm/model_executor/models/module_mapping.py` — 模块名映射
