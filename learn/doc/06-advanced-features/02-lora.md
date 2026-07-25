# LoRA 适配器

## 文件位置

`vllm/lora/` 目录

## 核心类

| 类 | 文件 | 用途 |
|----|------|------|
| `LoRAModelManager` | `model_manager.py` (53KB) | 多 LoRA 适配器生命周期的管理器 |
| `LoRAModel` | `lora_model.py` (12KB) | 单个 LoRA 适配器的包装 |
| `LoRALayerWeights` | `lora_weights.py` (9KB) | LoRA 权重参数（A、B 矩阵） |
| `AdapterLRUCache` | `model_manager.py` | 适配器的 LRU 缓存，淘汰时 deactivate |

## 多 LoRA 管理

`LoRAModelManager` 的核心职责：

1. **注册**：模型启动时注册所有可用 LoRA 适配器
2. **切换**：每个请求携带 `lora_request`，执行时切换到对应的适配器
3. **缓存**：热点适配器常驻 GPU，冷门适配器自动卸载
4. **融合**：将 LoRA 权重与原模型权重融合计算

```python
class LoRAModelManager:
    def __init__(self, model, ...):
        self.lora_slots = ...  # GPU 上的 LoRA 插槽
    
    def add_adapter(self, lora_request):
        # 加载 LoRA 权重到 GPU 插槽
    
    def remove_adapter(self, lora_request):
        # 卸载 LoRA 权重，释放 GPU 插槽
```

## 集成到推理管线

`lora/lora_model_runner_mixin.py` 将 LoRA 注入 `GPUModelRunner`：

```python
class LoRAModelRunnerMixin:
    def execute_model(self, scheduler_output):
        # 前向时：根据每个请求的 lora_request
        # 将 LoRA 权重注入 Attention / MLP 层
```

## 搜索路径

`lora/resolver.py` 负责根据请求的 `lora_request` 找到对应的适配器：

- 策略：基于 `lora_name` 或 `lora_path` 匹配
