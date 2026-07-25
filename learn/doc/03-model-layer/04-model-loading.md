# 模型加载

> 模型加载管道负责将 HuggingFace 格式的权重文件转换为分布在多个 GPU 上的 vLLM 模型实例。它涉及权重下载、格式转换、张量并行切分、量化重排等复杂步骤。

---

## 1. 加载器架构

### 文件位置

`vllm/model_executor/model_loader/` 目录

### 加载器策略

```python
_LOAD_FORMAT_TO_MODEL_LOADER = {
    "auto": DefaultModelLoader,
    "hf": DefaultModelLoader,
    "safetensors": DefaultModelLoader,
    "bitsandbytes": BitsAndBytesModelLoader,
    "dummy": DummyModelLoader,
    "runai_streamer": RunAIStreamerModelLoader,
    "tensorizer": TensorizerModelLoader,
    "sharded_state": ShardedStateModelLoader,
    # ...
}
```

`--load-format` 参数控制使用哪个加载器。

### 加载器基类

```python
class BaseModelLoader(ABC):
    @abstractmethod
    def download_model(self, model_config, ...):
        """下载模型权重"""
        pass
    
    @abstractmethod
    def load_model(self, *, vllm_config, ...) -> nn.Module:
        """加载模型并返回 nn.Module 实例"""
        pass
```

### 扩展机制

第三方可以通过 `register_model_loader()` 注册自定义加载器：

```python
from vllm.model_executor.model_loader import register_model_loader
register_model_loader("my_format", MyCustomLoader)
```

---

## 2. 权重参数管理

### 文件位置

`vllm/model_executor/parameter.py` —— 约 22KB

### 参数类层次

```
torch.nn.Parameter
    └── BasevLLMParameter           — 基类，携带 weight_loader 回调
        ├── ModelWeightParameter    — 标准模型权重
        ├── PackedvLLMParameter     — 融合权重（QKV, gate+up）
        ├── PerTensorScaleParameter — 每张量量化 scale
        ├── ChannelQuantScaleParameter — 每通道量化 scale
        ├── GroupQuantScaleParameter — 每分组量化 scale
        ├── RowvLLMParameter        — 行并行权重
        └── PackedColumnParameter   — 列并行融合权重
```

### 核心设计：weight_loader 回调

```python
class BasevLLMParameter(torch.nn.Parameter):
    def __init__(self, ..., weight_loader: Callable):
        self._weight_loader = weight_loader
    
    @property
    def weight_loader(self):
        return self._weight_loader
    
    def load_column_parallel_weight(self, loaded_weight, ...):
        """加载列并行切分的权重"""
        self.weight_loader(self, loaded_weight, ...)
    
    def load_row_parallel_weight(self, loaded_weight, ...):
        """加载行并行切分的权重"""
        self.weight_loader(self, loaded_weight, ...)
```

每个参数知道：
- **它的 TP 切分方式**（行/列）
- **它的量化格式**
- **它在哪个 rank 上**

这允许权重加载逻辑分散到各参数，而非集中在一个大函数中。

### 分布式加载

```python
class ShardedStateLoader:
    """从 DeepSpeed/FSDP 格式的 sharded state 加载"""
    def load_model(self, ...):
        # 只加载当前 rank 需要的权重分片
        shard = self._get_shard_for_rank(rank, world_size)
        model.load_weights(shard)
```

---

## 3. 权重加载流程

从 HuggingFace 格式加载权重的完整流程：

```
权重文件 (model-00001-of-00002.safetensors, ...)
    │
    ├── 1. 下载（从 HuggingFace Hub / 本地缓存）
    │      weight_utils.py 处理下载和缓存
    │
    ├── 2. 读取 safetensors 文件
    │      获取 state dict 的元信息（每个 tensor 的 shape、dtype）
    │
    ├── 3. 创建模型实例（未初始化权重）
    │      model = LlamaForCausalLM(config)
    │      此时参数已创建但未加载实际权重
    │
    ├── 4. load_weights(weights)
    │      │
    │      ├── AutoWeightsLoader 遍历 state dict
    │      ├── 对每个 tensor：
    │      │   ├── 匹配权重名到参数路径
    │      │   ├── 如果有 TP 切分：只加载当前 rank 的分片
    │      │   ├── 如果有量化：加载 scale / zero_point
    │      │   └── 调用 parameter.weight_loader(...)
    │      │
    │      └── process_weights_after_loading()
    │            └── 后处理：重排、融合、量化打包
    │
    └── 5. 模型就绪，分配到 GPU
           model.cuda() 或 model.to(device)
```

---

## 4. 权重工具

### 文件位置

`vllm/model_executor/model_loader/weight_utils.py` —— 约 60KB

提供：

| 函数 | 用途 |
|------|------|
| `download_weights()` | 从 HuggingFace Hub 下载权重 |
| `get_model_architecture()` | 从 config.json 获取架构名 |
| `get_model_cls()` | 从架构名获取模型类 |
| `resolve_model_path()` | 解析模型路径（本地/Hub） |

---

## 5. 模型卸载（Offloading）

### 文件位置

`vllm/model_executor/offloader/` 目录

当模型权重超过 GPU 显存容量时，将部分权重暂时移到 CPU 内存：

- 哪个层被卸载：通常是**较早的层**或**不常用的层**
- 何时加载回 GPU：当模型执行到该层时
- 性能影响：增加 PCIe 传输延迟，但允许运行超显存模型

---

## 6. GPU 预热

### 文件位置

`vllm/model_executor/warmup/` 目录

模型加载后、正式推理前，执行预热步骤：

1. **CUDA Graph 捕获**：预先捕获模型前向的 CUDA Graph
2. **KV Cache 预热**：分配并初始化 KV Cache 块
3. **内存估算**：`determine_num_available_blocks()` 确定可用块数

---

> **代码参考**：
> - `vllm/model_executor/model_loader/` — 加载器目录
> - `vllm/model_executor/parameter.py` — 参数管理
> - `vllm/model_executor/model_loader/default_loader.py` — 默认加载器实现
> - `vllm/model_executor/model_loader/weight_utils.py` — 权重下载工具
> - `vllm/model_executor/model_loader/sharded_state_loader.py` — 分布式加载
