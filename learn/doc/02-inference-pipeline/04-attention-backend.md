# 注意力后端

> 注意力计算是 Transformer 推理的核心算子。vLLM 通过策略模式抽象注意力后端，支持 FlashAttention、FlashInfer、Triton 等多种实现，允许用户根据硬件和模型选择最优方案。

---

## 1. 架构总览

注意力后端分为三层抽象：

```
AttentionBackend (ABC)          — 后端的整体描述（元信息、能力声明）
    │
    ├── AttentionImplBase       — 注意力计算的具体实现（前向传播）
    │   └── AttentionImpl       — 标准注意力实现
    │   └── MLAAttentionImpl    — MLA（Multi-head Latent Attention）实现
    │
    ├── AttentionMetadata       — 注意力元数据基类
    │   └── CommonAttentionMetadata  — 通用批处理元数据
    │
    └── AttentionMetadataBuilder(ABC)  — 元数据构建器（从 SchedulerOutput 构建）
```

这种分层设计使得**后端选择**与**元数据构建**、**计算执行**三个维度各自独立，便于组合。

---

## 2. AttentionBackend 抽象

### 文件位置

`vllm/v1/attention/backend.py` —— 约 40KB

### 类级属性（能力声明）

每个后端通过类级属性声明其能力，供选择器决策：

```python
class AttentionBackend(ABC):
    # 支持的 dtype
    supported_dtypes: set[torch.dtype]
    supported_kv_cache_dtypes: set[str]
    
    # 能力查询方法
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool: ...
    @classmethod
    def supports_sliding_window(cls) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
    @classmethod
    def supports_sparse(cls) -> bool: ...
    @classmethod
    def supports_mla(cls) -> bool: ...
```

### 静态工厂方法

```python
@staticmethod
@abstractmethod
def get_name() -> str: ...

@staticmethod
@abstractmethod
def get_impl_cls() -> type[AttentionImplBase]: ...

@staticmethod
@abstractmethod
def get_builder_cls() -> type[AttentionMetadataBuilder]: ...
```

每个 `AttentionBackend` 子类必须告诉框架：它的名字是什么、用哪个类执行注意力计算、用哪个类构建元数据。

---

## 3. 注意力实现（AttentionImpl）

### AttentionImpl（标准注意力）

```python
class AttentionImpl(AttentionImplBase[T]):
    @abstractmethod
    def __init__(self, num_heads: int, head_size: int, scale: float, ...):
        ...
    
    @abstractmethod
    def forward(self, layer, query, key, value, kv_cache,
                attn_metadata, output, ...) -> torch.Tensor:
        ...
```

### MLAAttentionImpl（DeepSeek 式 MLA）

```python
class MLAAttentionImpl(AttentionImplBase[T]):
    @abstractmethod
    def __init__(self, ..., q_lora_rank: int, kv_lora_rank: int,
                 qk_nope_head_dim: int, ...):
        ...
    
    # Prefill 模式
    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, ...) -> None
    
    # Decode 模式
    def forward_mqa(self, q, kv_c_and_k_pe_cache, ...) -> Tuple[Tensor, ...]
```

MLA 将 KV 压缩为低维 latent 表示，大幅减少 KV Cache 显存占用。

---

## 4. 注意力元数据（AttentionMetadata）

### 通用元数据

`CommonAttentionMetadata` 包含一次批处理所需的全部描述信息：

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor    # 每个请求 query 的起始位置
    seq_lens: torch.Tensor           # 每个请求的序列长度
    num_reqs: int                    # 请求数
    max_query_len: int               # 最大 query 长度
    max_seq_len: int                 # 最大序列长度
    block_table_tensor: torch.Tensor # 块表（逻辑块→物理块映射）
    slot_mapping: torch.Tensor       # slot 映射（token→cache 位置）
    causal: bool                     # 因果掩码
    positions: torch.Tensor          # 位置编码
    is_prefilling: list[bool]        # 哪些是 Prefill
    # ...
```

`block_table_tensor` 是 vLLM 注意力计算的核心——它将连续的逻辑序列映射到分页存储的物理块，是 PagedAttention 在 GPU 端的数据基础。

### 各后端子类元数据

具体后端（如 FlashAttentionBackend）会继承 `CommonAttentionMetadata` 并添加后端特有的字段：

```python
class FlashAttentionMetadata(CommonAttentionMetadata):
    # FlashAttention 特有的元数据字段
    ...
```

---

## 5. 元数据构建器（AttentionMetadataBuilder）

每个后端提供自己的 `AttentionMetadataBuilder`，负责将 `SchedulerOutput` 和 `CommonAttentionMetadata` 转换为该后端所需的完整元数据：

```python
class AttentionMetadataBuilder(ABC, Generic[M]):
    @abstractmethod
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> M: ...
    
    # 可选：CUDA Graph 捕获模式
    def build_for_cudagraph_capture(self, ...) -> M: ...
```

构建器模式的引入使得元数据的构建逻辑与计算逻辑解耦。

---

## 6. 后端选择器

### 文件位置

`vllm/v1/attention/selector.py` —— 约 8KB

### 选择流程

```python
def get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, ...) -> type[AttentionBackend]:
    # 1. 构建选择器配置
    config = AttentionSelectorConfig(...)
    
    # 2. 检查 per-kind 覆盖（注意力类型级别的后端切换）
    if attention_config.backend_per_kind:
        spec_kind = get_attn_spec_kind(...)
        backend = attention_config.backend_per_kind.get(spec_kind)
        if backend:
            return resolve_backend(backend)
    
    # 3. 回退到全局后端
    backend = attention_config.backend or "FLASH_ATTN"
    
    # 4. 平台兼容性检查 + 缓存
    return _cached_get_attn_backend(platform, backend, config)
```

`_cached_get_attn_backend` 使用 `@cache` 装饰器缓存结果，避免重复解析。

### 选择优先级

1. **每个 attention 类型的后端覆盖**（`backend_per_kind`）—— 粒度最细
2. **全局默认后端**（`attention_config.backend`）
3. **平台默认**（`current_platform.get_attn_backend_cls()`）—— 最宽松

**强制指定**：用户可以通过环境变量 `VLLM_ATTENTION_BACKEND` 或启动参数 `--attention-backend` 强制选择。

---

## 7. 可用后端一览

### 标准注意力后端

`vllm/v1/attention/backends/` 目录下的实现：

| 后端 | 文件 | 说明 |
|------|------|------|
| `FLASH_ATTN` | `flash_attn.py` (67KB) | FlashAttention-2/3 |
| `FLASH_ATTN_DIFFKV` | `flash_attn_diffkv.py` (13KB) | DiffKV 变体 |
| `FLASH_INFER` | `flashinfer.py` (101KB) | FlashInfer |
| `TRITON_ATTN` | `triton_attn.py` (34KB) | Triton 实现 |
| `TRITON_ATTN_DIFFKV` | `triton_attn_diffkv.py` (10KB) | Triton DiffKV |
| `FLEX_ATTENTION` | `flex_attention.py` (58KB) | PyTorch FlexAttention |
| `HPC_ATTN` | `hpc_attn.py` (20KB) | Hopper GPU 优化 |
| `CPU_ATTN` | `cpu_attn.py` (17KB) | CPU 推理 |
| `TURBO_QUANT` | `turboquant_attn.py` (38KB) | 量化注意力 |

### MLA 后端

`backends/mla/` 子目录：

| 后端 | 说明 |
|------|------|
| `FLASH_INFER_MLA` | FlashInfer 的 MLA 支持 |
| `TRITON_MLA` | Triton 实现的 MLA |
| `CUTLASS_MLA` | CUTLASS 实现的 MLA |
| `FLASH_MLA` | FlashAttention 的 MLA 支持 |
| `FLASH_ATTN_MLA` | FlashAttn 的 MLA 变体 |

### SSM/Mamba 后端

| 后端 | 说明 |
|------|------|
| `MAMBA_ATTN` | Mamba/SSM 的注意力实现 |
| `MAMBA1_ATTN` | Mamba-1 专用 |
| `MAMBA2_ATTN` | Mamba-2 专用 |
| `LINEAR_ATTN` | 线性注意力 |
| `SHORT_CONV_ATTN` | 短卷积注意力 |

### ROCm（AMD）后端

| 后端 | 说明 |
|------|------|
| `ROCM_ATTN` | AMD ROCm 通用 |
| `AITER_FLASH_ATTN` | Aiter 方案的 FlashAttention |
| `AITER_UNIFIED_ATTN` | Aiter 统一注意力 |

### 注册机制

所有后端通过注册表自动发现：

```python
# AttentionBackendEnum 包含 40+ 个命名后端
class AttentionBackendEnum(str, Enum):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_INFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    # ...

# 通过 qualname 延迟解析
resolve_obj_by_qualname("vllm.v1.attention.backends.flash_attn.FlashAttentionBackend")
```

---

## 8. AttentionLayer Protocol

`AttentionLayer` 是一个 `Protocol`（结构类型）：

```python
class AttentionLayer(Protocol):
    _q_scale: torch.Tensor
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor
    
    def forward(self, query, key, value, kv_cache, attn_metadata):
        ...
```

任何满足此协议的 `nn.Module` 都可以作为注意力层插入模型，实现了计算层无关性。

---

> **代码参考**：
> - `vllm/v1/attention/backend.py` — 后端抽象
> - `vllm/v1/attention/selector.py` — 选择器
> - `vllm/v1/attention/backends/` — 各后端实现
> - `vllm/v1/attention/backends/mla/` — MLA 后端
> - `vllm/v1/attention/backends/registry.py` — 注册表
