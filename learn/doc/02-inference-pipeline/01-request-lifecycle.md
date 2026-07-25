# 请求生命周期

> 一条请求从进入引擎到输出 token 的完整生命历程——数据结构、状态转换和池化管理。

---

## 1. Request 数据模型

### 文件位置

`vllm/v1/request.py` —— 约 15KB

`Request` 是推理引擎中**请求的唯一载体**，贯穿从调度到执行的全程。

### 核心字段

| 字段 | 类型 | 用途 |
|------|------|------|
| `request_id` | `str` | 全局唯一标识 |
| `prompt_token_ids` | `list[int] \| None` | token 化的 prompt |
| `prompt_embeds` | `torch.Tensor \| None` | embedding 形式的 prompt |
| `sampling_params` | `SamplingParams \| None` | 采样参数 |
| `pooling_params` | `PoolingParams \| None` | 池化参数（embedding 模型） |
| `mm_features` | `list[MultiModalFeatureSpec]` | 多模态特征 |
| `status` | `RequestStatus` | 当前生命周期状态 |
| `num_computed_tokens` | `int` | 已计算的 token 数 |
| `_output_token_ids` / `_all_token_ids` | `list[int]` | 输出 / 全部 token ID |
| `block_hashes` | `list[BlockHash]` | 块哈希序列（前缀缓存用） |
| `num_preemptions` | `int` | 被抢占次数 |
| `max_tokens` | `int` | 最大生成长度 |
| `is_prefill_chunk` | `bool` | 是否为分块预填充 |

### 关键方法

| 方法 | 用途 |
|------|------|
| `append_output_token_ids(token_ids)` | 追加生成的 token |
| `update_block_hashes()` | 根据当前 token 内容重新计算块哈希 |
| `num_tokens() -> int` | 当前总 token 数（prompt + output） |
| `is_finished() -> bool` | 是否已完成 |
| `from_engine_core_request(EngineCoreRequest)` | 工厂方法：从引擎核心协议创建 |

---

## 2. 请求状态机

`RequestStatus` 枚举定义了请求的完整生命周期：

```
         add_request()
              │
              ▼
          WAITING ─────────────────────────────────────┐
              │                                        │
              │ schedule()                             │
              ▼                                        │
          RUNNING                                       │
              │                                        │
         ┌────┴────┐                                   │
         │         │                                   │
         │    memory不足                                │
         │    _preempt_request()                       │
         │         │                                   │
         │      PREEMPTED ───── 重新排队 ──────────────┘
         │
    token 耗尽/stop 条件触发
         │
         ▼
    FINISHED_STOPPED / FINISHED_LENGTH_CAPPED
    FINISHED_ABORTED / FINISHED_IGNORED
```

其他中间状态：
- `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` — 等待结构化输出的 grammar 编译
- `WAITING_FOR_STREAMING_INPUT` — 等待流式输入（流式会话）

---

## 3. 请求池（RequestPool）

### 文件位置

`vllm/v1/pool/` 目录

`RequestPool` 管理所有活跃请求的集合，提供 O(1) 的 `request_id → Request` 查找。

请求池同时也包含 Pooling 模型的支持：

```python
@dataclass
class PoolingCursor:
    """追踪 token 位置的游标，用于 pooling 模型的逐 token 处理"""
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor

@dataclass
class PoolingMetadata:
    prompt_lens: list[int]
    prompt_token_ids: list[list[int]]
    pooling_params: list[PoolingParams]
```

对于 Late Interaction 模型（如 ColBERT），`LateInteractionRunner` 维护了查询和文档嵌入的缓存：

```python
class LateInteractionRunner:
    _query_cache: dict    # query_key → embedding
    _query_uses: dict     # query_key → 剩余使用次数
```

---

## 4. 请求的输入来源

请求既可以来自 token ID，也可以来自 embedding：

```python
# Token ID 模式
Request(request_id="req-1", prompt_token_ids=[1, 2, 3, ...])

# Embedding 模式
Request(request_id="req-2", prompt_embeds=torch.Tensor(...))
```

Embedding 模式用于：
- 多模态模型的视觉/音频特征已编码为 embedding
- Prompt embedding API（直接传入 embedding 而非 token）

---

## 5. 流式会话

`StreamingUpdate` 类支持流式会话（持续对话）：

```python
@dataclass
class StreamingUpdate:
    mm_features: list
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams
    
    @classmethod
    def from_request(cls, request) -> StreamingUpdate
```

流式请求允许在同一个 `Request` 对象上持续追加对话内容（多轮对话），无需重建 KV Cache。

---

> **代码参考**：
> - `vllm/v1/request.py` — Request 数据模型
> - `vllm/v1/pool/` — 请求池与 pooling 管理
