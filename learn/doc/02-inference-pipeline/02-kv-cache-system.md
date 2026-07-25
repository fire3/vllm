# KV Cache 系统（PagedAttention）

> KV Cache 管理是 vLLM 的核心创新，也是区别于其他推理框架的关键所在。vLLM 将操作系统虚拟内存的分页思想引入 Transformer 推理，实现了接近零碎片的内存复用。本章深入分析 v1 架构下 KV Cache 系统的完整设计。

---

## 1. 问题背景

### 为什么需要 KV Cache？

自回归解码中，每个 token 生成时需要用到之前所有 token 的 Key 和 Value 矩阵。朴素实现每次重算所有历史 token 的 K/V，计算复杂度为 O(n²)。KV Cache 将已算好的 K/V 缓存下来，新 token 只计算当前步的 K/V，复杂度降为 O(1)。

### 传统方案的碎片化问题

传统 LLM 推理服务中每个请求预分配连续显存以容纳其最大可能长度。这导致两个问题：

- **内部碎片**：请求实际生成远短于最大长度，预留空间用不完
- **外部碎片**：不同长度的请求释放后，剩余空间因不连续无法被新请求使用

vLLM 的 PagedAttention 正是为解决此问题而生。

---

## 2. 体系结构总览

KV Cache 系统分四层，从下到上：

```
BlockPool                     — 物理块池（原始资源）
    │
SingleTypeKVCacheManager     — 单类型管理器（每种 attention 类型一个实例）
    │
KVCacheCoordinator           — 多类型协调器（管理多个管理器实例）
    │
KVCacheManager               — 顶层门面（调度器直接调用的入口）
```

此外，`KVCacheSpec` 体系在编译期描述每层 `attention` 的内存布局，`KVCacheConfig` 承载运行时配置。

这种分层设计使得混合架构模型（如 DeepSeek-V2 同时使用 FullAttention + MLA + MoE）能够被统一管理。

---

## 3. Spec 层：KVCacheSpec 体系

### 文件位置

`vllm/v1/kv_cache_interface.py` —— 约 39KB

### 设计目标

每种 attention 类型（FullAttention、MLA、SlidingWindow、Mamba/SSM 等）的 KV Cache 布局不同，需要的块大小、量化方式、页对齐规则也不同。`KVCacheSpec` 是一个 frozen dataclass，以不可变值对象的身份描述这一切。

### 类层次

```
KVCacheSpec (frozen dataclass)          — 基类：block_size、page_size_bytes、max_num_blocks_per_req
    ├── AttentionSpec                   — 添加 num_kv_heads、head_size、dtype、kv_quant_mode
    │   ├── FullAttentionSpec           — 添加 head_size_v、sliding_window、non_causal
    │   │   ├── TQFullAttentionSpec     — 添加 tq_slot_size（TensorQueue 布局）
    │   │   ├── MLAAttentionSpec        — 添加 compress_ratio、cache_dtype_str（DeepSeek MLA）
    │   │   ├── CrossAttentionSpec      — Encoder-Decoder 交叉注意力
    │   │   ├── SinkFullAttentionSpec   — 带 sink buffer 的注意力
    │   │   └── RSWASpec                — Restricted Sliding Window Attention
    │   ├── SlidingWindowSpec           — 滑动窗口注意力
    │   │   └── SlidingWindowMLASpec    — 滑动窗口 + MLA
    │   ├── ChunkedLocalAttentionSpec   — 分块局部注意力
    │   ├── EncoderOnlyAttentionSpec    — Encoder-only（如 Whisper）
    │   └── HiddenStateCacheSpec        — 隐藏状态缓存（复用 MLA 布局）
    └── MambaSpec                       — SSM 层缓存（state_size、expand_factor、conv_kernel）
```

`UniformTypeKVCacheSpecs` 是一个包装器：当所有层使用同一种 attention 类型时，它将多个 spec 条目包装为单一对象，供上层批处理优化。

### 关键方法

每个 AttentionSpec 实现三个核心方法：

| 方法 | 用途 |
|------|------|
| `real_page_size_bytes()` | 计算物理页大小（考虑量化、对齐、压缩比） |
| `max_memory_usage_bytes(max_num_blocks)` | 估算最大显存占用 |
| `merge(other_spec)` | 合并两个 spec（验证兼容性、取最大值） |

### 量化模式 (`KVQuantMode`)

```python
class KVQuantMode(IntEnum):
    NONE = auto()
    FP8_PER_TENSOR = auto()       # 逐张量 FP8
    INT8_PER_TOKEN_HEAD = auto()  # 逐 token/head INT8
    NVFP4 = auto()                # NVIDIA FP4
    # ...
```

量化模式影响 `page_size_bytes` 的计算逻辑：压缩后的 KV 数据所需物理页更小，且对齐规则不同。

---

## 4. BlockPool：物理块池

### 文件位置

`vllm/v1/core/block_pool.py` —— 约 33KB

### 核心类

`BlockPool` 是物理资源的管理者，维护以下核心数据结构：

```
BlockPool
├── blocks: list[KVCacheBlock]            — 所有物理块（总数 = num_gpu_blocks）
├── free_block_queue: FreeKVCacheBlockQueue  — 空闲块双向链表（LRU 淘汰顺序）
├── cached_block_hash_to_block: BlockHashToBlockMap  — 哈希 → 物理块 索引
└── cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]]  — 块 → 哈希集 反向索引
```

### 块分配 (`get_new_blocks`)

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
```

分配逻辑：
1. 从 `free_block_queue` 头部弹出 `num_blocks` 个空闲块
2. 如果空闲块不足，调用 `_maybe_evict_cached_block()` 淘汰缓存块
3. 返回可用的物理块列表

### 前缀缓存 (`cache_full_blocks` / `get_cached_block`)

**哈希计算**：每个 Request 创建时，`Request.__init__` 调用 `update_block_hashes()`，基于 token 内容计算每个块的哈希值。

**缓存写入** (`cache_full_blocks`)：
1. 遍历 request 的 `block_hashes`，结合 `kv_cache_group_id` 生成 `BlockHashWithGroupId`
2. 写入 `cached_block_hash_to_block` 字典
3. 同时维护 `cached_block_hashes_by_block` 反向索引

**缓存查询** (`get_cached_block`)：
```python
def get_cached_block(self, block_hash, kv_cache_group_ids) -> list[KVCacheBlock] | None:
```
需要为每个 `kv_cache_group_id` 找到对应的缓存块，且要求**全部命中**（all-or-nothing）。

### 引用计数与淘汰

- `touch(blocks)`：增加引用计数，从 LRU 淘汰候选列表中移除
- `free_blocks(ordered_blocks)`：引用计数减一，减到 0 时追加到空闲队列尾部
- `FreeKVCacheBlockQueue` 是双向链表，头节点是最先被淘汰的候选，尾节点是最近被释放的

### 设计模式

**对象池 + LRU 淘汰 + 哈希辅助索引**。`BlockPool` 是典型的对象池模式，`BlockHashToBlockMap` 是叠加的辅助索引，不改变池本身的管理逻辑。

---

## 5. SingleTypeKVCacheManager

### 文件位置

`vllm/v1/core/single_type_kv_cache_manager.py` —— 约 82KB

### 类层次

```
SingleTypeKVCacheManager (ABC)
├── FullAttentionManager           — 标准全注意力
│   ├── RSWAManager                — 受限滑动窗口注意力
│   └── SinkFullAttentionManager   — Sink-aware 注意力
├── SlidingWindowManager           — 滑动窗口注意力
├── ChunkedLocalAttentionManager   — 分块局部注意力
├── MambaManager                   — SSM/Mamba
├── CrossAttentionManager          — Encoder-Decoder 交叉注意力
```

### 核心职责

每个 `SingleTypeKVCacheManager` 实例管理**一种 attention 类型**的 KV Cache。调度器通过 `KVCacheManager` 门面间接调用。

一个 vLLM 实例可能管理多个 `SingleTypeKVCacheManager` 实例（例如 FullAttention + Mamba）。

### 关键方法

| 方法 | 用途 |
|------|------|
| `find_longest_cache_hit(request)` | 在 `BlockPool` 中查找最长前缀缓存命中（子类实现不同策略） |
| `get_num_blocks_to_allocate(request_id, num_tokens, ...)` | 计算需要新分配的块数（考虑缓存命中、滑动窗口丢弃、CoW） |
| `add_local_computed_blocks(request_id, new_computed_blocks, ...)` | 增加已计算块（touch 缓存块、追加到 req_to_blocks） |
| `allocate_new_blocks(request_id, num_new_blocks)` | 从 BlockPool 获取新块，处理部分命中的 CoW |
| `free(request_id)` | 释放请求的所有块 |
| `cache_blocks(request, num_tokens_to_cache)` | 将块哈希写入 BlockPool（子类可限制窗口范围） |
| `remove_skipped_blocks(request_id, processed_computed_tokens, ...)` | 滑动窗口的块淘汰 |

### 各子类的缓存命中策略差异

| 子类 | 命中检测范围 | 窗口淘汰 |
|------|-----------|---------|
| `FullAttentionManager` | 全部 token，二分查找最长连续匹配 | 无 |
| `SlidingWindowManager` | 滑动窗口内的 token | 超出窗口的块被释放 |
| `ChunkedLocalAttentionManager` | attention_chunk_size 内 | 超出的块被释放 |
| `MambaManager` | 细粒度哈希匹配（非整块对齐） | 无 |
| `CrossAttentionManager` | 无缓存命中（encoder 输出不共享） | 无 |

### 工厂方法

```python
def get_manager_for_kv_cache_spec(kv_cache_spec, ...) -> SingleTypeKVCacheManager:
```

根据 `KVCacheSpec` 的类型分派到对应的子类构造函数。

---

## 6. KVCacheManager 门面

### 文件位置

`vllm/v1/core/kv_cache_manager.py` —— 约 37KB

### 定位

`KVCacheManager` 是**调度器直接调用的唯一入口**。它内部持有一个 `KVCacheCoordinator`，后者负责协调多个 `SingleTypeKVCacheManager` 实例。

### 关键方法

| 方法 | 用途 |
|------|------|
| `get_computed_blocks(request)` | 查找前缀缓存命中，返回 `KVCacheBlocks` |
| `allocate_slots(request, num_new_tokens, ...)` | 三段式分配：释放滑动窗口废弃块 → 分配已计算块 → 分配新块 |
| `free(request)` | 释放请求的全部 Cache 块 |
| `cache_blocks(request, num_computed_tokens)` | 缓存本次计算的块 |
| `new_step_starts()` | 重置每步状态（如 CoW 待办列表） |
| `take_events()` / `take_new_block_ids()` | 消费延迟操作事件 |

### KVCacheBlocks 数据传输对象

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # 每个 KV cache group 一个序列
```

它是调度器和 KVCacheManager 之间的数据契约：`allocate_slots` 返回 `KVCacheBlocks`，调度器将其中的块 ID 传递给 Model Runner。

### 三段式分配 (`allocate_slots`)

```
allocate_slots(request, num_new_tokens)
    │
    ├── 阶段一：KVCacheCoordinator.remove_skipped_blocks()
    │          释放滑动窗口淘汰的块
    │
    ├── 阶段二：KVCacheCoordinator.allocate_new_computed_blocks()
    │          增加已计算块（touch 缓存命中块，追加到 req_to_blocks）
    │
    └── 阶段三：KVCacheCoordinator.allocate_new_blocks()
               从 BlockPool 获取新物理块
```

如果任意阶段返回 `None`（内存不足），整个 `allocate_slots` 返回 `None`，触发调度器执行请求抢占。

---

## 7. KVCacheCoordinator 协调器

### 文件位置

`vllm/v1/core/kv_cache_coordinator.py` —— 约 37KB

### 定位

KVCacheCoordinator 管理多个 `SingleTypeKVCacheManager`，将 `KVCacheManager` 的统一调用分发到各管理器实例。

```python
class KVCacheCoordinator:
    managers: list[SingleTypeKVCacheManager]  # 每个 attention 类型一个
```

### 分发逻辑示例

```python
def allocate_new_blocks(self, request_id, num_new_blocks, ...):
    for manager in self.managers:
        manager.allocate_new_blocks(request_id, num_new_blocks[manager.group_id], ...)
```

---

## 8. 块表的 GPU 映射

### 文件位置

`vllm/v1/worker/block_table.py` —— 约 15KB

### 作用

调度器的 `KVCacheManager` 工作于 CPU 端，管理 `KVCacheBlock` 对象。但在 GPU 端，Attention kernel 需要一个扁平的整数张量：**block table**，将每个序列的逻辑块位置映射到物理块号。

```python
# 逻辑示意图
seq 0 的 block table: [3, 7, 12, 0]   # 逻辑块 0→物理块 3, 逻辑块 1→物理块 7, ...
seq 1 的 block table: [5, 9, 1, ...]
```

`BlockTable` 模块负责在 GPU Worker 执行模型前，将 `KVCacheBlocks` 转换为 GPU 上的 `block_table_tensor`。

---

## 9. 复制即写（Copy-on-Write）

### 触发场景

当两个请求共享相同的前缀 token，但前缀的末尾在一个**块中间**时，不能直接共享该块（因为后续内容不同）。此时：

1. 共享物理块保持不变（所有请求共用）
2. 为当前请求分配**新物理块**
3. 将共享块的内容复制到新块
4. 注册一次 `_pending_cow_copies` 事件，由 Worker 在 GPU 上执行复制

```python
# 流程示意
self._pending_cow_copies.append(
    KVCacheBlockCopy(source=shared_block, dest=new_block, ...)
)
```

### 实现位置

`SingleTypeKVCacheManager.allocate_new_blocks()` 中检测到部分命中时触发。

---

## 10. 分布式 Prefix Cache 事件

### 文件位置

`vllm/distributed/kv_events.py` —— 约 17KB

### 作用

在多 GPU 分布式部署中，不同 GPU 上的请求可能共享前缀。`BlockPool` 在缓存/淘汰块时生成 `KVCacheEvent`，通过 `take_events()` 方法消费，再通过分布式协调器广播到其他节点。

这使得跨 GPU 的前缀缓存得以同步。

---

## 11. 完整分配流程

以下是一次 Prefill 调度中 KV Cache 分配的完整路径：

```
Scheduler.schedule()
    │
    ▼
KVCacheManager.get_computed_blocks(request)
    ├── KVCacheCoordinator.find_longest_cache_hit()
    │       └── FullAttentionManager.find_longest_cache_hit()
    │               └── BlockPool.get_cached_block()  [二分查找哈希序列]
    │
    ▼
KVCacheManager.allocate_slots(request, num_new_tokens, ...)
    ├── remove_skipped_blocks()     [滑动窗口淘汰]
    ├── add_local_computed_blocks() [touch 缓存块]
    │       └── BlockPool.touch()
    ├── allocate_external_computed_blocks() [连接器计算的块]
    │       └── BlockPool.get_new_blocks()
    └── allocate_new_blocks()
            └── BlockPool.get_new_blocks() [新物理块]
    │
    ▼
(分配完成，返回 KVCacheBlocks)
    │
    ▼
SchedulerOutput → Worker.execute_model()
    │
    ▼
BlockTable 转换为 GPU tensor → Attention Kernel 使用
```

---

## 12. 设计模式总结

| 模式 | 体现位置 |
|------|---------|
| **门面（Facade）** | `KVCacheManager` 作为调度器的统一入口 |
| **策略（Strategy）** | `SingleTypeKVCacheManager` 各子类的差异化缓存命中策略 |
| **对象池（Object Pool）** | `BlockPool` 管理物理块分配与回收 |
| **空对象（Null Object）** | `BlockPool.null_block` 表示跳过/填充块 |
| **数据传输对象（DTO）** | `KVCacheBlocks` 作为层间契约 |
| **值对象（Value Object）** | `KVCacheSpec` 系列 frozen dataclass |
| **工厂方法（Factory）** | `get_manager_for_kv_cache_spec()` 按类型创建管理器 |
| **模板方法（Template Method）** | SingleTypeKVCacheManager 定义分配骨架，子类重写命中/淘汰逻辑 |

---

> **代码参考**：
> - `vllm/v1/kv_cache_interface.py` — KVCacheSpec 体系
> - `vllm/v1/core/block_pool.py` — 物理块池
> - `vllm/v1/core/kv_cache_manager.py` — 顶层管理门面
> - `vllm/v1/core/single_type_kv_cache_manager.py` — 单类型管理器（含子类）
> - `vllm/v1/core/kv_cache_coordinator.py` — 协调器
> - `vllm/v1/kv_cache_spec_registry.py` — Spec 注册
> - `vllm/v1/worker/block_table.py` — GPU 块表映射
> - `vllm/distributed/kv_events.py` — 分布式 Cache 事件
