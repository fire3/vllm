# 调度器

> 调度器是推理引擎的**决策中枢**——它决定每个推理步执行哪些请求、每个请求分配多少计算资源，以及在内存不足时如何优雅降级。

---

## 1. 文件结构

`vllm/v1/core/sched/` 目录下的文件：

| 文件 | 大小 | 用途 |
|------|------|------|
| `scheduler.py` | ~132KB | 主调度器实现 |
| `interface.py` | ~10KB | `SchedulerInterface` 抽象接口 |
| `output.py` | ~11KB | `SchedulerOutput` 及数据类 |
| `request_queue.py` | ~7KB | `RequestQueue` 优先级队列 |
| `async_scheduler.py` | ~3KB | 异步调度辅助 |
| `utils.py` | ~4KB | 工具函数 |
| `__init__.py` | 0B | |

---

## 2. SchedulerInterface 抽象

### 文件位置

`vllm/v1/core/sched/interface.py`

`SchedulerInterface` 定义了调度器对外暴露的全部方法：

| 方法 | 用途 |
|------|------|
| `schedule() -> SchedulerOutput` | 主调度方法，返回当前步需要执行的请求列表 |
| `update_from_output(scheduler_output, model_runner_output)` | 后处理：追加 token、更新哈希、释放完成的请求 |
| `add_request(request)` | 将新请求加入等待队列 |
| `finish_requests(request_ids, finished_status)` | 终止或完成请求 |
| `get_num_unfinished_requests() -> int` | 获取未完成的请求数 |
| `get_grammar_bitmask(request_ids) -> list[list[int]]` | 获取结构化输出的 grammar 位掩码 |
| `reset_prefix_cache() -> bool` | 重置前缀缓存 |
| `make_stats() -> SchedulerStats` | 收集调度统计指标 |

---

## 3. Scheduler 实现

### 文件位置

`vllm/v1/core/sched/scheduler.py` —— 约 132KB

### 核心调度循环 (`schedule` 方法)

```python
def schedule(self, throttle_prefills=False) -> SchedulerOutput:
```

每次 `schedule()` 调用对应引擎的一次 `step()`，大致流程：

```
schedule()
    │
    ├── 1. kv_cache_manager.new_step_starts()
    │      重置步状态（CoW 待办、事件队列）
    │
    ├── 2. 调度 running 请求（轮转）
    │      for req in running_requests:
    │          allocate_slots(req, num_new_tokens)
    │          扣除 token_budget
    │
    ├── 3. 调度 waiting 请求
    │      for req in waiting_requests:
    │          get_computed_blocks(req)        [前缀缓存查询]
    │          allocate_slots(req, num_new_tokens)
    │          if OOM:
    │              _preempt_request(running_req)  [抢占]
    │          else:
    │              move to running
    │
    ├── 4. 处理 preempted 请求
    │
    └── 5. 构建 SchedulerOutput
         ├── NewRequestData  (新一轮 Prefill)
         ├── CachedRequestData (继续 Decode)
         └── ...
```

### Token Budget（预算控制）

调度器维护 `token_budget`（每步最大处理的 token 数），防止单步计算量过大导致延迟抖动。

```
token_budget = min(
    max_num_batched_tokens,
    max_num_seqs * average_tokens_per_seq
)
```

每分配一个请求的 token 就扣减预算，预算耗尽时停止本轮调度。

### 抢占机制 (`_preempt_request`)

当内存不足（`allocate_slots` 返回 `None`）时触发：

```python
def _preempt_request(self, request, timestamp):
    # 1. 弹出请求的 KV 块（释放回 BlockPool）
    # 2. 重置 num_computed_tokens
    # 3. 移回 waiting 队列
    # 4. 递增抢占计数
```

每次被抢占的请求回到 waiting 队列末尾，等待下一次调度的重新调度。

### 请求队列

`RequestQueue` 实现基于 `SchedulingPolicy` 的优先级排序：

```python
class SchedulingPolicy(Enum):
    FCFS = "fcfs"          # 先来先服务
    PRIORITY = "priority"  # 优先级队列
    # ...
```

`create_request_queue(policy)` 工厂函数根据策略返回对应的队列实现。

---

## 4. SchedulerOutput 数据传输

### 文件位置

`vllm/v1/core/sched/output.py`

每个 `schedule()` 返回一个 `SchedulerOutput`，包含当前步需要执行的所有请求信息：

```python
@dataclass
class SchedulerOutput:
    requests: list[Request]                # 本次执行的请求
    scheduled_requests: list[NewRequestData | CachedRequestData]
    num_scheduled_tokens: int              # 总 token 数
    preempted_requests: list[Request]       # 被抢占的请求
    finished_requests: list[Request]        # 刚完成的请求
    # ...
```

`NewRequestData` 包含首次进入的请求的完整信息（prompt、采样参数等），`CachedRequestData` 只包含增量信息（继续 Decode 的请求）。

---

## 5. Prefill 与 Decode 的混合调度

一个关键设计是 **Prefill 和 Decode 在同一 batch 中混合执行**。调度策略需要权衡：

- **Prefill（预填充）**：计算密集型，处理大量 prompt token，产生初始 KV Cache
- **Decode（解码）**：内存带宽密集型，每步只处理 1 个 token

调度器通过 `token_budget` 在同一 batch 中同时包含 Prefill 和 Decode 请求，最大化 GPU 利用率。

```
一个典型 batch 的组成：
┌────────────────────────────────────────┐
│  Decode  Decode  Prefill  Decode  .... │
│  (1 tok) (1 tok) (512 tok) (1 tok)    │
└────────────────────────────────────────┘
```

---

## 6. 调度与 KV Cache 管理器的交互

调度器不是直接管理内存，而是调用 `KVCacheManager`：

```python
# 调度器中的典型调用序列
def schedule(self, ...):
    # 1. 前缀缓存查询
    computed_blocks, num_computed_tokens, shared_boundary = \
        self.kv_cache_manager.get_computed_blocks(request)
    
    # 2. 分配插槽
    kv_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...)
    
    if kv_blocks is None:
        # OOM: 触发抢占
        self._preempt_request(some_running_request)
        return self.schedule(...)  # 重试
    
    # 3. 缓存块
    self.kv_cache_manager.cache_blocks(request, num_computed_tokens)
```

---

## 7. 设计要点总结

| 方面 | 设计 |
|------|------|
| **调度策略** | 轮转（Round-Robin）+ 先来先服务 |
| **资源控制** | Token Budget + Watermark |
| **内存不足处理** | 抢占（Preemption）低优先级请求 |
| **Prefill/Decode 混合** | 同一 batch 混合，通过 budget 控制比例 |
| **与 Cache 的交互** | 通过 KVCacheManager 门面 |
| **输出契约** | SchedulerOutput 数据传输对象 |

---

> **代码参考**：
> - `vllm/v1/core/sched/scheduler.py` — 主调度器
> - `vllm/v1/core/sched/interface.py` — 调度器抽象接口
> - `vllm/v1/core/sched/output.py` — SchedulerOutput
> - `vllm/v1/core/sched/request_queue.py` — 请求优先级队列
