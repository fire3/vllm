# 引擎层

> 引擎是推理引擎的"心脏"——它连接调度器、Worker、API 层，驱动整个推理循环（Scheduling Loop），管理请求的排队、执行和后处理。

---

## 1. 引擎的定位

引擎在整个 vLLM 架构中处于**承上启下**的位置：

```
API Server / LLM API
    │
    ▼
Engine (v1/engine/)
    │
    ├── Scheduler (v1/core/sched/)     — 决策层
    ├── Executor (v1/executor/)        — 分布式执行层
    └── Worker (v1/worker/)            — 执行层
    │
    ▼
Model / Attention / Sampler
```

---

## 2. 引擎主循环

### 文件位置

`vllm/v1/engine/` 目录

引擎的核心是一个**事件循环**，驱动推理的所有步骤：

```
while not all_requests_finished:
    # 1. 调度
    scheduler_output = scheduler.schedule()
    
    # 2. 执行（分布式）
    model_runner_output = executor.execute_model(scheduler_output)
    
    # 3. 采样（如果 deferred）
    if model_runner_output is None:
        # 使用 grammar 输出约束 logits
        grammar_output = scheduler.get_grammar_bitmask(...)
        model_runner_output = executor.sample_tokens(grammar_output)
    
    # 4. 后处理
    scheduler.update_from_output(scheduler_output, model_runner_output)
    
    # 5. 回传结果给 API 层
    for finished_request in scheduler_output.finished_requests:
        api_server.stream_output(finished_request)
```

这个循环在 API 服务器中持续运行，直到所有请求完成或被取消。

---

## 3. 执行器（Executor）

### 文件位置

`vllm/v1/executor/` 目录

执行器管理**跨 Worker 的分布式执行**，将 `SchedulerOutput` 分发到各个 GPU Worker。

### 单 GPU 模式

单 GPU 模式下，执行器直接将 `SchedulerOutput` 传给本地的 `Worker.execute_model()`。

### 多 GPU 模式

多 GPU 模式下，执行器：

1. 将 `SchedulerOutput` 广播到所有 Worker
2. 收集各 Worker 的输出（PP 模式下只有尾节点有完整输出）
3. 归约采样结果

---

## 4. 引擎与 API 层的接口

### 文件位置

`vllm/engine/protocol.py` —— 约 8KB

引擎对外暴露的接口由 `EngineCoreProtocol` 或类似协议定义：

```
Engine API
├── add_request(request)              — 添加新请求
├── abort_request(request_id)         — 中止请求
├── step() → list[RequestOutput]     — 单步推进
├── get_num_unfinished_requests()    — 查询调度器状态
└── shutdown()                       — 关闭引擎
```

### LLM 类（离线 API）与引擎的关系

`vllm/entrypoints/llm.py` 中的 `LLM` 类内部封装了一个引擎实例：

```python
class LLM:
    def __init__(self, ...):
        self.engine = Engine(...)
    
    def generate(self, prompts, ...):
        for prompt in prompts:
            self.engine.add_request(prompt)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            yield outputs
```

---

## 5. 引擎的启动流程

引擎从初始化到就绪的完整流程：

```
engine.__init__()
    │
    ├── 1. 解析配置（EngineArgs → ModelConfig, CacheConfig, ...）
    │
    ├── 2. 初始化分布式环境（parallel_state）
    │
    ├── 3. 初始化 KV Cache 管理器（BlockPool + KVCacheManager）
    │
    ├── 4. 创建调度器（Scheduler）
    │
    ├── 5. 创建 Worker（并加载模型）
    │      ├── GPUWorker.__init__()
    │      └── GPUModelRunner.__init__() → 加载模型权重
    │
    ├── 6. 分配 KV Cache（determine_num_available_blocks）
    │
    ├── 7. 预热（warmup）
    │      ├── CUDA Graph 捕获
    │      └── 模型热身推理
    │
    └── 8. 就绪，等待请求
```

---

## 6. 数据流总结

以一次 `step()` 为例，完整的数据流如下：

```
┌────────────────────────────────────────────────────────┐
│                     Engine.step()                       │
│                                                        │
│  1. scheduler.schedule()                               │
│     → 返回 SchedulerOutput                              │
│       ├── scheduled_requests: 调度了哪些请求             │
│       ├── num_scheduled_tokens: 总 token 数             │
│       └── preempted/finished 请求列表                    │
│                                                        │
│  2. executor.execute_model(scheduler_output)           │
│     → 分发到所有 GPU Worker                              │
│       │                                                 │
│       └── GPUWorker.execute_model()                     │
│             └── GPUModelRunner.execute_model()          │
│                   ├── _update_states()                  │
│                   ├── _prepare_inputs()                 │
│                   ├── _model_forward()                  │
│                   │   ├── attention (via backend)       │
│                   │   ├── mlp / moe                     │
│                   │   └── lm_head → logits              │
│                   └── return None (deferred)            │
│                                                        │
│  3. sample_tokens()                                    │
│     → 消费 defered logits                               │
│       └── GPUModelRunner.sample_tokens()               │
│             ├── apply_grammar_bitmask()                 │
│             └── sampler.forward()                       │
│                   └── sample() → next tokens            │
│                                                        │
│  4. scheduler.update_from_output(scheduler_output,     │
│                                   model_runner_output)  │
│     → 更新调度器状态：追加 token、完成标记                │
│                                                        │
│  5. 返回 completed Outputs 给 API 层                    │
└────────────────────────────────────────────────────────┘
```

---

> **代码参考**：
> - `vllm/v1/engine/` — v1 引擎核心
> - `vllm/v1/executor/` — 分布式执行器
> - `vllm/engine/protocol.py` — 引擎协议接口
> - `vllm/engine/arg_utils.py` — 引擎参数
