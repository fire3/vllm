# 阶段二：核心推理管线（v1 架构）

> 学习目标：完整理解一条请求从进入引擎到输出 token 的完整生命周期，重点掌握 PagedAttention 的 KV Cache 管理、调度策略、注意力计算、采样过程以及 Worker/Model Runner 的执行编排。

---

## 2.1 请求生命周期全景

### 核心概念

一条请求在 vLLM 中的完整路径：

```
Client → API Server → Engine → Scheduler → Worker → Model Runner
    → Attention Backend → Sampler → Output → Client
```

关键阶段：
- **Prefill**：处理 prompt，计算第一个 token 的 KV Cache
- **Decode**：逐 token 自回归生成
- **Continuous Batching**：Prefill 和 Decode 可以在同一 batch 中混合执行

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/request.py` ~15KB | `Request` 数据类：请求的完整状态 |
| `v1/pool/` | `RequestPool`：管理所有活跃请求 |
| `v1/engine/` | v1 引擎核心 |
| `v1/outputs.py` ~14KB | 输出数据结构 |
| `vllm/inputs/` | 输入预处理 |

### 阅读要点

1. **`Request` 数据结构**：
   - 请求的完整状态字段：`request_id`, `prompt`, `sampling_params`, `state` 等
   - 状态机：`WAITING → RUNNING → FINISHED`
   - 与旧版本 `SequenceGroup` 的关系

2. **`RequestPool`**：
   - 如何跟踪所有活跃请求
   - 如何支持优先级排序

### 学习目标

- [ ] 画出请求的完整生命周期状态图
- [ ] 理解 Prefill 与 Decode 阶段的区别
- [ ] 知道 `Request` 包含哪些核心字段

---

## 2.2 KV Cache 系统（vLLM 的灵魂）

这是 vLLM 最核心的创新，**必须深入理解**。

### 核心概念

- **PagedAttention**：将 KV Cache 划分为固定大小的块（Block），类似操作系统的分页
- **逻辑块 ↔ 物理块映射**：连续的逻辑序列映射到不连续的物理内存
- **Copy-on-Write**：多个请求共享相同 prompt 前缀时共用物理块，写入时复制
- **Block Table**：每个序列维护一个块表，记录逻辑块对应的物理块号
- **Memory Pool**：物理块的预分配和回收

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/kv_cache_interface.py` ~39KB | KV Cache 接口定义 |
| `v1/kv_cache_spec_registry.py` ~8KB | Cache 规格注册 |
| `v1/core/block_pool.py` ~33KB | 物理块池：块的分配与回收 |
| `v1/core/kv_cache_manager.py` ~37KB | Cache 管理器：逻辑→物理映射 |
| `v1/core/single_type_kv_cache_manager.py` ~82KB | 单类型 Cache 管理器 |
| `v1/core/kv_cache_coordinator.py` ~37KB | 跨 GPU Cache 协调 |
| `v1/core/kv_cache_utils.py` ~93KB | Cache 工具函数 |
| `v1/core/kv_cache_metrics.py` ~3KB | Cache 指标 |
| `v1/worker/block_table.py` ~15KB | Block Table 实现 |

### 阅读要点

1. **`kv_cache_interface.py`**：
   - `KVCache` 抽象类定义了哪些方法
   - 不同 Cache 实现（`SelfAttnKVCache`, `CrossAttnKVCache` 等）的区别
   - `KVCacheConfig` 包含了哪些配置项

2. **`block_pool.py`**：
   - `BlockPool` 的数据结构：如何管理空闲块和已分配块
   - 块的分配策略（best-fit? first-fit?）
   - 块的回收机制（引用计数？）
   - `Block` 对象的元数据（block id, ref count, device 等）

3. **`kv_cache_manager.py`**：
   - 逻辑块到物理块的映射管理
   - Copy-on-Write 的实现：`fork()` 和 `fork_append()` 方法
   - 内存压力下的 Eviction 策略
   - `get_computed_blocks()`：哪些块已经计算完成

4. **`single_type_kv_cache_manager.py`**：
   - 与旧版 multi-type 的区别
   - 简化后的块分配逻辑
   - 性能优势

5. **`kv_cache_coordinator.py`**：
   - 多 GPU 场景下如何协调 KV Cache
   - 如何支持张量并行下的 Cache 共享

6. **`block_table.py`**：
   - Block Table 的数据结构（通常是一个数组或列表）
   - 如何通过 block table 快速定位一个 token 的物理位置

### 学习目标

- [ ] 能解释 PagedAttention 相比传统 KV Cache 的内存优势
- [ ] 理解逻辑块、物理块、Block Table 三者的关系
- [ ] 知道 Copy-on-Write 在什么场景下触发
- [ ] 理解 `single_type_kv_cache_manager` 的设计动机
- [ ] 能画出 PagedAttention 的内存寻址示意图

### 思考题

1. 如果有 10 个请求共享前缀 "The capital of France is"，PagedAttention 如何节省显存？
2. 当两个请求的 prompt 完全相同但需要独立生成时，什么时候触发 Copy-on-Write？
3. `block_size` 参数如何影响内存利用率和调度粒度？

---

## 2.3 调度器（Scheduler）

### 核心概念

- **调度策略**：决定每个推理步执行哪些请求
- **Prefill vs Decode 混合**：新请求的 Prefill 与进行中的 Decode 如何混合
- **抢占与恢复**：内存不足时暂停低优先级请求，释放内存后恢复
- **最大 token 预算**：`max_num_seqs` 和 `max_model_len` 的组合约束

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/core/sched/` | 调度器目录 |

### 阅读要点

1. **调度器输入**：当前所有 `RUNNING` 和 `WAITING` 状态的请求
2. **调度器输出**：一个 batch 中包含的请求列表，每个请求包含 `RequestId` 和阶段（Prefill/Decode）
3. **调度约束**：
   - 不超过 `max_num_seqs`（最大并行序列数）
   - 不超过剩余 KV Cache 内存
   - 不超过 `max_model_len`
4. **优先级策略**：FCFS? 基于请求长度? 基于 deadline?

### 学习目标

- [ ] 理解调度器如何混合 Prefill 和 Decode 请求
- [ ] 知道什么情况下会触发抢占
- [ ] 理解 `max_num_seqs` 和 `max_model_len` 对调度的影响

---

## 2.4 注意力后端（Attention Backend）

### 核心概念

- **注意力计算**：`Q @ K^T * scale → softmax → @ V`
- **PagedAttention** vs **FlashAttention** vs **FlashInfer**
- **后端选择**：根据硬件、模型、配置自动选择最佳后端
- **v1 注意力抽象**：统一接口，多种实现

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/attention/backend.py` ~40KB | 注意力后端抽象基类 |
| `v1/attention/selector.py` ~8KB | 后端自动选择器 |
| `v1/attention/backends/` | 具体后端实现 |
| `v1/attention/ops/` | 注意力操作 |
| `vllm/model_executor/layers/attention/` | 模型层的注意力封装 |

### 阅读要点

1. **`backend.py`**：
   - `AttentionBackend` 抽象类定义了哪些接口
   - `forward()` 方法的签名和语义
   - 不同后端的元数据（支持的 dtype、head 维度等）

2. **`selector.py`**：
   - 选择逻辑的优先级链
   - 通过环境变量 `VLLM_ATTENTION_BACKEND` 强制指定

3. **后端的核心区别**：
   - FlashAttention-2 vs FlashInfer vs PagedAttention 的算法差异
   - 各后端的硬件要求（CUDA compute capability）
   - 各后端的性能特征（Prefill vs Decode 的差异）

### 学习目标

- [ ] 理解不同注意力后端的核心算法差异
- [ ] 知道 vLLM 如何自动选择最佳后端
- [ ] 理解 PagedAttention 在注意力计算中如何利用 Block Table

---

## 2.5 采样器（Sampler）

### 核心概念

- **Logits 处理**：模型输出的 logits 到采样概率的转换
- **采样策略**：Greedy, Top-K, Top-P, Temperature, Min-P, Beam Search 等
- **拒绝采样（Rejection Sampling）**：投机解码中保证输出分布一致性
- **Thinking Budget**：推理模型（如 R1）的思考 token 预算控制

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/sample/sampler.py` ~18KB | 核心采样器 |
| `v1/sample/rejection_sampler.py` ~36KB | 拒绝采样器（投机解码用） |
| `v1/sample/logits_processor/` | Logits 处理器（惩罚、偏置等） |
| `v1/sample/ops/` | 采样 GPU Kernel 封装 |
| `v1/sample/metadata.py` | 采样元数据 |
| `v1/sample/thinking_budget_state.py` ~25KB | 思考预算管理 |
| `vllm/sampling_params.py` ~47KB | 采样参数定义 |
| `vllm/logits_process.py` | Logits 后处理 |
| `vllm/logprobs.py` | 对数概率处理 |

### 阅读要点

1. **`sampler.py`**：
   - 采样流程：`logits → (可选) logits_processor → softmax → 采样 → token_id`
   - 不同采样策略的组合规则（如 Top-K + Top-P 同时启用）
   - 批处理采样：如何高效处理 batch 中每个序列不同的采样参数

2. **`sampling_params.py`**：
   - `SamplingParams` 数据结构：temperature, top_k, top_p, frequency_penalty 等
   - 参数校验逻辑
   - 每个参数对生成结果的影响

3. **`rejection_sampler.py`**：
   - 投机解码的拒绝采样算法
   - 如何保证输出分布与直接采样一致

### 学习目标

- [ ] 理解采样参数的全集及各参数的作用
- [ ] 能说出 Greedy 和 Sampling 在代码中的实现差异
- [ ] 理解拒绝采样的数学原理和实现

---

## 2.6 Worker 与模型执行

### 核心概念

- **Worker**：实际执行模型推理的进程（每个 GPU 一个 Worker）
- **Model Runner**：Worker 内部负责编排模型前向传播的组件
- **GPU Model Runner**：准备输入 batch → 执行模型 → 提取输出
- **CUDA Graph**：将模型前向传播 capture 为图，重复执行减少框架开销
- **Input Batch**：将多个请求的输入组织为 GPU 可计算的 batch
- **Micro-batching**：在单个 batch 内进一步拆分以优化内存

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/worker/worker_base.py` ~13KB | Worker 基类 |
| `v1/worker/gpu_worker.py` ~58KB | GPU Worker |
| `v1/worker/gpu_model_runner.py` ~350KB | **GPU 模型运行器（全库最大文件）** |
| `v1/worker/gpu_input_batch.py` ~48KB | GPU 输入批处理 |
| `v1/worker/gpu_ubatch_wrapper.py` ~21KB | Micro-batch 包装 |
| `v1/worker/ubatching.py` ~8KB | Micro-batching 逻辑 |
| `v1/worker/ubatch_utils.py` ~10KB | Micro-batch 工具 |
| `v1/worker/block_table.py` ~15KB | Block Table 管理 |
| `v1/worker/cudagraph_dispatcher.py` ~15KB | CUDA Graph 调度 |
| `v1/worker/startup_plan.py` ~7KB | 启动规划 |
| `v1/worker/workspace.py` ~10KB | GPU Workspace 管理 |
| `v1/worker/encoder_cudagraph.py` ~34KB | Encoder CUDA Graph |
| `v1/worker/cpu_worker.py` ~11KB | CPU Worker |
| `v1/worker/cpu_model_runner.py` ~9KB | CPU Model Runner |
| `v1/worker/utils.py` ~22KB | Worker 工具函数 |
| `v1/worker/dp_utils.py` ~9KB | Data Parallel 工具 |
| `v1/worker/cp_utils.py` ~11KB | Context Parallel 工具 |
| `v1/worker/lora_model_runner_mixin.py` ~11KB | LoRA Runner Mixin |

### 阅读要点

1. **`worker_base.py`**：
   - `Worker` 基类的生命周期方法：`init()`, `determine_num_available_blocks()`, `execute_model()`
   - Worker 状态的初始化顺序

2. **`gpu_worker.py`**：
   - GPU Worker 的初始化：GPU 设备设置、模型加载、CUDA Graph 预热
   - `execute_model()` 方法：从 scheduler 接收 batch → 调用 model_runner → 返回输出
   - 如何处理不同并行策略下的 worker 通信

3. **`gpu_model_runner.py`**（**重点，350KB）**：
   - `prepare_inputs()`：将 Scheduler 的 batch 转换为 GPU tensor
     - Token IDs 的填充（padding）
     - Position IDs 的计算
     - Block Table 的索引
     - Sampling 参数的准备
   - `execute_model()`：实际的前向传播调用
   - CUDA Graph 相关的：
     - `capture_model()`：捕捉 CUDA Graph
     - `replay_model()`：回放 CUDA Graph
     - 多 CUDA Graph 的管理（不同 batch 大小）
   - 支持多种模型架构的条件分支

4. **`gpu_input_batch.py`**：
   - `GpuInputBatch` 的数据结构
   - 如何从 Request 列表转换到 Tensor batch
   - 填充和掩码的处理

5. **`cudagraph_dispatcher.py`**：
   - CUDA Graph 的命中策略：不同 batch size → 不同 Graph
   - 未命中时的 fallback 逻辑

### 学习目标

- [ ] 理解 Worker 和 Model Runner 的分工
- [ ] 理解 `prepare_inputs()` 到 `execute_model()` 的完整流程
- [ ] 知道 CUDA Graph 在 vLLM 中的使用策略
- [ ] 理解 Input Batch 的数据组织方式
- [ ] 能描述一个 batch 中多个请求如何并行执行前向传播

### 思考题

1. 为什么 GPU Model Runner 是 vLLM 最大的文件？它承担了哪些职责？
2. CUDA Graph 在什么条件下可以使用？什么条件下必须 fallback 到 eager 模式？
3. Micro-batching 如何防止 GPU 内存溢出？

---

## 2.7 引擎层

### 核心概念

- **引擎是 vLLM 的调度+执行中枢**，连接 API Server 和 Worker
- 管理请求的生命周期：接收→排队→调度→执行→输出→返回
- **执行器（Executor）**：跨 worker 的执行调度

### 代码路径

| 文件 | 说明 |
|------|------|
| `v1/engine/` | v1 引擎核心 |
| `v1/executor/` | 执行器（管理分布式执行） |

### 阅读要点

1. **引擎的核心循环**（类似 Reactor/Event Loop）：
   - `add_request()` → `step()` 循环 → 返回完成请求
   - 每次 `step()` 包含：调度 → 执行 → 后处理

2. **引擎与 Scheduler 的交互**：
   - 引擎调用 Scheduler 获取下一个 batch
   - 引擎将 batch 分发给 Worker

### 学习目标

- [ ] 理解引擎的主循环逻辑
- [ ] 知道引擎如何管理请求的多阶段生命周期

---

## 章节总结

### 核心流程数据流

```
Scheduler 决定 batch
    │
    ▼
GPUWorker.execute_model()
    │
    ├── ModelRunner.prepare_inputs()
    │   ├── Tokenize / Pad
    │   ├── Build Block Table indices
    │   └── Prepare sampling params
    │
    ├── ModelRunner.execute_model()
    │   ├── Embedding Lookup
    │   ├── Transformer Layers (with Attention)
    │   └── LM Head → logits
    │
    ├── Sampler.sample()
    │   ├── Logits Processing
    │   ├── Apply Sampling Strategies
    │   └── Return next token IDs
    │
    └── Output → Engine → Response
```

### 进一步阅读

- 继续阶段三：模型层 → `03-model-layer.md`
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- FlashAttention 论文：https://arxiv.org/abs/2205.14135

---

*对应 LEARNING_PLAN.md 第 4 章 | 基于 vLLM 主分支*
