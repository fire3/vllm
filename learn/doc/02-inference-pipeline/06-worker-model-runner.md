# Worker 与模型执行

> Worker 和 Model Runner 是推理引擎的**执行层**——它们将调度器的调度决策转换为 GPU 上的实际计算。GPU Model Runner 是 vLLM 全库最大的单个文件（~350KB），涵盖了输入准备、CUDA Graph 捕获、模型前向传播、采样等关键路径。

---

## 1. Worker 体系

### Worker 层次结构

```
WorkerBase (worker_base.py)             — 抽象基类
    └── Worker (gpu_worker.py)          — GPU Worker（主要实现）
    ├── CPUWorker (cpu_worker.py)      — CPU 推理
    ├── XPUWorker (xpu_worker.py)      — Intel XPU
    └── ...
```

### Worker 的职责

Worker 是**每个 GPU 进程**中运行的核心执行单元，职责包括：

1. 初始化 GPU 设备与模型加载
2. 接收 `SchedulerOutput` 并调用 Model Runner 执行
3. 流水线并行（PP）中负责中间张量的发送与接收
4. 返回执行结果（`ModelRunnerOutput`）

### GPUWorker.execute_model

```python
def execute_model(self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
```

执行流程：

```
execute_model(scheduler_output)
    │
    ├── 1. PP 非阻塞同步
    │      等待上一步的 PP 发送完成
    │
    ├── 2. PP 接收（非首节点）
    │      接收前一个 PP 阶段的 intermediate_tensors
    │
    ├── 3. model_runner.execute_model(scheduler_output, intermediate_tensors)
    │      ├── 返回 ModelRunnerOutput → 完成
    │      ├── 返回 None → 后续需要 sample_tokens()
    │      └── 返回 IntermediateTensors → PP 发送到下一节点
    │
    └── 4. PP 发送（非尾节点）
           非阻塞发送 intermediate_tensors 到下一 PP 阶段
```

---

## 2. GPU Model Runner

### 文件位置

`vllm/v1/worker/gpu_model_runner.py` —— ~350KB，**vLLM 全库最大文件**

### 类层次

```python
class GPUModelRunner(
    LoRAModelRunnerMixin,
    KVConnectorModelRunnerMixin,
    ECConnectorModelRunnerMixin,
):
```

通过三个 Mixin 组合了 LoRA、KV 连接器、编码器缓存连接器的支持。

### 核心状态

```python
class GPUModelRunner:
    model: nn.Module                          # 加载的模型
    sampler: Sampler                          # 采样器
    input_batch: GpuInputBatch                # 输入批处理
    attn_backends: list[AttentionBackend]     # attention 后端列表
    kv_cache: list[torch.Tensor]             # KV Cache tensor
    cudagraph_dispatcher: CUDAGraphDispatcher
    workspace: Workspace
```

---

## 3. 两阶段执行模型

vLLM v1 架构采用**两阶段执行**模式：

### 阶段一：`execute_model`

```python
def execute_model(self, scheduler_output, intermediate_tensors=None):
    # 1. 更新状态
    self._update_states(scheduler_output)
    
    # 2. 预处理
    logits_indices, spec_decode_metadata = self._prepare_inputs(
        scheduler_output)
    
    # 3. 前向传播
    hidden_states = self._model_forward(
        input_ids, positions, intermediate_tensors, ...)
    
    # 4. 如果是 PP 中间节点，返回 intermediate_tensors
    if not is_last_pp_rank:
        return IntermediateTensors(hidden_states, ...)
    
    # 5. 提取 logits（只取最后一个 token 的位置）
    logits = model.compute_logits(hidden_states[logits_indices])
    
    # 6. 存储临时状态，返回 None
    self._execute_model_state = ExecuteModelState(
        logits=logits,
        spec_decode_metadata=spec_decode_metadata,
        scheduler_output=scheduler_output,
    )
    return None  # 延迟到 sample_tokens
```

`execute_model` 返回 `None`，将采样延迟到下一步。这允许：

- **CUDA Graph 捕获**：不需要将采样器包含在 Graph 中
- **异步调度**：采样可以与下一轮的调度并行

### 阶段二：`sample_tokens`

```python
def sample_tokens(self, grammar_output):
    # 1. 获取临时状态
    state = self._execute_model_state
    
    # 2. 应用 grammar 位掩码
    logits = self.apply_grammar_bitmask(state.logits, grammar_output)
    
    # 3. 采样
    sampler_output = self._sample(logits, state.sampling_metadata)
    
    # 4. 更新状态
    self._update_states_after_model_execute(sampler_output)
    
    # 5. 返回结果
    return ModelRunnerOutput(sampler_token_ids=sampler_output.sampled_token_ids, ...)
```

---

## 4. 输入准备（`_prepare_inputs`）

### 文件定位

`GPUModelRunner._prepare_inputs()` — 约第 1952 行

```python
def _prepare_inputs(self, scheduler_output):
```

输入准备是 Model Runner 中计算最密集的 CPU 操作之一，分批将调度器的逻辑决策转换为 GPU 可消费的 tensor：

```
_prepare_inputs(scheduler_output)
    │
    ├── 1. 提交块表（与 CPU 操作重叠）
    │
    ├── 2. 计算请求索引、位置编码、token 索引
    │
    ├── 3. 从 input_batch 中 index-select token ID
    │
    ├── 4. 处理 prompt embeddings（多模态）
    │
    ├── 5. 构建 query_start_loc、seq_lens、slot_mapping
    │
    ├── 6. 处理 M-RoPE / XD-RoPE 位置
    │
    └── 7. 返回 logits_indices, spec_decode_metadata
```

返回的 `logits_indices` 指明了从 `hidden_states` 中提取 logits 的位置（通常每个序列最后一个 token 的位置）。

---

## 5. 模型前向传播（`_model_forward`）

```python
def _model_forward(self, input_ids, positions, intermediate_tensors, ...):
    with set_forward_context(attn_metadata):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_cache,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
        )
    return hidden_states
```

模型的前向传播由 PyTorch 原生的 `nn.Module.forward()` 执行。涉及：

1. **Embedding 层**：token ID → embedding vectors
2. **Transformer 层循环**：每层执行 Attention + MLP
3. **LM Head**：hidden states → vocabulary logits

Attention 层内部调用后端实现：

```python
# 每层 Transformer 中
class AttentionLayer:
    def forward(self, query, key, value, kv_cache, attn_metadata):
        return self.impl.forward(query, key, value, kv_cache, attn_metadata)
```

---

## 6. GPU Input Batch

### 文件位置

`vllm/v1/worker/gpu_input_batch.py` —— 约 48KB

`GpuInputBatch` 管理 GPU 上所有请求的 token 数据：

```python
class GpuInputBatch:
    token_ids_cpu_tensor: torch.Tensor   # CPU 上的 token ID 张量
    # ... 更多 GPU 相关的批处理状态
```

输入准备阶段，`_prepare_inputs` 从 `token_ids_cpu_tensor` 中 index-select 出当前步需要的 token ID。

---

## 7. CUDA Graph 调度

### 文件位置

`vllm/v1/cudagraph_dispatcher.py` —— 约 15KB

### 原理

CUDA Graph 将 CUDA kernel launch 的完整序列捕获为计算图，消除 PyTorch 框架的 Python 调用开销。

### 捕获条件

- 模型结构必须固定（权重不变）
- Batch size 在一定范围内
- 输入张量的 shape 在预设的 padding 范围内

### 多 Graph 管理

```python
class CUDAGraphDispatcher:
    graphs: dict[int, CUDAGraph]  # batch_size → CUDAGraph
```

对不同 batch size 预先捕获多个 Graph。执行时选择最近匹配的 Graph，未命中则 fallback 到 eager 模式。

### 在 Model Runner 中的集成

`_determine_batch_execution_and_padding()` 决定本次执行是否使用 CUDA Graph：

```python
def _determine_batch_execution_and_padding(self):
    if self.use_cudagraph and self._is_cudagraph_captured():
        # 使用 CUDA Graph 模式
        # 需要对 batch 做 padding 到预设的 batch size
    else:
        # 使用 eager 模式
```

---

## 8. Micro-batching

### 文件位置

`vllm/v1/worker/gpu_ubatch_wrapper.py` —— 约 21KB

当单个 batch 超出 GPU 显存容量时，将其拆分为多个 micro-batch 依次处理：

```python
class GPUUbatchWrapper:
    def execute_model(self, scheduler_output):
        for micro_batch in self.split(scheduler_output):
            output = self.model_runner.execute_model(micro_batch)
            self.accumulate(output)
        return self.merged_output()
```

---

## 9. 其他辅助组件

### Block Table（`v1/worker/block_table.py`）

在模型执行前，将 `KVCacheBlocks` 转换为 GPU 上的 block table tensor。Attention kernel 通过此张量查找物理块地址。

### Workspace（`v1/worker/workspace.py`）

管理 GPU 上的 workspace 内存池，用于临时张量分配。

### DP Utils（`v1/worker/dp_utils.py`）

Data Parallel 模式下，在多个模型副本间分发请求。

### CP Utils（`v1/worker/cp_utils.py`）

Context Parallel 模式下，将长上下文切分到多个 GPU。

---

> **代码参考**：
> - `vllm/v1/worker/gpu_worker.py` — GPU Worker
> - `vllm/v1/worker/gpu_model_runner.py` — GPU Model Runner（核心，350KB）
> - `vllm/v1/worker/gpu_input_batch.py` — GPU 输入批处理
> - `vllm/v1/worker/worker_base.py` — Worker 基类
> - `vllm/v1/worker/block_table.py` — 块表
> - `vllm/v1/cudagraph_dispatcher.py` — CUDA Graph 调度
> - `vllm/v1/worker/gpu_ubatch_wrapper.py` — Micro-batching
