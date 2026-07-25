# 性能优化

## CUDA Graph

CUDA Graph 将模型前向传播的 kernel launch 序列捕获为计算图，消除 PyTorch 框架的开销。

### 工作机制

```python
# 捕获阶段
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input_ids, positions, ...)

# 回放阶段（极低延迟）
graph.replay()
```

### 使用策略

`v1/cudagraph_dispatcher.py` 管理多个 CUDA Graph：

- 为不同的 batch size 分别捕获 Graph
- 执行时选择最匹配的 Graph
- 未命中时 fallback 到 eager 模式

### 限制

- 模型结构必须固定（权重不变）
- 输入张量 shape 必须在预设范围内
- 不支持分支逻辑（如 sampler）

## Torch Compile

`vllm/compilation/` 目录集成 PyTorch 2.0 的 `torch.compile`：

- 与 CUDA Graph 互补：Graph 消除框架开销，Compile 优化算子实现
- 支持 `@support_torch_compile` 装饰器（模型上使用）

## Micro-batching

当单 batch 超出 GPU 显存时，拆分为多个 micro-batch：

```python
class GPUUbatchWrapper:
    def execute_model(self, scheduler_output):
        for micro_batch in self.split(scheduler_output):
            output = self.model_runner.execute_model(micro_batch)
            self.accumulate(output)
```

## 算子融合

`layers/fusion/` 目录实现算子融合：

- `SiluAndMul`：SiLU 激活 + 逐元素乘法（SwiGLU）
- 融合的 RMSNorm
- Fused MoE：路由 + 专家计算 + 加权组合
