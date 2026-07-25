# 日志、追踪与监控

> vLLM 是一个**分布式系统**——在多 GPU 环境下，传统"printf 调试法"几乎无效。vLLM 的日志和监控系统专门针对分布式场景设计。

---

## 1. 分布式日志系统（`vllm/logger.py`）

### 文件位置

`vllm/logger.py`（~11KB）

### 核心问题

在多 GPU 推理中，每个 GPU 上运行一个独立的进程（rank）。如果所有进程都输出日志，控制台会变成不可读的乱码：

```
[rank0] Loading model...
[rank1] Loading model...
[rank0] KV Cache allocated 8192 blocks
[rank2] Loading model...
[rank1] KV Cache allocated 8192 blocks
```

vLLM 的日志系统通过 **rank 前缀** 来解决这个问题：

### 实现机制

```python
# vllm/logger.py (简化示意)
def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # 为每个 rank 添加前缀
    try:
        rank = get_distributed_rank()
    except:
        rank = None
    
    if rank is not None:
        formatter = logging.Formatter(
            f"[rank{rank}] %(message)s"
        )
    else:
        formatter = logging.Formatter("%(message)s")
    
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
```

### 使用方式

在每个模块中：

```python
from vllm.logger import init_logger

logger = init_logger(__name__)
logger.info("KV Cache allocated %d blocks", num_blocks)
```

### 关键要点

- **`get_distributed_rank()`**：从分布式环境中获取当前进程的 rank
- **单进程模式**：没有分布式环境时退化为标准日志
- **日志级别控制**：通过 `VLLM_LOG_LEVEL` 环境变量（`DEBUG`/`INFO`/`WARNING`/`ERROR`）
- **仅在 rank 0 输出**：某些日志只需要在 rank 0 上输出（如模型加载进度），通过 `if rank == 0` 控制

---

## 2. OpenTelemetry 分布式追踪（`vllm/tracing/`）

### 文件位置

`vllm/tracing/` 目录

### 什么是分布式追踪？

在分布式推理中，一个请求会经过多个 GPU 节点。分布式追踪**记录请求在每个阶段的耗时**，帮助定位性能瓶颈：

```
请求 12345 的时间线:
  ┌──────┬──────┬──────┬──────┬──────┬──────┐
  │排队   │调度  │Prefill│Decode│采样  │输出  │
  └──────┴──────┴──────┴──────┴──────┴──────┘
  0      5ms   10ms   150ms  200ms  205ms  210ms
```

### OpenTelemetry 集成

vLLM 使用 [OpenTelemetry](https://opentelemetry.io/) 标准来追踪请求生命周期：

```python
# 典型的 Span 创建
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("engine_step") as span:
    span.set_attribute("num_requests", len(batch))
    with tracer.start_as_current_span("scheduler"):
        batch = scheduler.schedule()
    with tracer.start_as_current_span("execute_model"):
        outputs = worker.execute_model(batch)
```

### 关键追踪阶段

| Span 名称 | 说明 | 关键属性 |
|-----------|------|---------|
| `engine_step` | 引擎单步迭代 | `num_requests`, `num_tokens` |
| `schedule` | 调度阶段 | `num_running`, `num_waiting` |
| `prepare_inputs` | 输入准备 | `batch_size`, `max_seq_len` |
| `execute_model` | 模型前向 | `num_layers`, `decode/prefill` |
| `sampler` | 采样阶段 | `sampling_strategy` |

### 启用方式

```bash
# 启动时启用 OpenTelemetry 追踪
vllm serve ... --otlp-traces-endpoint=http://localhost:4318/v1/traces
```

---

## 3. Prometheus 指标（`vllm/v1/metrics/`）

### 文件位置

`vllm/v1/metrics/` 目录（v1 架构）
`vllm/usage/` 目录（使用统计）

### 暴露的指标

vLLM 通过 `/metrics` 端点暴露 Prometheus 格式的指标，用于监控推理服务状态：

#### 吞吐量指标
| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm:num_requests_running` | Gauge | 当前正在处理的请求数 |
| `vllm:num_requests_waiting` | Gauge | 排队等待的请求数 |
| `vllm:request_prompt_tokens` | Histogram | 请求 prompt token 数分布 |
| `vllm:request_generation_tokens` | Histogram | 生成 token 数分布 |

#### 延迟指标
| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm:request_time_to_first_token` | Histogram | Prefill 到第一个 token 的延迟 |
| `vllm:request_e2e_time` | Histogram | 端到端请求延迟 |
| `vllm:time_per_output_token` | Histogram | 每 token 生成时间 |

#### KV Cache 指标
| 指标名 | 类型 | 说明 |
|--------|------|------|
| `vllm:kv_cache_usage` | Gauge | KV Cache 使用率 |
| `vllm:kv_cache_blocks_used` | Gauge | 已使用的 Cache 块数 |
| `vllm:kv_cache_blocks_free` | Gauge | 空闲 Cache 块数 |

### 使用方式

```bash
# 访问指标端点
curl http://localhost:8000/metrics

# 配合 Prometheus 和 Grafana 搭建监控面板
```

---

## 4. 使用统计（`vllm/usage/`）

### 核心目的

vLLM 会收集**匿名**的使用统计数据，帮助开发团队了解框架的使用情况（功能使用频率、常见配置等）。

### 收集的内容

- 使用的模型架构（如 `LlamaForCausalLM`）
- 启用的特性（量化、LoRA、多模态等）
- 分布式配置（TP 大小、PP 大小等）

### 控制方式

```bash
# 禁用使用统计
export DO_NOT_TRACK=1
```

---

## 学习产出清单

完成本节后，你应该能回答：

- [ ] vLLM 的日志系统如何处理多 GPU 输出的混乱问题？
- [ ] 如何启用 OpenTelemetry 追踪？追踪的各个 Span 代表什么阶段？
- [ ] Prometheus 指标中哪个指标表示"首 token 延迟"？
- [ ] 如何禁用使用统计？
- [ ] 日志级别如何通过环境变量控制？

## 思考题

1. **日志排错**：假设你的 8-GPU 服务中 rank 3 报错 `CUDA OOM`，但其他 rank 正常。vLLM 的日志系统如何帮助你快速定位问题？

2. **监控面板设计**：如果让你为一个高并发的 vLLM 服务设计 Grafana 监控面板，你会展示哪些关键指标？为什么？

## 下一步

继续阅读 [编译与构建系统](03-build-system.md)，了解 CUDA C++ 扩展如何集成到 Python 包中。

---

> **代码参考**：`vllm/logger.py`、`vllm/tracing/`、`vllm/v1/metrics/`、`vllm/usage/`
