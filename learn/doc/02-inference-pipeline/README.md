# 阶段二：核心推理管线（v1 架构）

> vLLM 的推理引擎核心——从请求进入系统到输出 token 的完整路径。这是整个框架最重要的一层，涵盖了 PagedAttention、调度策略、注意力计算、采样以及 GPU 模型执行的全过程。

## 章节列表

1. [请求生命周期](01-request-lifecycle.md) — Request 数据模型、状态机、请求池
2. [KV Cache 系统](02-kv-cache-system.md) — PagedAttention、BlockPool、KVCacheManager、前缀缓存
3. [调度器](03-scheduler.md) — 调度策略、预算控制、抢占与恢复
4. [注意力后端](04-attention-backend.md) — 后端抽象层、FlashAttention/FlashInfer 等实现
5. [采样器](05-sampler.md) — logits 处理、采样策略、拒绝采样
6. [Worker 与模型执行](06-worker-model-runner.md) — GPU Worker、Model Runner 的输入准备与执行管线
7. [引擎层](07-engine-layer.md) — 引擎主循环、执行器、请求排队

## 核心数据流

```
用户请求
    │
    ▼
Engine.add_request()
    │
    ▼ (schedule → execute 循环)
┌──────────────────────────────────────────────────┐
│              Engine Core 主循环                    │
│                                                   │
│  Scheduler.schedule()                             │
│    ├── KVCacheManager.get_computed_blocks()      │
│    ├── KVCacheManager.allocate_slots()           │
│    └── SchedulerOutput                           │
│         │                                         │
│         ▼                                         │
│  Worker.execute_model(scheduler_output)           │
│    └── ModelRunner.execute_model()               │
│         ├── _prepare_inputs()                     │
│         ├── _model_forward()                      │
│         │   ├── Attention (via backend)           │
│         │   ├── MLP / MoE                         │
│         │   └── LM Head → logits                  │
│         └── return None (deferred)                │
│              │                                     │
│  ModelRunner.sample_tokens()                      │
│    ├── apply_grammar_bitmask()                    │
│    ├── Sampler.forward(logits)                    │
│    └── SamplerOutput                              │
│         │                                         │
│  Scheduler.update_from_output()                   │
│    ├── append_output_token_ids()                  │
│    ├── update_block_hashes()                      │
│    └── free finished requests                     │
│                                                   │
└──────────────────────────────────────────────────┘
    │
    ▼
返回给用户
```

## 前置知识

- PyTorch Transformer 前向传播的基本概念（embedding、attention、FFN、layer norm）
- LLM 推理的基本流程：Prefill（预填充）和 Decode（逐 token 生成）
- 基本的 CUDA GPU 概念

## 推荐阅读顺序

按编号顺序阅读。如果时间有限，**优先阅读 02（KV Cache 系统）和 06（Worker 与模型执行）**，这两篇覆盖了 vLLM 最核心的创新和最大的代码文件。

每篇文档包含：
- 核心类/接口层次结构
- 关键代码路径与行号
- 设计模式与架构分析
- 数据流图与调用链
