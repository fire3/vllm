# vLLM 架构学习计划

> 目标：全面、系统地理解 vLLM 的架构设计、核心模块和工作原理。

---

## 目录

1. [全景概览](#1-全景概览)
2. [学习路线图](#2-学习路线图)
3. [阶段一：基础层（系统底座）](#3-阶段一基础层系统底座)
4. [阶段二：核心推理管线](#4-阶段二核心推理管线)
5. [阶段三：模型层](#5-阶段三模型层)
6. [阶段四：服务与入口层](#6-阶段四服务与入口层)
7. [阶段五：分布式与并行](#7-阶段五分布式与并行)
8. [阶段六：高级特性](#8-阶段六高级特性)
9. [阶段七：工具链与工程](#9-阶段七工具链与工程)
10. [推荐阅读路径](#10-推荐阅读路径)

---

## 1. 全景概览

### 1.1 什么是 vLLM

vLLM 是一个高性能、开源的大语言模型（LLM）推理引擎。核心能力包括：

- **PagedAttention**：高效管理 KV Cache 的内存调度算法（vLLM 的成名之作）
- **Continuous Batching**：连续批处理，最大化 GPU 利用率
- **模型并行**：支持张量并行、流水线并行、专家并行等
- **量化推理**：支持 FP8、INT4/INT8/AWQ/GPTQ 等多种量化方案
- **多模态**：支持图像、音频、视频等多模态输入
- **v1 架构**：新一代推理引擎，重新设计了调度和执行管线

### 1.2 顶层目录结构

```
vllm/
├── v1/                    # 新一代推理引擎（正在逐步取代旧架构）
│   ├── core/              #   KV Cache 调度核心（BlockPool, Scheduler）
│   ├── engine/            #   引擎层
│   ├── executor/          #   执行器
│   ├── worker/            #   工作节点（GPU/CPU/TPU Worker, Model Runner）
│   ├── attention/         #   注意力后端（FlashAttention, FlashInfer 等）
│   ├── sample/            #   采样器（Top-K, Top-P, Rejection Sampler）
│   ├── metrics/           #   指标收集
│   ├── kv_offload/        #   KV Cache 卸载
│   ├── spec_decode/       #   投机解码
│   ├── structured_output/ #   结构化输出
│   └── pool/              #   请求池管理
├── engine/                # 旧架构引擎层
│   ├── arg_utils.py       #   配置参数工具
│   ├── async_llm_engine.py
│   ├── llm_engine.py
│   └── protocol.py
├── model_executor/        # 模型执行核心
│   ├── layers/            #   神经网络层（Attention, Linear, MoE 等）
│   │   ├── attention/     #     注意力层
│   │   ├── fused_moe/     #     融合 MoE
│   │   ├── quantization/  #     量化层
│   │   └── mamba/         #     Mamba/SSM 层
│   ├── models/            #   各种模型的具体实现（100+ 模型）
│   ├── model_loader/      #   模型加载与权重管理
│   ├── kernels/           #   GPU Kernel 封装
│   ├── parameter.py       #   参数管理与优化
│   └── offloader/         #   模型卸载
├── models/                # 特定模型变体（DeepSeek, MiniMax 等）
├── entrypoints/           # 入口点 & API 服务
│   ├── openai/            #   OpenAI 兼容 API 服务
│   ├── serve/             #   HTTP 服务器
│   ├── generate/          #   离线推理入口
│   └── llm.py             #   LLM 类（高层 API）
├── distributed/           # 分布式通信
│   ├── parallel_state.py  #   并行状态管理（TP/PP/EP 等）
│   ├── device_communicators/ #   设备通信器
│   ├── kv_transfer/       #   KV Cache 跨节点传输
│   └── weight_transfer/   #   权重传输
├── multimodal/            # 多模态处理
│   ├── inputs.py          #   多模态输入处理
│   ├── parse.py           #   解析器
│   ├── processing/        #   预处理 pipeline
│   └── registry.py        #   多模态模型注册
├── kernels/               # GPU Kernels
├── lora/                  # LoRA 适配器支持
├── quantization/          # 量化工具（AWQ, GPTQ, FP8 等）
├── sampling_params.py     # 采样参数
├── inputs/                # 输入处理
├── reasoning/             # 推理（Reasoning/R1 相关）
├── tool_parsers/          # 工具调用解析器
└── transformers_utils/    # HuggingFace Transformers 兼容层

csrc/                      # C++/CUDA 内核源码
├── attention/             #   FlashAttention, PageAttention CUDA 实现
├── core/                  #   核心调度器
├── moe/                   #   MoE 相关（TopK 门控等）
├── quantization/          #   量化 CUDA 内核
├── cutlass_extensions/    #   CUTLASS 扩展
└── custom_all_reduce.cuh  #   自定义 AllReduce

tests/                     # 测试套件
benchmarks/                # 基准测试
docs/                      # 文档
```

---

## 2. 学习路线图

学习分 **7 个阶段**，由底层到上层，由核心到外围。每个阶段建议用 1-2 天深入阅读代码，总计约 2-3 周。

```
阶段一：系统底座
   ├── 环境与配置 (envs.py, arg_utils.py)
   ├── 日志与监控 (logger.py, tracing/, metrics/)
   ├── 工具基础 (utils/, connections.py)
   └── 编译与构建 (compilation/, setup.py, CMakeLists.txt)
   
阶段二：核心推理管线 (v1/)
   ├── 请求生命周期 (request.py, pool/)
   ├── KV Cache 管理 (core/block_pool, kv_cache_manager, kv_cache_coordinator)
   ├── 调度器 (core/sched/)
   ├── 注意力后端 (attention/backend.py, backends/)
   ├── 采样器 (sample/sampler.py, rejection_sampler.py)
   ├── 引擎 (engine/)
   └── Worker 与 Model Runner (worker/gpu_worker.py, gpu_model_runner.py)
   
阶段三：模型层
   ├── 模型架构模式 (model_executor/models/ 中的基类如 interfaces.py)
   ├── 神经网络层 (model_executor/layers/)
   ├── 模型加载 (model_executor/model_loader/)
   ├── 参数管理 (model_executor/parameter.py)
   └── 具体模型实现 (model_executor/models/llama.py, qwen2.py 等)
   
阶段四：服务与入口层
   ├── 离线推理 (entrypoints/llm.py)
   ├── OpenAI 兼容 API (entrypoints/openai/api_server.py)
   ├── HTTP 服务 (entrypoints/serve/)
   ├── CLi 入口 (entrypoints/cli/)
   └── 在线服务架构 (entrypoints/serve/engine/ 等)
   
阶段五：分布式与并行
   ├── 并行状态管理 (distributed/parallel_state.py)
   ├── 通信原语 (distributed/device_communicators/)
   ├── 张量并行 (Tensor Parallelism)
   ├── 流水线并行 (Pipeline Parallelism)
   ├── 专家并行 (Expert Parallelism / elastic_ep)
   └── KV Cache 传输 (distributed/kv_transfer/)
   
阶段六：高级特性
   ├── 量化 (model_executor/layers/quantization/)
   ├── LoRA (lora/)
   ├── 投机解码 (v1/spec_decode/)
   ├── 多模态 (multimodal/)
   ├── 结构化输出 (v1/structured_output/)
   └── KV Cache 卸载 (v1/kv_offload/)
   
阶段七：工具链与工程
   ├── 测试 (tests/)
   ├── 基准测试 (benchmarks/)
   ├── CUDA Kernels (csrc/)
   ├── 性能优化器 (compilation/, cudagraph_dispatcher.py)
   └── CI/Build 系统 (.buildkite/, cmake/)
```

---

## 3. 阶段一：基础层（系统底座）

### 3.1 配置系统

| 文件 | 说明 |
|------|------|
| `vllm/envs.py` | 所有环境变量定义（~100KB），管理 GPU 内存分配、并行度、调试开关等 |
| `vllm/engine/arg_utils.py` | 引擎参数 (`EngineArgs`)，从命令行到引擎配置的完整映射（~117KB） |
| `vllm/v1/engine/core.py` | v1 引擎核心配置 |
| `vllm/platforms/` | 各硬件平台（CUDA, ROCm, CPU, TPU, XPU）的抽象 |
| `vllm/config/` | 模型配置、设备配置等 |

**学习要点**：
- 理解 `EngineArgs` 如何解析命令行参数并传递给各组件
- 环境变量如何控制 KV Cache 分配、调度策略、调试模式
- 平台抽象如何支持多后端（NVIDIA, AMD, Intel, Google TPU, 昇腾 NPU）

### 3.2 日志、追踪与监控

| 文件 | 说明 |
|------|------|
| `vllm/logger.py` | 日志系统（分布式感知的日志，区分 rank） |
| `vllm/tracing/` | OpenTelemetry 分布式追踪 |
| `vllm/usage/` | 使用统计 |
| `vllm/v1/metrics/` | v1 架构下的 Prometheus 指标收集 |

### 3.3 编译与构建系统

| 文件/目录 | 说明 |
|-----------|------|
| `setup.py` | Python 包构建配置（50KB），CUDA 扩展注册 |
| `CMakeLists.txt` | CMake 构建（62KB），C++/CUDA 代码编译 |
| `cmake/` | CMake 模块 |
| `vllm/compilation/` | PyTorch JIT 编译优化器（torch.compile 相关） |
| `rust/` | Rust 组件（用于分布式协调） |
| `requirements/` | 依赖管理 |

**学习要点**：
- vLLM 如何通过 `setup.py` 将 CUDA C++ 扩展注册为 Python 模块
- `torch_bindings.cpp` 如何将 C++ 操作暴露给 Python

---

## 4. 阶段二：核心推理管线（v1 架构）

**这是 vLLM v1 新一代推理引擎，是学习的重中之重。**

### 4.1 请求与执行流程

```
用户请求
    │
    ▼
entrypoints/ (API Server / LLM API)
    │
    ▼
v1/engine/ (Engine: 调度 → 执行 → 输出)
    │
    ├── core/sched/ (Scheduler: 决定每步执行哪些请求)
    │
    ▼
v1/worker/ (GPUWorker: 执行模型推理)
    │
    ├── gpu_model_runner.py (Model Runner: 准备输入/执行模型/提取输出)
    │
    ▼
v1/attention/ (Attention Backend: 注意力计算)
    │
    ▼
v1/sample/ (Sampler: 从 logits 采样出 token)
    │
    ▼
v1/outputs.py (输出处理)
    │
    ▼
返回给用户
```

### 4.2 核心模块详解

#### 4.2.1 请求管理

| 文件 | 说明 |
|------|------|
| `v1/request.py` | `Request` 数据类：请求的全部状态 |
| `v1/pool/` | `RequestPool`：管理所有活跃请求 |

#### 4.2.2 KV Cache 系统（vLLM 的灵魂）

| 文件 | 说明 |
|------|------|
| `v1/kv_cache_interface.py` | KV Cache 接口定义（39KB） |
| `v1/core/block_pool.py` | Block Pool：物理块分配（33KB） |
| `v1/core/kv_cache_manager.py` | KV Cache 管理器（36KB）：逻辑到物理块的映射 |
| `v1/core/single_type_kv_cache_manager.py` | 单类型 KV Cache（82KB） |
| `v1/core/kv_cache_coordinator.py` | KV Cache 协调器（37KB）：跨 GPU 协调 |
| `v1/core/kv_cache_utils.py` | KV Cache 工具函数（93KB） |
| `v1/core/kv_cache_metrics.py` | Cache 指标 |

**核心概念**：
- **PagedAttention**：将 KV Cache 分页，类似操作系统的虚拟内存
- **逻辑块** ↔ **物理块** 映射，实现零碎片内存管理
- **Copy-on-Write**：共享 Prompt 时复用物理块
- 新的 `SingleTypeKVCacheManager` 简化了块类型管理

#### 4.2.3 调度器（Scheduler）

| 文件 | 说明 |
|------|------|
| `v1/core/sched/` | 调度器实现 |

调度策略决定：
- 每步调度哪些请求
- Prefill 与 Decode 阶段的混排
- 内存压力下的抢占策略

#### 4.2.4 注意力后端

| 文件 | 说明 |
|------|------|
| `v1/attention/backend.py` | 注意力后端抽象（40KB） |
| `v1/attention/selector.py` | 自动选择最佳后端 |
| `v1/attention/backends/` | 具体后端实现 |
|    ├── `flash_attn/` | FlashAttention |
|    ├── `flash_infer/` | FlashInfer |
|    └── other backends | |

#### 4.2.5 采样器

| 文件 | 说明 |
|------|------|
| `v1/sample/sampler.py` | 核心采样（18KB）：Top-K, Top-P, Temperature, Min-P 等 |
| `v1/sample/rejection_sampler.py` | 拒绝采样（36KB）：投机解码用 |
| `v1/sample/logits_processor/` | Logits 处理器 |
| `v1/sample/ops/` | 采样操作的 GPU Kernel 封装 |

#### 4.2.6 Worker 与模型执行

| 文件 | 说明 |
|------|------|
| `v1/worker/worker_base.py` | Worker 基类（13KB） |
| `v1/worker/gpu_worker.py` | GPU Worker（58KB） |
| `v1/worker/gpu_model_runner.py` | **GPU 模型运行器（350KB！）** — 最大的单个文件 |
| `v1/worker/gpu_input_batch.py` | GPU 输入批处理（48KB） |
| `v1/worker/cpu_worker.py` | CPU Worker |
| `v1/worker/tpu_input_batch.py` | TPU 输入批处理 |
| `v1/worker/block_table.py` | 块表管理 |
| `v1/worker/cudagraph_dispatcher.py` | CUDA Graph 调度 |
| `v1/worker/workspace.py` | GPU Workspace 管理 |

**`gpu_model_runner.py`** 是整个推理引擎最核心的文件（350KB），包含：
- 模型前向传播的编排
- Input 的 prepare（prompt 填充、位置编码等）
- CUDA Graph 的捕捉与回放
- 多模态数据的处理
- LoRA 适配

#### 4.2.7 引擎层

| 文件 | 说明 |
|------|------|
| `v1/engine/` | v1 引擎核心 |
| `v1/executor/` | 执行器（管理分布式执行） |

---

## 5. 阶段三：模型层

### 5.1 模型架构模式

| 文件 | 说明 |
|------|------|
| `model_executor/models/interfaces.py` | 模型接口：`SupportsPP`, `SupportsLoRA` 等 Mixin（55KB） |
| `model_executor/models/interfaces_base.py` | 基础接口 |
| `model_executor/models/registry.py` | 模型注册中心（60KB）：模型名 → 实现类的映射 |
| `model_executor/models/config.py` | 模型配置类（38KB）：从 HuggingFace config 到 vLLM 配置 |

### 5.2 神经网络层

| 文件/目录 | 说明 |
|-----------|------|
| `layers/linear.py` | 线性层（71KB）：支持多种量化、并行、权重类型 |
| `layers/activation.py` | 激活函数（30KB）：SwiGLU, GeGLU 等 |
| `layers/layernorm.py` | LayerNorm / RMSNorm |
| `layers/attention/` | 注意力层（PagedAttention 集成） |
| `layers/attention_layer_base.py` | 注意力基类 |
| `layers/vocab_parallel_embedding.py` | 并行 Embedding（22KB） |
| `layers/logits_processor.py` | Logits 处理器 |
| `layers/fused_moe/` | **融合 MoE 层**（专家混合模型的 FFN 计算） |
| `layers/quantization/` | 量化层 |
| `layers/mamba/` | Mamba/SSM 层 |
| `layers/rotary_embedding/` | RoPE 位置编码 |
| `layers/pooler/` | Pooling 层（用于 embedding 模型） |
| `layers/fusion/` | 算子融合 |

### 5.3 具体模型实现

`model_executor/models/` 目录包含 **100+ 个模型文件**，实现各类主流模型：

**关键学习示例**（从简单到复杂）：
1. `llama.py` — **标准的 Transformer 架构**（最佳起点）
2. `qwen2.py` — 类似 LLaMA 的结构
3. `deepseek_v2.py` — MLA（Multi-head Latent Attention）+ MoE
4. `qwen3_vl.py` / `llava.py` — 多模态模型
5. `mamba.py` / `mamba2.py` — SSM 架构

**学习要点**：
- 每个模型文件如何定义 `XXXModel`, `XXXForCausalLM` 等类
- 模型如何注册到注册表（`@ModelRegistry.register`）
- 权重加载逻辑（`load_weights` 方法）
- 如何支持张量并行（`ShardedStateLoader`）

### 5.4 模型加载

| 文件 | 说明 |
|------|------|
| `model_executor/model_loader/` | 模型加载器 |
| `model_executor/parameter.py` | 权重参数管理：`WeightsLoader`, `ShardedStateLoader` |
| `model_executor/offloader/` | 模型权重卸载（CPU offloading） |
| `model_executor/warmup/` | GPU 预热 |

---

## 6. 阶段四：服务与入口层

### 6.1 离线 API

| 文件 | 说明 |
|------|------|
| `entrypoints/llm.py` | **`LLM` 类**（41KB）：高层离线推理 API |
| `entrypoints/generate/` | 离线生成入口 |
| `entrypoints/offline_utils.py` | 离线推理工具 |

### 6.2 OpenAI 兼容在线 API

| 文件 | 说明 |
|------|------|
| `entrypoints/openai/api_server.py` | FastAPI 服务器（30KB） |
| `entrypoints/openai/cli_args.py` | CLI 参数解析 |
| `entrypoints/openai/completion/` | Completion API |
| `entrypoints/openai/chat_completion/` | Chat Completion API |
| `entrypoints/openai/dp_supervisor.py` | Data Parallel 监督器 |
| `entrypoints/openai/engine/` | API 引擎层 |
| `entrypoints/openai/models/` | 模型定义（API 级别的模型） |
| `entrypoints/openai/parser/` | 请求解析 |
| `entrypoints/openai/responses/` | 响应格式化 |
| `entrypoints/openai/run_batch.py` | 批量运行 |

### 6.3 HTTP 服务

| 文件 | 说明 |
|------|------|
| `entrypoints/serve/` | Serving 架构 |
| `entrypoints/serve/engine/` | Serving 引擎 |
| `entrypoints/serve/dev/` | 开发用服务器 |

### 6.4 其他入口

| 文件 | 说明 |
|------|------|
| `entrypoints/cli/` | 命令行接口 |
| `entrypoints/anthropic/` | Anthropic 兼容 API |
| `entrypoints/speech_to_text/` | 语音转文字 API |
| `entrypoints/pooling/` | Embedding/Pooling API |
| `entrypoints/mcp/` | Model Context Protocol |
| `entrypoints/scale_out/` | 弹性扩缩容 |
| `entrypoints/grpc_server.py` | gRPC 服务 |
| `entrypoints/launcher.py` | 启动器 |
| `entrypoints/chat_utils.py` | 聊天工具（74KB，处理对话模板） |

### 6.5 请求处理流程

```
HTTP Request
    │
    ▼
FastAPI 路由 (api_server.py)
    │
    ▼
请求解析与验证 (parser/)
    │
    ▼
Chat Template 处理 (chat_utils.py)
    │
    ▼
Engine API 调用
    │
    ▼
推理管线
    │
    ▼
响应格式化 (responses/)
    │
    ▼
HTTP Response (SSE / JSON)
```

---

## 7. 阶段五：分布式与并行

### 7.1 并行策略

| 策略 | 说明 | 相关文件 |
|------|------|---------|
| **TP** (Tensor Parallel) | 张量并行：拆分线性层的权重矩阵 | `parallel_state.py` |
| **PP** (Pipeline Parallel) | 流水线并行：按层切分模型 | |
| **EP** (Expert Parallel) | 专家并行：MoE 专家的分布式放置 | `elastic_ep/` |
| **DP** (Data Parallel) | 数据并行：多副本处理不同请求 | `dp_utils.py` |
| **CP** (Context Parallel) | 上下文并行：长上下文切分 | `cp_utils.py` |

### 7.2 核心模块

| 文件/目录 | 说明 |
|-----------|------|
| `distributed/parallel_state.py` | **并行状态管理**（85KB）：所有并行策略的初始化和协调 |
| `distributed/device_communicators/` | 设备通信器（NCCL, custom allreduce） |
| `distributed/communication_op.py` | 通信操作抽象 |
| `distributed/elastic_ep/` | 弹性专家并行 |
| `distributed/eplb/` | EP 负载均衡 |
| `distributed/kv_transfer/` | KV Cache 跨节点传输 |
| `distributed/weight_transfer/` | 权重传输 |
| `distributed/ec_transfer/` | 弹性上下文传输 |
| `distributed/utils.py` | 分布式工具 |
| `distributed/kv_events.py` | KV 事件 |
| `distributed/stateless_coordinator.py` | 无状态协调器 |
| `distributed/nixl_utils.py` | NVIDIA IxL 工具 |

### 7.3 关键概念

- **World Size / Rank**：分布式环境中的进程标识
- **TP Group / PP Group / EP Group**：不同并行策略的通信组
- **CPU AllReduce / Custom AllReduce**：优化的梯度同步方案
- **Ray** 集成：`vllm/ray/`

---

## 8. 阶段六：高级特性

### 8.1 量化（Quantization）

| 目录 | 说明 |
|------|------|
| `model_executor/layers/quantization/` | 量化层实现 |
| `csrc/quantization/` | 量化 CUDA Kernels |
| `_custom_ops.py` | 自定义量化操作 |

支持的量化方案：
- **FP8** (8-bit 浮点数)
- **AWQ** (Activation-aware Weight Quantization)
- **GPTQ** (Post-Training Quantization)
- **INT4/INT8** (对称/非对称量化)
- **SqueezeLLM, GGUF, bitsandbytes**

### 8.2 LoRA 适配器

| 文件/目录 | 说明 |
|-----------|------|
| `lora/model_manager.py` | LoRA 模型管理（53KB） |
| `lora/lora_model.py` | LoRA 模型包装 |
| `lora/lora_weights.py` | LoRA 权重管理 |
| `lora/worker_manager.py` | Worker 上的 LoRA 管理 |
| `lora/utils.py` | LoRA 工具 |
| `lora/request.py` | LoRA 请求 |
| `lora/resolver.py` | LoRA 解析器 |

### 8.3 投机解码（Speculative Decoding）

| 文件/目录 | 说明 |
|-----------|------|
| `v1/spec_decode/` | 投机解码实现 |

- Draft Model 与 Target Model 的协同
- Rejection Sampler 保证输出分布不变
- 多种投机策略（Medusa, MLP Speculator, Eagle）

### 8.4 多模态

| 文件/目录 | 说明 |
|-----------|------|
| `multimodal/inputs.py` | 多模态输入（32KB） |
| `multimodal/parse.py` | 输入解析（24KB） |
| `multimodal/processing/` | 预处理 pipeline |
| `multimodal/registry.py` | 多模态模型注册（13KB） |
| `multimodal/cache.py` | 多模态数据缓存（23KB） |
| `multimodal/audio.py` | 音频输入处理 |
| `multimodal/video.py` | 视频输入处理（75KB） |
| `multimodal/image.py` | 图像输入处理 |
| `multimodal/hasher.py` | 多模态数据哈希 |
| `multimodal/media/` | 媒体处理 |

支持的模态：**文本 + 图像 + 音频 + 视频**

### 8.5 结构化输出 & 推理逻辑

| 文件/目录 | 说明 |
|-----------|------|
| `v1/structured_output/` | 结构化输出（JSON Schema, Regex 约束等） |
| `reasoning/` | 推理能力（R1 风格的 Reasoning） |
| `tool_parsers/` | 工具调用解析（Function Calling） |

### 8.6 KV Cache 卸载

| 文件/目录 | 说明 |
|-----------|------|
| `v1/kv_offload/` | KV Cache 卸载到 CPU 内存 |
| `v1/simple_kv_offload/` | 简化版卸载方案 |

---

## 9. 阶段七：工具链与工程

### 9.1 CUDA/C++ Kernels

| 目录 | 说明 |
|------|------|
| `csrc/attention/` | PagedAttention CUDA 内核 |
| `csrc/core/` | 核心调度内核 |
| `csrc/moe/` | MoE 相关内核 |
| `csrc/quantization/` | 量化内核 |
| `csrc/cutlass_extensions/` | CUTLASS 模板扩展 |
| `csrc/custom_all_reduce.cuh` | 自定义 AllReduce |

**Python 绑定入口**：
- `vllm/_custom_ops.py` — 自定义操作的 Python 封装（118KB）
- `vllm/_aiter_ops.py` — 异步迭代器操作（100KB）

### 9.2 性能优化

| 文件/目录 | 说明 |
|-----------|------|
| `compilation/` | Torch Compile 集成 |
| `v1/cudagraph_dispatcher.py` | CUDA Graph 调度（15KB） |
| `v1/worker/gpu_ubatch_wrapper.py` | Micro-batch 包装 |
| `v1/worker/ubatching.py` | Micro-batching |
| `v1/worker/ubatch_utils.py` | Micro-batch 工具 |
| `csrc/spinloop.cpp` | 自旋锁 |
| `profiler/` | 性能分析 |

### 9.3 测试

| 目录 | 说明 |
|------|------|
| `tests/` | 41 个子目录，涵盖各模块 |
| `tests/evals/` | 模型评估测试 |

### 9.4 工程配置

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 项目配置 |
| `.pre-commit-config.yaml` | 代码风格检查 |
| `.buildkite/` | CI 流水线 |
| `.github/` | GitHub Actions |
| `docker/` | Docker 构建 |

---

## 10. 推荐阅读路径

### 🟢 入门（第 1-2 天）

从最基础的 "Hello World" 开始：

```
1.  README.md                                    → 了解项目概览
2.  entrypoints/llm.py (LLM 类)                 → 高层 API 使用
3.  sampling_params.py                          → 采样参数理解
4.  v1/request.py                               → 请求数据结构
5.  v1/outputs.py                               → 输出结构
6.  v1/kv_cache_interface.py                    → KV Cache 接口
7.  model_executor/models/llama.py              → 基础模型实现
8.  model_executor/models/interfaces.py         → 模型接口
```

### 🔵 核心推理（第 3-5 天）

深入推理管线的骨干：

```
9.  v1/core/kv_cache_manager.py                → Cache 管理
10. v1/core/block_pool.py                      → 物理块池
11. v1/core/sched/                              → 调度器
12. v1/worker/gpu_worker.py                    → Worker
13. v1/worker/gpu_model_runner.py             → ★ 模型运行器（最大文件）
14. v1/attention/backend.py                    → 注意力后端
15. v1/attention/backends/                     → 具体后端
16. v1/sample/sampler.py                       → 采样器
17. engine/arg_utils.py                         → 配置参数
```

### 🟡 服务部署（第 6-7 天）

```
18. entrypoints/openai/api_server.py           → API 服务
19. entrypoints/openai/chat_completion/        → 聊天补全
20. entrypoints/openai/completion/             → 文本补全
21. entrypoints/serve/                         → 服务架构
22. entrypoints/chat_utils.py                  → 聊天模板
```

### 🟠 分布式（第 8 天）

```
23. distributed/parallel_state.py              → 并行状态
24. distributed/device_communicators/          → 通信器
25. distributed/utils.py                       → 分布式工具
```

### 🔴 高级特性（第 9-12 天）

```
26. model_executor/layers/quantization/        → 量化
27. model_executor/layers/fused_moe/           → MoE
28. lora/                                       → LoRA
29. v1/spec_decode/                             → 投机解码
30. multimodal/inputs.py                        → 多模态输入
31. multimodal/processing/                      → 多模态处理
32. v1/structured_output/                       → 结构化输出
33. v1/kv_offload/                              → Cache 卸载
```

### ⚫ 底层与工程（第 13-14 天）

```
34. csrc/attention/                            → CUDA 注意力内核
35. csrc/custom_all_reduce.cuh                 → AllReduce
36. vllm/_custom_ops.py                       → C++ 绑定封装
37. vllm/compilation/                          → Torch Compile
38  v1/cudagraph_dispatcher.py                → CUDA Graph
39. tests/                                     → 测试结构
40. CMakeLists.txt + setup.py                  → 构建系统
```

---

## 附录：关键文件索引

### ⭐ 必须精读的文件（高亮优先）

| 优先级 | 文件 | 行数 | 理由 |
|--------|------|------|------|
| ★★★★★ | `v1/worker/gpu_model_runner.py` | ~350K | vLLM 推理执行的核心 |
| ★★★★★ | `v1/core/kv_cache_manager.py` | ~37K | PagedAttention 的核心 |
| ★★★★★ | `v1/core/block_pool.py` | ~33K | 物理内存管理 |
| ★★★★★ | `v1/core/sched/` | — | 调度策略 |
| ★★★★★ | `v1/attention/backend.py` | ~40K | 注意力抽象 |
| ★★★★☆ | `distributed/parallel_state.py` | ~85K | 分布式并行状态 |
| ★★★★☆ | `model_executor/layers/linear.py` | ~71K | 线性层（量化+并行） |
| ★★★★☆ | `model_executor/models/interfaces.py` | ~55K | 模型接口 |
| ★★★★☆ | `model_executor/models/registry.py` | ~60K | 模型注册 |
| ★★★★☆ | `engine/arg_utils.py` | ~117K | 全部引擎参数 |
| ★★★★☆ | `entrypoints/openai/api_server.py` | ~30K | 在线服务入口 |
| ★★★★☆ | `v1/worker/gpu_worker.py` | ~58K | GPU Worker |
| ★★★☆☆ | `lora/model_manager.py` | ~53K | LoRA 管理 |
| ★★★☆☆ | `v1/sample/sampler.py` | ~18K | 采样策略 |
| ★★★☆☆ | `vllm/envs.py` | ~104K | 环境变量大全 |
| ★★★☆☆ | `multimodal/inputs.py` | ~32K | 多模态输入 |

### 学习技巧

1. **从简单的模型开始**：先读 `llama.py`，再读 `qwen2.py`，最后读 `deepseek_v2.py`（MLA+MoE）
2. **追踪一条请求的生命周期**：从 `api_server.py` → `engine` → `worker` → `model_runner` → 返回
3. **善用 v1 架构**：新代码优先在 `v1/` 下，旧架构在 `vllm/engine/` 和 `vllm/worker/` 下
4. **关注 `__init__.py`**：了解模块导出了什么
5. **配合测试理解**：`tests/` 中的测试是理解模块行为的最佳文档
6. **逐步深入 C++**：先理解 Python 层的语义，再读 `csrc/` 中的 CUDA 内核

---

*编制日期：2025-07-25*
*基于 vllm-project/vllm 主分支*
