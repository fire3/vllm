# 阶段四：服务与入口层

> 学习目标：理解 vLLM 提供的各种使用方式——离线 Python API、OpenAI 兼容的 HTTP 服务、CLI 工具等，并掌握在线服务架构的组件与请求处理流程。

---

## 4.1 离线 API（LLM 类）

### 核心概念

- **`LLM` 类**：vLLM 的最高层 Python API，类似 HuggingFace `pipeline`
- **离线推理**：在同一个 Python 进程中完成模型加载 → 推理 → 输出
- **`generate()` vs `chat()`**：文本补全与对话两种模式

### 代码路径

| 文件 | 说明 |
|------|------|
| `entrypoints/llm.py` ~41KB | `LLM` 类的主实现 |
| `v1/engine/` | v1 引擎（LLM 类底层调用） |
| `vllm/sampling_params.py` | 采样参数 |

### 阅读要点

1. **`LLM` 类的生命周期**：
   - `__init__()`：引擎初始化、模型加载
   - `generate()`：同步推理入口
   - `generate_async()`：异步推理入口（`async_generator`）
   - `chat()`：对话模式的便捷封装

2. **使用方法**：
   - `llm.generate("Hello, my name is")` — 基础用法
   - `llm.generate([prompt1, prompt2])` — 批量推理
   - `llm.chat([{"role": "user", "content": "Hi"}])` — 对话模式

### 学习目标

- [ ] 掌握 `LLM` 类的核心 API 用法
- [ ] 理解离线推理与在线服务的本质区别
- [ ] 知道 `generate()` 与 `chat()` 的底层实现差异

---

## 4.2 OpenAI 兼容在线 API

### 核心概念

- **OpenAI 协议兼容**：提供 `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` 等端点
- **FastAPI 服务器**：基于 ASGI 的异步 HTTP 服务
- **流式输出（SSE）**：Server-Sent Events 实现逐 token 返回
- **Data Parallel 监督器**：多副本部署时的请求分发

### 代码路径

| 文件 | 说明 |
|------|------|
| `entrypoints/openai/api_server.py` ~30KB | FastAPI 主服务器 |
| `entrypoints/openai/cli_args.py` ~18KB | CLI 参数 |
| `entrypoints/openai/chat_completion/` | 聊天补全端点 |
| `entrypoints/openai/completion/` | 文本补全端点 |
| `entrypoints/openai/engine/` | API 引擎层 |
| `entrypoints/openai/models/` | API 模型定义 |
| `entrypoints/openai/parser/` | 请求解析 |
| `entrypoints/openai/responses/` | 响应格式化 |
| `entrypoints/openai/dp_supervisor.py` ~21KB | DP 监督器 |
| `entrypoints/openai/run_batch.py` ~31KB | 批量运行 |

### 阅读要点

1. **`api_server.py`**：
   - FastAPI app 的创建与中间件链
   - 路由注册：`@app.post("/v1/chat/completions")` 等
   - 启动流程：`uvicorn.run()` 的参数配置
   - 生命周期事件：`startup` (引擎初始化) 和 `shutdown` (资源清理)

2. **请求处理流**（`chat_completion/`）：
   - 请求解析：验证 JSON body，构建 `ChatCompletionRequest`
   - Chat Template 处理：将对话消息转为模型可识别的格式
   - Engine 调用：将请求提交给推理引擎
   - 流式响应：SSE 格式的逐 token 输出
   - 非流式响应：聚合所有 token 后返回完整结果

3. **`dp_supervisor.py`**：
   - Data Parallel 模式下如何分发请求
   - 多副本之间的负载均衡策略

### 学习目标

- [ ] 理解 OpenAI API 兼容层的架构设计
- [ ] 知道如何通过 CLI 启动 vLLM 服务
- [ ] 理解流式输出（SSE）和非流式输出的实现差异
- [ ] 知道请求从 HTTP 到引擎的完整路径

---

## 4.3 HTTP 服务层

### 核心概念

- **Serving 架构**：vLLM 的 HTTP 服务组件栈
- **`entrypoints/serve/`**：服务相关的工具和中间件

### 代码路径

| 文件 | 说明 |
|------|------|
| `entrypoints/serve/` | 服务层目录 |
| `entrypoints/serve/engine/` | 服务引擎 |
| `entrypoints/serve/dev/` | 开发用服务器 |
| `entrypoints/launcher.py` | 启动器统一入口 |

### 阅读要点

1. **serve 架构**：
   - 服务引擎的职责：管理请求队列、WebSocket 连接等
   - 健康检查、指标暴露的端点

### 学习目标

- [ ] 理解 HTTP 服务层的整体架构

---

## 4.4 其他入口

### 核心概念

vLLM 提供多种 API，不止是 OpenAI 兼容接口。

### 代码路径

| 入口 | 说明 |
|------|------|
| `entrypoints/cli/` | 命令行接口（`vllm serve`, `vllm chat` 等） |
| `entrypoints/anthropic/` | Anthropic 兼容 API |
| `entrypoints/speech_to_text/` | 语音转文字服务（whisper 等） |
| `entrypoints/pooling/` | Embedding/Pooling 服务（文本向量化） |
| `entrypoints/mcp/` | Model Context Protocol 服务 |
| `entrypoints/scale_out/` | 弹性扩缩容 |
| `entrypoints/grpc_server.py` ~6KB | gRPC 服务 |
| `entrypoints/chat_utils.py` ~74KB | 聊天模板工具 |

### 阅读要点

1. **`chat_utils.py`**（74KB，较大文件）：
   - 聊天模板的处理：Jinja2 模板渲染
   - 支持 HuggingFace `tokenizer.apply_chat_template`
   - 多模态消息的格式组织

2. **`grpc_server.py`**：
   - gRPC vs REST：不同场景的选择
   - gRPC 服务的 Protobuf 定义

### 学习目标

- [ ] 知道 vLLM 支持的所有入口类型
- [ ] 理解聊天模板的工作原理

---

## 4.5 请求处理完整流程

### 核心流程

```
用户请求 (HTTP / Python API / CLI)
    │
    ▼
┌─────────────────────────────────────────┐
│            API 入口层                     │
│  (OpenAI / Anthropic / gRPC / LLM API)  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           请求解析与校验                   │
│  Pascal → Request object → validate     │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│          聊天模板处理 (如需要)             │
│  Chat template → 模型输入格式            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           Engine API 接口                │
│  Engine.add_request()                    │
└────────────────┬────────────────────────┘
                 │
                 ▼ (进入阶段二推理管线)
┌─────────────────────────────────────────┐
│              推理引擎                     │
│  Scheduler → Worker → Model Runner      │
│  → Attention → Sampler → Output         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│           响应格式化                      │
│  Output → OpenAI / SSE / JSON 格式       │
└────────────────┬────────────────────────┘
                 │
                 ▼
            返回给客户端
```

### 代码路径

- 全程涉及：`entrypoints/openai/api_server.py` → `v1/engine/` → ... → 返回
- 关键接口：`Engine.add_request()` 和异步的 `generate()` 方法

### 学习目标

- [ ] 能画出请求从客户端到输出的完整流程图
- [ ] 知道每个步骤涉及的代码文件和关键函数

---

## 章节总结

### 服务层层次结构

```
CLI (vllm serve)          Python SDK (LLM类)         外部 HTTP/SDK 客户端
    │                           │                           │
    ▼                           ▼                           ▼
┌───────────────────────────────────────────────────────────────┐
│                  服务启动器 (launcher.py)                       │
├───────────────────────────────────────────────────────────────┤
│     OpenAI API    │   Anthropic    │  gRPC  │    Pooling     │
│  (api_server.py)  │   (anthropic/) │ (grpc) │  (pooling/)    │
├───────────────────────────────────────────────────────────────┤
│                     Chat Utils (chat_utils.py)                 │
├───────────────────────────────────────────────────────────────┤
│                     Engine API 接口                            │
├───────────────────────────────────────────────────────────────┤
│                     推理引擎 (v1/engine/)                      │
└───────────────────────────────────────────────────────────────┘
```

### 进一步阅读

- 继续阶段五：分布式与并行 → `05-distributed-parallel.md`
- FastAPI 官方文档
- OpenAI API 协议规范

---

*对应 LEARNING_PLAN.md 第 6 章 | 基于 vLLM 主分支*
