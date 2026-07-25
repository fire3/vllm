# 其他入口

> vLLM 提供多种 API 入口以适配不同的使用场景和协议。

## CLI 入口

`entrypoints/cli/` 目录提供命令行工具：

| 命令 | 用途 |
|------|------|
| `vllm serve` | 启动 OpenAI 兼容 HTTP 服务 |
| `vllm chat` | 交互式聊天 |
| `vllm generate` | 命令行文本生成 |

入口通过 `entrypoints/launcher.py` 统一调度。

## Anthropic 兼容 API

`entrypoints/anthropic/` 目录实现 Anthropic 消息协议：

- `POST /v1/messages` — 消息补全
- 流式与非流式支持
- 与 OpenAI API 共享底层引擎

## gRPC 服务

`entrypoints/grpc_server.py`（~6KB）提供 gRPC 接口：

- 适用于微服务架构的推理调用
- Protobuf 定义的请求/响应格式
- 比 REST 更低的序列化开销

## 语音服务

`entrypoints/speech_to_text/` 目录：

- 基于 Whisper 等模型的语音转文字
- 支持实时音频流

## Pooling/Embedding

`entrypoints/pooling/` 目录处理向量化请求：

| 端点 | 用途 |
|------|------|
| `POST /v1/embeddings` | 文本向量化 |
| 其他 pooling 端点 | 分类、重排序等 |

## 弹性扩缩容

`entrypoints/scale_out/` 目录：

- 自动检测请求量变化
- 动态增加/减少推理节点
- 与 Kubernetes 等编排平台集成
