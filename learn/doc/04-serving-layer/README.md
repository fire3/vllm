# 阶段四：服务与入口层

> 服务层是 vLLM 与用户交互的界面。它提供多种访问方式——从离线 Python API 到 OpenAI 兼容的 HTTP 服务，从命令行工具到 gRPC 接口。

## 章节列表

1. [离线 API](01-offline-api.md) — LLM 类的设计与用法
2. [OpenAI 兼容服务](02-openai-api.md) — FastAPI 服务器架构
3. [HTTP 服务](03-http-server.md) — 服务层组件与启动流程
4. [其他入口](04-other-entrypoints.md) — CLI、Anthropic、gRPC、语音等
5. [请求处理流程](05-request-flow.md) — 从 HTTP 到引擎的完整路径
