# HTTP 服务

## 服务组件

`entrypoints/serve/` 目录下的组件：

| 组件 | 用途 |
|------|------|
| `serve/engine/` | 服务引擎（管理请求队列和 WebSocket） |
| `serve/dev/` | 开发用服务器 |
| `launcher.py` | 统一启动入口 |

## 启动流程

```
CLI: vllm serve ...
    │
    ├── 1. make_arg_parser() 解析参数
    ├── 2. 创建 AsyncEngineArgs
    ├── 3. 初始化 EngineClient（与推理引擎通信）
    ├── 4. 创建 FastAPI app + 注册中间件
    ├── 5. 注册端点插件
    ├── 6. uvicorn.run(app, ...)
    └── 7. 进入事件循环
```

## 其他入口

| 入口 | 说明 |
|------|------|
| `entrypoints/anthropic/` | Anthropic 兼容 API |
| `entrypoints/speech_to_text/` | 语音转文字服务 |
| `entrypoints/pooling/` | Embedding/Pooling 服务 |
| `entrypoints/mcp/` | Model Context Protocol |
| `entrypoints/scale_out/` | 弹性扩缩容 |
| `entrypoints/grpc_server.py` | gRPC 服务 |
| `entrypoints/chat_utils.py` (~74KB) | 聊天模板处理（Jinja2 渲染） |

## 聊天模板

`chat_utils.py` 处理 OpenAI 消息格式到模型输入的转换：

- 使用 HuggingFace `tokenizer.apply_chat_template()`
- 支持 Jinja2 模板
- 处理多模态消息中的图像/音频占位符
