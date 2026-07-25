# OpenAI 兼容服务

## 文件位置

`vllm/entrypoints/openai/api_server.py` —— 约 30KB

## 服务器架构

基于 FastAPI 的异步 HTTP 服务，支持 OpenAI 协议的全部核心端点：

| 端点 | 用途 |
|------|------|
| `POST /v1/chat/completions` | 聊天补全 |
| `POST /v1/completions` | 文本补全 |
| `POST /v1/embeddings` | Embedding |
| `GET /v1/models` | 模型列表 |
| `GET /health` | 健康检查 |
| `GET /metrics` | Prometheus 指标 |

## 插件机制

服务器启动时通过 `_attach_endpoint_plugins()` 动态发现并注册端点插件：

```python
def _attach_endpoint_plugins(app, supported_tasks):
    plugins = load_endpoint_plugins()
    for plugin in plugins:
        if plugin.task in supported_tasks:
            plugin.attach(app)
```

## 中间件链路

```python
app.add_middleware(CORSMiddleware, ...)   # CORS
app.add_middleware(ScalingMiddleware, ...) # 弹性缩放
# ...

# 异常处理器
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(...): ...

@app.exception_handler(Exception)
async def generic_exception_handler(...): ...
```

## 子目录结构

```
entrypoints/openai/
├── api_server.py        — FastAPI 主服务器
├── cli_args.py          — CLI 参数定义
├── chat_completion/     — 聊天补全实现
├── completion/          — 文本补全实现
├── engine/              — API 引擎层
├── models/              — API 模型定义
├── parser/              — 请求解析
├── responses/           — 响应格式化
├── dp_supervisor.py     — Data Parallel 请求分发
└── run_batch.py         — 批量请求处理
```

## 请求处理链

```
HTTP Request → FastAPI 路由
    → parser 解析 JSON body
    → chat template 处理对话格式
    → Engine API 调用
    → 推理引擎执行
    → responses 格式化输出（SSE / JSON）
    → HTTP Response
```
