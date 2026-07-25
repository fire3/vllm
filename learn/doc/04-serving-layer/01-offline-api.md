# 离线 API（LLM 类）

## 文件位置

`vllm/entrypoints/llm.py` —— 约 41KB

## 类结构

`LLM` 通过 Mixin 组合了多个能力：

```python
class LLM(
    BeamSearchOfflineMixin,     # 束搜索
    PoolingOfflineMixin,        # Embedding 池化
    OfflineInferenceMixin,      # 离线推理
):
```

## 关键方法

| 方法 | 用途 |
|------|------|
| `__init__(model, ...)` | 接收约 40 个命名参数（模型路径、并行度、量化等），创建 `EngineArgs` → 初始化 `LLMEngine` |
| `generate(prompts, sampling_params)` | 同步推理入口，返回 `list[RequestOutput]` |
| `enqueue(prompts, ...)` | 异步入队：返回 request_id，不阻塞等待结果 |
| `chat(messages, ...)` | 对话模式入口，自动处理 chat template |

## 设计模式

- **Mixin** — 通过多继承组合正交能力
- **建造者** — `EngineArgs` 统一管理配置
- **门面** — 对下层引擎的简化封装

## 使用示例

```python
llm = LLM(model="meta-llama/Llama-2-7b")
outputs = llm.generate("Hello, my name is")
for output in outputs:
    print(output.outputs[0].text)
```
