# 结构化输出

## 文件位置

`vllm/v1/structured_output/` 目录

## 架构

```
StructuredOutputManager
    └── 持有某个后端：StructuredOutputBackend
        ├── GuidanceBackend
        ├── LmFormatEnforcerBackend
        ├── OutlinesBackend
        └── XgrammarBackend
```

## 原理

结构化输出通过**在采样时约束 logits**实现：

1. 将 JSON Schema / Regex 编译为有限状态机（FSM）
2. 在每一步解码时，根据当前状态计算允许的 token 集合
3. 将不允许的 token 的 logits 设为 `-inf`
4. Sampler 只能从允许的 token 中采样

```python
# 结构化输出的 logits 约束
logits[forbidden_token_ids] = -float('inf')
sampled_token = sampler.sample(logits)
```

## 后端对比

| 后端 | 语法支持 | 特点 |
|------|---------|------|
| Outlines | JSON Schema, Regex, CFG | 最成熟，社区广泛使用 |
| Guidance | JSON Schema | Microsoft 出品 |
| LM Format Enforcer | JSON Schema | 性能优化 |
| XGrammar | JSON Schema, Regex | 异步 grammar 编译 |

## 思考预算

`v1/sample/thinking_budget_state.py` 管理推理模型（如 DeepSeek-R1）的思考 token 预算：

- 限制模型在 "thinking" 阶段的 token 消耗
- 达到上限时自动切换到 "answer" 阶段
