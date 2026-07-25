# 阶段三：模型层

> 模型层是 vLLM 推理引擎的"语义核心"——定义了 vLLM 支持的所有模型架构的实现方式。从模型接口体系到具体模型文件，从神经网络层库到权重加载，本章逐一拆解。

## 章节列表

1. [模型接口与注册](01-model-interfaces.md) — Mixin 能力体系、模型注册表、配置映射
2. [神经网络层](02-neural-layers.md) — Linear、Attention、MoE、量化层等核心算子
3. [模型实现分析](03-model-implementations.md) — 以 LLaMA 为起点的模型架构实现
4. [模型加载](04-model-loading.md) — 权重加载器、参数管理、分布式加载

## 层次关系

```
┌─────────────────────────────────────────────┐
│           模型抽象层                          │
│  interfaces.py (Mixin) + registry.py (注册)  │
├─────────────────────────────────────────────┤
│           神经网络层库                        │
│  layers/linear.py, layers/attention/, ...    │
├─────────────────────────────────────────────┤
│           具体模型实现                        │
│  models/llama.py, qwen2.py, deepseek_v2.py  │
├─────────────────────────────────────────────┤
│           模型加载管道                        │
│  model_loader/, parameter.py                │
└─────────────────────────────────────────────┘
```

## 前置知识

- Transformer 架构基础（embedding、self-attention、FFN、layer norm）
- 张量并行（Tensor Parallelism）的基本概念
- PyTorch `nn.Module` 和参数管理
