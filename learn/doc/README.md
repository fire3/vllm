---
home: true
title: 首页
heroText: vLLM 源码分析
tagline: 逐步深入高性能 LLM 推理框架的源码世界
actions:
  - text: 开始学习 →
    link: /01-overview/
    type: primary
features:
  - title: 阶段一 · 系统底座
    details: 配置系统、日志/追踪/监控、编译与构建系统，了解 vLLM 的基础设施
  - title: 阶段二 · 核心推理管线
    details: 深入 KV Cache（PagedAttention）、调度器、注意力后端、采样器、Worker 与 Model Runner
  - title: 阶段三 · 模型层
    details: 模型接口体系、神经网络层库、100+ 模型的具体实现与加载机制
  - title: 阶段四 · 服务与入口层
    details: 离线 API、OpenAI 兼容服务、HTTP 服务、多种入口的统一请求处理流程
  - title: 阶段五 · 分布式与并行
    details: TP/PP/EP/DP/CP 五种并行策略、通信机制、KV Cache 传输与协调
  - title: 阶段六 · 高级特性
    details: 量化（AWQ/GPTQ/FP8）、LoRA、投机解码、多模态、结构化输出
  - title: 阶段七 · 工具链与工程
    details: CUDA Kernels、性能优化（CUDA Graph）、测试体系、CI 与构建
footer: MIT Licensed | Copyright © vLLM Project
---

## 项目概述

[vLLM](https://github.com/vllm-project/vllm) 是一个高性能的大语言模型（LLM）推理引擎，由加州大学伯克利分校开发并开源。其核心创新包括：

- **PagedAttention**：受操作系统分页思想启发的 KV Cache 管理算法，近乎零碎片的内存利用
- **Continuous Batching**：连续批处理，Prefill 和 Decode 阶段在同一个 batch 中混合执行
- **极致推理性能**：通过 CUDA Graph、算子融合、量化等优化达到业界领先的推理吞吐量
- **广泛模型支持**：100+ 模型架构（LLaMA、Qwen、DeepSeek、Gemma、Mistral、Phi 等）
- **多模态**：文本 + 图像 + 音频 + 视频的统一推理
- **分布式并行**：张量并行、流水线并行、专家并行等全面的分布式推理能力

本学习项目按 **七个阶段** 渐进式深入源码，从系统底座到上层服务，从单 GPU 推理到分布式部署，适合有一定 PyTorch 和 Transformer 基础的开发者。

## 如何使用

```bash
# 安装依赖后本地运行 VuePress
cd learn/doc
pnpm install
pnpm run dev
```

每个阶段的分析文档都包含：
- 要阅读的具体代码文件路径（含文件大小参考）
- 关键概念解释和架构流程图
- 学习产出清单与思考题
