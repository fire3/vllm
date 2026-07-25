# vLLM 架构学习 — 章节大纲目录

> 本文档是 [LEARNING_PLAN.md](../LEARNING_PLAN.md) 的具体化，每个章节对应一个大纲文件，列出该章需要覆盖的详细内容要点、代码路径和学习目标。

---

## 章节列表

| 章节 | 文件名 | 核心主题 |
|------|--------|---------|
| **阶段一** | [01-system-foundation.md](01-system-foundation.md) | 配置系统、日志/追踪/监控、编译与构建 |
| **阶段二** | [02-inference-pipeline.md](02-inference-pipeline.md) | 请求生命周期、KV Cache、调度器、注意力后端、采样器、Worker/Model Runner、引擎层 |
| **阶段三** | [03-model-layer.md](03-model-layer.md) | 模型架构模式、神经网络层、模型实现、模型加载 |
| **阶段四** | [04-serving-layer.md](04-serving-layer.md) | 离线 API、OpenAI 兼容服务、HTTP 服务、其他入口 |
| **阶段五** | [05-distributed-parallel.md](05-distributed-parallel.md) | 并行策略、通信器、KV 传输、协调器 |
| **阶段六** | [06-advanced-features.md](06-advanced-features.md) | 量化、LoRA、投机解码、多模态、结构化输出、Cache 卸载 |
| **阶段七** | [07-toolchain-engineering.md](07-toolchain-engineering.md) | CUDA Kernels、性能优化、测试体系、工程配置 |

---

## 每章大纲模板说明

每个章节大纲文件包含以下内容结构：

```
1. 章节概述
   - 本章的定位（在整个架构中的位置）
   - 学习前置依赖

2. 各子节大纲
   每个子节包含：
   - 核心概念（需要理解的关键思想）
   - 代码路径（文件:行号范围的引用）
   - 阅读要点（逐段阅读时需要关注的重点）
   - 学习目标（学完本节能回答什么问题）
   - 思考题（检验理解的题目）

3. 章节总结
   - 知识点地图
   - 推荐的进一步阅读
```

---

## 建议学习顺序

- **线性阅读**：按阶段一→七顺序，由底层到上层
- **跳跃阅读**：如已有基础，可直接从阶段二（核心推理管线）开始
- **随用随查**：遇到具体问题时查阅对应章节

---

*基于 vLLM 主分支 | 编制日期：2025-07-25*
