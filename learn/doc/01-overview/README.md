# 阶段一：系统底座

> 目标：理解 vLLM 启动时发生了什么、配置如何流转、日志/监控如何工作、C++/CUDA 扩展如何集成到 Python 包中。

## 章节列表

1. [配置系统](01-config-system.md) — `envs.py`、`arg_utils.py`、平台抽象
2. [日志、追踪与监控](02-logging-monitoring.md) — 分布式日志、OpenTelemetry、Prometheus 指标
3. [编译与构建系统](03-build-system.md) — `setup.py`、CMake、CUDA 扩展绑定、Rust 组件

## 前置知识

- 基本的 Python 包管理（`setup.py`、`pyproject.toml`）
- 基础的 CUDA / GPU 概念（显存、cuda compute capability）
- PyTorch 基础（`torch.utils.cpp_extension`）

## 推荐阅读方法

按编号顺序依次阅读。如果时间有限，**至少阅读第 1 篇**（配置系统），因为了解配置是一切的基础。

每篇文档都包含：
- 要阅读的具体代码文件
- 关键概念解释
- 学习产出清单（Checklist）
- 思考题
