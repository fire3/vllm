# 阶段一：基础层（系统底座）

> 学习目标：理解 vLLM 启动时发生了什么、配置如何流转、日志/监控如何工作、C++/CUDA 扩展如何集成到 Python 包中。

---

## 1.1 配置系统

### 核心概念

- **双层配置架构**：环境变量（`envs.py`）→ 引擎参数（`arg_utils.py`）
- `EngineArgs` 是所有引擎配置的统一入口，从 CLI args 解析并分发到各个子模块
- 平台抽象层（`platforms/`）允许同一套代码适配 CUDA/ROCm/CPU/TPU/XPU

### 代码路径

| 文件 | 定位 | 说明 |
|------|------|------|
| `vllm/envs.py` | 全文 ~104KB | 所有环境变量的定义、默认值、文档 |
| `vllm/engine/arg_utils.py` | 全文 ~117KB | `EngineArgs` 类：管理所有引擎启动参数 |
| `vllm/v1/engine/core.py` | 待定位 | v1 引擎核心配置 |
| `vllm/config/` | 目录 | 模型配置、设备配置 |
| `vllm/platforms/` | 目录 | 各硬件平台的抽象层 |

### 阅读要点

1. **`envs.py` 结构**：每个环境变量的定义模式 — `env_var_name → default_value → description`
   - 重点关注：`VLLM_*` 前缀的变量（如 `VLLM_ATTENTION_BACKEND`, `VLLM_USE_V1`）
   - GPU 内存控制相关：`VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_KV_CACHE_DTYPE` 等

2. **`arg_utils.py` 结构**：
   - `EngineArgs` 类的字段定义（模型路径、并行度、量化、调度参数等）
   - `EngineArgs.create_engine_configs()` 方法如何生产 `ModelConfig`、`CacheConfig`、`ParallelConfig`、`SchedulerConfig`
   - 参数校验逻辑（如 `max_model_len` 与 `max_num_seqs` 的关系）

3. **平台抽象**：
   - `Platform` 基类定义了哪些虚拟方法
   - CUDA vs ROCm vs CPU 平台的区别
   - `platforms/__init__.py` 的自动检测逻辑

### 学习目标

学完本节能回答：
- [ ] vLLM 有哪些环境变量控制 KV Cache 的分配？
- [ ] `EngineArgs` 到各 Config 对象的转换路径是什么？
- [ ] 平台抽象如何影响 Attention Backend 的选择？
- [ ] `gpu_memory_utilization` 与 `max_model_len` 的关系是什么？

### 思考题

1. 如果我想让 vLLM 使用 90% 的 GPU 显存而非默认的 90%，需要修改哪个环境变量或参数？
2. 新增一个硬件平台需要实现 `Platform` 的哪些方法？
3. `EngineArgs` 中哪个参数会影响最大并发请求数？

---

## 1.2 日志、追踪与监控

### 核心概念

- **分布式感知的日志系统**：自动在日志前缀添加 rank 信息，区分不同 GPU 进程的输出
- **OpenTelemetry 追踪**：端到端的请求延迟追踪
- **Prometheus 指标**：v1 架构下的结构化指标收集

### 代码路径

| 文件 | 说明 |
|------|------|
| `vllm/logger.py` | 日志系统实现 |
| `vllm/tracing/` | OpenTelemetry 分布式追踪 |
| `vllm/usage/` | 使用统计数据收集 |
| `vllm/v1/metrics/` | v1 架构的 Prometheus 指标 |
| `vllm/logging_utils/` | 日志工具 |

### 阅读要点

1. **日志系统**：
   - 如何为不同 rank 的进程添加日志前缀
   - 日志级别控制（环境变量或代码配置）
   - `init_logger()` 函数的全局单例模式

2. **OpenTelemetry 追踪**：
   - Span 的生命周期：从请求到达开始，直到响应返回
   - 关键 Span 名称与阶段对应关系

3. **v1 Metrics**：
   - 收集哪些指标（吞吐量、延迟、KV Cache 命中率等）
   - 如何通过 Prometheus 暴露

### 学习目标

- [ ] 理解 vLLM 的分布式日志架构
- [ ] 知道如何启用 OpenTelemetry 追踪
- [ ] 知道如何通过 `/metrics` 端点获取性能指标

---

## 1.3 编译与构建系统

### 核心概念

- **`setup.py` 驱动的构建**：通过 `torch.utils.cpp_extension` 将 CUDA C++ 代码编译为 Python 可调用的扩展模块
- **CMake 子构建**：`CMakeLists.txt` 管理 C++/CUDA 编译，`setup.py` 调用 CMake
- **预编译轮（precompiled wheel）**：`VLLM_USE_PRECOMPILED` 加速安装
- **Rust 组件**：用于分布式协调的高性能并发原语
- **`torch_bindings.cpp`**：C++ 到 Python 的绑定入口

### 代码路径

| 文件 | 说明 |
|------|------|
| `setup.py` ~50KB | Python 包构建、CUDA 扩展注册 |
| `CMakeLists.txt` ~63KB | CMake 构建配置 |
| `cmake/` | CMake 模块查找脚本 |
| `csrc/torch_bindings.cpp` | C++ 操作的 PyBind 绑定 |
| `rust/` | Rust 组件 |
| `vllm/compilation/` | Torch Compile 相关 |
| `pyproject.toml` | 项目元数据与构建系统配置 |
| `requirements/` | 依赖管理文件 |
| `build_rust.sh` | Rust 组件构建脚本 |

### 阅读要点

1. **`setup.py` 结构**：
   - `torch.utils.cpp_extension.CUDAExtension` 的注册
   - 不同平台（Linux/Windows）的条件编译
   - `ext_modules` 列表中的各个扩展模块名与源文件的对应

2. **`CMakeLists.txt` 结构**：
   - CUDA 架构目标的设置（`CMAKE_CUDA_ARCHITECTURES`）
   - 各子目录（`csrc/attention`, `csrc/moe` 等）的编译
   - 依赖查找（FlashAttention, CUTLASS 等）

3. **`torch_bindings.cpp`**：
   - `PYBIND11_MODULE` 宏的使用
   - 注册了哪些 C++ 函数，对应 Python 中的哪个命名空间

### 学习目标

- [ ] 理解 `setup.py` 如何将 CUDA C++ 代码构建为 Python 模块
- [ ] 知道 `torch_bindings.cpp` 在架构中的角色
- [ ] 理解 `VLLM_USE_PRECOMPILED` 的作用
- [ ] 知道 Rust 组件用于什么场景

### 思考题

1. 新增一个 CUDA kernel 文件需要修改哪些构建文件？
2. `torch_bindings.cpp` 中的函数与 `vllm/_custom_ops.py` 是什么关系？
3. 为什么 vLLM 要用 Rust 而非 C++ 来实现某些分布式组件？

---

## 章节总结

### 知识点地图

```
用户输入 (CLI args)
    │
    ▼
EngineArgs (arg_utils.py) ─── envs.py (环境变量覆盖)
    │
    ├──▶ ModelConfig
    ├──▶ CacheConfig
    ├──▶ ParallelConfig
    ├──▶ SchedulerConfig
    └──▶ ...
    │
    ▼
Platform 检测 ──→ 选择 Attention Backend
    │
    ▼
进程启动 ──→ Logger (rank 感知)
    │
    ▼
模型加载 ──→ 构建系统已就绪 (C++ 扩展已导入)
```

### 进一步阅读

- 继续阶段二：核心推理管线 → `02-inference-pipeline.md`
- PyTorch C++ Extension 官方文档
- CMake + CUDA 构建最佳实践

---

*对应 LEARNING_PLAN.md 第 3 章 | 基于 vLLM 主分支*
