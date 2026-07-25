# 阶段五：分布式与并行

> 学习目标：理解 vLLM 支持的各种并行策略（TP/PP/EP/DP/CP）、分布式通信机制、以及如何在多 GPU/多节点环境中高效协调推理。

---

## 5.1 并行策略全景

### 核心概念

| 策略 | 英文 | 缩写 | 说明 |
|------|------|------|------|
| **张量并行** | Tensor Parallelism | TP | 拆分单个算子的权重矩阵到多 GPU |
| **流水线并行** | Pipeline Parallelism | PP | 按层切分模型到不同 GPU |
| **专家并行** | Expert Parallelism | EP | 将 MoE 专家分布到不同 GPU |
| **数据并行** | Data Parallelism | DP | 多份模型副本处理不同请求 |
| **上下文并行** | Context Parallel | CP | 将长上下文切分到多个 GPU |

### 组合关系

```
                序列
                 │
                 ▼
        ┌────────────────┐
        │    DP Group    │  ← 数据并行（请求级别）
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │    PP Group    │  ← 流水线并行（层级别）
        └───────┬────────┘
                │
        ┌───────▼────────┐
        │ TP / EP Group  │  ← 张量/专家并行（算子级别）
        └────────────────┘
```

### 代码路径

| 文件 | 说明 |
|------|------|
| `distributed/parallel_state.py` ~85KB | **并行状态管理的核心** |
| `v1/worker/dp_utils.py` | DP 工具 |
| `v1/worker/cp_utils.py` ~11KB | CP 工具 |

### 阅读要点

1. **`parallel_state.py`**（**重点，85KB**）：
   - `World` 与 `GroupCoordinator`：分布式环境的抽象
   - `get_tp_group()` / `get_pp_group()` / `get_ep_group()`：获取各组通信器
   - 初始化流程：根据 `--tensor-parallel-size`, `--pipeline-parallel-size` 等参数创建通信组
   - 进程 rank 的分配与管理

### 学习目标

- [ ] 理解五种并行策略的原理与适用场景
- [ ] 知道 `parallel_state.py` 是分布式配置的中枢
- [ ] 能解释 TP/PP/EP/DP/CP 之间的关系

---

## 5.2 分布式通信

### 核心概念

- **NCCL**：NVIDIA 的集体通信库（AllReduce, AllGather, ReduceScatter 等）
- **Custom AllReduce**：vLLM 自定义的 AllReduce 实现，减少同步开销
- **Device Communicators**：封装不同通信后端

### 代码路径

| 文件 | 说明 |
|------|------|
| `distributed/device_communicators/` | 设备通信器目录 |
| `distributed/communication_op.py` ~1KB | 通信操作抽象 |
| `csrc/custom_all_reduce.cuh` ~23KB | 自定义 AllReduce CUDA 实现 |
| `csrc/custom_quickreduce.cu` | 快速 Reduce |

### 阅读要点

1. **设备通信器**：
   - `NCCLCommunicator`：标准 NCCL 封装
   - `CustomAllReduceCommunicator`：自定义 AllReduce 的 Python 端
   - 何时选择自定义 AllReduce vs 标准 NCCL

2. **`custom_all_reduce.cuh`**：
   - 实现原理：利用 NVLink/NVSwitch 的直接 GPU-GPU 通信
   - 相比 NCCL 的优势：更低延迟，更少同步点

### 学习目标

- [ ] 理解 NCCL 集体通信操作（AllReduce, Broadcast, AllGather）
- [ ] 知道 vLLM 自定义 AllReduce 的设计动机和优势
- [ ] 理解设备通信器的分层设计

---

## 5.3 KV Cache 与权重传输

### 核心概念

- **KV Cache 跨节点传输**：在分布式推理中，不同节点可能需要共享 KV Cache
- **权重传输**：模型权重在 GPU 之间的分发

### 代码路径

| 文件 | 说明 |
|------|------|
| `distributed/kv_transfer/` | KV Cache 跨节点传输 |
| `distributed/weight_transfer/` | 权重传输 |
| `distributed/kv_events.py` ~17KB | KV 事件 |
| `distributed/stateless_coordinator.py` ~27KB | 无状态协调器 |
| `distributed/elastic_ep/` | 弹性专家并行 |
| `distributed/eplb/` | EP 负载均衡 |
| `distributed/ec_transfer/` | 弹性上下文传输 |
| `distributed/nixl_utils.py` | NVIDIA IxL 工具 |

### 阅读要点

1. **KV 传输**：
   - 何时需要跨节点传输 KV Cache（如 prefix caching 跨服务节点）
   - 传输协议的实现（gRPC? NCCL? 共享内存?）

2. **弹性专家并行**：
   - 专家在节点间的动态放置
   - 负载均衡：`eplb/` 中的算法

### 学习目标

- [ ] 理解 KV Cache 传输的使用场景
- [ ] 知道弹性专家并行的基本思想

---

## 5.4 分布式协调

### 核心概念

- **无状态协调器（Stateless Coordinator）**：分布式节点间的轻量级协调
- **KV 事件**：KV Cache 状态变更的事件通知

### 代码路径

| 文件 | 说明 |
|------|------|
| `distributed/stateless_coordinator.py` | 协调器 |
| `distributed/kv_events.py` | KV 事件 |
| `vllm/ray/` | Ray 集成 |
| `vllm/connections.py` ~12KB | 连接管理 |

### 阅读要点

1. **无状态协调器**：
   - 职责：节点发现、状态同步、故障恢复
   - 为什么不使用中心化的协调服务？

### 学习目标

- [ ] 理解分布式协调的基本机制
- [ ] 知道 Ray 在 vLLM 分布式中的角色

---

## 章节总结

### 分布式架构全景

```
多节点 / 多 GPU 环境
    │
    ▼
┌─────────────────────────────────────────────┐
│          分布式协调 (stateless_coordinator)  │
├─────────────────────────────────────────────┤
│          并行状态 (parallel_state)           │
│    TP Group  PP Group  EP Group  DP Group   │
├────────────┬──────────┬──────────┬──────────┤
│  NCCL Comm │ CustomAR │ KV Xfer  │ Wt Xfer  │
│ (communicat│ (csrc/)  │(kv_transf)│(weight_tr)│
│  -ors/)    │          │          │          │
└────────────┴──────────┴──────────┴──────────┘
```

### 进一步阅读

- 继续阶段六：高级特性 → `06-advanced-features.md`
- NCCL 文档：https://docs.nvidia.com/deeplearning/nccl/
- Megatron-LM 论文（TP/PP 的原创工作）

---

*对应 LEARNING_PLAN.md 第 7 章 | 基于 vLLM 主分支*
