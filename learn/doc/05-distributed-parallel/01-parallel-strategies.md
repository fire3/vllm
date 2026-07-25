# 并行策略

## 五种并行概述

### 张量并行（TP）

将单个算子的权重矩阵切分到多个 GPU。每个 GPU 只存储和计算一部分权重。

```python
# ColumnParallelLinear: 输出特征按列切分
# TP=2 时，rank0 计算前一半输出，rank1 计算后一半

# RowParallelLinear: 输入特征按行切分
# 结果需要 all-reduce 汇总
```

- **适用**：所有线性层、Attention 的 QKV 投影
- **通信模式**：AllReduce（行并行）或无通信（列并行后接 all-gather）
- **典型规模**：2-8 GPU

### 流水线并行（PP）

按层切分模型。每个 GPU 负责连续若干层，前一个 GPU 的输出传给下一个 GPU。

- **适用**：深层网络（>40 层）
- **通信模式**：P2P send/recv（仅层边界）
- **典型规模**：2-4 阶段

### 专家并行（EP）

将 MoE 层中的专家分布到不同 GPU，每个 token 通过路由发往目标 GPU 上的专家计算。

- **适用**：MoE 模型（Mixtral、DeepSeek-V2）
- **通信模式**：All-to-All（token 分发 + 结果收集）
- **典型规模**：4-64 GPU

### 数据并行（DP）

多份完整的模型副本，分别处理不同的请求。

- **适用**：高并发场景
- **通信模式**：无（各副本独立推理）

### 上下文并行（CP）

将长上下文切分到多个 GPU，每个 GPU 处理序列的不同段。

- **适用**：超长序列（>128K tokens）
- **通信模式**：Ring Attention 等

## 状态管理

### 文件位置

`vllm/distributed/parallel_state.py` —— 约 85KB

`GroupCoordinator` 是并行状态的核心类：

```python
class GroupCoordinator:
    """管理一组通信进程"""
    rank: int              # 组内 rank
    world_size: int        # 组大小
    ranks: list[int]       # 全局 rank 列表
    cpu_group: ProcessGroup    # CPU 通信组
    device_group: ProcessGroup # GPU 通信组
```

并行初始化时会创建多个 `GroupCoordinator` 实例，分别对应 TP Group、PP Group、EP Group 等：

```python
# 根据 --tensor-parallel-size, --pipeline-parallel-size 等参数
tp_group = GroupCoordinator(tensor_parallel_rank_group)
pp_group = GroupCoordinator(pipeline_parallel_rank_group)
```

## 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tensor-parallel-size` | `1` | TP 并行度 |
| `--pipeline-parallel-size` | `1` | PP 并行度 |
| `--data-parallel-size` | `1` | DP 并行度 |
| `--expert-parallel-size` | `1` | EP 并行度 |

这些参数传递给 `EngineArgs` → `ParallelConfig` → `parallel_state.py` 初始化。
