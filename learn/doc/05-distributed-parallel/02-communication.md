# 通信实现

## 通信后端

`distributed/device_communicators/` 目录封装了多种通信器：

| 通信器 | 用途 |
|--------|------|
| `NCCLCommunicator` | 标准 NCCL 通信（AllReduce、AllGather 等） |
| `CustomAllReduceCommunicator` | vLLM 自定义 AllReduce |

## 自定义 AllReduce

### 文件位置

`csrc/custom_all_reduce.cuh` —— 约 23KB

标准 NCCL AllReduce 需要 CPU 参与调度，延迟较高。vLLM 的自定义 AllReduce 利用 NVLink/NVSwitch 的直接 GPU-GPU 通信：

```python
class CustomAllReduceCommunicator:
    def all_reduce(self, tensor):
        # 调用自定义 CUDA kernel
        # 不经过 CPU 调度
        # 适用于同节点内的 GPU 通信
```

**优势**：
- 延迟更低（省去 CPU 调度开销）
- 更适合小张量的频繁同步（如 TP 中的激活值）

**限制**：
- 仅支持同节点内的 GPU
- 需要 NVLink 连接

## 通信模式对比

| 操作 | NCCL | Custom AllReduce | 适用场景 |
|------|------|-----------------|---------|
| AllReduce | `ncclAllReduce` | 自定义 kernel | 行并行、梯度同步 |
| AllGather | `ncclAllGather` | — | 列并行输出收集 |
| ReduceScatter | `ncclReduceScatter` | — | 梯度分片 |
| P2P Send/Recv | `ncclSend`/`ncclRecv` | — | 流水线并行 |
| All-to-All | `ncclAllToAll` | — | 专家并行 |
