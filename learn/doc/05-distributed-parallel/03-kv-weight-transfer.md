# KV 与权重传输

## KV Cache 跨节点传输

### 文件位置

`vllm/distributed/kv_transfer/` 目录

### 使用场景

- 分布式 Prefix Cache：节点 A 计算过的前缀缓存，节点 B 可以直接复用
- PD 分离部署：Prefill 节点计算 KV Cache 后传输给 Decode 节点

### 传输机制

```python
class KVTransferEngine:
    def send_kv_cache(self, blocks, target_rank):
        # 将 KV 块发送到目标 GPU
        ...
    
    def recv_kv_cache(self, source_rank) -> KVCacheBlocks:
        # 从源 GPU 接收 KV 块
        ...
```

## 权重传输

### 文件位置

`vllm/distributed/weight_transfer/` 目录

用于在弹性场景下将模型权重同步到新加入的节点。

## 弹性专家并行

### 文件位置

`vllm/distributed/elastic_ep/` 目录

在推理过程中动态调整专家的分布：

- 监测各 GPU 的负载
- 将过载 GPU 上的专家迁移到空闲 GPU
- 更新路由表以反映专家位置变化

## 分布式事件

### 文件位置

`vllm/distributed/kv_events.py` —— 约 17KB

KV Cache 的分配和释放会产生事件，通过分布式协调器广播到其他节点，维护全局一致的前缀缓存状态。

## 无状态协调器

### 文件位置

`vllm/distributed/stateless_coordinator.py` —— 约 27KB

轻量级分布式协调组件：

- 节点发现：新节点加入时自动注册
- 状态同步：定期心跳检测
- 容错：节点失联时重新分配负载

## Ray 集成

`vllm/ray/` 目录提供 Ray 框架的集成：

- 基于 Ray 的分布式任务调度
- 多节点集群管理
- 动态资源分配
