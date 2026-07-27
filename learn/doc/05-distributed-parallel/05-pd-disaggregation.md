# PD 分离部署

> PD 分离（Prefill-Decouple Disaggregation）将推理的 Prefill 和 Decode 阶段部署到不同的 GPU 集群上，通过跨节点 KV Cache 传输打破 Prefill 与 Decode 的资源争抢，是 vLLM 面向大规模生产部署的核心能力。

## 概述

### 问题背景

在传统的单节点推理中，Prefill 和 Decode 共享同一组 GPU：

| 阶段 | 计算特点 | 资源需求 | 耗时 |
|------|---------|---------|------|
| Prefill | 计算密集型（大量矩阵乘法） | GPU Compute | 与 prompt 长度成比例 |
| Decode | 访存密集型（读取 KV Cache） | GPU Memory Bandwidth + KV Cache 容量 | 逐个 token 生成 |

当两者在同一 GPU 上混合执行时：
- **资源抢占**：长 prompt 的 prefill 会阻塞 decode 流，增加首 token 延迟（TTFT）
- **内存竞争**：大量 KV Cache 占用显存，限制了并发 decode 的 batch size
- **利用率不均匀**：prefill 阶段 GPU compute 满载但 decode 时利用率低，反之亦然

### 核心思想

PD 分离的核心是将 Prefill 和 Decode 拆分到不同的 GPU 实例上运行：

```
传统模式：
  GPU 0: [Prefill A] [Decode A] [Prefill B] [Decode B] [Decode A] ...
  资源争抢，利用率不均

PD 分离：
  Prefill GPU 集群: [Prefill A] [Prefill B] [Prefill C] ...
                                       │ KV Cache 传输
                                       ▼
  Decode GPU 集群:                     [Decode A] [Decode B] [Decode C] ...
  专注 decode，可容纳更大的 batch
```

### 优势

| 指标 | 传统模式 | PD 分离 |
|------|---------|---------|
| TTFT（首 token 延迟） | 受 decode batch 影响 | 仅受 prefill 集群负载影响 |
| TPOT（每 token 延迟） | 受 prefill 计算影响 | 仅受 decode 集群负载影响 |
| Decode Batch 大小 | 受 prefill 的 KV Cache 挤压 | 独立扩容，可支持更大并发 |
| 资源利用率 | Prefill/Decode 互相拖累 | 各自优化，可独立扩缩容 |
| 针对性优化 | 难以单独优化某阶段 | Prefill: 高计算密度；Decode: 低延迟访存 |

## 架构设计

### 核心抽象：三层传输栈

```
┌──────────────────────────────────────────────────────────┐
│                    KV Connector                          │
│  连接 vLLM 调度器/Worker，管理请求级别的 KV 传输生命周期   │
├──────────────────────────────────────────────────────────┤
│                    KV Lookup Buffer                      │
│   以 token 为 key 的查找缓冲区，支持乱序 KV 消费          │
├──────────────────────────────────────────────────────────┤
│                    KV Pipe                               │
│   FIFO 管道，在分布式节点间传输 torch.Tensor              │
└──────────────────────────────────────────────────────────┘
```

#### KV Pipe（可绕过层）

最底层的张量传输管道，提供 `send_tensor()` / `recv_tensor()` 接口。

```python
# 抽象接口
class KVPipe:
    def send_tensor(self, tensor: torch.Tensor, target_rank: int) -> None: ...
    def recv_tensor(self, source_rank: int) -> torch.Tensor: ...
```

- 默认基于 NCCL P2P 或自定义 RDMA 传输
- **可绕过**：如果上层通信服务（如 RDMA 数据库、Redis）已支持键值查找，可以直接跳过这一层

#### KV Lookup Buffer（可绕过层）

以 token ID 块为 key 的查找缓冲区，解决 FIFO 管道的乱序消费问题。

```
场景：高 QPS 下请求处理顺序不一致
  Prefill 处理顺序: A → B → C
  Decode 处理顺序:  C → A → B  (因网络延迟或负载不均衡)

  FIFO 方案: 需要等待 A、B 到来后才能消费 C → 延迟
  Lookup Buffer 方案: 以 token hash 为 key 直接查找 C 的 KV → 零等待
```

```python
# 抽象接口
class KVLookupBuffer:
    def insert(self, key: str, kv_cache: torch.Tensor) -> None: ...
    def drop_select(self, key: str) -> torch.Tensor | None: ...
    # 语义类似 SQL: INSERT INTO buffer VALUES (...) / SELECT ... WHERE key=...
```

#### KV Connector（核心层）

连接 vLLM 内部的调度器和模型执行器，分为两个角色：

| 角色 | 进程 | 职责 |
|------|------|------|
| `KVConnectorRole.SCHEDULER` | 调度器进程 | 绑定请求元数据，决定 KV 加载策略 |
| `KVConnectorRole.WORKER` | Worker 进程 | 实际执行 KV Cache 的加载和存储 |

### 关键配置

```python
# vllm/config/kv_transfer.py
@config
class KVTransferConfig:
    kv_connector: str | None = None          # 连接的名称（如 "NixlConnector"）
    kv_role: KVRole | None = None            # kv_producer / kv_consumer / kv_both
    kv_rank: int | None = None               # 0=prefill, 1=decode
    kv_parallel_size: int = 1                # 并行实例数
    kv_ip: str = "127.0.0.1"                 # 连接 IP
    kv_port: int = 14579                     # 连接端口
    kv_buffer_device: str = "cuda"           # 缓冲区设备（cuda/cpu/xpu）
    kv_buffer_size: float = 1e9              # 缓冲区大小（字节）
    kv_load_failure_policy: str = "fail"     # 加载失败策略（fail/recompute）
```

`kv_role` 定义了实例在 PD 拓扑中的角色：

```python
KVProducer = Literal["kv_producer", "kv_both"]    # Prefill 节点
KVConsumer = Literal["kv_consumer", "kv_both"]    # Decode 节点
```

## 调度器与 Connector 的协作

调度器端和 Worker 端各有一个独立的 `KVConnector` 实例，角色不同：

### 数据流全景

```
Scheduler（调度器进程）
  │
  ├─ 1. get_computed_blocks_for_connector()  本地前缀缓存命中查询
  ├─ 2. get_num_new_matched_tokens()         询问 Connector 远端匹配的 token 数
  ├─ 3. allocate_slots(external_tokens, delay_cache_blocks)  分配 slots
  ├─ 4. build_connector_meta()               生成 KVConnectorMetadata
  │     └─ 放入 SchedulerOutput.kv_connector_metadata
  │
  ▼
GPU Model Runner（Worker 进程）
  │
  ├─ 5. bind_connector_metadata()            应用调度计划
  ├─ 6. start_load_kv()                      异步启动远端 KV 加载
  ├─ 7. 模型前向执行（attention 层内调用 wait_for_layer_load / save_kv_layer）
  ├─ 8. wait_for_save()                      等待 KV 保存完成
  ├─ 9. get_finished()                       获取传输完成的请求 ID
  │     └─ 封装为 KVConnectorOutput
  │
  ▼
Scheduler.update_from_output()
  │
  ├─ 10. _handle_invalid_blocks()            处理加载失败的 block
  ├─ 11. _update_from_kv_xfer_finished()     处理传输完成
  │      ├─ finished_recving → 请求状态从 WAITING_FOR_REMOTE_KV 恢复为 WAITING
  │      └─ finished_sending → 释放 blocks
  └─ 12. 收集 stats、events
```

### 调度决策流程

```python
# vllm/v1/core/sched/scheduler.py（简化流程）

def schedule_prefills_and_decode_reqs(self):
    for request in new_requests:
        # 1. 本地前缀缓存查找
        local_tokens = kv_cache_manager.get_computed_blocks_for_connector(request)

        # 2. 远端 KV 匹配查询
        ext_tokens, load_async = self.connector.get_num_new_matched_tokens(
            request, len(local_tokens))

        if ext_tokens is None:
            # Connector 尚未准备好 → 放回等待队列，下次再试
            waiting_queue.append(request)
            continue

        # 3. 总计算 token 数 = 本地 + 远端
        num_computed_tokens = local_tokens + (ext_tokens or 0)

        # 4. 分配 slots，标记 external tokens 区域
        if load_async:
            allocate_slots(delay_cache_blocks=True)  # 异步加载，预分配但不缓存
        else:
            allocate_slots()  # 同步加载，直接填入 KV
```

### 请求状态管理

PD 分离引入了一个新的请求状态 `WAITING_FOR_REMOTE_KV`：

```
WAITING ──→ SCHEDULED ──→ PREFILL ──→ DECODE ──→ FINISHED
  │                                              ↑
  └──→ 远端 KV 未就绪                                   │
       WAITING_FOR_REMOTE_KV ────────────→ cache_blocks()
       （异步传输完成后恢复）                       
```

- `finished_recving` 标记异步 KV 接收完成的请求 ID
- 完成后调用 `_update_waiting_for_remote_kv()` 将 blocks 加入缓存，状态恢复为 `WAITING`
- `invalid_block_ids` 机制提供容错：加载失败的 block 触发生效 token 数回退和重算

## Worker 端 KV 传输生命周期

Worker 端的 KV Connector 通过混入类（mixin）集成到模型执行流程中：

```python
# vllm/v1/worker/kv_connector_model_runner_mixin.py

def _get_kv_connector_output(self, scheduler_output):
    # 1. 应用调度器元数据
    kv_connector.bind_connector_metadata(
        scheduler_output.kv_connector_metadata)

    # 2. 异步启动远端 KV 加载
    kv_connector.start_load_kv(get_forward_context())

    try:
        yield  # → 进入模型前向执行
    finally:
        # 3. 等待保存完成
        kv_connector.wait_for_save()

        # 4. 获取传输完成信号
        finished_sending, finished_recving = \
            kv_connector.get_finished(finished_req_ids)

        # 5. 收集加载失败的 blocks
        invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

        # 6. 收集统计和事件
        ...
```

在 Attention 层内部，逐层操作实现流水线化：

```python
def forward(self, layer_name, kv_cache, attn_metadata):
    # 加载：等待当前层的 KV 加载到显存
    self.kv_connector.wait_for_layer_load(layer_name)

    # 执行注意力计算
    output = attention_forward(kv_cache, ...)

    # 保存：将当前层 KV 异步传输出去
    self.kv_connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
```

这种逐层（layer-by-layer）的设计使得 KV 传输可以与注意力计算流水线化，隐藏传输延迟。

## 传输模式

### Pull 模式（拉模式）

Decode 端主动从 Prefill 端拉取 KV Cache。

```
Prefill (producer)           Decode (consumer)
     │                            │
     │     (1) 生成 KV Cache       │
     │     (2) 注册 PUSH_REG       │
     │◄───────────────────────────│  (3) 发送 PUSH_REG 通知
     │     (4) WRITE transfer     │
     │───────────────────────────►│  (4) 数据直接从 Prefill GPU → Decode GPU
     │                            │
```

- **适用场景**：Decode 端主动调度，Prefill 端被动响应
- **典型实现**：`NixlPullConnector`（默认）、`LMCacheConnectorV1`（consumer 角色）

### Push 模式（推模式）

Prefill 端主动将 KV Cache 推送到 Decode 端。

```
Prefill (producer)           Decode (consumer)
     │                            │
     │     (1) 生成 KV Cache       │
     │     (2) 等待 D 的注册       │
     │◄───────────────────────────│  (3) PUSH_REG 注册
     │     (4) WRITE transfer     │
     │───────────────────────────►│  (4) Prefill 主动推送
     │                            │
```

- **适用场景**：Decode 端无调度器/地址不公开的场景
- **典型实现**：`NixlPushConnector`
- **技术细节**：Push 模式运行一个专用的 `nixl-push-writer` 线程，通过事件驱动模型管理未匹配的推送块

### Connector 注册表

vLLM 通过工厂模式管理多种 Connector 实现：

```python
# factory.py 中的注册
KVConnectorFactory.register_connector("NixlConnector", ...)
KVConnectorFactory.register_connector("NixlPullConnector", ...)
KVConnectorFactory.register_connector("NixlPushConnector", ...)
KVConnectorFactory.register_connector("LMCacheConnectorV1", ...)
KVConnectorFactory.register_connector("MooncakeConnector", ...)
KVConnectorFactory.register_connector("MoRIIOConnector", ...)
KVConnectorFactory.register_connector("MultiConnector", ...)
...
```

通过 `--kv-transfer-config '{"kv_connector":"...", "kv_role":"..."}'` 选择。

## Connector 实现

### NIXL Connector（推荐生产方案）

NIXL（NVIDIA 互联库）是 vLLM 最成熟且功能最完整的 Connector 实现。

**核心特性**：
- 基于 NVIDIA NIXL 的 GPU 直接通信（避免 CPU 中转）
- 支持 Pull（拉）和 Push（推）两种传输模式
- 支持 TP 映射：当 Prefill 和 Decode 的 TP size 不一致时自动调整数据传输映射
- 支持 PP 分片感知：PP 各阶段的 Connector 只传输该阶段所属层的 KV
- 跨层 KV Cache 支持：统一管理所有 attention 层的 KV

**TP 映射机制**：

```python
# TransferTopology.tp_ratio() 处理 TP size 不匹配
# 例：Prefill TP=2, Decode TP=4
# tp_ratio = -(4/2) = -2
# 每个 Prefill worker 向 2 个 Decode worker 发送数据
```

**Block Size 适配**：
- Prefill 通常使用较小的 block size（如 16 tokens/block）
- Decode 使用较大的 block size（如 64 tokens/block）
- 接收端通过 `kv_postprocess_blksize_on_receive()` 自动转换布局

### LMCache Connector

使用 [LMCache](https://github.com/LMCache/LMCache) 作为 KV 缓存后端，支持多级缓存（GPU → CPU → Disk）。

```
Prefill  ──►  LMCache ──► Decode
                   │
              CPU / Disk
            （冷数据卸载）
```

- 支持 NIXL 作为底层传输（配置 `enable_nixl: True`）
- 多进程模式：`LMCacheMPConnector` 在独立进程中管理缓存
- 适用于需要 KV 缓存持久化和跨节点共享的场景

### Mooncake Connector

基于 [Mooncake](https://github.com/kvcache-ai/Mooncake) 的 KV 缓存传输方案，专为多节点集群设计。

- 支持 RDMA 高速传输
- 内置分布式存储层（Mooncake Store）
- 适合大规模集群部署（跨机架、跨交换机）

## 部署拓扑

### 1P1D（最简拓扑）

最简单的 PD 分离：1 个 Prefill 实例 + 1 个 Decode 实例。

```bash
# Prefill 节点
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8100 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}'

# Decode 节点
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8200 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}'
```

### XpYd（多实例拓扑）

通过 Proxy 层实现多个 Prefill 和 Decode 实例的负载均衡。

```
                   ┌──────────────────┐
                   │   Proxy Server   │
                   │ (Round-Robin)    │
                   └────────┬─────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ Prefill 0│    │ Prefill 1│    │ Prefill N│
     └────┬─────┘    └────┬─────┘    └────┬─────┘
          │ KV Cache      │ KV Cache      │ KV Cache
          ├───────────────┼───────────────┘
          │               │
     ┌────▼─────┐    ┌────▼─────┐
     │ Decode 0 │    │ Decode 1 │
     └──────────┘    └──────────┘
```

Proxy 的请求路由逻辑：

```python
# 1. Prefill 阶段：将请求发送到某个 Prefill 实例，max_tokens=1
#    只做 prefill，生成 KV Cache
prefill_request = request.copy()
prefill_request["max_tokens"] = 1
await forward(prefill_instance, prefill_request)

# 2. Decode 阶段：将完整请求发送到某个 Decode 实例
#    Decode 实例通过 KV Connector 获取 Prefill 阶段生成的 KV Cache
response = await forward(decode_instance, request)
```

参考实现：`examples/disaggregated/disaggregated_serving/disagg_proxy_demo.py`

### 编码器分离（Encoder PD）

对于 encoder-decoder 模型（如 T5、Whisper），支持将 Encoder 计算也进行分离：

```
Encoder ──► EC Transfer ──► Prefill ──► KV Transfer ──► Decode
```

实现文件：`vllm/distributed/ec_transfer/`

## 容错与失败处理

### KV 加载失败

```python
# KVTransferConfig
kv_load_failure_policy: "recompute" | "fail"
```

| 策略 | 行为 |
|------|------|
| `fail` | 立即失败请求，返回错误 finish reason |
| `recompute` | 将请求重新调度，从远端 KV 失败的 blocks 开始自行计算 |

**实现细节**：
1. Worker 端记录 `invalid_block_ids`（加载失败的 block ID）
2. 上传到 Scheduler 的 `KVConnectorOutput`
3. Scheduler 调用 `_handle_invalid_blocks()` 重新计算有效 token 数
4. 请求回到调度队列，缺失部分通过本地 prefill 补充

### 请求完成处理

```python
# 请求完成时
def request_finished(request, block_ids):
    # 返回 (延迟释放blocks, KV传输参数)
    return should_delay_free, kv_transfer_params

# 延迟释放：Connector 异步发送完成后，通过 get_finished() 通知
# Scheduler 收到 finished_sending 信号后才释放 blocks
```

### Connector 生命周期管理

```python
def shutdown(self):
    """关闭时确保所有异步操作完成"""
    ...

def get_finished(self, finished_req_ids):
    """返回已完成异步发送/接收的请求 ID"""
    return finished_sending_ids, finished_recving_ids
```

## 配置参考

### 启动参数

| 参数 | 格式 | 说明 |
|------|------|------|
| `--kv-transfer-config` | JSON 字符串 | KV 传输配置，包含所有 KVTransferConfig 字段 |
| `--disable-hybrid-kv-cache-manager` | 标志 | 若 Connector 不支持 HMA 时需要设置 |

### 环境变量

| 变量 | 说明 |
|------|------|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | V1 多进程模式（PD 分离需要设为 1） |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker 进程启动方式（推荐 `spawn`） |

### Connector 选择指南

| Connector | 传输方式 | 适用场景 |
|-----------|---------|---------|
| `NixlConnector` / `NixlPullConnector` | GPU 直接 RDMA（Pull） | 同集群内 P/D 分离，推荐生产 |
| `NixlPushConnector` | GPU 直接 RDMA（Push） | Decode 无公开地址的网络拓扑 |
| `LMCacheConnectorV1` | GPU→CPU→Disk 多级缓存 | 需要 KV 持久化、缓存共享 |
| `LMCacheMPConnector` | 多进程 LMCache | 高并发、需要独立缓存进程 |
| `MooncakeConnector` | RDMA + 分布式存储 | 大规模跨节点集群 |
| `MoRIIOConnector` | 自定义 RDMA IO | 特定硬件优化场景 |
| `FlexKVConnectorV1` | 灵活配置的 KV 缓存 | 实验/测试场景 |
| `OffloadingConnector` | GPU→CPU Offload | 显存不足时的 KV 卸载 |
| `ExampleConnector` | 本地文件系统 | 学习调试（不用于生产） |
| `MultiConnector` | 多 Connector 组合 | 同时使用多种传输策略 |

## 关键文件索引

| 文件 | 大小 | 说明 |
|------|------|------|
| `vllm/config/kv_transfer.py` | 3KB | KV 传输配置类 |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 9KB | Connector 工厂与注册 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 25KB | KVConnectorBase 抽象类 |
| `vllm/distributed/kv_transfer/kv_connector/utils.py` | 24KB | 传输工具（拓扑、块复制、布局转换） |
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` | — | NIXL Connector 实现（75K+ 代码） |
| `vllm/v1/core/sched/scheduler.py` | — | 调度器中的 Connector 集成 |
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | 13KB | Worker 端 Connector 运行时的生命周期 |
| `vllm/v1/engine/core.py` | — | Engine Core 中的初始化与协调 |
| `vllm/v1/outputs.py` | — | KVConnectorOutput 数据类 |
| `vllm/v1/core/kv_cache_manager.py` | — | 支持 external_tokens 的缓存管理器 |
| `examples/disaggregated/` | — | 部署示例脚本 |

## 思考题

1. PD 分离中，`get_num_new_matched_tokens()` 返回 `None` 时调度器做了什么？为什么要这样设计？
2. `delay_cache_blocks=True` 和常规的 block 分配有什么区别？异步传输完成后如何将 blocks 加入缓存？
3. 当 Prefill 的 TP size = 4，Decode 的 TP size = 2 时，NIXL Connector 如何做 KV 数据映射？
4. Push 模式和 Pull 模式在处理请求完成时的生命周期有何不同？
5. `MultiConnector` 如何同时使用多个 KV 传输后端？其 `SupportsHMA` 检测逻辑是怎样的？
6. KV 加载失败的 `recompute` 策略是如何实现的？是否会引入额外的延迟？
