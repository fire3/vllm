# 协调机制

## 初始化流程

分布式环境的初始化由 `parallel_state.py` 中的 `initialize_model_parallel()` 函数完成：

```python
def initialize_model_parallel(
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    ...
):
    # 1. 划分 TP 组
    # 2. 在 TP 组内划分 PP 组
    # 3. 可选：划分 EP 组
    # 4. 初始化各组的 GroupCoordinator
    # 5. 注册全局组名
```

进程 rank 分配示例（TP=2, PP=2, 共 4 GPU）：

```
rank 0: TP group 0, PP group 0  (前 6 层)
rank 1: TP group 1, PP group 0  (前 6 层)
rank 2: TP group 0, PP group 1  (后 6 层)
rank 3: TP group 1, PP group 1  (后 6 层)
```

## 组协调器（GroupCoordinator）

`GroupCoordinator` 是每个并行组的通信入口：

```python
group = get_tp_group()
group.all_reduce(tensor)   # 在 TP 组内执行 all-reduce

group = get_pp_group()
group.send(tensor, next_rank)  # 发送到下一 PP 阶段
```
