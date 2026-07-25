# 量化

## 文件位置

`vllm/model_executor/layers/quantization/` 目录

## 架构设计

量化系统采用**策略模式**：

```
QuantizeMethodBase (abstract)       — 每个量化层的计算接口
    ├── create_weights()            — 创建量化的权重参数
    ├── apply()                     — 前向传播（含量化计算）
    └── process_weights_after_loading() — 后处理

QuantizationConfig (abstract)       — 配置解析与工厂
    ├── get_name()                  — 量化方案名
    ├── get_min_capability()        — 最低 GPU 算力要求
    └── get_quant_method()          — 返回对应的 QuantizeMethodBase
```

## 支持的量化方案

| 方案 | 精度 | 文件 | 特点 |
|------|------|------|------|
| **AWQ** | W4A16 | `auto_awq.py` (37KB) | Activation-aware 权重量化，按通道缩放 |
| **GPTQ** | W4A16 | `auto_gptq.py` (31KB) | Hessian 矩阵优化的量化 |
| **FP8** | FP8 | `fp8.py` (34KB) | 原生 FP8 推理（H100 支持） |
| **BitsAndBytes** | INT8/INT4 | `bitsandbytes.py` (20KB) | 基于 bitsandbytes 库 |
| **ModelOpt** | 多种 | `modelopt.py` (98KB) | TensorRT 模型优化器 |

## 与线性层的集成

量化通过 `LinearMethodBase` 注入到 `ColumnParallelLinear` / `RowParallelLinear` 等线性层：

```python
# 线性层创建时指定量化方法
linear = ColumnParallelLinear(
    in_features, out_features,
    quant_method=FP8LinearMethod(...)
)
```

前向传播时，量化方法负责反量化 + 矩阵乘法融合：

```python
def apply(self, layer, x):
    # 对权重反量化并计算
    return F.linear(x, self.dequantize(layer.weight))
```
