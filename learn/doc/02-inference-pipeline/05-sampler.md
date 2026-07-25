# 采样器

> 采样器将模型输出的 logits 转换为最终的 token ID。vLLM 的采样器是一个流水线式的处理链，支持丰富的采样策略和 logits 后处理。

---

## 1. Sampler 架构

### 文件位置

`vllm/v1/sample/sampler.py` —— 约 18KB

### 核心类

```python
class Sampler(nn.Module):
    def __init__(self, logprobs_mode="raw_logprobs",
                 use_fp64_gumbel=False):
        self.top_k_top_p_sampler = TopKTopPSampler()
```

`Sampler` 是一个 `nn.Module`，内部持有一个 `TopKTopPSampler` 子模块。

### 完整处理管线（`forward` 方法）

```python
def forward(self, logits, sampling_metadata,
            predict_bonus_token=False, logprobs_mode_override=None):
```

管线包含 9 个步骤：

```
logits
    │
    ├── 1. 提取 logprobs（如果请求了 logprobs）
    ├── 2. 转换为 float32（如果原始 dtype 不是 fp32）
    ├── 3. 应用 whitelist（仅允许某些 token ID）
    ├── 4. 添加 bad words（禁止某些 token ID）
    ├── 5. 应用非-argmax-invariant 处理器（排斥惩罚等）
    ├── 6. 应用 penalties（重复惩罚、频率惩罚、存在惩罚）
    ├── 7. 应用 argmax-invariant 处理器（温度缩放等）
    ├── 8. 执行 top-k / top-p 采样
    └── 9. 收集 logprobs
```

---

## 2. 核心采样方法（`sample`）

```python
def sample(self, logits, sampling_metadata):
```

采样流程：

```
sample()
    │
    ├── 判断：是否所有请求都使用 greedy？
    │   ├── 是 → greedy_sample(logits)
    │   └── 否 ↓
    │
    ├── apply_temperature(logits, temp, all_random)
    │   # 温度缩放，greedy 请求的温度设为 1（跳过）
    │
    ├── 应用 argmax-invariant logits processors
    │   # 拓扑结构不变的处理器（如 top-k 截断）
    │
    ├── TopKTopPSampler(top_k, top_p, random_generators)
    │   # 标准 top-k + top-p 采样
    │
    └── temperature gate: 选择 greedy 或 random 路径
        # 每个请求独立选择
```

### Greedy Sample

```python
def greedy_sample(self, logits):
    return logits.argmax(dim=-1)
```

### Temperature 处理

```python
def apply_temperature(self, logits, temp, all_random):
    # 对 logits 进行逐行除法
    # temperature = 0 → greedy（跳过）
    # temperature = 1 → 原始分布
    # temperature > 1 → 更均匀分布
```

---

## 3. Logits 处理器

### 文件位置

`vllm/v1/sample/logits_processor/` 目录

Logits 处理器分为两类：

1. **Non-argmax-invariant**：会改变 argmax 结果
   - 排斥惩罚（repetition penalty）
   - 频率惩罚（frequency penalty）
   - 存在惩罚（presence penalty）

2. **Argmax-invariant**：不改变 argmax 结果
   - Temperature 缩放
   - Top-k 截断（只保留 top-k 的 logits）
   - Top-p 截断（累积概率达到阈值）

这种分类是为了与 **CUDA Graph** 兼容：argmax-invariant 的处理器可以在 Graph 中捕获。

### 应用顺序

```
1. Bad words mask           [非 argmax-invariant]
2. Repetition penalty       [非 argmax-invariant]
3. Frequency penalty        [非 argmax-invariant]
4. Presence penalty         [非 argmax-invariant]
5. Temperature              [argmax-invariant]
6. Top-k / Top-p mask       [argmax-invariant]
```

---

## 4. 拒绝采样器

### 文件位置

`vllm/v1/sample/rejection_sampler.py` —— 约 36KB

### 用途

拒绝采样器用于**投机解码**（Speculative Decoding）。当 draft model 生成候选 token 后，target model 并行计算所有候选 logits，通过拒绝采样决定接受哪些、拒绝哪些。

### 数学原理

给定 target model 的分布 `p(x)` 和 draft model 的分布 `q(x)`：

1. 从 `q(x)` 采样 `x`
2. 以概率 `min(1, p(x) / q(x))` 接受 `x`
3. 如果拒绝，从 `max(0, p(x) - q(x))` 归一化的分布中重新采样

这个过程的输出分布**等于直接采样 `p(x)`**，保证了投机解码的数学正确性。

---

## 5. SamplingParams

### 文件位置

`vllm/sampling_params.py` —— 约 47KB

`SamplingParams` 定义了用户可配置的全部采样参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | `float` | `1.0` | 采样温度 |
| `top_k` | `int` | `-1` | top-k 截断（-1 表示禁用） |
| `top_p` | `float` | `1.0` | top-p 截断（1.0 表示禁用） |
| `min_p` | `float` | `0.0` | min-p 采样 |
| `repetition_penalty` | `float` | `1.0` | 重复惩罚系数 |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚 |
| `presence_penalty` | `float` | `0.0` | 存在惩罚 |
| `max_tokens` | `int` | `16` | 最大生成 token 数 |
| `stop_token_ids` | `list[int]` | `[]` | 停止 token ID |
| `logprobs` | `int` | `0` | 返回的 logprobs 数 |

---

## 6. Logits 与 Logprobs

### 文件位置

`vllm/logprobs.py` —— 约 8KB

`compute_logprobs(logits)` 计算 log_softmax：

```python
def compute_logprobs(self, logits):
    # 在 float32 精度下计算 log_softmax
    return torch.log_softmax(logits.float(), dim=-1)
```

`gather_logprobs(logprobs, num_logprobs, token_ids)` 收集 top-k 的对数概率：

```python
def gather_logprobs(self, logprobs, num_logprobs, token_ids):
    # 对每个请求、每个输出 token，收集 top-k logprobs
    # 同时计算 token 在 top-k 中的排名
```

---

## 7. 思考预算（Thinking Budget）

### 文件位置

`vllm/v1/sample/thinking_budget_state.py` —— 约 25KB

用于推理模型（如 DeepSeek-R1）的思考 token 预算控制：

- 限制模型在 "thinking" 阶段的 token 数
- 达到预算上限时强制切换到生成阶段
- 控制 思考 与 回答 之间的切换

---

> **代码参考**：
> - `vllm/v1/sample/sampler.py` — 采样器核心
> - `vllm/v1/sample/rejection_sampler.py` — 拒绝采样器
> - `vllm/v1/sample/logits_processor/` — Logits 处理器
> - `vllm/sampling_params.py` — 采样参数
> - `vllm/v1/sample/thinking_budget_state.py` — 思考预算
