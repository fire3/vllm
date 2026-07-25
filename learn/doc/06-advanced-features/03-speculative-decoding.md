# 投机解码

## 文件位置

`vllm/v1/spec_decode/` 目录

## 核心思想

用小模型（Draft Model）快速生成候选 token，大模型（Target Model）并行验证。由于验证是并行的，Decode 速度可提升 2-3x：

```
常规解码（每步 1 个 token）:
    Step:  D1 → D2 → D3 → D4 → D5
    延迟:  t    t    t    t    t

投机解码（每步 N 个候选）:
    Step:  [D1, D2, D3] → [D4, D5, D6]
             (并行验证)      (并行验证)
    延迟:     t (3 tok)      t (3 tok)
```

## 策略类实现

`SpecDecodeBaseProposer` 是所有投机策略的基类：

| 策略 | 文件 | 说明 |
|------|------|------|
| **Eagle** | `eagle.py` | 使用自回归草案模型 |
| **Medusa** | `medusa.py` | 多头并行预测 |
| **MLP Speculator** | `models/mlp_speculator.py` | 轻量 MLP 草案模型 |
| **N-gram** | `ngram_proposer.py` | 基于检索的 n-gram 匹配 |
| **D-Flash** | `dflash.py` | 动态投机解码 |

## 拒绝采样保证

`v1/sample/rejection_sampler.py` 实现拒绝采样，保证输出分布与直接采样一致：

- Draft 模型分布 `q(x)` 预测候选
- Target 模型分布 `p(x)` 验证
- 接受概率 = `min(1, p(x)/q(x))`
- 拒绝后从 `max(0, p-q)` 重新采样
