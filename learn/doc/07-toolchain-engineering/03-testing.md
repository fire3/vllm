# 测试体系

## 文件位置

`tests/` 目录（41 个子目录）

## 测试分层

| 层级 | 目录 | 说明 |
|------|------|------|
| **单元测试** | `tests/kernels/` | 单个 CUDA kernel 的正确性 |
| **集成测试** | `tests/v1/`、`tests/lora/` 等 | 模块间的交互 |
| **端到端测试** | `tests/models/` | 完整模型推理的正确性 |
| **评估测试** | `tests/evals/` | 模型输出质量评估 |

## 运行方法

```bash
# 运行单文件测试
.venv/bin/python -m pytest tests/v1/core/test_block_pool.py -v

# 运行所有测试
.venv/bin/python -m pytest tests/ -v

# 安装测试依赖
uv pip install -r requirements/test/cuda.in
```

## 模型测试

模型正确性测试通过对比 vLLM 输出与 HuggingFace 参考实现来验证：

```python
def test_llama_output():
    # 加载 vLLM 模型
    vllm_output = vllm_llama.generate(prompt)
    # 加载 HuggingFace 参考
    hf_output = hf_llama.generate(prompt)
    # 断言输出一致
    assert vllm_output == hf_output
```
