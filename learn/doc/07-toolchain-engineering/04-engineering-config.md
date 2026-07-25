# 工程配置

## CI/CD

`.buildkite/` 和 `.github/` 目录定义 CI 流水线：

```
Pipeline 阶段:
    Lint (Ruff, MyPy)
      → Build (setup.py + CMake)
        → Test (pytest)
          → Benchmark (benchmarks/)
            → Package (Docker / wheel)
```

## 代码质量

| 工具 | 用途 | 配置位置 |
|------|------|---------|
| **Ruff** | Python linter + formatter | `.pre-commit-config.yaml` |
| **MyPy** | 类型检查 | `.pre-commit-config.yaml` |
| **Pre-commit** | 提交前自动检查 | `.pre-commit-config.yaml` |

运行方式：

```bash
# 安装 lint 依赖
uv pip install -r requirements/lint.txt
pre-commit install

# 检查所有文件
pre-commit run --all-files
```

## Docker

`docker/` 目录提供 Dockerfile，支持：

- CUDA 12.x 基础镜像
- 预编译 wheel 加速安装
- 多阶段构建减小镜像体积

## 版本发布

`RELEASE.md` 定义版本策略：

- 语义化版本（SemVer）
- 每月或按需发布
- RC 版用于社区测试
