# DeepSeek V4 SM89 (Ada) 移植说明

本分支（`v0.27.0-dsv4-sm89`）在 vLLM `v0.27.0` 基础上携带 DeepSeek V4
(Flash) 稀疏 MLA 在 NVIDIA Ada 架构（SM89，如 L40S）上的完整移植，并与
配套的 FlashInfer 分支（`v0.6.16.post3-dev-sm89-dsv4`）一一对应。开发过程
中的调试/排查类文档（`DSV4_SM89_*.md`）已从分支中移除，相关中间记录仍
保留在 git 历史中，需要时可从对应提交找回。

## 1. 背景与动机

上游 vLLM 的 DeepSeek V4 稀疏 MLA 路径依赖 DeepGEMM、FlashMLA 与
CuTe-DSL，只覆盖 SM90/SM100/SM120，在 Ada（SM89）上完全不满足支持条件：

| 依赖 | 上游支持范围 | SM89 上的情况 |
| --- | --- | --- |
| DeepGEMM | SM90/SM100/SM120 | 不编译、不可用 |
| FlashMLA C++ 扩展 | SM100/SM120 | 不可用 |
| CuTe-DSL (`cutlass`) | SM90+（含 Ada 之外的平台） | Ada 上不兼容，需禁用 |
| FlashInfer sparse MLA | SM120（vLLM 侧） | 需配套分支提供 Ada 内核 |

目标是让 SM89 设备（如 L40S）以 `FLASHINFER_MLA_SPARSE_DSV4` 后端运行
DeepSeek V4 Flash，使用 FP8 KV cache（`fp8_ds_mla` 布局），支持稀疏 MLA
的 prefill + decode、CUDA graph 与 DSpark 投机解码。

## 2. 移植思路

### 2.1 内核层：FlashInfer 提供 Ada sparse MLA decode

稀疏 MLA 的 decode 内核由配套 FlashInfer 分支实现（`flashinfer/mla` 的
`_resolve_dsv4_sparse_mla_backend` 在 SM89 返回 `"sparse"`）。vLLM 侧只做
能力探测与路由：

- `has_flashinfer_sparse_mla_sm89()`：确认当前 FlashInfer 构建确实在 SM89
  上启用了 sparse MLA decode（`vllm/utils/flashinfer.py`）。
- `FLASHINFER_MLA_SPARSE_DSV4` 后端在 SM89 上沿用 SM120 的
  `DeepseekV4FlashInferSM120Attention`（类名保留，实际按设备能力分支），
  复用 `fp8_ds_mla` 的 KV cache 布局与 `get_kv_cache_shape`。

### 2.2 索引器（indexer）：Triton FP8 MQA logits 回退

稀疏注意力需要先用 indexer 为每个 token 挑 top-k 候选。上游用 DeepGEMM 的
`fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`，SM89 上没有，因此在
`vllm/v1/attention/ops/triton_fp8_mqa_logits.py` 中提供两个 Triton 内核：

- `fp8_mqa_logits_triton`：prefill 稠密 logits（`[M, N]` fp32）。
- `fp8_paged_mqa_logits_triton`：decode 分页 KV 的 logits。

后者直接以平铺字节布局读取 indexer K 缓存（每 block 连续存放
`[block_size*head_dim]` FP8 数据 + `[block_size*4]` fp32 scale），用
`as_strided` 复用原始字节，避免每 decode 步对整层索引 KV 做
`.contiguous()` 拷贝（262k 模型长度下单层可达数十 MB，纯固定开销）。

为控制长上下文下的固定成本，还做了三项语义保持的优化：

- 按调度器给出的实际最大上下文长度裁剪稠密 logits 宽度与 launch grid；
- `VLLM_SM89_INDEXER_BLOCK_COL` 环境变量调节分页内核的列块大小
  （64/128/256，默认 64）；
- prefill 稠密路径去掉 `-inf` 预填充（`top_k_per_row_prefill` 只消费
  `[cu_starts, cu_ends)` 且内核写满该区间，改为 `torch.empty`）。

### 2.3 o_proj：Triton 分组 BF16 回退

MLA 的 o_proj（逆 RoPE + `wo_a` 低秩 + `wo_b`）上游走 DeepGEMM 的
`fp8_einsum`。SM89 上回退为 Triton 分组 BF16 矩阵乘
（`vllm/models/deepseek_v4/nvidia/ops/o_proj.py`），权重转置缓存到模块上，
不重复计算。

### 2.4 其他依赖回退（SM89 缺件时也能跑）

- `tf32_hc_prenorm_gemm`：DeepGEMM 缺失时提供 torch 回退（匹配输出契约，
  支持 `num_split`），见 `vllm/utils/deep_gemm.py`；
- FlashMLA 缺失时 `get_mla_metadata` 返回空的 `FlashMLASchedMeta` 占位，
  见 `vllm/v1/attention/ops/flashmla.py`；
- `w8a8_triton_block_scaled_mm` 的 E8M0 指数缩放从“仅 ROCm/XPU”放宽到
  全平台（Triton 无法直接绑定 E8M0，先转 fp32），见
  `vllm/model_executor/layers/quantization/utils/fp8_utils.py`；
- `has_cutedsl()` 在 SM89 CUDA 上返回 False，压缩器改走 Triton 路径，
  见 `vllm/utils/import_utils.py` 与 `vllm/models/deepseek_v4/compressor.py`。

### 2.5 索引预算与 DSpark 细节

- **index_topk 归一化**：`DeepSeek-V4-Flash` 配置的 `index_topk=512` 在长
  上下文下会让 Triton 索引器（召回率约 97% 左右）丢掉系统提示中的指令，
  导致畸形工具调用。`normalize_dsv4_sm89_index_topk()` 在模型与 metadata
  builder 初始化时统一归一化到 2048（与 SM120 一致），可用
  `VLLM_DSV4_SM89_INDEX_TOPK=512/1024/2048` 覆盖做 A/B。
- **DSpark 非因果 SWA 索引宽度**：只对齐到内核实际实例化的 topk 值
  {128, 512, 1024}，多余槽位保持 -1，由 `decode_swa_lens` 截断。
- **CUDA graph 重放**：`_prepare_dflash_inputs_kernel` 对 padding 行补齐
  input_ids/positions，并对 bonus_token 做 `>=0` 夹取，避免陈旧/哨兵值
  被草稿 embedding 当查询 id 造成异步 OOB。

### 2.6 无关的通用修复

`vllm/entrypoints/openai/responses/utils.py` 支持 `ResponseCustomToolCall`
转换，避免 DSV4 Flash 工具调用链路崩溃。与 SM89 无直接关系，但属于该
分支发布所需的修复，故保留在分支中。

## 3. 分支对应与依赖

| 仓库 | 分支 / 版本 | 说明 |
| --- | --- | --- |
| vllm | `v0.27.0-dsv4-sm89`（本分支） | 基于 `v0.27.0` |
| flashinfer | `v0.6.16.post3-dev-sm89-dsv4` | 提供 SM89 sparse MLA decode |

FlashInfer 侧的内核实现、构建与验证细节见 [FlashInfer 仓库
`v0.6.16.post3-dev-sm89-dsv4` 分支的 README_SM89.md](https://github.com/fire3/flashinfer/blob/v0.6.16.post3-dev-sm89-dsv4/README_SM89.md)，
本文档与其互为引用。

建议依赖基线（可按部署环境调整）：

- Python 3.12、CUDA 13.0 / PyTorch `2.13.0+cu130`；
- SM89 GPU（如 L40S）；
- 关键依赖：`tokenspeed-mla`、`humming-kernels`、`tilelang`、
  `flashinfer-python`（配套分支）。

开发阶段推荐以 editable 方式安装：vllm 用
`cu130_build_wheel.sh --editable`，flashinfer 用 `pip install -e .`；
Python 改动即时生效，C++/CUDA 改动需重新构建对应扩展。

## 4. 编译与安装

### 4.1 conda 环境构建（推荐）

```bash
conda activate <env>                       # 已配置 CUDA 13.0 + PyTorch cu130 的环境
source scripts/build/cu130_env.sh          # CUDA_HOME、arch 列表等
bash scripts/build/cu130_install_deps.sh   # torch 2.13.0+cu130 + 构建依赖
bash scripts/build/prepare_deps.sh         # 按当前 checkout 的 CMake 依赖清单准备 .deps
bash scripts/build/cu130_build_wheel.sh    # 构建 dist/vllm-*.whl
bash scripts/build/cu130_verify.sh         # 校验嵌入的内核
```

开发期直接 editable 安装（Python 改动即时生效；C++/CUDA 改动需要重新
执行构建脚本）：

```bash
bash scripts/build/cu130_build_wheel.sh --editable
```

要点：

- `TORCH_CUDA_ARCH_LIST` 默认 `8.9+PTX`（sm_89 SASS + compute_89 PTX，
  Ada 及更新架构可 JIT）；只需 Ada 时可改为 `8.9`。
- 构建必须 `--no-build-isolation`，用 conda 环境内的 cu130 torch。
- 本地 wheel 版本由 git tag 推导（setuptools-scm），可用
  `VLLM_VERSION_OVERRIDE` 显式指定，如 `0.27.0+cu130sm89`。
- 换 vLLM 分支/release tag 后要重跑 `prepare_deps.sh`，可用
  `--gpu --list` 查看依赖清单、`--check` 校验现有 `.deps`。

### 4.2 Docker 开发环境（可选）

宿主机不想装 conda/CUDA 时：

```bash
bash scripts/build/docker/setup_host.sh       # 一次性：Docker + GPU 透传
bash scripts/build/docker/build_dev_image.sh  # 构建开发镜像
bash scripts/build/docker/run_dev.sh up       # 进入容器
```

容器内复用同一套 `prepare_deps.sh` / `cu130_build_wheel.sh`
（`CONDA_PREFIX=/opt/venv`）。详见 `scripts/build/docker/README.md`。

### 4.3 FlashInfer JIT 注意事项（重要）

若使用 FlashInfer 的运行时 JIT 编译（不安装预编译 `flashinfer-cubin`），
需要：

- `nvcc` 在 `PATH` 中（JIT 编译要用）；
- 按需设置 `LD_LIBRARY_PATH` 指向 Python 环境的 lib（避免 ICU 等第三方库
  版本冲突导致加载失败）；
- `FLASHINFER_DISABLE_VERSION_CHECK=1`；
- `flashinfer/data/cccl/` 目录非空：JIT include 路径引用
  `data/cccl/{cub, libcudacxx/include, thrust}`。仓库里的 `3rdparty/cccl`
  submodule 未初始化时该目录是空的，会导致 `<cuda/cmath>` 找不到而
  编译失败。修复方式：

  ```bash
  # 方式一：从任意已装 flashinfer 的环境拷贝（python 版本按实际环境调整）
  cp -a <env>/lib/python3.12/site-packages/flashinfer/data/cccl/. \
        <flashinfer 仓库路径>/flashinfer/data/cccl/
  # 方式二：初始化 submodule 后按 pyproject.toml 的映射拷贝
  git submodule update --init 3rdparty/cccl
  ```

已知坑：`include/flashinfer/fastdiv.cuh` 的 `cuda::fast_mod_div` 包装依赖
`<cuda/cmath>`；cccl 缺失时 JIT 编译失败，曾有一版被临时改成普通整数除法
（性能回退约 1.76~2.1x）。已恢复上游实现并完成正确性验证（9 万+ 除数 ×
2048 个 n，共 1.84 亿次 divmod，0 错误）。

## 5. 启动服务

以下为 TP=8（8 卡）部署的参考启动方式，模型路径、端口与资源参数按实际
部署调整。

### 5.1 环境变量（建议在启动脚本中设置）

```bash
# 优先使用 conda 环境的 libstdc++（含 CXXABI_1.3.15），否则系统旧版本
# 会导致 ICU 78 加载失败（sqlite3 -> _sqlite3 -> libicui18n）
export LD_LIBRARY_PATH="${CONDA_PREFIX:-<env>}/lib:$LD_LIBRARY_PATH"
# 关闭 P2P 传输（参考脚本中的默认设置，按部署网络情况保留或去掉）
export NCCL_P2P_DISABLE=1
# flashinfer 为 JIT 模式（未安装 flashinfer-cubin）时关闭版本检查
export FLASHINFER_DISABLE_VERSION_CHECK=1
# nvcc 必须在 PATH 中：flashinfer JIT 编译依赖它
export PATH="/usr/local/cuda/bin:$PATH"
```

### 5.2 启动命令

```bash
# 模型目录，如 /data1/DeepSeek-V4-Flash-0731
vllm serve <model_dir> \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8_ds_mla \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --tokenizer-mode deepseek_v4 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --host 0.0.0.0 --port 8091
```

日志可按需重定向，例如追加
`2>&1 | tee -a /tmp/vllm_serve_dspark.log`。

### 5.3 DSpark 投机解码：当前不可用

**当前 SM89 移植尚未把 DSpark（dflash 草稿路径）作为可用功能，请勿添加
`--speculative-config '{"method":"dspark",...}'` 参数。** 参考启动脚本中
该参数保持注释状态：

- DSpark 草稿路径在 SM89 上仍有未根除的稳定性问题：CUDA graph 重放时
  草稿输入的异步越界断言在长时运行下仍可能出现，代码侧的输入补齐修复
  只覆盖了已复现场景，尚未完成完整验证；
- 该参数开启后还会要求草稿头随目标权重内嵌（DeepSeek-V4 DSpark 复用完整
  DSV4 配置），SM89 上未做端到端验证；
- 待 DSpark 路径在 SM89 上验证稳定后再启用，届时同步更新本文档。

## 6. 验证

### 6.1 内核精度

`sm89_precision_verify.py`（需 GPU）：在 vllm/flashinfer 仓库目录之外
（如 `/tmp`）用环境内 Python 运行，避免本地目录被当作 namespace package
解析；运行环境需 `PATH` 含 nvcc、`LD_LIBRARY_PATH` 指向环境 lib，并设置
`FLASHINFER_DISABLE_VERSION_CHECK=1`。

- decode + prefill 共 **37/37 PASS**（NH=8/16/32、topk=128/512、
  nt<=2048、sink 开/关、变长 topk_length）；
- 输出误差 <= 2.1e-3，LSE <= 1e-4，远低于 5e-2 阈值。

结论：SM89 sparse-MLA 内核数学（含 prefill 与 NH=8 填充路径）无系统性
偏差。若仍有精度/智能体问题，优先排查索引召回、FP8 KV 量化噪声与
vLLM 侧工具调用/parser 链路。

### 6.2 索引召回

`sm89_index_recall_test.py`（无模型，需 GPU）：

- 索引合并管线（SWA 窗口 + C128A/top-k 合并、local→global 映射、-1
  padding）与朴素参考逐元素一致；
- SM89 Triton indexer fallback 与精确公式误差 <= 5e-5（舍入级）；
- FP8 量化相对 BF16 精确打分的 top-k 稳定性 recall@512 ≈ 0.97。

剩余未验证项：真实模型下 indexer 与全注意力 top-k 的重叠率（需加载模型
对比）。

### 6.3 构建 / fastdiv

`test_fastdiv.cu`：`nvcc -arch=sm_89 ... -o test_fastdiv && ./test_fastdiv`
可复测 fastdiv 正确性与性能。

## 7. 关键改动文件

| 文件 | 内容 |
| --- | --- |
| `vllm/v1/attention/ops/triton_fp8_mqa_logits.py` | SM89 indexer 的 Triton FP8 MQA logits 内核（prefill 稠密 + decode 分页）与相关优化 |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 无 DeepGEMM 时路由到 Triton 回退，放宽 CUDA 支持检查 |
| `vllm/models/deepseek_v4/nvidia/ops/o_proj.py` | DeepGEMM 缺失时的 Triton o_proj 回退 |
| `vllm/models/deepseek_v4/sparse_mla.py` | `normalize_dsv4_sm89_index_topk`（默认 2048 + env 覆盖） |
| `vllm/utils/flashinfer.py` | `has_flashinfer_sparse_mla_sm89` 能力探测 |
| `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` | SM89 后端支持、KV cache 布局、启动检查 |
| `vllm/utils/deep_gemm.py` / `vllm/v1/attention/ops/flashmla.py` / `fp8_utils.py` | torch 级回退与 E8M0 处理 |
| `vllm/v1/attention/backends/mla/sparse_swa.py` | DSpark 非因果 SWA 宽度对齐 |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | CUDA graph 重放输入补齐 |
| `scripts/build/*` | CUDA 13.0 / SM89 构建脚本与 Docker 开发环境 |

## 8. 维护与同步说明

- **分支基线**：基于 vLLM `v0.27.0` tag。升级上游时建议以新的 release tag
  为基线重新移植，改动集中在第 7 节列出的文件中，便于 diff 与回放。
- **与 FlashInfer 强绑定**：SM89 能力探测依赖配套分支
  `_resolve_dsv4_sparse_mla_backend` 的实现，两仓库需同步升级；
  `requirements/cuda.txt` 默认走 flashinfer JIT（不装
  `flashinfer-cubin`）。
- **提交历史**：本分支保留原始开发历史（含部分调试往返提交），便于追溯
  问题；调试/排查文档已移出工作树，需要时从 git 历史找回。
- **改动验证**：涉及 SM89 内核/JIT 的修改，务必同时验证 decode 与 prefill
  两条路径，避免 mbarrier / `cp.async.bulk` 兼容回退类问题。
