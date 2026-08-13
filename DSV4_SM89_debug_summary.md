# DeepSeek V4 SM89 远程调试总结

> 时间：2026-08-13。本文档是对"vLLM 在远程 SM89 环境启动卡死"整个排查过程的
> 复盘总结，重点记录**远程环境本身的情况**（目录结构、安装方式、工具链、坑），
> 方便以后在这个环境上继续工作时不重复踩坑。
> 逐项排查记录见 [`DSV4_SM89_flashinfer_hang_debug.md`](DSV4_SM89_flashinfer_hang_debug.md)。

## 1. 最终结论

- 卡死根因：flashinfer sparse-MLA decode 内核在 SM89 兼容路径上漏了 mbarrier
  arrive（`expect_tx` 空操作 + `cp.async.bulk` 退化为普通 `cp.async`，`mbar_full`
  永远不翻转），math 线程在 `mbarrier_wait_parity` 永久自旋。
- 修复后 vLLM 可正常启动，推理冒烟测试通过（短问答 + 128 tokens 长生成，0 ERROR）。
- 修复已提交 push 到本地 flashinfer fork（`38d77a3c`、`f195a814`）。

## 2. 远程环境画像（重要）

### 2.1 访问与基础环境

| 项 | 值 |
|---|---|
| SSH | `ssh -p 6000 user@106.14.69.40`（主机名 gserver） |
| conda | 必须显式 `source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate vllm-dsv4-sm89` |
| conda env 变量 | `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`（libstdc++ CXXABI_1.3.15） |
| GPU | 8x NVIDIA L40S（sm89 / compute capability 8.9） |
| CUDA toolkit | `/usr/local/cuda`（CUDA 13.0；nvcc 需显式加入 PATH） |
| torch | 2.13.0+cu130；torchvision 0.28.0+cu130（0.26.0 不匹配，已升） |
| flashinfer | 0.6.16.post3，JIT 模式（已卸载 flashinfer-cubin） |
| flashinfer 安装方式 | **editable**（`pip install -e ~/flashinfer`，2026-08-13 改；原手工副本备份为 `site-packages/flashinfer_manual_copy_backup`） |
| 模型 | `/data1/DeepSeek-V4-Flash-0731`（served name `deepseek-v4-flash`） |
| vLLM | `~/vllm`，editable 安装，分支 `v0.27.0-dsv4-sm89` |
| API | `vllm serve ... --host 0.0.0.0 --port 8091`；当前实际跑在 **8000**（脚本续行 bug，用户表示不关心 8091） |

### 2.2 flashinfer 代码有三层副本（本环境最大的坑）

同一个文件可能存在于三个位置，各自用途不同，**改错位置 = 改了没生效**：

| 位置 | 是否 git 跟踪 | 用途 |
|---|---|---|
| `~/flashinfer/include/` | ✅ 是（仓库源） | flashinfer 源码头文件 |
| `~/flashinfer/flashinfer/data/`（含 include/ 和 csrc/） | ❌ gitignore（`.gitignore: flashinfer/data/`） | **JIT 编译实际使用的源**：`build.ninja` 里 `-isystem .../flashinfer/data/include`，源文件也来自 `.../flashinfer/data/csrc` |
| `site-packages/flashinfer/` | ❌ | **已改为 editable**（`pip install -e ~/flashinfer`，导入走 `~/flashinfer/flashinfer`）；原手工副本已备份为 `site-packages/flashinfer_manual_copy_backup` |

结论：

- 改 `.cuh`/`.cu` → 必须同步 `~/flashinfer/flashinfer/data/`（至少 include 头文件），
  否则 JIT 重编译还是旧代码（本次卡死 2 小时就是这个原因）。
- 改 `flashinfer/mla/*.py` → **editable 后直接生效**，无需再同步 site-packages。
- git 提交只提交 `include/` 和 `flashinfer/mla/` 等仓库内路径；`data/` 不入库，
  由打包/同步过程生成。

注意：editable 安装的版本号从 `0.6.16.post3+sm89` 变成 `0.6.16.post3`，
JIT 缓存路径随之变为 `~/.cache/flashinfer/0.6.16.post3/89/...`
（旧缓存 `0.6.16.post3+sm89/` 仍保留，当前运行中的服务在用）。

### 2.3 工具链情况

- `pytest` **未安装**；跑 flashinfer 测试可用 python 直接调用测试函数 +
  pytest stub（`sys.modules["pytest"]` 打桩）。
- `cuobjdump` / `nvdisasm` 在 `/usr/local/cuda/bin`（不在默认 PATH，要用绝对路径）。
- `nvcc` 需要 `export PATH=/usr/local/cuda/bin:$PATH`（launch 脚本已加，
  vllm 的 `has_flashinfer()` 在无 cubin 时需要 nvcc 探测）。
- 无 cuda-gdb / nsight 确认（本次没用上）。

## 3. 问题时间线（问题 → 根因 → 修复 → 验证）

1. **`vllm --version` 报错** → torchvision 0.26.0 与 torch 2.13+cu130 不匹配
   → 升 0.28.0+cu130。
2. **libstdc++ 符号缺失** → conda env `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`。
3. **JIT 模式探测不到 flashinfer / nvcc** → launch 脚本加 PATH、`"$@"` 透传。
4. **fastdiv JIT 编译失败** → `fastdiv.cuh` 去掉对 vendored CCCL
   `cuda::fast_mod_div` 的依赖，改为自包含实现。
5. **DeepGEMM 缺失** → vllm `deep_gemm.py` 加 torch 回退（数值验证通过）。
6. **E8M0 KeyError** → `fp8_utils.py` E8M0 upcast 扩展到所有平台。
7. **FlashMLA 扩展缺失** → vllm `flashmla.py` 提供 Python dataclass 占位。
8. **decode 内核卡死（GPU 100% 自旋）** → SM89 mbarrier arrive 缺失
   （见 hang_debug 文档 §5/§10）；补丁最初打错目录，同步到
   `flashinfer/data/include` 后冒烟测试全过。
9. **DSPark speculator warmup 报 tvm 错误（6 tokens 进 prefill 编排）** →
   按用户指示先注释 `--speculative-config` 关闭 dspark（遗留事项：分派待修）。
10. **CUDA graph capture 报 CPU→CUDA 拷贝非法** → `_sparse_mla_cache_signature`
    的 `torch.tensor(..., device=device)` 改 CPU 张量（同步仓库 + site-packages）。
11. **启动脚本续行 bug** → 给带 `\` 的选项行加 `#` 注释把 `--host/--port "$@"`
    整段吞掉；删掉注释行后恢复正常（端口最终用了默认 8000，用户表示不关心）。
12. **推理验证** → `/v1/models` OK；`17*23` → `391`；128 tokens 段落正常。

## 4. 远程环境的坑与调试技巧

### 4.1 进程清理

```bash
# 括号技巧防 pkill 自匹配（模式出现在自己命令行里会误杀本 shell）
pkill -9 -f 'VLLM::Worke[r]'
pkill -9 -f 'VLLM::EngineCor[e]'
pkill -9 -f 'vllm serv[e]'
pkill -9 -f 'DeepSeek-V4-Flash-073[1]'
```

- 注意：ssh 单条命令里如果后续部分出现**字面量**进程名（如 `cat xxx.sh`），
  pkill 会杀掉自己的远程 shell，ssh 直接 255 退出。
- 上次崩溃可能留下孤儿 `VLLM::EngineCore`（ppid=1），要一起清。

### 4.2 抓挂起栈

```bash
# 启动时带 PYTHONFAULTHANDLER=1
WPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
kill -ABRT $WPID
grep -A 45 'Current thread' /tmp/vllm_serve.log
```

### 4.3 验证补丁真的编进了内核

```bash
rm -rf ~/.cache/flashinfer/0.6.16.post3+sm89/89/cached_ops/sparse_mla_sm89
rm -rf ~/.cache/vllm/flashinfer_autotune_cache
# 重编译后反汇编检查新增指令
/usr/local/cuda/bin/cuobjdump --dump-sass \
  ~/.cache/flashinfer/0.6.16.post3+sm89/89/cached_ops/sparse_mla_sm89/sparse_mla_sm89.so \
  | grep -c 'BAR.SYNC 0x4'
```

- SASS 事实：sm89 目标下没有 `MBARRIER` 指令，`mbarrier.arrive` 落成
  `ATOMS.ARRIVE.64` + `BSYNC`；`barrier.cta.sync 4, 32` 落成 `BAR.SYNC 0x4, 0x20`。

### 4.4 冒烟测试（无 pytest）

```bash
cd ~/flashinfer && python - <<'PY'
# 用 pytest stub 顶掉 import（tests 模块顶层 import pytest）
# 手动调用 test_sparse_mla_sm89_decode.py 里的测试函数
PY
```

### 4.5 其他

- GPU 是否空闲：`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`。
- 看进程真实参数：`tr '\0' ' ' < /proc/<pid>/cmdline`（ps 会截断/合并）。
- 端口确认：`ss -tlnp | grep <port>`（服务"HTTP server started"但端口没监听 =
  参数没传进去，查 cmdline）。
- 远程 flashinfer 仓库里有未跟踪备份 `.bak`（fastdiv、_sparse_mla_sm120.py），
  不入库；清理时注意别删 git 仓库内文件。

## 5. 遗留事项（详见 hang_debug 文档 §7）

1. DSPark speculator 的 decode 分派修复（重新启用 dspark 前必须做）。
2. 固化 `flashinfer/data/include` 的同步方式（或让 JIT 直接指 `include/`）。
3. flashinfer 建议改 editable 安装或写 site-packages 同步脚本。
   **已完成（2026-08-13）**：editable 安装落地，冒烟测试通过。
4. autotune 开/关两条路径的显式验证。
5. 收尾清理（.bak、旧服务、备份脚本）。
6. vllm 侧改动本地入库状态核对。

> 更新（2026-08-14）：**prefill 崩溃已修复**（flashinfer `273041d1`，
> 见 hang_debug 文档 §14）。长 prompt 推理恢复正常，全链路回归通过。
> 遗留事项相应减少为：DSPark 分派、JIT include 同步固化、autotune 显式验证、
> 收尾清理。
