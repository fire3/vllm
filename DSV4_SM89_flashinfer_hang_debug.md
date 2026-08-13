# DeepSeek V4 SM89 FlashInfer 卡死问题排查记录

> 状态：**卡死已解决，vLLM 已能启动并通过推理冒烟测试**（2026-08-13 23:50 更新）。
> 本文档记录当前所有已确认事实、已做修改和下一步计划，供后续排查直接续接。

## 1. 环境

- 远程测试机：`ssh -p 6000 user@106.14.69.40`（主机名 gserver）
- conda 环境：`vllm-dsv4-sm89`（必须显式 `conda activate`）
- vLLM 源码：`~/vllm`（editable，分支 `v0.27.0-dsv4-sm89`）
- FlashInfer fork：`~/flashinfer`（分支 `v0.6.16.post3-dev-sm89-dsv4`），
  JIT 模式（已卸载 flashinfer-cubin），JIT 缓存
  `~/.cache/flashinfer/0.6.16.post3+sm89/89/cached_ops/sparse_mla_sm89/`
- GPU：SM89（Ada，compute capability 8.9），CUDA 13.0（torch 2.13.0+cu130）

## 2. 启动命令

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh && conda activate vllm-dsv4-sm89
cd ~/vllm && PYTHONFAULTHANDLER=1 nohup ./vllm_serve_dsv4flash.sh \
  --no-enable-flashinfer-autotune > /tmp/vllm_serve.log 2>&1 < /dev/null &
```

## 3. 已经解决的历史问题（本轮之前）

1. **torchvision 版本不匹配**：torch 2.13.0+cu130 配 0.26.0 → 升级到
   `0.28.0+cu130`。
2. **LD_LIBRARY_PATH**：`conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
   （libstdc++ CXXABI_1.3.15）。
3. **launch 脚本** `~/vllm/vllm_serve_dsv4flash.sh`：
   - 加 `export PATH=/usr/local/cuda/bin:$PATH`（vllm 的 `has_flashinfer()` 在无
     cubin 时需要 nvcc 在 PATH）；
   - 末尾加 `"$@"` 以便传 `--no-enable-flashinfer-autotune`。
   已备份 .bak/.bak2。
4. **requirements/cuda.txt**：注释 `flashinfer-cubin==0.6.16.post3`，走 JIT dev 模式。
5. **FlashInfer JIT 编译**：卸载 flashinfer-cubin；`fastdiv.cuh` 改自包含实现
   （去掉 `cuda::fast_mod_div`）；修好 decode/prefill JIT 编译。
6. **DeepGEMM**：sm89 无 vendored deep_gemm，在 `vllm/utils/deep_gemm.py` 增加
   `_tf32_hc_prenorm_gemm_torch` torch 回退（数值验证通过）。
7. **E8M0 KeyError**：`fp8_utils.py` 把 E8M0 upcast 扩展到所有平台。
8. **FlashMLA**：`vllm/v1/attention/ops/flashmla.py` 提供纯 Python
   `FlashMLASchedMeta` dataclass + `get_mla_metadata` 占位（sm89 仅作占位）。

## 4. 当前卡死问题：现象与已确认事实

### 4.1 现象

- 启动后日志停在 `Warming up DeepSeek V4 sparse MLA attention for mixed tokens=16`；
- GPU 8 卡中 1 卡 100%，worker CPU ~90%；
- 无 nvcc/ninja 子进程、无 autotune 日志、无 tqdm（说明不是编译期/autotune 卡住）；
- `--no-enable-flashinfer-autotune` 关闭 autotune 后同样挂起 → **是内核本身问题**。

### 4.2 抓栈结果

`PYTHONFAULTHANDLER=1` + `kill -ABRT <worker_pid>`，确认卡在：

```
_sparse_mla_sm120.py:908 forward
  → module.sparse_mla_sm120_decode_dsv4  (JIT 内核调用)
  → sparse_mla_sm120_decode_dsv4 (line 1391)
  → _paged_attention (line 375)
  → _sparse_mla_sm120_paged_attention (line 528)
```

即 Python 侧就是阻塞在 CUDA 内核调用上，GPU 100% = 内核内部在自旋。

### 4.3 最小复现（冒烟测试）

`~/flashinfer/tests/attention/test_sparse_mla_sm89_decode.py` 的 dsv4 decode 测试
（plain python 复刻，pytest stub 掉 import）即可复现：

```bash
cd ~/flashinfer && PYTHONFAULTHANDLER=1 timeout 900 python - <<'PY'
# 构造 num_tokens=1/8, 16 heads, topk=128, d_qk=d_v=512, page_block_size=64
# 调用 _sparse_mla_sm120_paged_attention(...)
PY
```

- 首次会重编译全部 5 个 CUDA 文件（约 2-3 分钟）；
- 编译成功后内核调用仍挂死（GPU 0 100%），`kill -ABRT` 后栈同 4.2。

## 5. 根本原因分析（已确认部分）

### 5.1 SM89 兼容层的问题

`arch/barrier.cuh` 中：

```cpp
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
#if SPARSE_MLA_USE_SM89_PRIMS
  (void)mbar; (void)tx_bytes;   // ← SM89 下是空操作
#else
  ... mbarrier.arrive.expect_tx ...
#endif
}
```

`cp_async_bulk_g2s` 在 SM89 下退化为普通 `cp.async` 16B 循环（不驱动 mbarrier 的
tx 计数）。因此：

- SM120：`issue_gather` 中 lane0 `arrive_expect_tx(mbar_full, TX_BYTES)`，bulk 完成
  后 tx 归零，`mbar_full` phase 翻转 → math 侧 `mbarrier_wait_parity` 通过。
- SM89：expect_tx 是空操作，且 cp.async 不产生 tx 计数 → `mbar_full` 计数永远为 0，
  math 侧消费者 `mbarrier_wait_parity` 永久自旋。**这是第一层根因。**

对照：prefill 路径 `common/kv_cache_io.cuh` 在 SM89 提交中补了
`cp_async_mbarrier_arrive_noinc`；**decode 的两个内联 issue_gather 漏补**。

### 5.2 SASS 级证据（关键新发现）

`cuobjdump --dump-sass sparse_mla_sm89.so`（5 个 sm_89 cubin，共约 98 万行）：

- **没有任何 `MBARRIER` SASS 指令**（sm90+ 原生 mbarrier）；
- `mbarrier.arrive` 在 sm89 上落成 `ATOMS.ARRIVE.64 RZ, [addr]` + `BSYNC`（55 处）；
- `BAR.SYNC` 分布：ID 2 (count 0x100=256)、ID 3 (count 0x100/0x180)、ID 1
  (count 0x180)；未发现明显 `MBARRIER.TEST_WAIT` 自旋；
- 说明该 toolkit 对 mbarrier 有一套 sm89 兼容的原子落地方式，
  `mbarrier.test_wait.parity` 是共享内存上的轮询（需要再定位）。

结论：mbarrier 语义在 sm89 上"可用但非硬件原生"，补 arrive 的思路本身没有错，
但**补完之后依然卡死**，说明卡点可能不止 mbar_full，或 phase/顺序还有问题。

## 6. 已做的补丁（尚未验证通过）

远程修改了 4 份文件（源码仓库 + site-packages 各 2 份，已 diff 确认一致）：

- `~/flashinfer/include/flashinfer/attention/sparse_mla_sm120/decode_dsv4_kernel.cuh`
- `~/flashinfer/include/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh`
- `site-packages/flashinfer/data/include/...` 同路径

补丁内容（两文件结构相同）：

```cpp
#if SPARSE_MLA_USE_SM89_PRIMS
    // expect_tx is a no-op on SM89; the arrive is issued after the copies.
#else
    if (lane == 0) {
      mbarrier_arrive_expect_tx(sm.mbar_full(buf), ...TX_BYTES);
    }
#endif
    ... cp_async_bulk_g2s 循环 ...
#if SPARSE_MLA_USE_SM89_PRIMS
    // cp.async emulation has no mbarrier tx tracking: wait for the IO warp's
    // copies, then arrive once so the math side can proceed.
    cp_async_wait_all();
    bar_sync_t<4, DSV4_IO_THREADS>();   // dsv3_2 用 DSV3_2_IO_THREADS
    if (lane == 0) {
      mbarrier_arrive(sm.mbar_full(buf));
    }
#endif
```

### 6.1 已核对的正确性

- `DSV4_IO_THREADS = DSV3_2_IO_THREADS = 32`，`is_io` 线程恰好 1 warp，全部执行
  `issue_gather`，`barrier.cta.sync 4, 32` 的计数正确；
- 原代码已用 `bar_sync_t<3, DSV4_MATH_THREADS>`（仅 math 线程参与），证明
  带计数的 `barrier.cta.sync` 允许子块参与 → ID 4 子块同步是合法用法；
- barrier ID 4 在文件中无其他占用（仅 ID 1/2/3 被使用）；
- 语义：`cp_async_wait_all()`（每 IO 线程等自己的 cp.async）→ bar4 使 smem 对
  lane0 可见 → lane0 arrive（计数 1，正好等于 mbar_full 的 init count 1）。

### 6.2 仍卡死 → 候选原因（按怀疑度排序）

1. **卡点其实不在 mbar_full，而在别处**：需要内核级定位（全局内存打点 + 主机轮询，
   或 cuda-gdb / nsight）。之前"卡在 mbar_full"只是推断，没有设备侧直接证据。
2. **phase/parity 时序问题**：sm89 的 `mbarrier.test_wait.parity` 轮询语义与
   sm120 原生实现不完全一致（例如 phase 初始/翻转计数），需要实测。
3. **mbarrier 在 sm89 上的 arrive 有顺序/可见性要求**：例如 `ATOMS.ARRIVE` 需要
   `MEMBAR`/`BSYNC` 组合才保证 smem 对 consumer 可见（SASS 里 arrive 前后确实有
   `MEMBAR.ALL.CTA` + `BSYNC`，说明编译器已经生成了对应 fence）。
4. **dsv3_2 与 dsv4 的差异**：dsv3_2 的 gather 没有 scalar scale 阶段，结构更简单；
   冒烟测试只覆盖了 dsv4。

## 7. 下一步计划

> 本节为**当前遗留事项**（2026-08-13 更新，替代最初的"定位卡点"计划——卡点已解决）。

### 遗留事项清单

1. **DSPark speculator 的 decode 分派（若要重新启用 dspark）**
   - 现象：6 token 的 decode 走到 flashinfer C++ 编排入口（仅允许 prefill，
     `num_tokens > 64`），报 `tvm.error.InternalError`。
   - 待查方向：speculator 路径的 `num_heads/topk` 或 KV page block size 不在
     `_decode_dsv4_dispatchable()` 的表内，或该路径绕过了 Python 分派直接调
     C++ 编排；需要打点确认调用参数后修 vllm 侧分派。
2. **固化 flashinfer JIT 头文件副本的同步方式**
   - JIT 编译实际用 `~/flashinfer/flashinfer/data/include`（gitignore 的打包
     副本），改 `include/` 后必须同步这份，否则"改了但没编进去"。
   - 建议：写一个小脚本或 Makefile 目标，从 `include/` 增量同步到
     `flashinfer/data/include`；或让 JIT 生成器直接指向 `include/`。
3. **flashinfer 运行时 Python 是 site-packages 真实副本（非 editable）**
   - 改 `~/flashinfer/flashinfer/mla/*.py` 后必须同步
     `site-packages/flashinfer/mla/`，否则运行时不生效。
   - **已完成（2026-08-13）**：`pip install -e ~/flashinfer` 已落地并验证
     （冒烟测试通过），无需再同步 site-packages。
4. **autotune 开启的完整验证**
   - 当前成功启动的实例实际**没有**吃到 `--no-enable-flashinfer-autotune`
     （脚本续行 bug 吞掉了 `"$@"`），即默认 autotune 开启也能跑通 warmup；
     但建议在脚本修好后再显式验证一次 autotune 开/关两条路径。
5. **收尾清理**
   - 远程 `~/flashinfer` 未跟踪备份文件：`include/flashinfer/fastdiv.cuh.bak`、
     `flashinfer/mla/_sparse_mla_sm120.py.bak_capture`（不入库，可删）；
   - 远程 vllm 启动脚本备份 `.bak/.bak2/.bak3` 确认无需保留后可清理；
   - 测试结束记得停掉服务并确认 GPU 归零。
6. **vllm 侧改动入库状态核对**
   - deep_gemm.py / fp8_utils.py / flashmla.py / requirements / build 脚本等
     已同步远程；本地 vllm 仓库是否已全部 commit 需要核对（flashinfer 侧已
     在独立仓库提交）。

## 8. 常用命令速查

```bash
# 清残留进程
pkill -9 -f "VLLM::Worke[r]"; pkill -9 -f "DeepSeek-V4-Flash-073[1]"
# 确认 GPU 空闲
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# 挂起时抓栈（PYTHONFAULTHANDLER=1 前提下）
WPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1); kill -ABRT $WPID
# 清 JIT 缓存强制重编译
rm -rf ~/.cache/flashinfer/0.6.16.post3+sm89/89/cached_ops/sparse_mla_sm89
rm -rf ~/.cache/vllm/flashinfer_autotune_cache
# SASS 反汇编
cuobjdump --dump-sass ~/.cache/flashinfer/0.6.16.post3+sm89/89/cached_ops/sparse_mla_sm89/sparse_mla_sm89.so
```

## 9. 远程-本地一致性提醒

- vllm 侧修改（deep_gemm.py / fp8_utils.py / flashmla.py / requirements / build
  脚本等）已 scp 同步到远程；本地 `/home/fire3/SRC/vllm` 同步过。
- flashinfer 修复（decode 内核 + fastdiv + cache signature）已同步到本地
  `/home/fire3/SRC/flashinfer` 并提交 push（`38d77a3c`、`f195a814`）。
- flashinfer 是独立 git 仓库，提交在 `~/flashinfer` 做，不要提交进 vllm 仓库。

## 10. 卡死问题的真正根因（补丁目录用错）——已解决

第一次补丁打在 `~/flashinfer/include/...` 和
`site-packages/flashinfer/data/include/...`，但 **JIT 编译实际使用
`-isystem /home/user/flashinfer/flashinfer/data/include`**（gitignore 的打包
副本，第三份）。因此冒烟测试重编译后 SASS 里仍无 `BAR.SYNC 0x4`，依然卡死。

把补丁同步到 `~/flashinfer/flashinfer/data/include/...` 后：

- 冒烟测试全部通过：`test_sparse_mla_sm89_decode_dsv4_distributed_scales`
  (num_tokens=1/8/64) + `min_ue8m0_scale` 数值对比一致；
- SASS 确认出现 `BAR.SYNC 0x4, 0x20`（110 处）；
- `~/flashinfer` 已提交并 push 到
  `origin/v0.6.16.post3-dev-sm89-dsv4`：
  - `38d77a3c fix(sm89): make JIT sparse-MLA decode work on Ada`
    （fastdiv 自包含实现 + decode dsv4/dsv3_2 mbarrier 补 arrive）；
  - `f195a814 fix(sm89): keep sparse-MLA cache signature on CPU for CUDA graph capture`

## 11. 卡死后暴露的两个后续问题（均已处理/绕过）

### 11.1 DSPark speculator warmup 报错（按用户指示先禁用 dspark）

卡死修复后首次启动在 warmup 的 `worker_sample_tokens` 阶段报：

```
tvm.error.InternalError: Check failed: num_tokens > 64 (6 vs. 64) :
Decode (num_tokens <= 64) must go through sparse_mla_sm120_decode_dsv3_2
or sparse_mla_sm120_decode_dsv4; got num_tokens=6
```

- 根因：DSPark speculator 的 6 token decode 走到了 flashinfer C++ 编排入口
  （`sparse_mla_sm120.cu` 仅允许 prefill，num_tokens>64），说明 vllm 侧该路径
  的 decode 分派没命中（待查，可能与 speculator 的 KV page block size /
  num_heads/topk 不在 decode dispatch 表有关）。
- 处理：`vllm_serve_dsv4flash.sh` 中注释掉
  `--speculative-config '{"method":"dspark",...}'`（已备份 .bak3）。
- **待办**：若需重新启用 dspark，需修复该分派（后续再排查）。

### 11.2 CUDA graph capture 期间 CPU→CUDA 拷贝报错——已修复

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

- 根因：`flashinfer/mla/_sparse_mla_sm120.py::_sparse_mla_cache_signature`
  用 `torch.tensor([...], device=device)` 在 GPU 上建缓存签名张量，graph
  capture 期间非法。
- 修复：改为 CPU 张量（该签名只用于 `.tolist()` 缓存 key）。
- 同步位置：`~/flashinfer/flashinfer/mla/_sparse_mla_sm120.py`（git 仓库）+
  `site-packages/flashinfer/mla/_sparse_mla_sm120.py`（当时的运行时副本，
  当时 flashinfer 是手工拷贝安装）。
- **2026-08-13 已改为 editable 安装**（`pip install -e ~/flashinfer`），
  原 site-packages 手工副本备份为 `site-packages/flashinfer_manual_copy_backup`；
  以后改 `~/flashinfer/flashinfer/**/*.py` 直接生效，无需再同步 site-packages。
- 已随 `f195a814` 提交并 push。

## 12. 启动脚本续行被注释吞掉的坑

给脚本里带行尾 `\` 的选项行加 `#` 注释时，`#` 后的内容会被 bash 当注释吞掉，
导致后续 `--host/--port`、`"$@"` 整段丢失（服务会以默认 8000 端口启动、参数
不生效）。修法：注释行不能插在续行链中间，直接删掉该行。

## 13. 当前验证结论（2026-08-13 23:50）

- vLLM 已在远程 **8000 端口**成功启动（dspark 关闭、0 ERROR）；
- 推理冒烟测试通过：
  - `GET /v1/models` 返回 `deepseek-v4-flash`；
  - `17*23` → `391`（2 tokens，0.7s）；
  - 128 tokens 段落生成正常（98 completion tokens，3.2s，内容连贯）。
- 仓库同步（2026-08-13 追加）：
  - 远程 flashinfer：HEAD = `f195a814`（= origin），editable 安装，JIT 冒烟通过；
  - 远程 vllm：HEAD = `a135ec2c5`（= origin），本地 2 个 commit 已 ff 拉取；
  - 两个仓库工作区均干净（仅剩环境相关未跟踪文件：启动脚本、requirements-lock.txt）。
- 待办：修复 DSPark speculator 的 decode 分派后重新启用 dspark；
  以及把 JIT 头文件副本（`flashinfer/data/include`）的同步方式固化，
  避免再次出现"改了 include/ 但 JIT 没编到"的问题。

## 14. prefill 崩溃修复（2026-08-14 追加）

- 现象：长 prompt（prefill，num_tokens>64）500；
  `Unsupported sparse-MLA prefill configuration: num_heads=8`；
  之后排查出 prefill 在 sm89 上从未真正跑通过（NH=16 也崩）。
- 根因与修复（flashinfer commit `273041d1`，含 4 个问题）：
  1. DSV4 prefill 分派缺 NH=8 → 移植上游 padded-H8 MG 方案（#4380）；
  2. `cudaLaunchKernelExC` + `__grid_constant__` 在 sm89 启动被拒 → 改 `<<<>>>`；
  3. `cp.async.mbarrier.arrive.noinc`（LDGSTSBAR）在 sm89 触发 SM 异常 → 改
     decode 同款 `cp_async_wait_all + bar4 + mbarrier_arrive`；
  4. cp.async L2::128B hint 在 sm89 规避（保守）。
- 验证：DSV4 prefill NH=8/16（含 sink）、v32 SG prefill 数值对比通过；
  vLLM 长 prompt（prefill）推理恢复，0 ERROR；全链路推理回归通过
  （短问答/长生成/推理题/completions/并发/tool call/stream）。
- 已同步：本地 flashinfer 入库 push，远程仓库 ff 到 `273041d1`。
