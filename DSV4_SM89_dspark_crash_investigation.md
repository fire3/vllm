# DeepSeek V4 SM89 DSPark 崩溃排查记录

> 时间：2026-08-14。症状：启用 `--speculative-config`（dspark，6 个草稿
> token）+ EP 的会话在生成中途崩溃，所有 worker 报
> `CUDA error: device-side assert triggered` +
> `indexSelectSmallIndex: Assertion srcIndex < srcSelectDimSize failed`。

## 1. 现象与复现

- 崩溃是 ATen `torch.index_select` 的**越界索引**设备断言，且是异步错误：
  真实越界 kernel 运行后，错误在任意后续 CUDA 调用才浮现（例如 DSPark
  `precompute_and_store_context_kv` 的 `fused_wqa_wkv`、或 tensor 析构）。
- 复现条件：**并发长上下文**（如 3 路并行 25k 上下文 + 长生成）。单请求
  负载不触发；`CUDA_LAUNCH_BLOCKING=1` 或每步 host 同步（`.item()/
  .tolist()`）会改变批次调度时序并**掩盖**崩溃。
- 首次会话（修复前）在并发下第一轮即崩；修复后 3 轮并发（18 个长生成）
  全过，但长时间运行后仍出现过一次同类残留崩溃。

## 2. 根因分析（代码层面）

DFlash/DSPark 的草稿前向在 CUDA graph 里回放，输入由 eager 的
`_prepare_dflash_inputs_kernel` 写入：

1. **图回放用 padded num_reqs，但内核只初始化真实 query 行**：
   `_prepare_dflash_inputs_kernel` 只对真实 query 写 `input_ids`/
   `positions`；padding 只覆盖了 `query_slot_mapping` 和 sample 缓冲。
   当真实请求数 < 图档位数（如并发 3 路 → 档位 4）时，padded 行的
   `input_ids` 是**陈旧数据**，会被草稿前向的 `F.embedding` 当索引使用，
   一旦越界即触发 `indexSelectSmallIndex`。
2. **`bonus_token` 可能为 -1**：`last_sampled`/`next_prefill_tokens` 在请求
   结束/padding 时可能含 -1，被直接写成 query input id。
3. 采样路径 `_sample_sequential` 同样按 padded num_reqs 处理，`sample_
   idx_mapping` 的 padded 行是 -1（gumbel 内核用 `is_valid_req` 掩掉，
   但 `markov_embed(prev)` 的 `prev` 来自 padded 行的陈旧 input id）。

## 3. 修复（`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`）

`_prepare_dflash_inputs_kernel`：

1. padded 行的 `input_ids`/`query_positions` 显式写 0（与
   `query_slot_mapping` 的 PAD 对齐），避免图回放喂陈旧数据给 embedding；
2. `bonus_token = tl.maximum(bonus_token, 0)`，防止 -1 进入 query input id。

## 4. 验证结果

- 修复前：并发 3 路负载第一轮即崩。
- 修复后：3 轮 × 3 路并发长生成（18 个请求）全部通过、服务健康；
  5 路并发混合通过；fixture 双轮 replay 通过。
- 残留：长时间（~25 个长生成）后仍出现一次同类断言，说明还有另一个
  触发路径未被覆盖；由于任何 host 同步都会掩盖问题，建议用
  compute-sanitizer / cuda-gdb 在 worker 上抓取真实越界 kernel 进一步定位。

## 5. 相关文件

- 修复：`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
- 复现脚本：`dspark_repro/repro_long.py`（25k 上下文 + 长生成，支持并发）
- 复现说明：`dspark_repro/README.md`
