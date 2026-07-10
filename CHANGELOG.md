# Changelog

## 2026-07-10 - FlashInfer 0.6.14 sparse MLA on SM89

### 中文

- 将 SM89 sparse MLA prefill/decode 切换到 FlashInfer 0.6.14 SM89 JIT fork，并为 release 增加匹配的 FlashInfer wheel。
- Lightning Indexer scheduler metadata 改为按 `is_deep_gemm_supported()` 判断；安装了 DeepGEMM 包但硬件不支持的 SM89 不再调用其 metadata API。
- 本次不更新 `confidence_head`，不包含 per-request adaptive ℓ，DSpark 固定使用 `ℓ=6`。
- 4× RTX 4090、TP=4、单并发、每组 5 次全部成功。`8K / 32K / 128K -> 1K` 的 Prefill TPS 为 3515.72 / 4881.18 / 3812.00，Decode TPS 为 286.82 / 344.63 / 313.57。

### English

- Switched SM89 sparse MLA prefill/decode to the FlashInfer 0.6.14 SM89 JIT fork and added the matching FlashInfer wheel to the release.
- Gated Lightning Indexer scheduler metadata with `is_deep_gemm_supported()` so SM89 does not call the metadata API merely because the DeepGEMM package is installed.
- Left `confidence_head` unchanged and excluded per-request adaptive ℓ; DSpark remains fixed at `ℓ=6`.
- On 4× RTX 4090, TP=4, single concurrency, all five requests per case passed. For `8K / 32K / 128K -> 1K`, Prefill TPS is 3515.72 / 4881.18 / 3812.00 and Decode TPS is 286.82 / 344.63 / 313.57.

## 2026-07-06 - SM80/A800 DSpark test adaptation

### 中文

- 增加 DeepSeek-V4-Flash 的 SM80/A800 测试性适配说明。SM80 路径仅用于自测和实验，不代表生产级支持。
- 增加 DSpark 推测解码说明，测试参数为 `method=dspark`、`num_speculative_tokens=6`、`draft_sample_method=greedy`。
- 构建环境切换到 CUDA 13.0 / PyTorch cu130，wheel 构建显式使用 `/usr/local/cuda-13.0`。
- 记录 4× A800 上的 decode 结果：8k 输入、1k 输出、单并发为 229.8 tok/s/req；32k 输入、1k 输出、单并发为 274.2 tok/s/req。对应无 DSpark `mbt16k` 基线分别为 57.6 和 58.1 tok/s/req。

### English

- Added notes for the SM80/A800 DeepSeek-V4-Flash test adaptation. The SM80 path is for experiments and self-testing only, not production-grade support.
- Documented DSpark speculative decoding with `method=dspark`, `num_speculative_tokens=6`, and `draft_sample_method=greedy`.
- Moved the documented build environment to CUDA 13.0 / PyTorch cu130, with wheel builds explicitly using `/usr/local/cuda-13.0`.
- Recorded decode-side 4× A800 results: 229.8 tok/s/req for 8k input -> 1k output, single concurrency; 274.2 tok/s/req for 32k input -> 1k output, single concurrency. The matching no-DSpark `mbt16k` baselines are 57.6 and 58.1 tok/s/req.
