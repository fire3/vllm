# SM89 DSv4 sparse MLA port benchmarks

## Environment

See `ENV_SM120.txt` for the exact local environment. The benchmark host is an
RTX PRO 6000 Blackwell workstation GPU (`compute_cap=12.0`) with:

- PyTorch `2.11.0+cu130`
- FlashInfer `0.6.14` (`flashinfer-cubin` `0.6.13`)
- vLLM `0.23.1rc1.dev889+g587bd29cc.d20260707`

FlashInfer must be imported with `FLASHINFER_DISABLE_VERSION_CHECK=1` in this
environment because `flashinfer-python==0.6.14` has no matching
`flashinfer-cubin==0.6.14` release.

## Sparse MLA decode

Command:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda-13.2 \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --num-heads 64 --num-slots 65536 \
  --out benchmarks/sparse_mla_sm89_port/results_decode_h64.json

FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda-13.2 \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --num-heads 16 --num-slots 65536 \
  --out benchmarks/sparse_mla_sm89_port/results_decode_h16.json
```

The input cache uses the DSv4 584-byte footer layout expected by the FlashInfer
SM120 DSV4 path and by the vendored Triton reference. FlashInfer internally
quantizes Q to FP8, while the Triton reference uses BF16/F32 Q math; the measured
relative diff is therefore recorded but not treated as bitwise equivalence.

| heads | tokens | FlashInfer ms | Triton ms | Triton/FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 0.0827 | 0.0540 | 0.65 |
| 64 | 64 | 0.0811 | 0.1580 | 1.95 |
| 64 | 128 | 0.0317 | 0.2459 | 7.76 |
| 64 | 256 | 0.0473 | 0.4558 | 9.64 |
| 16 | 1 | 0.0787 | 0.0538 | 0.68 |
| 16 | 64 | 0.0760 | 0.0684 | 0.90 |
| 16 | 128 | 0.0207 | 0.0793 | 3.83 |
| 16 | 256 | 0.0210 | 0.1520 | 7.23 |

Summary: Triton is faster for tiny decode batches on this synthetic SWA-only
shape. FlashInfer becomes faster at larger batches and is much faster after the
SM120 FlashInfer path routes through its prefill orchestrator (`num_tokens > 64`).

Source install check:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda-13.2 \
  FLASHINFER_NVCC=/usr/local/cuda-13.2/bin/nvcc \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --num-heads 64 --num-slots 65536 \
  --out benchmarks/sparse_mla_sm89_port/results_decode_source_h64.json
```

This rerun used editable `flashinfer-python==0.6.14` from
`/home/yyf/flashinfer`. The stable rerun stayed close to the wheel baseline for
the FlashInfer path: 0.0779 ms at 1 token, 0.0752 ms at 64 tokens, 0.0343 ms at
128 tokens, and 0.0474 ms at 256 tokens.

SM89-primitives-on-SM120 check:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 FLASHINFER_SPARSE_MLA_FORCE_SM89_PRIMS=1 \
  CUDA_HOME=/usr/local/cuda-13.2 FLASHINFER_NVCC=/usr/local/cuda-13.2/bin/nvcc \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_decode.py \
  --out benchmarks/sparse_mla_sm89_port/results_decode_sm89prims_on_sm120.json
```

This rerun forced the ported SM89 primitive path on the SM120 workstation. It is
an upper-bound probe for the eventual SM89 path, not a final SM89 measurement.
The same Q-quantization caveat applies; relative diff is high for small-token
synthetic cases and drops to about 1.3-1.5% once the FlashInfer orchestrator path
is active.

| heads | tokens | FlashInfer ms | Triton ms | Triton/FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 0.0801 | 0.0540 | 0.67 |
| 64 | 8 | 0.0841 | 0.0620 | 0.74 |
| 64 | 32 | 0.0772 | 0.0842 | 1.09 |
| 64 | 64 | 0.1764 | 0.1635 | 0.93 |
| 64 | 128 | 0.0465 | 0.2700 | 5.80 |
| 64 | 256 | 0.0668 | 0.4993 | 7.48 |

## Sparse MLA prefill

Command:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 CUDA_HOME=/usr/local/cuda-13.2 \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_sparse_mla_prefill.py \
  --num-heads 64 --num-slots 65536 \
  --out benchmarks/sparse_mla_sm89_port/results_prefill.json
```

| heads | tokens | FlashInfer ms | Triton ms | Triton/FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 256 | 0.0485 | 0.4800 | 9.90 |
| 64 | 1024 | 0.2634 | 1.9666 | 7.47 |
| 64 | 4096 | 1.1800 | 7.7987 | 6.61 |
| 64 | 8192 | 2.2429 | 16.0824 | 7.17 |

Summary: for prefill-sized batches on SM120, FlashInfer is consistently faster
than the old Triton sparse MLA reference by roughly 6.6x to 9.9x on this probe.

## MXFP4 MoE

Command:

```bash
FLASHINFER_DISABLE_VERSION_CHECK=1 \
  .venv-sm120/bin/python benchmarks/sparse_mla_sm89_port/bench_mxfp4_moe.py \
  --out benchmarks/sparse_mla_sm89_port/results_mxfp4.json
```

The external `deep-gemm` package install was attempted with the Tsinghua mirror,
but `deep-gemm==1.0.0` failed to build because its sdist did not include the
required CUTLASS submodule. The benchmark therefore uses vLLM's vendored
DeepGEMM runtime. DeepGEMM is timed with synthetic packed MXFP4 tensors and is a
throughput probe only; output finite-ness is recorded and false for the synthetic
DeepGEMM payload. Marlin uses vLLM's existing MXFP4 Marlin test helper and
produces finite outputs.

Shape: hidden size 4096, intermediate size 2048, 8 experts, top-k 2.

| tokens | DeepGEMM ms | Marlin ms | Marlin/DeepGEMM | notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.2294 | 0.0829 | 0.36 | DeepGEMM output not finite |
| 16 | n/a | 0.0988 | n/a | DeepGEMM assertion |
| 64 | 0.2219 | 0.0822 | 0.37 | DeepGEMM output not finite |
| 256 | 0.2181 | 0.1348 | 0.62 | DeepGEMM output not finite |
| 1024 | 0.2448 | 0.3682 | 1.50 | DeepGEMM output not finite |
| 4096 | 0.8219 | 1.3506 | 1.64 | DeepGEMM output not finite |

Summary: this local run confirms vendored DeepGEMM FP4 and Marlin MXFP4 can both
execute same-shape throughput probes on SM120, but the DeepGEMM side still needs
a model-faithful MXFP4 packing source before treating the values as correctness
or final performance evidence. For SM89 planning, Marlin remains the realistic
fallback path because DeepGEMM FP4 is Blackwell/Hopper-oriented and not expected
to run on Ada SM89.
