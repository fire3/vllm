# Build a cu130 vLLM wheel for SM 8.9 (PTX)

Helper scripts to build a CUDA 13.0 wheel from this checkout that only carries
SM 8.9 (NVIDIA Ada) support, embedded as PTX.

## Prerequisites

- An activated conda environment (the scripts work in conda; see
  `cu130_install_deps.sh`).
- CUDA 13.0 toolkit with `nvcc` under `/usr/local/cuda`.

## Usage

```bash
conda activate vllm-dev
source scripts/build/cu130_env.sh        # CUDA_HOME, arch list, etc.
bash scripts/build/cu130_install_deps.sh # torch 2.11.0+cu130 + build deps
bash scripts/build/cu130_build_wheel.sh  # builds dist/vllm-*.whl
bash scripts/build/cu130_verify.sh       # confirms the embedded kernels
```

## Choosing the architecture

`TORCH_CUDA_ARCH_LIST` controls what nvcc emits (default `8.9+PTX`):

| Value    | Contents                             | Runs on                  |
|----------|--------------------------------------|--------------------------|
| `8.9`    | sm_89 SASS only                      | Ada only                 |
| `8.9+PTX`| sm_89 SASS + compute_89 PTX (JIT)    | Ada and newer (default)  |

Set it before building to override:

```bash
export TORCH_CUDA_ARCH_LIST=8.9
```

## Installing the wheel on the target machine

The wheel links against cu130 PyTorch, so install it with the cu130 index:

```bash
pip install dist/vllm-*.whl --extra-index-url https://download.pytorch.org/whl/cu130
```

## Notes

- `--no-build-isolation` is required so the build uses the conda env's cu130
  torch instead of downloading the CPU torch from PyPI.
- PyTorch is installed from the `cu130` index only; the other build deps
  (cmake, ninja, setuptools-rust, ...) are installed from the default index, so
  keep `--index-url` off for those two commands.
- The local wheel version is derived from git tags (setuptools-scm). To tag it
  explicitly (e.g. `0.26.0+cu130sm89`), export `VLLM_VERSION_OVERRIDE`.
