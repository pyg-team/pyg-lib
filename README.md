[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&color=4B26A4
[contributing-url]: https://github.com/pyg-team/pytorch_geometric/blob/master/.github/CONTRIBUTING.md
[coverage-image]: https://codecov.io/gh/pyg-team/pyg-lib/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pyg-lib?branch=master
[docs-image]: https://readthedocs.org/projects/pyg-lib/badge/?version=latest
[docs-url]: https://pyg-lib.readthedocs.io/en/latest/?badge=latest
[slack-image]: https://img.shields.io/badge/slack-join-white.svg?logo=slack&color=4B26A4
[slack-url]: https://data.pyg.org/slack.html

# pyg-lib

<div align="center">

[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Slack][slack-image]][slack-url]
[![Contributing][contributing-image]][contributing-url]

</div>

## Installation

We provide pre-built Python wheels for all major OS/PyTorch/CUDA combinations from Python 3.10 till 3.13, see [here](https://data.pyg.org/whl).

To install the wheels for CPU/CUDA backend, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

* `${TORCH}` should be replaced by either `2.8.0`, `2.9.0`, or `2.10.0`
* `${CUDA}` should be replaced by either `cpu`, `cu126`, `cu128`, `cu129`, or `cu130`

The following combinations are supported:

| PyTorch 2.10 | `cpu` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      |       | ✅      |
| **Windows**  | ✅    | ✅      | ✅      |       | ✅      |
| **macOS**    | ✅    |         |         |        |        |

| PyTorch 2.9  | `cpu` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      |       | ✅      |
| **Windows**  | ✅    | ✅      | ✅      |       | ✅      |
| **macOS**    | ✅    |         |         |        |        |

| PyTorch 2.8  | `cpu` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      | ✅      |       |
| **Windows**  | ✅    | ✅      | ✅      | ✅      |       |
| **macOS**    | ✅    |         |         |        |        |

For ROCM backend, there is an external [`pyg-rocm-build` repository](https://github.com/Looong01/pyg-rocm-build) provides wheels and detailed instructions on how to install PyG for ROCm.
If you have any questions about it, please open an issue [here](https://github.com/Looong01/pyg-rocm-build/issues).

**Note:** ROCM backend only support Linux up to now.

### Build from source on a ROCm machine (Linux)

The following steps build and install `pyg-lib` with ROCm/HIP support from source.
Ensure your ROCm installation includes `hipblaslt`, `rocblas`, `rocprim`,
`rocthrust`, and `composable_kernel`.

1. Install system build tools:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-pip cmake ninja-build
```

2. Install Python build dependencies:

```bash
python3 -m pip install --upgrade pip setuptools wheel ninja
```

3. Install a ROCm-enabled PyTorch build (matching your ROCm stack):

```bash
# Example:
# python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3
```

4. Configure environment variables:

```bash
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH="${ROCM_PATH};${ROCM_PATH}/lib/cmake"
export FORCE_ROCM=1
export FORCE_CUDA=0

# Set your GPU architecture, for example gfx90a/gfx942/gfx1100:
export PYTORCH_ROCM_ARCH="gfx1100;gfx950;gfx942;gfx90a;gfx908;gfx1201;gfx1101;gfx1030"
# Alternatively, you can use:
# export AMDGPU_TARGETS=gfx90a;gfx950;gfx942;gfx90a;gfx908;gfx1201;gfx1101;gfx1030
# If your hipcc does not recognize one of the targets, remove that target.

# Optional: disable CK grouped matmul path (enabled by default).
# export PYG_ROCM_MATMUL_USE_CK=0
# Optional: require CK path (fail fast if fallback would happen).
# export PYG_ROCM_MATMUL_REQUIRE_CK=1
```

`grouped_matmul` / `segment_matmul` behavior on ROCm:

- **Important:** The CK backend in `pyg-lib` only provides native kernels for
  `bf16` and `fp16`.
- `fp16` input: use CK FP16 grouped GEMM path.
- `bf16` input: use CK BF16 grouped GEMM path.
- `fp32` input: CK does not run native FP32 kernels. `pyg-lib` first converts
  to `bf16` and tries CK BF16, then converts to `fp16` and tries CK FP16.
- Since `fp32` uses reduced-precision conversion on the CK path, numerical
  differences at `bf16/fp16` precision are expected.
- `PYG_ROCM_MATMUL_USE_CK=0`: disable CK grouped matmul and use ATen matmul.
- `PYG_ROCM_MATMUL_REQUIRE_CK=1`: strict mode. If no CK path is accepted, an
  error is raised instead of falling back.
- Without strict mode, unsupported CK shapes/targets fall back to `at::mm_out`
  with a warning that includes the reason.
- On architectures without CK XDL support for the selected path (for example
  some `gfx10` targets), fallback warnings are expected.

5. Build and install:

```bash
python3 -m pip install -v .
```

For editable/development install:

```bash
python3 -m pip install -v -e .
```

Optional check:

```bash
python3 -c "import torch; print(torch.version.hip)"
```

### From nightly

Nightly wheels are provided for Linux from Python 3.10 till 3.13:

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### From master

```
pip install ninja wheel
pip install --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git
```
