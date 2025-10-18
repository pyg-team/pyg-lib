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

We provide pre-built Python wheels for all major OS/PyTorch/CUDA/ROCm(HIP) combinations from Python 3.10 till 3.13, see [here](https://data.pyg.org/whl).
Note that currently, Windows wheels are not supported (we are working on fixing this as soon as possible).

To install the wheels for CPU/CUDA backend, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

* `${TORCH}` should be replaced by either `2.6.0`, `2.7.0`, `2.8.0`, or `2.9.0`
* `${CUDA}` should be replaced by either `cpu`, `cu118`, `cu124`, `cu126`, `cu128`, `cu129`, or `cu130`

The following combinations are supported:

| PyTorch 2.9  | `cpu` | `cu118` | `cu124` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         | ✅      | ✅      |       | ✅      |
| **Windows**  | ✅    |         |         | ✅      | ✅      |       | ✅      |
| **macOS**    | ✅    |         |         |         |         |        |        |

| PyTorch 2.8  | `cpu` | `cu118` | `cu124` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         | ✅      | ✅      | ✅      |       |
| **Windows**  | ✅    |         |         | ✅      | ✅      | ✅      |       |
| **macOS**    | ✅    |         |         |         |         |        |        |

| PyTorch 2.7  | `cpu` | `cu118` | `cu124` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      |         | ✅      | ✅      |         |       |
| **Windows**  | ✅    | ✅      |         | ✅      | ✅      |         |       |
| **macOS**    | ✅    |         |         |         |         |         |        |

| PyTorch 2.6  | `cpu` | `cu118` | `cu124` | `cu126` | `cu128` | `cu129` | `cu130` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      | ✅      |         |         |       |
| **Windows**  | ✅    | ✅      | ✅      | ✅      |         |         |       |
| **macOS**    | ✅    |         |         |         |         |         |        |


For ROCM backend, there is an external [`pyg-rocm-build` repository](https://github.com/Looong01/pyg-rocm-build) provides wheels and detailed instructions on how to install PyG for ROCm.
If you have any questions about it, please open an issue [here](https://github.com/Looong01/pyg-rocm-build/issues).

**Note:** ROCM backend only support Linux up to now.

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
