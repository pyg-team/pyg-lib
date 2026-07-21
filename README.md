# pyg-lib

<div align="center">

[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]
[![Slack][slack-image]][slack-url]
[![Contributing][contributing-image]][contributing-url]

</div>

## Installation

We provide pre-built Python wheels for all major OS/PyTorch/CUDA
combinations, see [here](https://data.pyg.org/whl). Each wheel supports CPython
3.10 through 3.14 via CPython's stable ABI. CI checks every repaired wheel with
`abi3audit --strict` and tests it on each supported CPython version.

To install the wheels, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

- `${TORCH}` should be replaced by either `2.11.0`, `2.12.0`, or `2.13.0`
- `${CUDA}` should be replaced by either `cpu`, `cu126`, `cu128`, `cu130`, or `cu132`

The following combinations are supported:

| PyTorch 2.13 | `cpu` | `cu126` | `cu128` | `cu130` | `cu132` |
| ------------ | ----- | ------- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      |         | ✅      | ✅      |
| **Windows**  | ✅    | ✅      |         | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |         |

| PyTorch 2.12 | `cpu` | `cu126` | `cu128` | `cu130` | `cu132` |
| ------------ | ----- | ------- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      |         | ✅      | ✅      |
| **Windows**  | ✅    | ✅      |         | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |         |

| PyTorch 2.11 | `cpu` | `cu126` | `cu128` | `cu130` |
| ------------ | ----- | ------- | ------- | ------- |
| **Linux**    | ✅    | ✅      | ✅      | ✅      |
| **Windows**  | ✅    | ✅      | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |

### From nightly

Nightly wheels are provided for Linux and support CPython 3.10 through 3.14:

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### From source

```
pip install ninja wheel
pip install --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git
```

[contributing-image]: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&color=4B26A4
[contributing-url]: https://github.com/pyg-team/pytorch_geometric/blob/master/.github/CONTRIBUTING.md
[coverage-image]: https://codecov.io/gh/pyg-team/pyg-lib/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pyg-lib?branch=main
[docs-image]: https://readthedocs.org/projects/pyg-lib/badge/?version=latest
[docs-url]: https://pyg-lib.readthedocs.io/en/latest/?badge=latest
[slack-image]: https://img.shields.io/badge/slack-join-white.svg?logo=slack&color=4B26A4
[slack-url]: https://data.pyg.org/slack.html
