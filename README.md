[testing-image]: https://github.com/pyg-team/pyg-lib/actions/workflows/testing.yml/badge.svg
[testing-url]: https://github.com/pyg-team/pyg-lib/actions/workflows/testing.yml
[docs-image]: https://readthedocs.org/projects/pyg-lib/badge/?version=latest
[docs-url]: https://pyg-lib.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/pyg-team/pyg-lib/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pyg-lib?branch=master

# pyg-lib

[![Testing Status][testing-image]][testing-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

* [Installation](#installation)

## Installation

We provide pre-built Python wheels for all major OS/PyTorch/CUDA combinations from Python 3.7 till 3.11, see [here](https://data.pyg.org/whl).
Note that currently, Windows wheels are not supported (we are working on fixing this as soon as possible).

To install the wheels, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

* `${TORCH}` should be replaced by either `1.11.0`, `1.12.0`, `1.13.0` or `2.0.0`
* `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, `cu115`, `cu116`, `cu117` or `cu118`

The following combinations are supported:

| PyTorch 2.0  | `cpu` | `cu102` | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         |         | ✅      | ✅      |
| **Windows**  |       |         |         |         |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 1.13 | `cpu` | `cu102` | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      |         |
| **Windows**  |       |         |         |         |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 1.12 | `cpu` | `cu102` | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      |         | ✅      |         |         |
| **Windows**  |       |         |         |         |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 1.11 | `cpu` | `cu102` | `cu113` | `cu115` | `cu116` | `cu117` | `cu118` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    | ✅      | ✅      | ✅      |         |         |         |
| **Windows**  |       |         |         |         |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

### Form nightly

Nightly wheels are provided for Linux from Python 3.7 till 3.11:

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### From master

```
pip install git+https://github.com/pyg-team/pyg-lib.git
```
