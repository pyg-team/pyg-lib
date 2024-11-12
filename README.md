[python-testing-image]: https://github.com/pyg-team/pyg-lib/actions/workflows/python_testing.yml/badge.svg
[python-testing-url]: https://github.com/pyg-team/pyg-lib/actions/workflows/python_testing.yml
[cpp-testing-image]: https://github.com/pyg-team/pyg-lib/actions/workflows/cpp_testing.yml/badge.svg
[cpp-testing-url]: https://github.com/pyg-team/pyg-lib/actions/workflows/cpp_testing.yml
[docs-image]: https://readthedocs.org/projects/pyg-lib/badge/?version=latest
[docs-url]: https://pyg-lib.readthedocs.io/en/latest/?badge=latest
[coverage-image]: https://codecov.io/gh/pyg-team/pyg-lib/branch/master/graph/badge.svg
[coverage-url]: https://codecov.io/github/pyg-team/pyg-lib?branch=master

# pyg-lib

[![Python Testing Status][python-testing-image]][python-testing-url]
[![CPP Testing Status][cpp-testing-image]][cpp-testing-url]
[![Docs Status][docs-image]][docs-url]
[![Code Coverage][coverage-image]][coverage-url]

* [Installation](#installation)

## Installation

We provide pre-built Python wheels for all major OS/PyTorch/CUDA combinations from Python 3.9 till 3.12, see [here](https://data.pyg.org/whl).
Note that currently, Windows wheels are not supported (we are working on fixing this as soon as possible).

To install the wheels, simply run

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

where

* `${TORCH}` should be replaced by either `1.12.0`, `1.13.0`, `2.0.0`, `2.1.0`, `2.2.0`, `2.3.0`, `2.4.0` or `2.5.0`
* `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, `cu116`, `cu117`, `cu118`, `cu121`, or `cu124`

The following combinations are supported:

| PyTorch 2.5  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      | ✅      |
| **Windows**  | ✅    |         |         |         | ✅      | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 2.4  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      | ✅      |
| **Windows**  | ✅    |         |         |         | ✅      | ✅      | ✅      |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 2.3  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      |         |
| **Windows**  | ✅    |         |         |         | ✅      | ✅      |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 2.2  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      |         |
| **Windows**  | ✅    |         |         |         | ✅      | ✅      |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 2.1  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         |         | ✅      | ✅      |         |
| **Windows**  | ✅    |         |         |         | ✅      | ✅      |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 2.0  | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         |         | ✅      | ✅      | ✅      |         |
| **Windows**  | ✅    |         |         | ✅      | ✅      |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 1.13 | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------|---------|---------|
| **Linux**    | ✅    |         | ✅      | ✅      |         |         |         |
| **Windows**  | ✅    |         | ✅      | ✅      |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

| PyTorch 1.12 | `cpu` | `cu113` | `cu116` | `cu117` | `cu118` | `cu121` | `cu124` |
|--------------|-------|---------|---------|---------|---------| --------|---------|
| **Linux**    | ✅    | ✅      | ✅      |         |         |         |         |
| **Windows**  | ✅    | ✅      | ✅      |         |         |         |         |
| **macOS**    | ✅    |         |         |         |         |         |         |

### Form nightly

Nightly wheels are provided for Linux from Python 3.9 till 3.12:

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-${TORCH}+${CUDA}.html
```

### From master

```
pip install ninja wheel
pip install git+https://github.com/pyg-team/pyg-lib.git
```
