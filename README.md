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

### Form nightly

Nightly wheels are provided for Linux from Python 3.7 till 3.10:

#### PyTorch 1.12

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-1.12.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu116` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu102` | `cu113` | `cu116` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |

#### PyTorch 1.11

```
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-1.11.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

|             | `cpu` | `cu102` | `cu113` | `cu115` |
|-------------|-------|---------|---------|---------|
| **Linux**   | ✅    | ✅      | ✅      | ✅      |

### From master

```
pip install git+https://github.com/pyg-team/pyg-lib.git
```
