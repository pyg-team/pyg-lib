import importlib
import os
import os.path as osp
import warnings

__version__ = '0.0.0'

# * `libpyg.so`: The name of the shared library file.
# * `torch.ops.pyg`: The used namespace.
# * `pyg_lib`: The name of the Python package.
# TODO Make naming more consistent.


def load_library(lib_name: str):
    if bool(os.getenv('BUILD_DOCS', 0)):
        return

    import torch

    loader_details = (importlib.machinery.ExtensionFileLoader,
                      importlib.machinery.EXTENSION_SUFFIXES)

    path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    ext_finder = importlib.machinery.FileFinder(path, loader_details)
    spec = ext_finder.find_spec(lib_name)

    if spec is None:
        warnings.warn(f"Could not find shared library '{lib_name}'")
    else:
        torch.ops.load_library(spec.origin)


load_library('libpyg')


def cuda_version() -> int:
    r"""Returns the CUDA version for which :obj:`pyg_lib` was compiled with."""
    return torch.ops.pyg.cuda_version()


__all__ = [
    '__version__',
    'cuda_version',
]
