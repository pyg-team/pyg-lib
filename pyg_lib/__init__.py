import os.path as osp
import importlib
import torch

__version__ = '0.0.0'

loader_details = (importlib.machinery.ExtensionFileLoader,
                  importlib.machinery.EXTENSION_SUFFIXES)

path = osp.abspath(osp.join(osp.dirname(__file__), '..'))
ext_finder = importlib.machinery.FileFinder(path, loader_details)
spec = ext_finder.find_spec('libpyg')
torch.ops.load_library(spec.origin)

cuda_version = torch.ops.pyg.cuda_version

__all__ = [
    '__version__',
    'cuda_version',
]
