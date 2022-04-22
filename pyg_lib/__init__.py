import os.path as osp
import imp
import importlib
import torch

__version__ = '0.0.0'

# loader_details = (importlib.machinery.ExtensionFileLoader,
#                   importlib.machinery.EXTENSION_SUFFIXES + ['.dylib'])
# print(loader_details)

ext = importlib.machinery.FileFinder(osp.dirname(__file__))
print(ext)
spec = ext.find_spec('libpyg')
print('---------')
print(spec.origin)
print('---------')
torch.ops.load_library(spec.origin)

cuda_version = torch.ops.pyg.cuda_version

__all__ = [
    '__version__',
]
