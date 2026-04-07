import glob
import os

import torch

# Find and load the built shared library.
pattern = os.path.join(os.path.dirname(__file__), 'build', '**', '_C*.so')
matches = glob.glob(pattern, recursive=True)
if matches:
    torch.ops.load_library(matches[0])
else:
    # Editable install puts it in the source dir.
    pattern = os.path.join(os.path.dirname(__file__), '_C*.so')
    matches = glob.glob(pattern)
    if matches:
        torch.ops.load_library(matches[0])
    else:
        raise RuntimeError(
            'Could not find _C shared library. Run: pip install -e . --no-build-isolation',
        )

v = torch.ops.pyg.cuda_version()
print(f'cuda_version: {v}')
assert isinstance(v, int), f'Expected int, got {type(v)}'
print('Passed!')
