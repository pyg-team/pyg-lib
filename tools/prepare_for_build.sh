#!/bin/bash

# Usage:
# ./prepare_for_build.sh <torch_version> <cuda_version>
set -e

TORCH_VERSION="${1:?Specify torch version, e.g. 2.1.0}"
CUDA_VERSION="${2:?Specify cuda version, e.g. cu121}"

pip install torch=="${TORCH_VERSION}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\nCXX11 ABI: {torch.compiled_with_cxx11_abi()}')"

sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver log,log@GLIBC_2.2.5");\n#endif\n/' third_party/METIS/GKlib/gk_proto.h
sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver pow,pow@GLIBC_2.2.5");\n#endif\n/' third_party/METIS/libmetis/metislib.h

pip install setuptools ninja wheel
