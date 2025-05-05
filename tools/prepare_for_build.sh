#!/bin/bash

# Usage:
# ./prepare_for_build.sh <torch_version> <cuda_version>
set -e
uname -a

TORCH_VERSION="${1:?Specify torch version, e.g. 2.6.0}"
CUDA_VERSION="${2:?Specify cuda version, e.g. cu126}"

# Install Python development packages
if [[ -d "/opt/conda" ]]; then
    # We're in a conda environment
    conda install -y python=${PYTHON_VERSION} python-dev
else
    # System Python
    dnf install -y python3.11-devel python3.11-libs
fi

# Install pthread development package
dnf install -y glibc-devel

# create virtual env:
pip install setuptools ninja wheel build --progress-bar off
pip install torch=="${TORCH_VERSION}" --extra-index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" --progress-bar off
python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\nCXX11 ABI: {torch.compiled_with_cxx11_abi()}')"

sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver log,log@GLIBC_2.2.5");\n#endif\n/' third_party/METIS/GKlib/gk_proto.h
sed -i '1s/^/#if defined(__linux__) \&\& defined(__x86_64__)\n__asm__(".symver pow,pow@GLIBC_2.2.5");\n#endif\n/' third_party/METIS/libmetis/metislib.h

######################

# Install CUDA 12.8
# aria2c -s 16 -x 16 https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-rhel8-12-8-local-12.8.0_570.86.10-1.x86_64.rpm
# rpm -i cuda-repo-rhel8-12-8-local-12.8.0_570.86.10-1.x86_64.rpm
# dnf clean all && dnf -y install cuda-toolkit-12-8

# Install CUDA 11.8
# Download if not found:
# if [ ! -f cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.x86_64.rpm ]; then
#   dnf install -y aria2
#   aria2c -s 16 -x 16 https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
# else
#   echo "cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.x86_64.rpm already exists"
# fi
# rpm -i cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
# dnf clean all && dnf -y module install nvidia-driver:latest-dkms && dnf -y install cuda

ls -al /usr/local/ | grep -i cuda

# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

nvcc --version
