#!/bin/bash
set -ex

CUDA_VERSION="${1:?Specify cuda version, e.g. 12.9}"

# See https://developer.nvidia.com/cuda-toolkit-archive
if [ "$CUDA_VERSION" == "13.0" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-rhel8-13-0-local-13.0.0_580.65.06-1.x86_64.rpm
    PACKAGE_NAME=cuda-toolkit-13-0
elif [ "$CUDA_VERSION" == "12.9" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-rhel8-12-9-local-12.9.1_575.57.08-1.x86_64.rpm
    PACKAGE_NAME=cuda-toolkit-12-9
elif [ "$CUDA_VERSION" == "12.8" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-rhel8-12-8-local-12.8.1_570.124.06-1.x86_64.rpm
    PACKAGE_NAME=cuda-toolkit-12-8
elif [ "$CUDA_VERSION" == "12.6" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-rhel8-12-6-local-12.6.3_560.35.05-1.x86_64.rpm
    PACKAGE_NAME=cuda-toolkit-12-6
elif [ "$CUDA_VERSION" == "12.4" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-rhel8-12-4-local-12.4.1_550.54.15-1.x86_64.rpm
    PACKAGE_NAME=cuda-toolkit-12-4
elif [ "$CUDA_VERSION" == "12.1" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-rhel8-12-1-local-12.1.1_530.30.02-1.x86_64.rpm
    PACKAGE_NAME=cuda
elif [ "$CUDA_VERSION" == "11.8" ]; then
    URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-rhel8-11-8-local-11.8.0_520.61.05-1.x86_64.rpm
    PACKAGE_NAME=cuda
elif [ "$CUDA_VERSION" == "cpu" ]; then
    echo "No need to install CUDA"
else
    echo "Invalid CUDA version: $CUDA_VERSION"
    exit 1
fi

echo "Installing CUDA version: $CUDA_VERSION"
# aria2c $URL -o /tmp/cuda.rpm
aria2c -x 16 -s 16 -k 1M --file-allocation=none "$URL" -o /tmp/cuda.rpm
rpm -i /tmp/cuda.rpm
dnf clean all
dnf --setopt=install_weak_deps=False -y install $PACKAGE_NAME
