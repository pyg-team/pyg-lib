#!/bin/bash
set -ex

CUDA_VERSION="${1:?Specify cuda version, e.g. cpu, cu128}"
PYTHON_VERSION="${2:?Specify python version, e.g. 3.12}"
TORCH_VERSION="${3:?Specify torch version, e.g. 2.7.0}"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "PYTHON_VERSION: ${PYTHON_VERSION//./}"
echo "TORCH_VERSION: ${TORCH_VERSION}"

source ./.github/workflows/cuda/Linux-env.sh ${CUDA_VERSION}
echo "FORCE_CUDA: ${FORCE_CUDA}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

export CIBW_BUILD="cp${PYTHON_VERSION//./}-manylinux_x86_64"
export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
# pyg-lib doesn't have torch as a dependency, so we need to explicitly install it when running tests.
export CIBW_BEFORE_TEST="pip install pytest && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"

if [[ "${CUDA_VERSION}" == "cu"* ]]; then
  # Use CUDA-pre-installed image
  export CIBW_MANYLINUX_X86_64_IMAGE=akihironitta/manylinux:${CUDA_VERSION}
else
  export CIBW_MANYLINUX_X86_64_IMAGE=quay.io/pypa/manylinux_2_28_x86_64
fi

rm -rf Testing libpyg.so build dist outputs  # for local testing
python -m cibuildwheel --output-dir dist
ls -ahl dist/
python -m auditwheel show dist/*.whl

unzip dist/*.whl -d debug/
ldd debug/libpyg.so
readelf -d debug/libpyg.so
