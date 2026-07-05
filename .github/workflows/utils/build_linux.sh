#!/bin/bash
set -ex

CUDA_VERSION="${1:?Specify cuda version, e.g. cpu, cu130}"
PYTHON_VERSION="${2:?Specify python version, e.g. 3.14}"
TORCH_VERSION="${3:?Specify torch version, e.g. 2.11.0}"
ARCH="${4:-x86_64}"
case "${ARCH}" in
  x86_64 | aarch64)
    ;;
  arm64)
    ARCH=aarch64
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    exit 1
    ;;
esac

echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "PYTHON_VERSION: ${PYTHON_VERSION//./}"
echo "TORCH_VERSION: ${TORCH_VERSION}"
echo "ARCH: ${ARCH}"

source ./.github/workflows/cuda/Linux-env.sh ${CUDA_VERSION}
echo "FORCE_CUDA: ${FORCE_CUDA}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

export CIBW_BUILD="cp${PYTHON_VERSION//./}-manylinux_${ARCH}"
export CIBW_ARCHS_LINUX="${ARCH}"
# pyg-lib doesn't have torch as a dependency, so we need to explicitly install it when running tests.
if [[ "${TORCH_VERSION}" == "2.13.0" ]]; then
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools MarkupSafe && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"
  export CIBW_BEFORE_TEST="pip install pytest MarkupSafe && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"
else
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools MarkupSafe && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
  export CIBW_BEFORE_TEST="pip install pytest MarkupSafe && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

if [[ "${CUDA_VERSION}" == "cu"* ]]; then
  # Use CUDA-pre-installed image
  export "CIBW_MANYLINUX_${ARCH^^}_IMAGE=ghcr.io/pyg-team/pyg-lib/manylinux_2_28_${ARCH}:${CUDA_VERSION}"
else
  export "CIBW_MANYLINUX_${ARCH^^}_IMAGE=quay.io/pypa/manylinux_2_28_${ARCH}"
fi

rm -rf Testing pyg_lib/libpyg.so build dist outputs  # for local testing
python -m cibuildwheel --output-dir dist
ls -ahl dist/
python -m auditwheel show dist/*.whl

unzip dist/*.whl -d debug/
ldd debug/pyg_lib/libpyg.so
readelf -d debug/pyg_lib/libpyg.so
