#!/bin/bash
set -ex

TORCH_VERSION="${1:?Specify PyTorch version, e.g. 2.5.0}"
DEST_DIR="${2:?Specify destination directory, e.g. dist}"
WHEEL="${3:?Specify wheel file, e.g. dist/pyg_lib-0.1.0-cp311-cp311-linux_x86_64.whl}"

# PyTorch 2.5.0 is the first version that supports manylinux 2.28.
if [ "${TORCH_VERSION}" = "$(echo -e "${TORCH_VERSION}\n2.5.0" | sort -V | tail -n1)" ]; then
  auditwheel repair -w ${DEST_DIR} ${WHEEL} --plat manylinux_2_28_x86_64 \
    --exclude libcudart.so* \
    --exclude libcublasLt.so* \
    --exclude libcublas.so* \
    --exclude libcublas_api.so* \
    --exclude libcudnn.so* \
    --exclude libcusparse.so* \
    --exclude libcufft.so* \
    --exclude libcupti.so*
else
  mv ${WHEEL} ${DEST_DIR}
fi
