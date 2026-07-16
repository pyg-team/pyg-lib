#!/bin/bash
set -ex

CUDA_VERSION="${1:?Specify cuda version, e.g. cpu, cu130}"
TORCH_VERSION="${2:?Specify torch version, e.g. 2.13.0}"

echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "TORCH_VERSION: ${TORCH_VERSION}"

source ./.github/workflows/cuda/Windows-env.sh "${CUDA_VERSION}"
echo "FORCE_CUDA: ${FORCE_CUDA:-0}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST:-}"

# pyg-lib doesn't have torch as a dependency, so we need to explicitly install
# it when building and testing wheels.
if [[ "${TORCH_VERSION}" == "2.14.0" ]]; then
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools packaging && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"
  export CIBW_BEFORE_TEST="pip install pytest packaging && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/${CUDA_VERSION}"
else
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
  export CIBW_BEFORE_TEST="pip install pytest && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

rm -rf Testing pyg_lib/libpyg.pyd build dist outputs  # for local testing
python -m cibuildwheel --output-dir dist
ls -ahl dist/
WHEELS=(dist/*.whl)
if [[ ${#WHEELS[@]} -ne 1 || ${WHEELS[0]} != *-cp310-abi3-* ]]; then
  echo "Expected exactly one cp310-abi3 wheel, found: ${WHEELS[*]}"
  exit 1
fi
