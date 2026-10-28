#!/bin/bash
set -ex

TORCH_VERSION="${1:?Specify torch version, e.g. 2.15.0}"
echo "TORCH_VERSION: ${TORCH_VERSION}"

# PyTorch 2.15 and newer require Python 3.11; retain 3.10 for older versions.
IFS=. read -r TORCH_MAJOR TORCH_MINOR TORCH_PATCH <<< "${TORCH_VERSION}"
if (( TORCH_MAJOR > 2 || (TORCH_MAJOR == 2 && TORCH_MINOR >= 15) )); then
  export CIBW_BUILD='cp3{11,12,13,14}-*'
fi

# pyg-lib doesn't have torch as a dependency, so we need to explicitly install it when running tests.
if [[ "${TORCH_VERSION}" == "2.16.0" ]]; then
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
  export CIBW_BEFORE_TEST="pip install pytest && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
else
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
  export CIBW_BEFORE_TEST="pip install pytest && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
fi

rm -rf Testing pyg_lib/libpyg.so build debug dist outputs  # for local testing
python -m cibuildwheel --output-dir dist
ls -ahl dist/
WHEELS=(dist/*.whl)
if [[ ${#WHEELS[@]} -ne 1 || ${WHEELS[0]} != *-cp310-abi3-* ]]; then
  echo "Expected exactly one cp310-abi3 wheel, found: ${WHEELS[*]}"
  exit 1
fi
delocate-listdeps -vv --all dist/*.whl

unzip dist/*.whl -d debug/
otool -L debug/pyg_lib/libpyg.so
otool -l debug/pyg_lib/libpyg.so
