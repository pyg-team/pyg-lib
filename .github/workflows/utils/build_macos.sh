#!/bin/bash
set -ex

PYTHON_VERSION="${1:?Specify python version, e.g. 3.13}"
TORCH_VERSION="${2:?Specify torch version, e.g. 2.10.0}"
echo "PYTHON_VERSION: ${PYTHON_VERSION//./}"
echo "TORCH_VERSION: ${TORCH_VERSION}"

export CIBW_BUILD="cp${PYTHON_VERSION//./}-macosx_arm64"
# pyg-lib doesn't have torch as a dependency, so we need to explicitly install it when running tests.
if [[ "${TORCH_VERSION}" == "2.11.0" ]]; then
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
  export CIBW_BEFORE_TEST="pip install pytest && pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu"
else
  export CIBW_BEFORE_BUILD="pip install ninja wheel setuptools && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
  export CIBW_BEFORE_TEST="pip install pytest && pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu"
fi

rm -rf Testing libpyg.so build dist outputs  # for local testing
python -m cibuildwheel --output-dir dist
ls -ahl dist/
delocate-listdeps -vv --all dist/*.whl

unzip dist/*.whl -d debug/
otool -L debug/libpyg.so
otool -l debug/libpyg.so
