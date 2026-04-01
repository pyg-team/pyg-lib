#!/bin/bash

CUDA_VERSION=${1}

if [[ "${CUDA_VERSION}" == "cu"* ]]; then
  export FORCE_CUDA=1
  # Extract version digits, e.g. cu130 -> 13.0
  CUDA_DIGITS="${CUDA_VERSION#cu}"
  CUDA_MAJOR="${CUDA_DIGITS:0:2}"
  CUDA_MINOR="${CUDA_DIGITS:2}"
  export PATH="/usr/local/cuda-${CUDA_MAJOR}.${CUDA_MINOR}/bin:${PATH}"
fi
