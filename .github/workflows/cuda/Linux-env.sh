#!/bin/bash

case ${1} in
  cu115)
    export CUDA_HOME=/usr/local/cuda-11.5
    ;;
  cu113)
    export CUDA_HOME=/usr/local/cuda-11.3
    ;;
  cu102)
    export CUDA_HOME=/usr/local/cuda-10.2
    ;;
  cpu)
    exit 0
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

export FORCE_CUDA=1
export PATH=${CUDA_HOME}/bin:${PATH}
