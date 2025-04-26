#!/bin/bash

case ${1} in
  cu126)
    export CUDA_HOME=/usr/local/cuda-12.6
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export CUDA_HOME=/usr/local/cuda-12.4
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export CUDA_HOME=/usr/local/cuda-12.1
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export CUDA_HOME=/usr/local/cuda-11.8
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export CUDA_HOME=/usr/local/cuda-11.7
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export CUDA_HOME=/usr/local/cuda-11.6
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export CUDA_HOME=/usr/local/cuda-11.5
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export CUDA_HOME=/usr/local/cuda-11.3
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
    ;;
  cu102)
    export CUDA_HOME=/usr/local/cuda-10.2
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5"
    ;;
  *)
    export FORCE_CUDA=0
    ;;
esac

if [ "${1}" != "cpu" ] ; then
  export FORCE_CUDA=1
  export PATH=${CUDA_HOME}/bin:${PATH}
  export CUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME}
  export CUDA_NVCC_EXECUTABLE=${CUDA_HOME}/bin/nvcc
  export CUDA_INCLUDE_DIRS=${CUDA_HOME}/include
  export CUDA_CUDART_LIBRARY=${CUDA_HOME}/lib64/stubs/libcuda.so
fi
