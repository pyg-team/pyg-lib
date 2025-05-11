#!/bin/bash

case ${1} in
  cu128)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-12.8/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
    ;;
  cu126)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-12.6/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu124)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-12.4/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu121)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-12.1/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu118)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.8/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  *)
    ;;
esac
