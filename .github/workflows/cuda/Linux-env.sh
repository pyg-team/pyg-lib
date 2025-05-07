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
  cu117)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.7/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu116)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu115)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu113)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu102)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5+PTX"
    ;;
  *)
    ;;
esac
