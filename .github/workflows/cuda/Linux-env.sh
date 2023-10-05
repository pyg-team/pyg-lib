#!/bin/bash

case ${1} in
  cu121)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-12.1/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.8/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.7/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu116)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu115)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu113)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
    ;;
  cu102)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
    ;;
  *)
    ;;
esac
