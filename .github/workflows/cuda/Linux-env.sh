#!/bin/bash

case ${1} in
  cu128)
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
    ;;
  cu126)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu124)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu121)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu118)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6;9.0+PTX"
    ;;
  cu117)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu116)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu115)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu113)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5;8.0;8.6+PTX"
    ;;
  cu102)
    export TORCH_CUDA_ARCH_LIST="6.0;7.0;7.5+PTX"
    ;;
  *)
    ;;
esac

if [ "${1}" != "cpu" ] ; then
  export FORCE_CUDA=1
  export PATH=/usr/local/cuda/bin:${PATH}
else
  export FORCE_CUDA=0
fi
