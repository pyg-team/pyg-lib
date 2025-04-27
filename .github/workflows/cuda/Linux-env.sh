#!/bin/bash

case ${1} in
  cu126)
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
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
