#!/bin/bash

case ${1} in
  cu126)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.6/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu124)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu121)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.1/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu118)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6;9.0"
    ;;
  cu117)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    export TORCH_CUDA_ARCH_LIST="6.0+PTX;7.0;7.5;8.0;8.6"
    ;;
  *)
    ;;
esac

if [ "${1}" != "cpu" ] ; then
  export FORCE_CUDA=1
else
  export FORCE_CUDA=0
fi
