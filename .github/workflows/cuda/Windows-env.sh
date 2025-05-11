#!/bin/bash

case ${1} in
  cu128)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.8/bin:${PATH}
    ;;
  cu126)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.6/bin:${PATH}
    ;;
  cu124)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4/bin:${PATH}
    ;;
  cu121)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.1/bin:${PATH}
    ;;
  cu118)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8/bin:${PATH}
    ;;
  *)
    ;;
esac
