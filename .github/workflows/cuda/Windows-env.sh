#!/bin/bash

case ${1} in
  cu130)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v13.0/bin:${PATH}
    # CUDA 13.0 dropped sm_50 support. Without this, PyTorch's auto-detection
    # fails on CI (no GPU) and falls back to a default list that includes
    # compute_50, causing nvcc to error out.
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
    ;;
  cu129)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.9/bin:${PATH}
    ;;
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
  cu117)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    ;;
  cu116)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu115)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu113)
    export FORCE_CUDA=1
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  *)
    ;;
esac
