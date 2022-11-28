#!/bin/bash

case ${1} in
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

export PATH=/c/Program\ Files\ \(x86\)/Microsoft\ Visual\ Studio/2017/BuildTools/MSBuild/15.0/Bin:$PATH
