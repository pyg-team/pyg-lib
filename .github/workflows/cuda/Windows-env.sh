#!/bin/bash

case ${1} in
  cu126)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.6/bin:${PATH}
    ;;
  cu124)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.4/bin:${PATH}
    ;;
  cu121)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v12.1/bin:${PATH}
    ;;
  cu118)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.8/bin:${PATH}
    ;;
  cu117)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.7/bin:${PATH}
    ;;
  cu116)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu115)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  cu113)
    export PATH=/c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v11.3/bin:${PATH}
    ;;
  *)
    ;;
esac

if [ "${1}" != "cpu" ] ; then
  export FORCE_CUDA=1
else
  export FORCE_CUDA=0
fi
