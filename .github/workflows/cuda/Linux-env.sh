#!/bin/bash

case ${1} in
  cu116)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.6/bin:${PATH}
    ;;
  cu115)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.5/bin:${PATH}
    ;;
  cu113)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-11.3/bin:${PATH}
    ;;
  cu102)
    export FORCE_CUDA=1
    export PATH=/usr/local/cuda-10.2/bin:${PATH}
    ;;
  *)
    ;;
esac
