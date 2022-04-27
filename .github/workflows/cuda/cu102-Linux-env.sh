#!/bin/bash

export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PATH=${CUDA_HOME}/bin:${PATH}

export FORCE_CUDA=1
export CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5"
