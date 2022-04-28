#!/bin/bash

OS=ubuntu1804

wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600

CUDA=11.5
APT_KEY=${OS}-${CUDA/./-}-local
FILENAME=cuda-repo-${APT_KEY}_${CUDA}.2-495.29.05-1_amd64.deb
URL=https://developer.download.nvidia.com/compute/cuda/${CUDA}.2/local_installers

wget -nv ${URL}/${FILENAME}
sudo dpkg -i ${FILENAME}
sudo apt-key add /var/cuda-repo-${APT_KEY}/7fa2af80.pub

sudo apt-get -qq update
sudo apt install cuda-nvcc-${CUDA/./-} cuda-libraries-dev-${CUDA/./-} cuda-command-line-tools-${CUDA/./-}
sudo apt clean

rm -f ${FILENAME}
