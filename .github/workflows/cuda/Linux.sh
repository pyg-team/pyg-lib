#!/bin/bash

case ${1} in
  cu126)
    CUDA=12.6
    CUDA_PATCH=${CUDA}.0
    CUDA_ID=${CUDA_PATCH}-560.28.03
    ;;
  cu124)
    CUDA=12.4
    CUDA_PATCH=${CUDA}.1
    CUDA_ID=${CUDA_PATCH}-550.54.15
    ;;
  cu121)
    CUDA=12.1
    CUDA_PATCH=${CUDA}.1
    CUDA_ID=${CUDA_PATCH}-530.30.02
    ;;
  cu118)
    CUDA=11.8
    CUDA_PATCH=${CUDA}.0
    CUDA_ID=${CUDA_PATCH}-520.61.05
    ;;
  cu117)
    CUDA=11.7
    CUDA_PATCH=${CUDA}.1
    CUDA_ID=${CUDA_PATCH}-515.65.01
    ;;
  cpu)
    exit 0
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

if [ "$CIBUILDWHEEL" = "1" ]; then

  FILENAME=cuda_${CUDA_ID//-/_}_linux.run

  yum install -y wget
  wget --quiet "https://developer.download.nvidia.com/compute/cuda/${CUDA_PATCH}/local_installers/${FILENAME}"
  sh "${FILENAME}" --silent --toolkit
  rm -f "${FILENAME}"

else

  OS=ubuntu2204
  APT_KEY=${OS}-${CUDA/./-}-local
  URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_PATCH}/local_installers
  FILENAME=cuda-repo-${APT_KEY}_${CUDA_ID}-1_amd64.deb

  wget -nv https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
  sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget -nv "${URL}/${FILENAME}"
  sudo dpkg -i "${FILENAME}"
  sudo cp "/var/cuda-repo-${APT_KEY}/cuda-*-keyring.gpg /usr/share/keyrings/"
  sudo apt-get -qq update
  sudo apt install cuda-nvcc-${CUDA/./-} cuda-libraries-dev-${CUDA/./-} cuda-command-line-tools-${CUDA/./-}
  sudo apt clean
  sudo ln -sf /usr/local/cuda-${CUDA} /usr/local/cuda
  rm -f "${FILENAME}"

fi
