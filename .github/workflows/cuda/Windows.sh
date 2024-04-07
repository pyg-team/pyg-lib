#!/bin/bash

case ${1} in
  cu121)
    CUDA_SHORT=12.1
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_531.14_windows.exe
    ;;
  cu118)
    CUDA_SHORT=11.8
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_522.06_windows.exe
    ;;
  cu117)
    CUDA_SHORT=11.7
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.1/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.1_516.94_windows.exe
    ;;
  cu116)
    CUDA_SHORT=11.3
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_465.89_win10.exe
    ;;
  cu115)
    CUDA_SHORT=11.3
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_465.89_win10.exe
    ;;
  cu113)
    CUDA_SHORT=11.3
    CUDA_URL=https://developer.download.nvidia.com/compute/cuda/${CUDA_SHORT}.0/local_installers
    CUDA_FILE=cuda_${CUDA_SHORT}.0_465.89_win10.exe
    ;;
  *)
    echo "Unrecognized CUDA_VERSION=${1}"
    exit 1
    ;;
esac

# Install NVIDIA drivers, see:
# https://github.com/pytorch/vision/blob/master/packaging/windows/internal/cuda_install.bat#L99-L102
# curl -k -L "https://ossci-windows.s3.us-east-1.amazonaws.com/builder/additional_dlls.zip" --output "/tmp/gpu_driver_dlls.zip"
# 7z x "/tmp/gpu_driver_dlls.zip" -o"/c/Windows/System32"

curl -k -L "${CUDA_URL}/${CUDA_FILE}" --output "${CUDA_FILE}"
echo ""
echo "Installing from ${CUDA_FILE}..."
PowerShell -Command "Start-Process -FilePath \"${CUDA_FILE}\" -ArgumentList \"-s nvcc_${CUDA_SHORT} cuobjdump_${CUDA_SHORT} nvprune_${CUDA_SHORT} cupti_${CUDA_SHORT} cublas_dev_${CUDA_SHORT} cudart_${CUDA_SHORT} cufft_dev_${CUDA_SHORT} curand_dev_${CUDA_SHORT} cusolver_dev_${CUDA_SHORT} cusparse_dev_${CUDA_SHORT} thrust_${CUDA_SHORT} npp_dev_${CUDA_SHORT} nvrtc_dev_${CUDA_SHORT} nvml_dev_${CUDA_SHORT}\" -Wait -NoNewWindow"
echo "Done!"
rm -f "${CUDA_FILE}"

echo Installing NvToolsExt...
curl -k -L https://ossci-windows.s3.us-east-1.amazonaws.com/builder/NvToolsExt.7z --output "/tmp/NvToolsExt.7z"
7z x "/tmp/NvToolsExt.7z" -o"/tmp/NvToolsExt"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
mkdir -p "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
echo "-------"
ls "/tmp/NvToolsExt"
echo "-------"
ls "/tmp/NvToolsExt/bin"
echo "-------"
ls "/tmp/NvToolsExt/bin/x64"
echo "-------"
ls "/tmp/NvToolsExt/include"
echo "-------"
ls "/tmp/NvToolsExt/lib"
echo "-------"
ls "/tmp/NvToolsExt/lib/x64"
cp -r /tmp/NvToolsExt/bin/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"
cp -r /tmp/NvToolsExt/include/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/include"
cp -r /tmp/NvToolsExt/lib/x64/* "/c/Program Files/NVIDIA Corporation/NvToolsExt/lib/x64"
export NVTOOLSEXT_PATH="/c/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64"

export CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;7.0;7.5;8.0;8.6"
