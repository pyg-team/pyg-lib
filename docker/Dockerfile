FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.10-2 AS builder
ARG CUDA_VERSION=12.8

COPY install_cuda.sh /tmp/
RUN bash /tmp/install_cuda.sh ${CUDA_VERSION}

FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.10-2
ARG CUDA_VERSION=12.8

COPY --from=builder /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda-${CUDA_VERSION}

# The base image comes with gcc 14. However, NVIDIA officially supports:
# GCC<=14 for CUDA 12.9
# GCC<=14 for CUDA 12.8
# GCC<=13 for CUDA 12.6
# GCC<=13 for CUDA 12.4
# GCC<=12 for CUDA 12.1
# GCC<=11 for CUDA 11.8
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
RUN dnf install -y gcc-toolset-11-toolchain
# Hmmm this shouldn't be necessary:
RUN dnf install -y python3.11-devel

ENV PATH=/opt/rh/gcc-toolset-11/root/usr/bin:/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-11/root/usr/lib64:/opt/rh/gcc-toolset-11/root/usr/lib:/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}
