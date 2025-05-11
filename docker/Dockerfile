FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.10-2 AS builder

ARG CUDA_VERSION=12.8

# Copy local files
COPY install_cuda.sh /tmp/

# Install CUDA
RUN bash /tmp/install_cuda.sh ${CUDA_VERSION}

# Final stage
FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.10-2
ARG CUDA_VERSION=12.8

# Copy CUDA and Python installations from builder
COPY --from=builder /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda-${CUDA_VERSION}

RUN dnf install -y python3.11-devel  # Hmmm this shouldn't be necessary

ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:${LD_LIBRARY_PATH}
