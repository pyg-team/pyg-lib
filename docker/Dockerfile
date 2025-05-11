FROM quay.io/pypa/manylinux_2_28_x86_64:2025.05.10-2 AS builder

ARG CUDA_VERSION=cu128
ARG TORCH_VERSION=2.7.0

# Copy local files
COPY install_cuda.sh /tmp/

# Install CUDA
RUN bash /tmp/install_cuda.sh ${CUDA_VERSION}

# Install Python build dependencies
RUN /opt/python/cp311-cp311/bin/python -m pip install --progress-bar off -q setuptools ninja wheel build

# Install PyTorch
RUN /opt/python/cp311-cp311/bin/python -m pip install --progress-bar off torch==$TORCH_VERSION --index-url https://download.pytorch.org/whl/$CUDA_VERSION

# Final stage
FROM quay.io/pypa/manylinux_2_28_x86_64

# Copy CUDA and Python installations from builder
COPY --from=builder /usr/local/cuda-12.8 /usr/local/cuda-12.8
COPY --from=builder /opt/python/cp311-cp311 /opt/python/cp311-cp311

RUN dnf install -y python3.11-devel python3.11-libs

# Set environment variables
ENV PATH=/usr/local/cuda-12.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}

# Verify installation
RUN /opt/python/cp311-cp311/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}\nCUDA: {torch.version.cuda}\nCXX11 ABI: {torch.compiled_with_cxx11_abi()}')" && \
    echo "CUDA_HOME: $CUDA_HOME" && \
    echo "PATH: $PATH" && \
    echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
