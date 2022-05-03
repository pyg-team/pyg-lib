#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace pyg {
namespace utils {

int threads() {
  const auto props = at::cuda::getCurrentDeviceProperties();
  return std::min(props->maxThreadsPerBlock, 1024);
}

int blocks(int numel) {
  const auto props = at::cuda::getCurrentDeviceProperties();
  const auto blocks_per_sm = props->maxThreadsPerMultiProcessor / 256;
  const auto max_blocks = props->multiProcessorCount * blocks_per_sm;
  const auto max_threads = threads();
  return std::min(max_blocks, (numel + max_threads - 1) / max_threads);
}

#define CUDA_1D_KERNEL_LOOP(scalar_t, i, n)                           \
  for (scalar_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

}  // namespace utils
}  // namespace pyg
