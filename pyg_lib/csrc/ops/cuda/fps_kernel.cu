#include "../fps.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define FPS_THREADS 256

// Explicit non-template comparison/min functions to avoid NVCC ambiguous
// operator overload errors from c10::SymInt (error #3343).
__device__ __forceinline__ bool scalar_gt(float a, float b) {
  return a > b;
}
__device__ __forceinline__ bool scalar_gt(double a, double b) {
  return a > b;
}
__device__ __forceinline__ bool scalar_lt(float a, float b) {
  return a < b;
}
__device__ __forceinline__ bool scalar_lt(double a, double b) {
  return a < b;
}
__device__ __forceinline__ float scalar_min(float a, float b) {
  return fminf(a, b);
}
__device__ __forceinline__ double scalar_min(double a, double b) {
  return fmin(a, b);
}

template <typename scalar_t>
__global__ void fps_cuda_kernel(const scalar_t* src,
                                const int64_t* ptr,
                                const int64_t* out_ptr,
                                const int64_t* start,
                                scalar_t* dist,
                                int64_t* out,
                                int64_t dim) {
  const int64_t thread_idx = threadIdx.x;
  const int64_t batch_idx = blockIdx.x;

  const int64_t start_idx = ptr[batch_idx];
  const int64_t end_idx = ptr[batch_idx + 1];

  __shared__ scalar_t best_dist[FPS_THREADS];
  __shared__ int64_t best_dist_idx[FPS_THREADS];

  if (thread_idx == 0) {
    out[out_ptr[batch_idx]] = start_idx + start[batch_idx];
  }

  for (int64_t m = out_ptr[batch_idx] + 1; m < out_ptr[batch_idx + 1]; m++) {
    __syncthreads();
    int64_t old = out[m - 1];

    scalar_t best = (scalar_t)-1.;
    int64_t best_idx = 0;

    for (int64_t n = start_idx + thread_idx; n < end_idx; n += FPS_THREADS) {
      scalar_t tmp, dd = (scalar_t)0.;
      for (int64_t d = 0; d < dim; d++) {
        tmp = src[dim * old + d] - src[dim * n + d];
        dd += tmp * tmp;
      }
      dd = scalar_min(dist[n], dd);
      dist[n] = dd;
      if (scalar_gt(dd, best)) {
        best = dd;
        best_idx = n;
      }
    }

    best_dist[thread_idx] = best;
    best_dist_idx[thread_idx] = best_idx;

    for (int64_t i = 1; i < FPS_THREADS; i *= 2) {
      __syncthreads();
      if ((thread_idx + i) < FPS_THREADS &&
          scalar_lt(best_dist[thread_idx], best_dist[thread_idx + i])) {
        best_dist[thread_idx] = best_dist[thread_idx + i];
        best_dist_idx[thread_idx] = best_dist_idx[thread_idx + i];
      }
    }

    __syncthreads();
    if (thread_idx == 0) {
      out[m] = best_dist_idx[0];
    }
  }
}

at::Tensor fps_cuda(const at::Tensor& src,
                    const at::Tensor& ptr,
                    double ratio,
                    bool random_start) {
  TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
  TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
  TORCH_CHECK(ptr.is_cuda(), "ptr must be a CUDA tensor");

  int64_t batch_size = ptr.numel() - 1;
  int64_t D = src.size(1);

  auto deg = ptr.narrow(0, 1, batch_size) - ptr.narrow(0, 0, batch_size);
  auto out_ptr = deg.to(at::kFloat) * ratio;
  out_ptr = out_ptr.ceil().to(at::kLong).cumsum(0);
  out_ptr = at::cat({at::zeros({1}, ptr.options()), out_ptr}, 0);

  at::Tensor start;
  if (random_start) {
    start = at::rand({batch_size}, src.options());
    start = (start * deg.to(at::kFloat)).to(at::kLong);
  } else {
    start = at::zeros({batch_size}, ptr.options());
  }

  auto dist = at::full({src.size(0)}, 5e4, src.options());

  int64_t out_total;
  cudaMemcpy(&out_total, out_ptr[-1].data_ptr<int64_t>(), sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto out = at::empty({out_total}, out_ptr.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "fps_cuda", [&] {
    fps_cuda_kernel<scalar_t><<<batch_size, FPS_THREADS, 0, stream>>>(
        src.data_ptr<scalar_t>(), ptr.data_ptr<int64_t>(),
        out_ptr.data_ptr<int64_t>(), start.data_ptr<int64_t>(),
        dist.data_ptr<scalar_t>(), out.data_ptr<int64_t>(), D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::fps"), TORCH_FN(fps_cuda));
}

}  // namespace ops
}  // namespace pyg
