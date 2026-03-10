#include "../graclus.h"
#include "utils.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define GRACLUS_THREADS 256
#define GRACLUS_BLOCKS(N) ((N) + GRACLUS_THREADS - 1) / GRACLUS_THREADS
#define BLUE_P 0.53406

__device__ bool done_d;

__global__ void init_done_kernel() {
  done_d = true;
}

__global__ void colorize_kernel(int64_t* out,
                                const float* bernoulli,
                                int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (out[idx] < 0) {
      out[idx] = (int64_t)bernoulli[idx] - 2;
      done_d = false;
    }
  }
}

bool colorize(at::Tensor out) {
  auto stream = at::cuda::getCurrentCUDAStream();
  init_done_kernel<<<1, 1, 0, stream>>>();

  auto numel = out.size(0);
  auto props = at::full({numel}, BLUE_P, out.options().dtype(at::kFloat));
  auto bernoulli = props.bernoulli();

  colorize_kernel<<<GRACLUS_BLOCKS(numel), GRACLUS_THREADS, 0, stream>>>(
      out.data_ptr<int64_t>(), bernoulli.data_ptr<float>(), numel);

  bool done_h;
  cudaMemcpyFromSymbol(&done_h, done_d, sizeof(done_h), 0,
                       cudaMemcpyDeviceToHost);
  return done_h;
}

__global__ void propose_kernel(int64_t* out,
                               int64_t* proposal,
                               const int64_t* rowptr,
                               const int64_t* col,
                               int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (out[idx] != -1)
      return;

    bool has_unmatched_neighbor = false;

    for (int64_t i = rowptr[idx]; i < rowptr[idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true;

      if (out[v] == -2) {
        proposal[idx] = v;
        break;
      }
    }

    if (!has_unmatched_neighbor)
      out[idx] = idx;
  }
}

template <typename scalar_t>
__global__ void weighted_propose_kernel(int64_t* out,
                                        int64_t* proposal,
                                        const int64_t* rowptr,
                                        const int64_t* col,
                                        const scalar_t* weight,
                                        int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (out[idx] != -1)
      return;

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = rowptr[idx]; i < rowptr[idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true;

      if (out[v] == -2 && scalar_ge(weight[i], w_max)) {
        v_max = v;
        w_max = weight[i];
      }
    }

    proposal[idx] = v_max;

    if (!has_unmatched_neighbor)
      out[idx] = idx;
  }
}

void propose(at::Tensor out,
             at::Tensor proposal,
             at::Tensor rowptr,
             at::Tensor col,
             const std::optional<at::Tensor>& weight) {
  auto stream = at::cuda::getCurrentCUDAStream();

  if (!weight.has_value()) {
    propose_kernel<<<GRACLUS_BLOCKS(out.numel()), GRACLUS_THREADS, 0, stream>>>(
        out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
        rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto w = weight.value();
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "_", [&] {
      weighted_propose_kernel<scalar_t>
          <<<GRACLUS_BLOCKS(out.numel()), GRACLUS_THREADS, 0, stream>>>(
              out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
              rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
              w.data_ptr<scalar_t>(), out.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
}

__global__ void respond_kernel(int64_t* out,
                               const int64_t* proposal,
                               const int64_t* rowptr,
                               const int64_t* col,
                               int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (out[idx] != -2)
      return;

    bool has_unmatched_neighbor = false;

    for (int64_t i = rowptr[idx]; i < rowptr[idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true;

      if (out[v] == -1 && proposal[v] == idx) {
        int64_t m = idx < v ? idx : v;
        out[idx] = m;
        out[v] = m;
        break;
      }
    }

    if (!has_unmatched_neighbor)
      out[idx] = idx;
  }
}

template <typename scalar_t>
__global__ void weighted_respond_kernel(int64_t* out,
                                        const int64_t* proposal,
                                        const int64_t* rowptr,
                                        const int64_t* col,
                                        const scalar_t* weight,
                                        int64_t numel) {
  const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) {
    if (out[idx] != -2)
      return;

    bool has_unmatched_neighbor = false;
    int64_t v_max = -1;
    scalar_t w_max = 0;

    for (int64_t i = rowptr[idx]; i < rowptr[idx + 1]; i++) {
      auto v = col[i];

      if (out[v] < 0)
        has_unmatched_neighbor = true;

      if (out[v] == -1 && proposal[v] == idx && scalar_ge(weight[i], w_max)) {
        v_max = v;
        w_max = weight[i];
      }
    }

    if (v_max >= 0) {
      int64_t m = idx < v_max ? idx : v_max;
      out[idx] = m;
      out[v_max] = m;
    }

    if (!has_unmatched_neighbor)
      out[idx] = idx;
  }
}

void respond(at::Tensor out,
             at::Tensor proposal,
             at::Tensor rowptr,
             at::Tensor col,
             const std::optional<at::Tensor>& weight) {
  auto stream = at::cuda::getCurrentCUDAStream();

  if (!weight.has_value()) {
    respond_kernel<<<GRACLUS_BLOCKS(out.numel()), GRACLUS_THREADS, 0, stream>>>(
        out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
        rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(), out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    auto w = weight.value();
    AT_DISPATCH_FLOATING_TYPES(w.scalar_type(), "_", [&] {
      weighted_respond_kernel<scalar_t>
          <<<GRACLUS_BLOCKS(out.numel()), GRACLUS_THREADS, 0, stream>>>(
              out.data_ptr<int64_t>(), proposal.data_ptr<int64_t>(),
              rowptr.data_ptr<int64_t>(), col.data_ptr<int64_t>(),
              w.data_ptr<scalar_t>(), out.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
}

at::Tensor graclus_cuda(const at::Tensor& rowptr,
                        const at::Tensor& col,
                        const std::optional<at::Tensor>& weight) {
  TORCH_CHECK(rowptr.is_cuda() && col.is_cuda(), "Inputs must be CUDA tensors");

  int64_t num_nodes = rowptr.numel() - 1;
  auto out = at::full({num_nodes}, -1, rowptr.options());
  auto proposal = at::full({num_nodes}, -1, rowptr.options());

  while (!colorize(out)) {
    propose(out, proposal, rowptr, col, weight);
    respond(out, proposal, rowptr, col, weight);
  }

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::graclus_cluster"), TORCH_FN(graclus_cuda));
}

}  // namespace ops
}  // namespace pyg
