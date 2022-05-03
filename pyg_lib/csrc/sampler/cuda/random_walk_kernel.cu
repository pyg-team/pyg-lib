#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace sampler {

namespace {

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

template <typename scalar_t>
__global__ void random_walk_kernel_impl(
    const scalar_t* __restrict__ rowptr_data,
    const scalar_t* __restrict__ col_data,
    const scalar_t* __restrict__ seed_data,
    const float* __restrict__ rand_data,
    scalar_t* __restrict__ out_data,
    int64_t num_seeds,
    int64_t walk_length) {
  CUDA_1D_KERNEL_LOOP(scalar_t, i, num_seeds) {
    auto v = seed_data[i];
    out_data[i] = v;

    for (scalar_t j = 0; j < walk_length; ++j) {
      auto row_start = rowptr_data[v], row_end = rowptr_data[v + 1];
      if (row_end - row_start > 0) {
        auto rand = rand_data[j * num_seeds + i];
        v = col_data[row_start + scalar_t(rand * (row_end - row_start))];
      }
      // For isolated nodes, this will add a fake self-loop.
      // This does not do any harm when used in within a `node2vec` model.
      out_data[(j + 1) * num_seeds + i] = v;
    }
  }
}

at::Tensor random_walk_kernel(const at::Tensor& rowptr,
                              const at::Tensor& col,
                              const at::Tensor& seed,
                              int64_t walk_length,
                              double p,
                              double q) {
  TORCH_CHECK(rowptr.is_cuda(), "'rowptr' must be a CUDA tensor");
  TORCH_CHECK(col.is_cuda(), "'col' must be a CUDA tensor");
  TORCH_CHECK(seed.is_cuda(), "'seed' must be a CUDA tensor");
  TORCH_CHECK(p == 1 && q == 1, "Uniform sampling required for now");

  const auto stream = at::cuda::getCurrentCUDAStream();
  // Ensure contiguous access by transposing `out` matrix:
  const auto out = rowptr.new_empty({walk_length + 1, seed.size(0)});
  const auto rand =
      at::rand({walk_length, seed.size(0)}, seed.options().dtype(at::kFloat));

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "random_walk_kernel", [&] {
    const auto rowptr_data = rowptr.data_ptr<scalar_t>();
    const auto col_data = col.data_ptr<scalar_t>();
    const auto seed_data = seed.data_ptr<scalar_t>();
    const auto rand_data = rand.data_ptr<float>();
    auto out_data = out.data_ptr<scalar_t>();

    random_walk_kernel_impl<<<blocks(seed.size(0)), threads(), 0, stream>>>(
        rowptr_data, col_data, seed_data, rand_data, out_data, seed.size(0),
        walk_length);
  });

  return out.t().contiguous();
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::random_walk"),
         TORCH_FN(random_walk_kernel));
}

}  // namespace sampler
}  // namespace pyg
