#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define THREADS 1024
#define CDIV(N, M) ((N) + (M)-1) / (M)

enum FnType { ADD, SUB, MUL, DIV };
const std::map<std::string, FnType> to_fn_type = {
    {"add", ADD},
    {"sub", SUB},
    {"mul", MUL},
    {"div", DIV},
};

template <typename scalar_t>
__global__ void sampled_op_kernel_impl(const scalar_t* __restrict__ left,
                                       const scalar_t* __restrict__ right,
                                       scalar_t* __restrict__ out,
                                       const int64_t* __restrict__ left_index,
                                       const int64_t* __restrict__ right_index,
                                       const FnType fn_type,
                                       const bool has_left_index,
                                       const bool has_right_index,
                                       const int64_t num_feats,
                                       const int64_t numel) {
  int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_idx >= numel)
    return;

  int64_t i = thread_idx / num_feats;
  if (has_left_index) {
    i = left_index[i];
  }

  int64_t j = thread_idx / num_feats;
  if (has_right_index) {
    j = right_index[j];
  }

  int64_t k = thread_idx % num_feats;

  scalar_t a = left[i * num_feats + k];
  scalar_t b = right[j * num_feats + k];

  scalar_t c;
  if (fn_type == ADD) {
    c = a + b;
  } else if (fn_type == SUB) {
    c = a - b;
  } else if (fn_type == MUL) {
    c = a * b;
  } else if (fn_type == DIV) {
    c = a / b;
  }

  out[thread_idx] = c;
}

at::Tensor sampled_op_kernel(const at::Tensor& left,
                             const at::Tensor& right,
                             const at::optional<at::Tensor> left_index,
                             const at::optional<at::Tensor> right_index,
                             const std::string fn) {
  auto dim_size = left.size(0);
  if (left_index.has_value() && !right_index.has_value()) {
    dim_size = right.size(0);
  } else if (left_index.has_value() && right_index.has_value()) {
    dim_size = left_index.value().size(0);
  }
  const auto num_feats = left.size(1);
  const auto numel = dim_size * num_feats;

  const auto out = left.new_empty({dim_size, num_feats});

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_ALL_TYPES(left.scalar_type(), "sampled_kernel_impl", [&] {
    const auto left_data = left.data_ptr<scalar_t>();
    const auto right_data = right.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    int64_t* left_index_data = NULL;
    if (left_index.has_value()) {
      left_index_data = left_index.value().data_ptr<int64_t>();
    }
    int64_t* right_index_data = NULL;
    if (right_index.has_value()) {
      right_index_data = right_index.value().data_ptr<int64_t>();
    }

    sampled_op_kernel_impl<scalar_t>
        <<<CDIV(numel, THREADS), THREADS, 0, stream>>>(
            left_data, right_data, out_data, left_index_data, right_index_data,
            to_fn_type.at(fn), left_index.has_value(), right_index.has_value(),
            num_feats, numel);
  });
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::sampled_op"), TORCH_FN(sampled_op_kernel));
}

}  // namespace ops
}  // namespace pyg
