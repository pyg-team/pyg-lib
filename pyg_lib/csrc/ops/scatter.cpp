#include "scatter.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

at::Tensor broadcast_index(at::Tensor index, const at::Tensor& src, int64_t dim) {
  if (dim < 0) {
    dim = src.dim() + dim;
  }
  
  if (index.dim() == 1) {
    for (int64_t i = 0; i < dim; i++) {
      index = index.unsqueeze(0);
    }
  }
  
  for (int64_t i = index.dim(); i < src.dim(); i++) {
    index = index.unsqueeze(-1);
  }
  
  index = index.expand(src.sizes());
  return index;
}

}  // namespace

PYG_API at::Tensor scatter_add(const at::Tensor& src,
                               const at::Tensor& index,
                               const int64_t dim,
                               const at::optional<at::Tensor> out,
                               const at::optional<int64_t> dim_size) {
  at::TensorArg src_arg{src, "src", 0};
  at::TensorArg index_arg{index, "index", 1};
  at::CheckedFrom c{"scatter_add"};

  at::checkAllDefined(c, {src_arg, index_arg});
  at::checkContiguous(c, src_arg);
  at::checkContiguous(c, index_arg);
  at::checkScalarType(c, index_arg, at::ScalarType::Long);

  // Broadcast index to match src dimensions
  auto broadcasted_index = broadcast_index(index, src, dim);

  at::Tensor result;
  if (out.has_value()) {
    result = out.value();
    at::TensorArg out_arg{result, "out", 2};
    at::checkSameType(c, src_arg, out_arg);
    at::checkContiguous(c, out_arg);
    
    // Validate dimensions match except for dim
    for (int64_t i = 0; i < result.dim(); i++) {
      if (i != dim) {
        at::checkSize(c, src_arg, i, out_arg->size(i));
      }
    }
  } else {
    // Create output tensor
    auto sizes = src.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (broadcasted_index.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = broadcasted_index.max().item<int64_t>() + 1;
    }
    result = at::zeros(sizes, src.options());
  }

  // Dispatch to optimized kernel
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("pyg::scatter_add_kernel", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&, 
                                        const int64_t, at::Tensor)>();
  return op.call(src, broadcasted_index, dim, result);
}

PYG_API at::Tensor scatter_mean(const at::Tensor& src,
                                const at::Tensor& index,
                                const int64_t dim,
                                const at::optional<at::Tensor> out,
                                const at::optional<int64_t> dim_size) {
  // First compute scatter_add
  auto sum_result = scatter_add(src, index, dim, out, dim_size);
  
  // Compute counts for mean
  auto ones = at::ones_like(src);
  auto counts = scatter_add(ones, index, dim, at::nullopt, sum_result.size(dim));
  
  // Avoid division by zero
  counts = counts.clamp_min(1);
  
  // Broadcast counts to match sum_result dimensions
  auto broadcasted_counts = broadcast_index(counts, sum_result, dim);
  
  // Compute mean
  if (sum_result.is_floating_point()) {
    sum_result.div_(broadcasted_counts);
  } else {
    sum_result = sum_result.div(broadcasted_counts, "floor");
  }
  
  return sum_result;
}

// Register high-level operators
TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def("scatter_add(Tensor src, Tensor index, int dim, Tensor? out=None, int? dim_size=None) -> Tensor");
  m.def("scatter_mean(Tensor src, Tensor index, int dim, Tensor? out=None, int? dim_size=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(pyg, CompositeExplicitAutograd, m) {
  m.impl("scatter_add", scatter_add);
  m.impl("scatter_mean", scatter_mean);
}

}  // namespace ops
}  // namespace pyg