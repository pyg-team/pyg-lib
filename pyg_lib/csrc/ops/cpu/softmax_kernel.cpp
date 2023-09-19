#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace pyg {
namespace ops {

namespace {

void check_arguments(const at::Tensor& values,
                     const at::optional<at::Tensor> ptr,
                     const int64_t dim) {
  TORCH_CHECK(values.dim() == 2, "Only 2D `values` are currently supported.");
  TORCH_CHECK(ptr.has_value(),
              "`ptr` is currently only way to specify groups.");
  TORCH_CHECK(dim == 0, "Only first dimension is currently supported.");
}

void check_arguments(const at::Tensor& values,
                     const at::Tensor& values_grad,
                     const at::optional<at::Tensor> ptr,
                     const int64_t dim) {
  TORCH_CHECK(values_grad.dim() == 2,
              "Only 2D `values_grad` is currently supported.");
  check_arguments(values, ptr, dim);
}

std::vector<int64_t> create_per_thread_groups(const int64_t* groups_ptr,
                                              const int64_t n_groups,
                                              const int64_t n_rows) {
  std::vector<int64_t> new_groups = {0};
  const auto avg_work_per_thread = at::divup(n_rows, at::get_num_threads());
  int64_t cur_work = 0;
  for (int64_t i = 0; i < n_groups; ++i) {
    cur_work += groups_ptr[i + 1] - groups_ptr[i];
    if (cur_work >= avg_work_per_thread) {
      new_groups.push_back(i + 1);
      cur_work = 0;
    }
  }
  new_groups.push_back(n_groups);

  return new_groups;
}

at::Tensor softmax_forward_kernel_ptr_dim0_impl(const at::Tensor& src,
                                                const at::Tensor& groups) {
  auto out = at::zeros_like(src);

  AT_DISPATCH_FLOATING_TYPES(
      src.scalar_type(), "softmax_forward_kernel_ptr_dim0_impl", [&] {
        const auto n_groups = groups.size(0) - 1;
        const auto n_heads = src.size(-1);
        auto max = at::full({n_groups, n_heads},
                            std::numeric_limits<scalar_t>::lowest());
        auto sum = at::zeros({n_groups, n_heads});

        const auto src_ptr = src.data_ptr<scalar_t>();
        const auto groups_ptr = groups.data_ptr<int64_t>();
        auto out_ptr = out.data_ptr<scalar_t>();
        auto max_ptr = max.data_ptr<scalar_t>();
        auto sum_ptr = sum.data_ptr<scalar_t>();
        const auto new_groups = std::move(
            create_per_thread_groups(groups_ptr, n_groups, src.size(0)));

        at::parallel_for(
            0, new_groups.size() - 1, 1, [&](int64_t beg, int64_t end) {
              // each thread may cover several groups
              for (auto group_id = new_groups[beg]; group_id < new_groups[end];
                   ++group_id) {
                const auto row_beg = groups_ptr[group_id];
                const auto row_end = groups_ptr[group_id + 1];
                const auto rows_in_group = row_end - row_beg;
                const auto inout_offset = row_beg * n_heads;
                const auto aux_offset = group_id * n_heads;
                const auto src_beg_ptr = src_ptr + inout_offset;
                auto out_beg_ptr = out_ptr + inout_offset;
                auto max_beg_ptr = max_ptr + aux_offset;
                auto sum_beg_ptr = sum_ptr + aux_offset;

                if (rows_in_group == 1) {
                  std::fill(out_beg_ptr, out_beg_ptr + n_heads,
                            static_cast<scalar_t>(1.0));
                } else {
                  // calculate max
                  for (int64_t i = 0; i < rows_in_group * n_heads; ++i) {
                    const auto aux_id = i % n_heads;
                    max_beg_ptr[aux_id] =
                        std::max(max_beg_ptr[aux_id], src_beg_ptr[i]);
                  }
                  // calculate sum
                  for (int64_t i = 0; i < rows_in_group * n_heads; ++i) {
                    const auto aux_id = i % n_heads;
                    const auto value =
                        std::exp(src_beg_ptr[i] - max_beg_ptr[aux_id]);
                    sum_beg_ptr[aux_id] += value;
                    out_beg_ptr[i] = value;
                  }
                  // unify
                  for (int64_t i = 0; i < rows_in_group * n_heads; ++i) {
                    const auto aux_id = i % n_heads;
                    out_beg_ptr[i] /= sum_beg_ptr[aux_id];
                  }
                }
              }
            });
      });

  return out;
}

at::Tensor softmax_forward_kernel(const at::Tensor& src,
                                  const at::optional<at::Tensor> index,
                                  const at::optional<at::Tensor> ptr,
                                  const at::optional<int64_t> num_nodes,
                                  const int64_t dim) {
  check_arguments(src, ptr, dim);

  return softmax_forward_kernel_ptr_dim0_impl(src, ptr.value());
}

at::Tensor softmax_backward_kernel_ptr_dim0_impl(const at::Tensor& out,
                                                 const at::Tensor& out_grad,
                                                 const at::Tensor& ptr) {
  auto in_grad = at::zeros_like(out);

  // TODO: not implemented yet

  return in_grad;
}

at::Tensor softmax_backward_kernel(const at::Tensor& out,
                                   const at::Tensor& out_grad,
                                   const at::optional<at::Tensor> index,
                                   const at::optional<at::Tensor> ptr,
                                   const at::optional<int64_t> num_nodes,
                                   const int64_t dim) {
  check_arguments(out, out_grad, ptr, dim);

  return softmax_backward_kernel_ptr_dim0_impl(out, out_grad, ptr.value());
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::softmax_forward"),
         TORCH_FN(softmax_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::softmax_backward"),
         TORCH_FN(softmax_backward_kernel));
}

}  // namespace ops
}  // namespace pyg
