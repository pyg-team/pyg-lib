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

std::vector<int64_t> create_per_thread_groups(const int64_t* groups_ptr,
                                              const int64_t n_groups,
                                              const int64_t dim_size) {
  std::vector<int64_t> new_groups = {0};
  const auto avg_work_per_thread = at::divup(dim_size, at::get_num_threads());
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

std::pair<std::vector<int64_t>, std::vector<int64_t>>
precompute_data_access_patterns(const int64_t outer_range,
                                const int64_t inner_range,
                                const int64_t global_dim_size,
                                const int64_t dim_stride) {
  std::vector<int64_t> data_ids(outer_range * inner_range);
  std::vector<int64_t> aux_ids(outer_range * inner_range);
  for (int64_t i = 0; i < outer_range; ++i) {
    const auto contiguous_offset = i * global_dim_size * dim_stride;
    for (int64_t j = 0; j < inner_range; ++j) {
      const auto k = i * inner_range + j;
      const auto data_id = j + contiguous_offset;
      const auto aux_id = k % dim_stride + (dim_stride * (k / inner_range));
      data_ids[k] = data_id;
      aux_ids[k] = aux_id;
    }
  }

  return {std::move(data_ids), std::move(aux_ids)};
}

at::Tensor softmax_csr_forward_kernel_impl(const at::Tensor& src,
                                           const at::Tensor& groups,
                                           const int64_t dim) {
  auto out = at::zeros_like(src);

  AT_DISPATCH_FLOATING_TYPES(
      src.scalar_type(), "softmax_csr_forward_kernel_impl", [&] {
        const auto n_groups = groups.size(0) - 1;
        const auto n_heads = src.numel() / src.size(dim);
        auto max = at::full({n_groups, n_heads},
                            std::numeric_limits<scalar_t>::lowest());
        auto sum = at::zeros({n_groups, n_heads});
        const auto groups_ptr = groups.data_ptr<int64_t>();
        const auto src_base_ptr = src.data_ptr<scalar_t>();
        auto out_base_ptr = out.data_ptr<scalar_t>();
        auto max_base_ptr = max.data_ptr<scalar_t>();
        auto sum_base_ptr = sum.data_ptr<scalar_t>();
        const auto global_dim_size = src.size(dim);
        const auto new_groups = std::move(
            create_per_thread_groups(groups_ptr, n_groups, global_dim_size));

        at::parallel_for(
            0, new_groups.size() - 1, 1, [&](int64_t beg, int64_t end) {
              // each thread may cover several groups
              for (auto group_id = new_groups[beg]; group_id < new_groups[end];
                   ++group_id) {
                const auto dim_beg = groups_ptr[group_id];
                const auto dim_end = groups_ptr[group_id + 1];
                const auto local_dim_size = dim_end - dim_beg;
                const auto dim_stride = src.stride(dim);
                // outer_range says how many data jumps we need to make
                const auto outer_range = [&src, dim]() {
                  int64_t range = 1;
                  for (int64_t i = 0; i < dim; ++i)
                    range *= src.size(i);
                  return range;
                }();
                // inner_range says how many contiguous elements we can visit
                const auto inner_range = local_dim_size * dim_stride;
                const auto inout_offset = dim_beg * dim_stride;
                const auto aux_offset = group_id * n_heads;

                const auto src_ptr = src_base_ptr + inout_offset;
                auto out_ptr = out_base_ptr + inout_offset;
                auto max_ptr = max_base_ptr + aux_offset;
                auto sum_ptr = sum_base_ptr + aux_offset;

                const auto indices = precompute_data_access_patterns(
                    outer_range, inner_range, global_dim_size, dim_stride);
                const auto& data_ids = indices.first;
                const auto& aux_ids = indices.second;

                if (local_dim_size == 1) {
                  for (int64_t i = 0; i < outer_range; ++i) {
                    const auto k = i * inner_range;
                    const auto data_id = data_ids[k];
                    std::fill(out_ptr + data_id,
                              out_ptr + data_id + inner_range,
                              static_cast<scalar_t>(1.0));
                  }
                } else {
                  // calculate max
                  for (int64_t i = 0; i < outer_range; ++i) {
                    for (int64_t j = 0; j < inner_range; ++j) {
                      const auto k = i * inner_range + j;
                      const auto data_id = data_ids[k];
                      const auto aux_id = aux_ids[k];
                      max_ptr[aux_id] =
                          std::max(max_ptr[aux_id], src_ptr[data_id]);
                    }
                  }

                  // calculate sum
                  for (int64_t i = 0; i < outer_range; ++i) {
                    for (int64_t j = 0; j < inner_range; ++j) {
                      const auto k = i * inner_range + j;
                      const auto data_id = data_ids[k];
                      const auto aux_id = aux_ids[k];
                      const auto value =
                          std::exp(src_ptr[data_id] - max_ptr[aux_id]);
                      sum_ptr[aux_id] += value;
                      out_ptr[data_id] = value;
                    }
                  }

                  // unify
                  for (int64_t i = 0; i < outer_range; ++i) {
                    for (int64_t j = 0; j < inner_range; ++j) {
                      const auto k = i * inner_range + j;
                      const auto data_id = data_ids[k];
                      const auto aux_id = aux_ids[k];
                      out_ptr[data_id] /= sum_ptr[aux_id];
                    }
                  }
                }
              }
            });
      });

  return out;
}

at::Tensor softmax_csr_backward_kernel_impl(const at::Tensor& out,
                                            const at::Tensor& out_grad,
                                            const at::Tensor& groups,
                                            const int64_t dim) {
  auto in_grad = at::zeros_like(out);

  AT_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "softmax_csr_backward_kernel_impl", [&] {
        const auto n_groups = groups.size(0) - 1;
        const auto n_heads = out.numel() / out.size(dim);
        auto sum = at::zeros({n_groups, n_heads});
        const auto groups_ptr = groups.data_ptr<int64_t>();
        const auto out_base_ptr = out.data_ptr<scalar_t>();
        const auto out_grad_base_ptr = out_grad.data_ptr<scalar_t>();
        auto in_grad_base_ptr = in_grad.data_ptr<scalar_t>();
        auto sum_base_ptr = sum.data_ptr<scalar_t>();
        const auto global_dim_size = out.size(dim);
        const auto new_groups = std::move(
            create_per_thread_groups(groups_ptr, n_groups, global_dim_size));

        at::parallel_for(
            0, new_groups.size() - 1, 1, [&](int64_t beg, int64_t end) {
              for (auto group_id = new_groups[beg]; group_id < new_groups[end];
                   ++group_id) {
                const auto dim_beg = groups_ptr[group_id];
                const auto dim_end = groups_ptr[group_id + 1];
                const auto local_dim_size = dim_end - dim_beg;
                const auto dim_stride = out.stride(dim);
                // outer_range says how many data jumps we need to make
                const auto outer_range = [&out, dim]() {
                  int64_t range = 1;
                  for (int64_t i = 0; i < dim; ++i)
                    range *= out.size(i);
                  return range;
                }();
                // inner_range says how many contiguous elements we can visit
                const auto inner_range = local_dim_size * dim_stride;
                const auto inout_offset = dim_beg * dim_stride;
                const auto sum_offset = group_id * n_heads;

                const auto out_ptr = out_base_ptr + inout_offset;
                const auto out_grad_ptr = out_grad_base_ptr + inout_offset;
                auto in_grad_ptr = in_grad_base_ptr + inout_offset;
                auto sum_ptr = sum_base_ptr + sum_offset;

                const auto indices = precompute_data_access_patterns(
                    outer_range, inner_range, global_dim_size, dim_stride);
                const auto& data_ids = indices.first;
                const auto& aux_ids = indices.second;

                // calculate sum of out * out_grad
                for (int64_t i = 0; i < outer_range; ++i) {
                  for (int64_t j = 0; j < inner_range; ++j) {
                    const auto k = i * inner_range + j;
                    const auto data_id = data_ids[k];
                    const auto aux_id = aux_ids[k];
                    sum_ptr[aux_id] += out_ptr[data_id] * out_grad_ptr[data_id];
                  }
                }

                // calculate out * (out_grad - sum)
                for (int64_t i = 0; i < outer_range; ++i) {
                  for (int64_t j = 0; j < inner_range; ++j) {
                    const auto k = i * inner_range + j;
                    const auto data_id = data_ids[k];
                    const auto aux_id = aux_ids[k];
                    in_grad_ptr[data_id] =
                        out_ptr[data_id] *
                        (out_grad_ptr[data_id] - sum_ptr[aux_id]);
                  }
                }
              }
            });
      });

  return in_grad;
}

at::Tensor softmax_csr_forward_kernel(const at::Tensor& src,
                                      const at::Tensor& ptr,
                                      const int64_t dim) {
  return softmax_csr_forward_kernel_impl(src, ptr, dim);
}

at::Tensor softmax_csr_backward_kernel(const at::Tensor& out,
                                       const at::Tensor& out_grad,
                                       const at::Tensor& ptr,
                                       const int64_t dim) {
  return softmax_csr_backward_kernel_impl(out, out_grad, ptr, dim);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::softmax_csr"),
         TORCH_FN(softmax_csr_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::softmax_csr_backward"),
         TORCH_FN(softmax_csr_backward_kernel));
}

}  // namespace ops
}  // namespace pyg
