#include "../spmm.h"

#include <torch/autograd.h>
#include <torch/library.h>

#include <vector>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

enum class Reduction { Sum, Mean, Min, Max };

at::Tensor row_from_rowptr(const at::Tensor& rowptr, const at::Tensor& col) {
  auto rowptr_c = rowptr.contiguous();
  auto row = at::empty({col.numel()}, col.options());

  const auto* rowptr_data = rowptr_c.data_ptr<int64_t>();
  auto* row_data = row.data_ptr<int64_t>();
  const int64_t M = rowptr_c.numel() - 1;
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t e = rowptr_data[m]; e < rowptr_data[m + 1]; ++e)
      row_data[e] = m;
  }

  return row;
}

at::Tensor edge_vector_view(const at::Tensor& value, const at::Tensor& target) {
  auto sizes = std::vector<int64_t>(target.dim(), 1);
  sizes[target.dim() - 2] = -1;
  return value.view(sizes);
}

at::Tensor edge_index_view(const at::Tensor& index, const at::Tensor& target) {
  auto sizes = std::vector<int64_t>(target.dim(), 1);
  sizes[target.dim() - 2] = -1;
  return index.view(sizes).expand(target.sizes());
}

std::vector<int64_t> all_dims_except_edge(const at::Tensor& input) {
  std::vector<int64_t> dims;
  const int64_t edge_dim = input.dim() - 2;
  for (int64_t dim = 0; dim < input.dim(); ++dim) {
    if (dim != edge_dim)
      dims.push_back(dim);
  }
  return dims;
}

at::Tensor row_count_for_edges(const at::Tensor& rowptr,
                               const at::Tensor& row,
                               const at::TensorOptions& options) {
  auto rowptr_c = rowptr.contiguous();
  auto row_count = rowptr_c.narrow(0, 1, rowptr_c.numel() - 1) -
                   rowptr_c.narrow(0, 0, rowptr_c.numel() - 1);
  row_count = row_count.clamp_min(1).to(options);
  return row_count.index_select(0, row);
}

at::Tensor spmm_value_backward(const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const at::Tensor& mat,
                               const at::Tensor& grad_out,
                               Reduction reduce) {
  auto row = row_from_rowptr(rowptr, col);
  auto mat_selected = mat.index_select(-2, col);
  auto grad_selected = grad_out.index_select(-2, row);
  auto prod = mat_selected * grad_selected;
  if (reduce == Reduction::Mean) {
    auto count = row_count_for_edges(rowptr, row, prod.options());
    prod = prod / edge_vector_view(count, prod);
  }
  return prod.sum(all_dims_except_edge(prod));
}

at::Tensor spmm_mat_backward(const at::Tensor& rowptr,
                             const at::Tensor& col,
                             const at::Tensor& value,
                             const at::Tensor& mat,
                             const at::Tensor& grad_out,
                             bool has_value,
                             Reduction reduce) {
  auto row = row_from_rowptr(rowptr, col);
  auto contrib = grad_out.index_select(-2, row);
  at::Tensor weight;
  if (has_value) {
    weight = value;
  } else if (reduce == Reduction::Mean) {
    weight = at::ones({col.numel()}, grad_out.options());
  }
  if (reduce == Reduction::Mean) {
    auto count = row_count_for_edges(rowptr, row, grad_out.options());
    weight = weight / count;
  }
  if (weight.defined())
    contrib = contrib * edge_vector_view(weight, contrib);

  auto grad_mat = at::zeros_like(mat);
  grad_mat.scatter_add_(-2, edge_index_view(col, contrib), contrib);
  return grad_mat;
}

class SpMMSum : public torch::autograd::Function<SpMMSum> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const std::optional<at::Tensor>& value,
                               const at::Tensor& mat) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = spmm_sum(rowptr, col, value, mat);
    auto saved_value = value.has_value() ? value.value() : col;
    ctx->saved_data["has_value"] = value.has_value();
    ctx->save_for_backward({rowptr, col, saved_value, mat});
    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto rowptr = saved[0];
    const auto col = saved[1];
    const auto value = saved[2];
    const auto mat = saved[3];
    const bool has_value = ctx->saved_data["has_value"].toBool();

    at::Tensor grad_value;
    if (has_value && value.requires_grad())
      grad_value =
          spmm_value_backward(rowptr, col, mat, grad_out, Reduction::Sum);

    at::Tensor grad_mat;
    if (mat.requires_grad())
      grad_mat = spmm_mat_backward(rowptr, col, value, mat, grad_out, has_value,
                                   Reduction::Sum);

    return {at::Tensor(), at::Tensor(), grad_value, grad_mat};
  }
};

at::Tensor spmm_sum_autograd(const at::Tensor& rowptr,
                             const at::Tensor& col,
                             const std::optional<at::Tensor>& value,
                             const at::Tensor& mat) {
  return SpMMSum::apply(rowptr, col, value, mat)[0];
}

class SpMMMean : public torch::autograd::Function<SpMMMean> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const std::optional<at::Tensor>& value,
                               const at::Tensor& mat) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto out = spmm_mean(rowptr, col, value, mat);
    auto saved_value = value.has_value() ? value.value() : col;
    ctx->saved_data["has_value"] = value.has_value();
    ctx->save_for_backward({rowptr, col, saved_value, mat});
    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto rowptr = saved[0];
    const auto col = saved[1];
    const auto value = saved[2];
    const auto mat = saved[3];
    const bool has_value = ctx->saved_data["has_value"].toBool();

    at::Tensor grad_value;
    if (has_value && value.requires_grad())
      grad_value =
          spmm_value_backward(rowptr, col, mat, grad_out, Reduction::Mean);

    at::Tensor grad_mat;
    if (mat.requires_grad())
      grad_mat = spmm_mat_backward(rowptr, col, value, mat, grad_out, has_value,
                                   Reduction::Mean);

    return {at::Tensor(), at::Tensor(), grad_value, grad_mat};
  }
};

at::Tensor spmm_mean_autograd(const at::Tensor& rowptr,
                              const at::Tensor& col,
                              const std::optional<at::Tensor>& value,
                              const at::Tensor& mat) {
  return SpMMMean::apply(rowptr, col, value, mat)[0];
}

std::tuple<at::Tensor, at::Tensor> spmm_minmax_backward(
    const at::Tensor& col,
    const at::Tensor& value,
    const at::Tensor& mat,
    const at::Tensor& arg_out,
    const at::Tensor& grad_out,
    bool has_value) {
  const auto invalid_arg_mask = arg_out == col.size(0);
  auto arg = arg_out.masked_fill(invalid_arg_mask, 0);
  auto arg_flat = arg.flatten();

  at::Tensor grad_value;
  if (has_value && value.requires_grad()) {
    auto ind = col.index_select(0, arg_flat).view_as(arg);
    auto out = mat.gather(-2, ind);
    out = out * grad_out;
    out = out.masked_fill(invalid_arg_mask, 0);

    grad_value = at::zeros_like(value);
    grad_value.scatter_add_(0, arg_flat, out.flatten());
  }

  at::Tensor grad_mat;
  if (mat.requires_grad()) {
    at::Tensor contrib;
    if (has_value) {
      contrib = value.index_select(0, arg_flat).view_as(arg) * grad_out;
    } else {
      contrib = grad_out.clone();
    }
    contrib = contrib.masked_fill(invalid_arg_mask, 0);
    auto ind = col.index_select(0, arg_flat).view_as(arg);

    grad_mat = at::zeros_like(mat);
    grad_mat.scatter_add_(-2, ind, contrib);
  }

  return std::make_tuple(grad_value, grad_mat);
}

class SpMMMin : public torch::autograd::Function<SpMMMin> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const std::optional<at::Tensor>& value,
                               const at::Tensor& mat) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto result = spmm_min(rowptr, col, value, mat);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    auto saved_value = value.has_value() ? value.value() : col;
    ctx->saved_data["has_value"] = value.has_value();
    ctx->save_for_backward({col, saved_value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto saved = ctx->get_saved_variables();
    const auto col = saved[0];
    const auto value = saved[1];
    const auto mat = saved[2];
    const auto arg_out = saved[3];
    const bool has_value = ctx->saved_data["has_value"].toBool();
    auto grads =
        spmm_minmax_backward(col, value, mat, arg_out, grad_outs[0], has_value);

    return {at::Tensor(), at::Tensor(), std::get<0>(grads), std::get<1>(grads)};
  }
};

std::tuple<at::Tensor, at::Tensor> spmm_min_autograd(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& value,
    const at::Tensor& mat) {
  auto result = SpMMMin::apply(rowptr, col, value, mat);
  return std::make_tuple(result[0], result[1]);
}

class SpMMMax : public torch::autograd::Function<SpMMMax> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& rowptr,
                               const at::Tensor& col,
                               const std::optional<at::Tensor>& value,
                               const at::Tensor& mat) {
    at::AutoDispatchBelowADInplaceOrView g;

    auto result = spmm_max(rowptr, col, value, mat);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);
    auto saved_value = value.has_value() ? value.value() : col;
    ctx->saved_data["has_value"] = value.has_value();
    ctx->save_for_backward({col, saved_value, mat, arg_out});
    ctx->mark_non_differentiable({arg_out});
    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto saved = ctx->get_saved_variables();
    const auto col = saved[0];
    const auto value = saved[1];
    const auto mat = saved[2];
    const auto arg_out = saved[3];
    const bool has_value = ctx->saved_data["has_value"].toBool();
    auto grads =
        spmm_minmax_backward(col, value, mat, arg_out, grad_outs[0], has_value);

    return {at::Tensor(), at::Tensor(), std::get<0>(grads), std::get<1>(grads)};
  }
};

std::tuple<at::Tensor, at::Tensor> spmm_max_autograd(
    const at::Tensor& rowptr,
    const at::Tensor& col,
    const std::optional<at::Tensor>& value,
    const at::Tensor& mat) {
  auto result = SpMMMax::apply(rowptr, col, value, mat);
  return std::make_tuple(result[0], result[1]);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_sum"), TORCH_FN(spmm_sum_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_mean"), TORCH_FN(spmm_mean_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_min"), TORCH_FN(spmm_min_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_max"), TORCH_FN(spmm_max_autograd));
}

}  // namespace ops
}  // namespace pyg
