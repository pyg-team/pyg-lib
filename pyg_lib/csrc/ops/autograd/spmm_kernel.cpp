#include "../spmm.h"

#include <torch/autograd.h>
#include <torch/library.h>

#include <vector>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

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

at::Tensor spmm_sum_value_backward(const at::Tensor& rowptr,
                                   const at::Tensor& col,
                                   const at::Tensor& mat,
                                   const at::Tensor& grad_out) {
  auto row = row_from_rowptr(rowptr, col);
  auto mat_selected = mat.index_select(-2, col);
  auto grad_selected = grad_out.index_select(-2, row);
  auto prod = mat_selected * grad_selected;
  return prod.sum(all_dims_except_edge(prod));
}

at::Tensor spmm_sum_mat_backward(const at::Tensor& rowptr,
                                 const at::Tensor& col,
                                 const at::Tensor& value,
                                 const at::Tensor& mat,
                                 const at::Tensor& grad_out,
                                 bool has_value) {
  auto row = row_from_rowptr(rowptr, col);
  auto contrib = grad_out.index_select(-2, row);
  if (has_value)
    contrib = contrib * edge_vector_view(value, contrib);

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
      grad_value = spmm_sum_value_backward(rowptr, col, mat, grad_out);

    at::Tensor grad_mat;
    if (mat.requires_grad())
      grad_mat =
          spmm_sum_mat_backward(rowptr, col, value, mat, grad_out, has_value);

    return {at::Tensor(), at::Tensor(), grad_value, grad_mat};
  }
};

at::Tensor spmm_sum_autograd(const at::Tensor& rowptr,
                             const at::Tensor& col,
                             const std::optional<at::Tensor>& value,
                             const at::Tensor& mat) {
  return SpMMSum::apply(rowptr, col, value, mat)[0];
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_sum"), TORCH_FN(spmm_sum_autograd));
}

}  // namespace ops
}  // namespace pyg
