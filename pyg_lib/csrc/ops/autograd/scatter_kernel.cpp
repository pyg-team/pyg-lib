#include "../scatter.h"
#include "../utils.h"

#include <torch/autograd.h>

#include <tuple>

namespace pyg {
namespace ops {

namespace {

using torch::autograd::variable_list;

// Autograd Function for `pyg::scatter_sum`.
//
// The C++ dispatcher returns a fresh tensor (when `out=None`) or the caller's
// buffer mutated in place (when `out=` is supplied). The `mark_dirty` call in
// the latter case lets gradcheck pick up the in-place mutation; the actual
// backward gradient is computed the same way regardless.
class ScatterSum : public torch::autograd::Function<ScatterSum> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    // Normalize `dim` once so saved metadata is consistent with what backward
    // sees from `torch.gather`.
    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` (records unsqueeze/expand on the
    // autograd graph so backward can call `.gather` directly).
    auto index_b = broadcast(index, src, dim_norm);

    auto out = scatter_sum(src, index_b, dim_norm, optional_out, dim_size);

    ctx->save_for_backward({index_b});
    ctx->saved_data["dim"] = dim_norm;

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto index_b = saved[0];
    const auto dim = ctx->saved_data["dim"].toInt();

    // `grad_src[i] = grad_out[index[i]]` — exactly `gather` along `dim`.
    auto grad_src = grad_out.gather(dim, index_b);

    // 5 input slots: (src, index, dim, out, dim_size). Only `src` is a real
    // differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor scatter_sum_autograd(const at::Tensor& src,
                                const at::Tensor& index,
                                int64_t dim,
                                const std::optional<at::Tensor>& optional_out,
                                std::optional<int64_t> dim_size) {
  return ScatterSum::apply(src, index, dim, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::scatter_mul`.
//
// Backward formula (upstream pytorch_scatter `scatter.cpp:101-112`):
//
//   grad_src = (grad_out * out).gather(dim, index_b) / src
//
// where `out` is the forward result. At positions where `src == 0`, this
// expression is ill-defined (0/0 NaN); upstream produces a NaN and then
// masks it with `masked_fill_(isnan, 0)`. We use `torch::where(src != 0,
// ..., zeros)` to produce zeros directly without going through NaN.
class ScatterMul : public torch::autograd::Function<ScatterMul> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` so that backward's `gather` can
    // use it directly. Recording the broadcast on the autograd graph keeps
    // gradient connectivity (though `index` itself is non-differentiable).
    auto index_b = broadcast(index, src, dim_norm);

    auto out = scatter_mul(src, index_b, dim_norm, optional_out, dim_size);

    // Backward needs `src` (denominator), `index_b` (gather axis), and `out`
    // (so that `grad_out * out` can be gathered at the contributing indices).
    ctx->save_for_backward({src, index_b, out});
    ctx->saved_data["dim"] = dim_norm;

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto src = saved[0];
    const auto index_b = saved[1];
    const auto out = saved[2];
    const auto dim = ctx->saved_data["dim"].toInt();

    // Note: `(grad_out * out)` is the gradient that arrives at the bucket;
    // gathering it back to the source positions multiplies by the product
    // of all *other* contributors, which is `out / src`. Hence the divide.
    auto gathered = (grad_out * out).gather(dim, index_b);
    auto grad_src = at::where(src != 0, gathered / src, at::zeros_like(src));

    // 5 input slots: (src, index, dim, out, dim_size).
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor scatter_mul_autograd(const at::Tensor& src,
                                const at::Tensor& index,
                                int64_t dim,
                                const std::optional<at::Tensor>& optional_out,
                                std::optional<int64_t> dim_size) {
  return ScatterMul::apply(src, index, dim, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::scatter_mean`.
//
// Composite op: forward dispatches to `scatter_sum` twice (numerator over
// `src` and denominator over a `ones`-tensor sized like `index`) and then
// divides. Because there is no dedicated CPU/CUDA kernel, the wrapper is
// registered to `CompositeExplicitAutograd` rather than `Autograd`: that
// key routes the call here for *all* backends and regardless of whether
// any input requires grad. `Function::apply` itself decides whether to
// record on the autograd tape based on input grad state, so this single
// registration handles both `requires_grad=True` and `requires_grad=False`
// callers correctly.
//
// Backward formula (upstream pytorch_scatter `scatter.cpp:149-160`):
//
//   grad_src = (grad_out.gather(dim, index_b)) / count.gather(dim, index_b)
//
// where `count` is the broadcast per-bucket count tensor saved from forward.
class ScatterMean : public torch::autograd::Function<ScatterMean> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` for the sum dispatch and so
    // backward's `gather` can use it directly.
    auto index_b = broadcast(index, src, dim_norm);

    // Numerator: per-bucket sum of `src` values.
    auto out_tensor =
        scatter_sum(src, index_b, dim_norm, optional_out, dim_size);

    // Denominator: per-bucket count. Use the *original* (non-broadcast)
    // `index` against a 1-D `ones` of length `index.size(...)`. Upstream
    // mirrors the same trick to avoid double-counting when `index` is
    // broadcast across multiple non-`dim` axes. The reduction `dim` for
    // this auxiliary call is the last axis of `index` if `index` has
    // fewer dims than `dim_norm` (i.e. `index` is 1-D and the broadcast
    // adds leading singletons), else `dim_norm`.
    const int64_t count_dim =
        index.dim() <= dim_norm ? index.dim() - 1 : dim_norm;
    auto ones = at::ones(index.sizes(), src.options());
    auto count = scatter_sum(ones, index, count_dim, std::nullopt,
                             out_tensor.size(dim_norm));
    count.masked_fill_(count < 1, 1);

    // Broadcast `count` (1-D along `dim_norm`) up to `out_tensor.shape`
    // so it aligns for in-place division along the reduction axis.
    auto count_b = broadcast(count, out_tensor, dim_norm);

    if (out_tensor.is_floating_point()) {
      out_tensor.true_divide_(count_b);
    } else {
      out_tensor.div_(count_b, "floor");
    }

    // Save the *broadcast* count: backward's `gather(count_b, dim, index_b)`
    // expects `count` to be shaped like `out_tensor` so gather aligns.
    ctx->save_for_backward({index_b, count_b});
    ctx->saved_data["dim"] = dim_norm;

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out_tensor};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto index_b = saved[0];
    const auto count_b = saved[1];
    const auto dim = ctx->saved_data["dim"].toInt();

    // `grad_src[i] = grad_out[index[i]] / count[index[i]]`. Gather both
    // along `dim` using the broadcast index so per-position counts line up.
    auto count_gathered = count_b.gather(dim, index_b);
    auto grad_src = grad_out.gather(dim, index_b).true_divide_(count_gathered);

    // 5 input slots: (src, index, dim, out, dim_size).
    return {grad_src, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

at::Tensor scatter_mean_autograd(const at::Tensor& src,
                                 const at::Tensor& index,
                                 int64_t dim,
                                 const std::optional<at::Tensor>& optional_out,
                                 std::optional<int64_t> dim_size) {
  return ScatterMean::apply(src, index, dim, optional_out, dim_size)[0];
}

// Autograd Function for `pyg::scatter_min`.
//
// Forward returns `(out, arg_out)`. Only `out` is differentiable; `arg_out`
// is marked non-differentiable via `ctx->mark_non_differentiable({arg_out})`
// so PyTorch will not error when callers try to take gradients through it
// and so gradcheck only probes the value output (convention 8).
//
// Backward formula (upstream `pytorch_scatter/csrc/scatter.cpp:184-196`):
//
//   src_shape_plus = src_shape; src_shape_plus[dim] += 1
//   grad_in = zeros(src_shape_plus)
//   grad_in.scatter_(dim, arg_out, grad_out)   // sentinel lands in +1 slot
//   grad_in = grad_in.narrow(dim, 0, src_shape[dim])  // drop sentinel
//
// The `+1`/`narrow` trick routes gradient only to the source positions
// that produced the per-bucket min and silently drops any contribution
// from buckets whose `arg_out` is still at the sentinel (empty buckets).
class ScatterMin : public torch::autograd::Function<ScatterMin> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` so the kernel can index per-K
    // and so the saved `src_shape` aligns with the (per-position) backward
    // scatter. Recording this on the autograd graph keeps the broadcast
    // visible to any upstream graph rewriting, though `index` itself is
    // non-differentiable.
    auto index_b = broadcast(index, src, dim_norm);

    auto result = scatter_min(src, index_b, dim_norm, optional_out, dim_size);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    // Backward needs `arg_out` (where to deposit each `grad_out`), the
    // normalized `dim`, and the original `src.sizes()` (so the +1/narrow
    // trick reconstructs the right shape).
    ctx->save_for_backward({arg_out});
    ctx->saved_data["dim"] = dim_norm;
    ctx->saved_data["src_shape"] = src.sizes();

    // `arg_out` is integer / non-differentiable; mark it so PyTorch will
    // not attempt to propagate gradients through it.
    ctx->mark_non_differentiable({arg_out});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    // `grad_outs[0]` is the gradient w.r.t. `out`; `grad_outs[1]` is the
    // gradient w.r.t. `arg_out` and is ignored (non-differentiable).
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto arg_out = saved[0];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `+1`/`narrow` trick: extend `src_shape` along `dim` by one extra
    // sentinel slot, scatter `grad_out` into the extended buffer using
    // `arg_out` (sentinel entries land in the trailing slot and are
    // dropped by the subsequent `narrow`).
    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    // 5 input slots: (src, index, dim, out, dim_size). Only `src` is a
    // real differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> scatter_min_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  auto result = ScatterMin::apply(src, index, dim, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

// Autograd Function for `pyg::scatter_max`.
//
// Forward returns `(out, arg_out)`. Only `out` is differentiable; `arg_out`
// is marked non-differentiable via `ctx->mark_non_differentiable({arg_out})`
// so PyTorch will not error when callers try to take gradients through it
// and so gradcheck only probes the value output (convention 8).
//
// Backward formula (identical to `scatter_min`; upstream
// `pytorch_scatter/csrc/scatter.cpp:184-196` is symmetric for max):
//
//   src_shape_plus = src_shape; src_shape_plus[dim] += 1
//   grad_in = zeros(src_shape_plus)
//   grad_in.scatter_(dim, arg_out, grad_out)   // sentinel lands in +1 slot
//   grad_in = grad_in.narrow(dim, 0, src_shape[dim])  // drop sentinel
//
// The `+1`/`narrow` trick routes gradient only to the source positions
// that produced the per-bucket max and silently drops any contribution
// from buckets whose `arg_out` is still at the sentinel (empty buckets).
class ScatterMax : public torch::autograd::Function<ScatterMax> {
 public:
  static variable_list forward(torch::autograd::AutogradContext* ctx,
                               const at::Tensor& src,
                               const at::Tensor& index,
                               int64_t dim,
                               const std::optional<at::Tensor>& optional_out,
                               std::optional<int64_t> dim_size) {
    at::AutoDispatchBelowADInplaceOrView g;

    const int64_t dim_norm = dim < 0 ? src.dim() + dim : dim;

    // Broadcast `index` up to `src.shape` so the kernel can index per-K
    // and so the saved `src_shape` aligns with the (per-position) backward
    // scatter. Recording this on the autograd graph keeps the broadcast
    // visible to any upstream graph rewriting, though `index` itself is
    // non-differentiable.
    auto index_b = broadcast(index, src, dim_norm);

    auto result = scatter_max(src, index_b, dim_norm, optional_out, dim_size);
    auto out = std::get<0>(result);
    auto arg_out = std::get<1>(result);

    // Backward needs `arg_out` (where to deposit each `grad_out`), the
    // normalized `dim`, and the original `src.sizes()` (so the +1/narrow
    // trick reconstructs the right shape).
    ctx->save_for_backward({arg_out});
    ctx->saved_data["dim"] = dim_norm;
    ctx->saved_data["src_shape"] = src.sizes();

    // `arg_out` is integer / non-differentiable; mark it so PyTorch will
    // not attempt to propagate gradients through it.
    ctx->mark_non_differentiable({arg_out});

    if (optional_out.has_value()) {
      ctx->mark_dirty({optional_out.value()});
    }

    return {out, arg_out};
  }

  static variable_list backward(torch::autograd::AutogradContext* ctx,
                                variable_list grad_outs) {
    // `grad_outs[0]` is the gradient w.r.t. `out`; `grad_outs[1]` is the
    // gradient w.r.t. `arg_out` and is ignored (non-differentiable).
    const auto grad_out = grad_outs[0];
    const auto saved = ctx->get_saved_variables();
    const auto arg_out = saved[0];
    const auto dim = ctx->saved_data["dim"].toInt();
    auto src_shape = ctx->saved_data["src_shape"].toIntList().vec();

    // `+1`/`narrow` trick: extend `src_shape` along `dim` by one extra
    // sentinel slot, scatter `grad_out` into the extended buffer using
    // `arg_out` (sentinel entries land in the trailing slot and are
    // dropped by the subsequent `narrow`).
    src_shape[dim] += 1;
    auto grad_in = at::zeros(src_shape, grad_out.options());
    grad_in.scatter_(dim, arg_out, grad_out);
    grad_in = grad_in.narrow(dim, 0, src_shape[dim] - 1);

    // 5 input slots: (src, index, dim, out, dim_size). Only `src` is a
    // real differentiable tensor; the rest get undefined-Tensor grads.
    return {grad_in, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor> scatter_max_autograd(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  auto result = ScatterMax::apply(src, index, dim, optional_out, dim_size);
  return std::make_tuple(result[0], result[1]);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mul"),
         TORCH_FN(scatter_mul_autograd));
  // `scatter_mean` is also registered to `Autograd` (in addition to
  // `CompositeExplicitAutograd` below) to silence the runtime warning
  // emitted when PyTorch finds no Autograd-key kernel and falls back to
  // the not-implemented sentinel; the same wrapper handles both paths.
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mean"),
         TORCH_FN(scatter_mean_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_min"),
         TORCH_FN(scatter_min_autograd));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_max"),
         TORCH_FN(scatter_max_autograd));
}

// `scatter_mean` has no dedicated CPU/CUDA kernel; it is fully composed of
// other dispatcher calls. Registering to `CompositeExplicitAutograd` makes
// the dispatcher route the call to our wrapper for *every* backend (and
// regardless of `requires_grad`), so non-grad callers do not fall through
// to an empty backend kernel slot.
TORCH_LIBRARY_IMPL(pyg, CompositeExplicitAutograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mean"),
         TORCH_FN(scatter_mean_autograd));
}

}  // namespace ops
}  // namespace pyg
