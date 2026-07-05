#include "../scatter.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <limits>
#include <tuple>

namespace pyg {
namespace ops {

namespace {

// CPU implementation of `pyg::scatter_sum`.
//
// Layout: `src` is viewed as (B, E, K) where
//   * B = product of dim sizes before `dim`,
//   * E = src.size(dim),
//   * K = product of dim sizes after `dim`.
// `out` has the same layout with N replacing E (N = out.size(dim)). `index`
// must already be broadcast to `src.shape` (the dispatcher / autograd front
// guarantees this) and is forced contiguous here before the kernel scan.
//
// `out=` contract: when the caller supplies `optional_out`, we **accumulate**
// into it (no zero-init). This matches upstream pytorch_scatter and is what
// makes `gather_coo` / `gather_csr` backward efficient.
at::Tensor scatter_sum_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_sum: src.dim() must equal index.dim() "
              "after broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  // Normalize `dim` to a non-negative value relative to `src.dim()`.
  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter_sum: dim out of range");

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "scatter_sum: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + *index_c.max().data_ptr<int64_t>();
    }
    // Zero-init the freshly allocated buffer so that the accumulate loop
    // below produces the correct result.
    out = at::zeros(sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_sum_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();

        // Parallelize over the leading (B) dimension. Each `b` slice writes
        // to a disjoint sub-region of `out`, so no atomics are needed.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                for (int64_t e = 0; e < E; ++e) {
                  const int64_t idx = index_data[src_off + e * K];
                  // Note: when `index` is broadcast from a 1-D shape (i.e.
                  // shape
                  // `(E,)` expanded along K), the K positions in row `e` all
                  // hold the same index, so reading `index_data[src_off + e * K
                  // + 0]` is sufficient. The expand contract guarantees this;
                  // the upstream kernel matches.
                  //
                  // For the general case (index already had a K dimension
                  // before broadcast), each (e, k) pair may legitimately have a
                  // distinct index, so we still index per-k below.
                  if (K == 1) {
                    const opmath_t v =
                        static_cast<opmath_t>(src_data[src_off + e]);
                    out_data[out_off + idx] = static_cast<scalar_t>(
                        static_cast<opmath_t>(out_data[out_off + idx]) + v);
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const int64_t j = src_off + e * K + k;
                      const int64_t i = index_data[j];
                      const opmath_t v = static_cast<opmath_t>(src_data[j]);
                      out_data[out_off + i * K + k] = static_cast<scalar_t>(
                          static_cast<opmath_t>(out_data[out_off + i * K + k]) +
                          v);
                    }
                  }
                }
              }
            });
      });

  return out;
}

// CPU implementation of `pyg::scatter_mul`.
//
// Same (B, E, K) layout as `scatter_sum_kernel`. The only differences:
//   * The reduction is multiplicative.
//   * When `out=None`, the freshly allocated buffer is initialized with
//     **ones** (the multiplicative identity), not zeros.
//
// `out=` contract: when the caller supplies `optional_out`, we **multiply
// into** it (no ones-init). The caller is responsible for any non-default
// starting state. This matches upstream pytorch_scatter.
at::Tensor scatter_mul_kernel(const at::Tensor& src,
                              const at::Tensor& index,
                              int64_t dim,
                              const std::optional<at::Tensor>& optional_out,
                              std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_mul: src.dim() must equal index.dim() "
              "after broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  // Normalize `dim` to a non-negative value relative to `src.dim()`.
  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter_mul: dim out of range");

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "scatter_mul: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + *index_c.max().data_ptr<int64_t>();
    }
    // Ones-init the freshly allocated buffer so that the multiplicative
    // reduction starts from the multiplicative identity.
    out = at::ones(sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_mul_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();

        // Parallelize over the leading (B) dimension. Each `b` slice writes
        // to a disjoint sub-region of `out`, so no atomics are needed.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                for (int64_t e = 0; e < E; ++e) {
                  if (K == 1) {
                    const int64_t idx = index_data[src_off + e];
                    const opmath_t v =
                        static_cast<opmath_t>(src_data[src_off + e]);
                    out_data[out_off + idx] = static_cast<scalar_t>(
                        static_cast<opmath_t>(out_data[out_off + idx]) * v);
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const int64_t j = src_off + e * K + k;
                      const int64_t i = index_data[j];
                      const opmath_t v = static_cast<opmath_t>(src_data[j]);
                      out_data[out_off + i * K + k] = static_cast<scalar_t>(
                          static_cast<opmath_t>(out_data[out_off + i * K + k]) *
                          v);
                    }
                  }
                }
              }
            });
      });

  return out;
}

// CPU implementation of `pyg::scatter_min`.
//
// Same (B, E, K) layout as `scatter_sum_kernel`. Two output tensors:
//   * `out`: the per-bucket minimum value, init to `numeric_limits::max()`
//     when `out=None` so the first contributing element wins. Empty
//     buckets (no contributing source element) are reset to `0` after
//     the reduction loop (upstream convention).
//   * `arg_out`: the source position that produced each per-bucket min,
//     init to the sentinel `src.size(dim)` (one past the last valid
//     index along `dim`). Has dtype `int64` (matches `index.options()`).
//
// **Determinism:** the inner loop over `E` is sequential per `(B, K)`
// slice and uses strict `<` on the comparison, so on ties the kernel
// preserves *first-match* semantics. Parallelism is only over the
// leading `B` dim, where slices write to disjoint sub-regions of
// `out` / `arg_out`.
//
// `out=` contract: when the caller supplies `optional_out`, the running
// state begins from the caller's buffer (no max-init). The caller is
// responsible for any non-default starting state. This matches upstream.
std::tuple<at::Tensor, at::Tensor> scatter_min_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_min: src.dim() must equal index.dim() "
              "after broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  // Normalize `dim` to a non-negative value relative to `src.dim()`.
  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter_min: dim out of range");

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "scatter_min: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + *index_c.max().data_ptr<int64_t>();
    }
    // Allocate uninitialized; the dispatch below fills with
    // `numeric_limits<scalar_t>::max()` before the reduction loop.
    out = at::empty(sizes, src_c.options());
  }

  // `arg_out` is always freshly allocated (independent of `optional_out`)
  // and starts at the sentinel `src.size(dim)`. The sentinel both encodes
  // "empty bucket" for the `masked_fill_` post-step and feeds into the
  // backward's `+1`/`narrow` trick.
  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, index_c.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_min_cpu", [&] {
        // Init `out` to `numeric_limits::max()` only when the caller did
        // not supply `optional_out` (otherwise the caller's state is the
        // running state, per the `out=` contract).
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::max());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();

        // Parallelize over the leading (B) dimension only — the inner
        // loop over E must remain sequential to preserve first-match
        // argindex semantics on tied source values.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                for (int64_t e = 0; e < E; ++e) {
                  if (K == 1) {
                    const int64_t idx = index_data[src_off + e];
                    const scalar_t v = src_data[src_off + e];
                    auto* out_slot = out_data + out_off + idx;
                    if (v < *out_slot) {
                      *out_slot = v;
                      arg_out_data[out_off + idx] = e;
                    }
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const int64_t j = src_off + e * K + k;
                      const int64_t idx = index_data[j];
                      const scalar_t v = src_data[j];
                      auto* out_slot = out_data + out_off + idx * K + k;
                      if (v < *out_slot) {
                        *out_slot = v;
                        arg_out_data[out_off + idx * K + k] = e;
                      }
                    }
                  }
                }
              }
            });
      });

  // Empty buckets retain the sentinel in `arg_out` and either keep the
  // `numeric_limits::max()` placeholder (when `optional_out` was not
  // supplied) or the caller's initial value (when it was). For the
  // former, upstream resets the value to `0`; for the latter we leave
  // the caller's value alone so the `out=` contract is honored.
  if (!optional_out.has_value()) {
    out.masked_fill_(arg_out == sentinel, 0);
  }

  return std::make_tuple(out, arg_out);
}

// CPU implementation of `pyg::scatter_max`.
//
// Same (B, E, K) layout as `scatter_sum_kernel`. Two output tensors:
//   * `out`: the per-bucket maximum value, init to `numeric_limits::lowest()`
//     when `out=None` so the first contributing element wins. Empty
//     buckets (no contributing source element) are reset to `0` after
//     the reduction loop (upstream convention).
//   * `arg_out`: the source position that produced each per-bucket max,
//     init to the sentinel `src.size(dim)` (one past the last valid
//     index along `dim`). Has dtype `int64` (matches `index.options()`).
//
// **Determinism:** the inner loop over `E` is sequential per `(B, K)`
// slice and uses strict `>` on the comparison, so on ties the kernel
// preserves *first-match* semantics. Parallelism is only over the
// leading `B` dim, where slices write to disjoint sub-regions of
// `out` / `arg_out`.
//
// `out=` contract: when the caller supplies `optional_out`, the running
// state begins from the caller's buffer (no lowest-init). The caller is
// responsible for any non-default starting state. This matches upstream.
std::tuple<at::Tensor, at::Tensor> scatter_max_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    int64_t dim,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() == index.dim(),
              "scatter_max: src.dim() must equal index.dim() "
              "after broadcasting (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  // Normalize `dim` to a non-negative value relative to `src.dim()`.
  dim = dim < 0 ? src.dim() + dim : dim;
  TORCH_CHECK(dim >= 0 && dim < src.dim(), "scatter_max: dim out of range");

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "scatter_max: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      sizes[dim] = dim_size.value();
    } else if (index_c.numel() == 0) {
      sizes[dim] = 0;
    } else {
      sizes[dim] = 1 + *index_c.max().data_ptr<int64_t>();
    }
    // Allocate uninitialized; the dispatch below fills with
    // `numeric_limits<scalar_t>::lowest()` before the reduction loop.
    out = at::empty(sizes, src_c.options());
  }

  // `arg_out` is always freshly allocated (independent of `optional_out`)
  // and starts at the sentinel `src.size(dim)`. The sentinel both encodes
  // "empty bucket" for the `masked_fill_` post-step and feeds into the
  // backward's `+1`/`narrow` trick.
  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, index_c.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= src_c.size(i);
  const int64_t E = src_c.size(dim);
  const int64_t K = src_c.numel() / (B * E);
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "scatter_max_cpu", [&] {
        // Init `out` to `numeric_limits::lowest()` only when the caller did
        // not supply `optional_out` (otherwise the caller's state is the
        // running state, per the `out=` contract).
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::lowest());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();

        // Parallelize over the leading (B) dimension only — the inner
        // loop over E must remain sequential to preserve first-match
        // argindex semantics on tied source values.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                for (int64_t e = 0; e < E; ++e) {
                  if (K == 1) {
                    const int64_t idx = index_data[src_off + e];
                    const scalar_t v = src_data[src_off + e];
                    auto* out_slot = out_data + out_off + idx;
                    if (v > *out_slot) {
                      *out_slot = v;
                      arg_out_data[out_off + idx] = e;
                    }
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const int64_t j = src_off + e * K + k;
                      const int64_t idx = index_data[j];
                      const scalar_t v = src_data[j];
                      auto* out_slot = out_data + out_off + idx * K + k;
                      if (v > *out_slot) {
                        *out_slot = v;
                        arg_out_data[out_off + idx * K + k] = e;
                      }
                    }
                  }
                }
              }
            });
      });

  // Empty buckets retain the sentinel in `arg_out` and either keep the
  // `numeric_limits::lowest()` placeholder (when `optional_out` was not
  // supplied) or the caller's initial value (when it was). For the
  // former, upstream resets the value to `0`; for the latter we leave
  // the caller's value alone so the `out=` contract is honored.
  if (!optional_out.has_value()) {
    out.masked_fill_(arg_out == sentinel, 0);
  }

  return std::make_tuple(out, arg_out);
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_mul"),
         TORCH_FN(scatter_mul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_min"),
         TORCH_FN(scatter_min_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_max"),
         TORCH_FN(scatter_max_kernel));
}

}  // namespace ops
}  // namespace pyg
