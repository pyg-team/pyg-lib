#include "../scatter.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

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

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::scatter_sum"),
         TORCH_FN(scatter_sum_kernel));
}

}  // namespace ops
}  // namespace pyg
