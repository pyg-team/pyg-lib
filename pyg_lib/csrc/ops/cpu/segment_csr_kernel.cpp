#include "../segment_csr.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

// CPU implementation of `pyg::segment_sum_csr`.
//
// CSR ops fix the reduction axis at `dim = indptr.dim() - 1` (upstream
// `segment_csr_cpu.cpp:24`). Each row `r ∈ [0, indptr.size(dim) - 1)`
// consumes the source slice `src[..., indptr[r]:indptr[r+1], ...]` and writes
// the per-row sum to `out[..., r, ...]`. Empty rows leave the corresponding
// `out` slot at its zero-initialized (or caller-supplied) value.
//
// Layout (upstream `segment_csr_cpu.cpp:54-56`): we factor the work into
//   * N = out.size(dim) * (indptr.numel() / indptr.size(-1))
//         — total number of "rows" across all leading indptr dims,
//   * E = src.size(dim) — number of source entries along the reduction axis,
//   * K = out.numel() / N — product of trailing dims of `src` past
//         `indptr.dim()`.
//
// `out=` contract: when the caller supplies `optional_out`, we **accumulate**
// into it (no zero-init). Matches upstream and is what makes
// `GatherCSR::backward` efficient.
at::Tensor segment_sum_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_sum_csr: src.dim() must be >= indptr.dim() "
              "(got src.dim()=",
              src.dim(), ", indptr.dim()=", indptr.dim(), ")");

  const int64_t dim = indptr.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_sum_csr: indptr must have at least 1 dimension");

  // Broadcast `indptr` up to `src.shape[:indptr.dim()-1]` along its leading
  // dims (upstream `segment_csr_cpu.cpp:19-22`). The last indptr dim
  // (size = num_rows + 1) is preserved as-is. Then force contiguous so the
  // kernel can do flat pointer arithmetic.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i)
    sizes[i] = src.size(i);
  auto indptr_b = indptr.expand(sizes).contiguous();

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_sum_csr: out.size(",
                    i, ") must match src.size(", i, ")");
    }
    TORCH_CHECK(
        src_c.numel() == 0 || out.size(dim) == indptr_b.size(dim) - 1,
        "segment_sum_csr: out.size(dim) must equal indptr.size(-1) - 1");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_b.size(dim) - 1, 0);
    // Zero-init so empty rows (and the accumulate loop) produce the correct
    // result without any per-row pre-fill.
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  const int64_t E = src_c.size(dim);
  // `N` is the total row count across all leading indptr dims (upstream's
  // factoring: `out.size(dim) * (indptr.numel() / indptr.size(-1))`).
  const int64_t rows_per_slice = indptr_b.size(dim) - 1;
  const int64_t leading = indptr_b.numel() / indptr_b.size(-1);
  const int64_t N = out.size(dim) * leading;
  const int64_t K = out.numel() / N;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_sum_csr_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* indptr_data = indptr_b.data_ptr<int64_t>();

        // Parallelize over the flat row index `n ∈ [0, N)`. Each row writes
        // to a disjoint `out` slot (`out_data + n * K`) so no atomics are
        // needed. The inner accumulate loop is sequential per row.
        at::parallel_for(
            0, N, at::internal::GRAIN_SIZE, [&](int64_t n_beg, int64_t n_end) {
              for (int64_t n = n_beg; n < n_end; ++n) {
                // `n` indexes into the flattened (leading, rows_per_slice)
                // space; reconstruct the offsets into indptr and src.
                const int64_t slice = n / rows_per_slice;
                const int64_t row = n % rows_per_slice;
                // indptr layout: leading slices each of length
                // `indptr.size(-1)`
                // (= rows_per_slice + 1). Row `row` consumes
                // `[indptr[row], indptr[row+1])`.
                const int64_t indptr_off = slice * indptr_b.size(-1) + row;
                const int64_t row_start = indptr_data[indptr_off];
                const int64_t row_end = indptr_data[indptr_off + 1];
                // `src` per-slice stride along `dim` is `E * K`; within a
                // slice each row position is one `K`-block.
                const int64_t src_off = slice * E * K;

                if (K == 1) {
                  // Honor the `out=` accumulate contract: seed from the
                  // current `out` slot.
                  opmath_t acc = static_cast<opmath_t>(out_data[n]);
                  for (int64_t e = row_start; e < row_end; ++e) {
                    acc += static_cast<opmath_t>(src_data[src_off + e]);
                  }
                  out_data[n] = static_cast<scalar_t>(acc);
                } else {
                  std::vector<opmath_t> acc(K);
                  for (int64_t k = 0; k < K; ++k)
                    acc[k] = static_cast<opmath_t>(out_data[n * K + k]);
                  for (int64_t e = row_start; e < row_end; ++e) {
                    for (int64_t k = 0; k < K; ++k)
                      acc[k] +=
                          static_cast<opmath_t>(src_data[src_off + e * K + k]);
                  }
                  for (int64_t k = 0; k < K; ++k)
                    out_data[n * K + k] = static_cast<scalar_t>(acc[k]);
                }
              }
            });
      });

  return out;
}

// CPU implementation of `pyg::gather_csr`.
//
// CSR ops fix the gather axis at `dim = indptr.dim() - 1`. For each row `r`,
// the value `src[..., r, ...]` is **broadcast** to every output position
// `out[..., i, ...]` for `i ∈ [indptr[r], indptr[r+1])`.
//
// Strategy: per-row linear pass (upstream `segment_csr_cpu.cpp:146-158`).
// This avoids the per-output-position binary search the plan mentions: for
// each row we read one `src` value (per K) and write it to a contiguous
// stretch of `out`. Parallelism is over the flat row index `N`.
//
// Layout:
//   * N = src.size(dim) * (indptr.numel() / indptr.size(-1)),
//   * E = out.size(dim) — number of output positions along the gather axis,
//   * K = src.numel() / N — product of trailing dims past `indptr.dim()`.
//
// `out=` contract: **overwrite** the caller's buffer per element. Positions
// outside any `[indptr[r], indptr[r+1])` range (i.e. before `indptr[0]` or
// after `indptr[-1]`) are left untouched.
at::Tensor gather_csr_kernel(const at::Tensor& src,
                             const at::Tensor& indptr,
                             const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "gather_csr: src.dim() must be >= indptr.dim() (got src.dim()=",
              src.dim(), ", indptr.dim()=", indptr.dim(), ")");

  const int64_t dim = indptr.dim() - 1;
  TORCH_CHECK(dim >= 0, "gather_csr: indptr must have at least 1 dimension");

  // Broadcast `indptr` along leading dims to match `src.shape[:dim]`.
  auto sizes = indptr.sizes().vec();
  for (int64_t i = 0; i < indptr.dim() - 1; ++i)
    sizes[i] = src.size(i);
  auto indptr_b = indptr.expand(sizes).contiguous();

  TORCH_CHECK(src.size(dim) == 0 || src.size(dim) == indptr_b.size(dim) - 1,
              "gather_csr: src.size(dim) must equal indptr.size(-1) - 1");

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < src_c.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "gather_csr: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (src_c.numel() > 0) {
      // Output size along `dim` is the trailing entry of `indptr`. We read it
      // from the flattened-last entry of the broadcast indptr (all leading
      // slices share the same trailing value since the original indptr only
      // varied along `dim`).
      out_sizes[dim] = *indptr_b.flatten()[-1].data_ptr<int64_t>();
    } else {
      out_sizes[dim] = 0;
    }
    out = at::empty(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return out;
  }

  // Layout: N = (rows_per_slice) * (leading slices); per src entry we write
  // a contiguous stretch in `out` of length (row_end - row_start) * K.
  const int64_t rows_per_slice = src_c.size(dim);
  const int64_t leading = indptr_b.numel() / indptr_b.size(-1);
  const int64_t N = rows_per_slice * leading;
  const int64_t K = src_c.numel() / N;
  const int64_t E = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "gather_csr_cpu", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* indptr_data = indptr_b.data_ptr<int64_t>();

        // Parallelize over flat row index. Each row writes a disjoint
        // stretch of `out` so no atomics / coordination needed.
        at::parallel_for(
            0, N, at::internal::GRAIN_SIZE, [&](int64_t n_beg, int64_t n_end) {
              for (int64_t n = n_beg; n < n_end; ++n) {
                const int64_t slice = n / rows_per_slice;
                const int64_t row = n % rows_per_slice;
                const int64_t indptr_off = slice * indptr_b.size(-1) + row;
                const int64_t row_start = indptr_data[indptr_off];
                const int64_t row_end = indptr_data[indptr_off + 1];
                const int64_t out_off = slice * E * K;

                if (K == 1) {
                  const scalar_t v = src_data[n];
                  for (int64_t e = row_start; e < row_end; ++e) {
                    out_data[out_off + e] = v;
                  }
                } else {
                  // Cache row value across the inner E loop to avoid repeated
                  // loads from `src`. Matches upstream `vals[K]` cache pattern.
                  std::vector<scalar_t> vals(K);
                  for (int64_t k = 0; k < K; ++k)
                    vals[k] = src_data[n * K + k];
                  for (int64_t e = row_start; e < row_end; ++e) {
                    for (int64_t k = 0; k < K; ++k)
                      out_data[out_off + e * K + k] = vals[k];
                  }
                }
              }
            });
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_csr"),
         TORCH_FN(segment_sum_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"), TORCH_FN(gather_csr_kernel));
}

}  // namespace ops
}  // namespace pyg
