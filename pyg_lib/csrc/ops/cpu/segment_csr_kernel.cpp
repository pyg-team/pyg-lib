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

// CPU implementation of `pyg::segment_mean_csr`.
//
// Same layout/shape as `segment_sum_csr_kernel` (sequential per-row pass with
// `at::parallel_for` over rows), with an extra post-loop divide by the row
// length. Row length is `indptr[r+1] - indptr[r]`; empty rows have length 0
// so we clamp to 1 before dividing — the corresponding `out` slot is already
// zero (from the zero-init) and `0 / 1 = 0` is the desired empty-row value.
//
// `out=` contract: upstream MEAN does **not** honor an accumulate contract.
// We allocate `out` as `at::zeros(...)` and overwrite per row (sum into the
// zero, then divide). If the caller supplies `out=`, we still overwrite —
// the API surface accepts it for symmetry with sum.
//
// `AT_DISPATCH_FLOATING_TYPES_AND2` (not `_ALL_TYPES_AND2`) — mean returns a
// floating result; integer dispatch would silently truncate the divide.
at::Tensor segment_mean_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_mean_csr: src.dim() must be >= indptr.dim() "
              "(got src.dim()=",
              src.dim(), ", indptr.dim()=", indptr.dim(), ")");

  const int64_t dim = indptr.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_mean_csr: indptr must have at least 1 dimension");

  // Broadcast `indptr` up to `src.shape[:indptr.dim()-1]` (mirrors the sum
  // kernel) and force contiguous so we can use flat pointer arithmetic.
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
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_mean_csr: out.size(",
                    i, ") must match src.size(", i, ")");
    }
    TORCH_CHECK(
        src_c.numel() == 0 || out.size(dim) == indptr_b.size(dim) - 1,
        "segment_mean_csr: out.size(dim) must equal indptr.size(-1) - 1");
    // Mean overwrites; clear the buffer so the sum loop accumulates from 0
    // and empty rows land at 0 after the divide.
    out.zero_();
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_b.size(dim) - 1, 0);
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  const int64_t E = src_c.size(dim);
  const int64_t rows_per_slice = indptr_b.size(dim) - 1;
  const int64_t leading = indptr_b.numel() / indptr_b.size(-1);
  const int64_t N = out.size(dim) * leading;
  const int64_t K = out.numel() / N;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_csr_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* indptr_data = indptr_b.data_ptr<int64_t>();

        at::parallel_for(
            0, N, at::internal::GRAIN_SIZE, [&](int64_t n_beg, int64_t n_end) {
              for (int64_t n = n_beg; n < n_end; ++n) {
                const int64_t slice = n / rows_per_slice;
                const int64_t row = n % rows_per_slice;
                const int64_t indptr_off = slice * indptr_b.size(-1) + row;
                const int64_t row_start = indptr_data[indptr_off];
                const int64_t row_end = indptr_data[indptr_off + 1];
                const int64_t src_off = slice * E * K;

                // Row length: clamp to 1 for empty rows so the divide is a
                // no-op (the accumulator is still 0).
                const int64_t row_len = row_end - row_start;
                const opmath_t denom =
                    static_cast<opmath_t>(row_len > 0 ? row_len : 1);

                if (K == 1) {
                  opmath_t acc = static_cast<opmath_t>(0);
                  for (int64_t e = row_start; e < row_end; ++e) {
                    acc += static_cast<opmath_t>(src_data[src_off + e]);
                  }
                  out_data[n] = static_cast<scalar_t>(acc / denom);
                } else {
                  std::vector<opmath_t> acc(K, static_cast<opmath_t>(0));
                  for (int64_t e = row_start; e < row_end; ++e) {
                    for (int64_t k = 0; k < K; ++k)
                      acc[k] +=
                          static_cast<opmath_t>(src_data[src_off + e * K + k]);
                  }
                  for (int64_t k = 0; k < K; ++k)
                    out_data[n * K + k] = static_cast<scalar_t>(acc[k] / denom);
                }
              }
            });
      });

  return out;
}

// CPU implementation of `pyg::segment_min_csr`.
//
// Same (N, E, K) layout as `segment_sum_csr_kernel` with
// `dim = indptr.dim() - 1`. Per-row sequential pass: maintain a running
// minimum value and its first-match argindex per row. Two output tensors:
//   * `out`: the per-row minimum value, init to `numeric_limits::max()`
//     when `out=None` so the first contributing element wins. Empty rows
//     (no contributing source element) are reset to `0` after the
//     reduction loop (matched against the sentinel in `arg_out`, mirrors
//     `segment_min_coo_kernel`).
//   * `arg_out`: the source position that produced each per-row min, init
//     to the sentinel `src.size(dim)` (one past the last valid index along
//     `dim`). Has dtype `int64` (matches `indptr.options()`).
//
// **Determinism:** the inner E loop is sequential per row and uses strict
// `<` on the comparison, so on ties the kernel preserves *first-match*
// semantics. Parallelism is only over the flat row index `N` where rows
// write to disjoint sub-regions of `out` / `arg_out`.
//
// `out=` contract: when the caller supplies `optional_out`, the running
// state begins from the caller's buffer (no max-init), and the kernel
// **continues** the reduction over that state. The caller is responsible
// for any non-default starting value. This matches upstream.
std::tuple<at::Tensor, at::Tensor> segment_min_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_min_csr: src.dim() must be >= indptr.dim() "
              "(got src.dim()=",
              src.dim(), ", indptr.dim()=", indptr.dim(), ")");

  const int64_t dim = indptr.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_min_csr: indptr must have at least 1 dimension");

  // Broadcast `indptr` up to `src.shape[:indptr.dim()-1]` and force contiguous
  // so the kernel can do flat pointer arithmetic.
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
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_min_csr: out.size(",
                    i, ") must match src.size(", i, ")");
    }
    TORCH_CHECK(
        src_c.numel() == 0 || out.size(dim) == indptr_b.size(dim) - 1,
        "segment_min_csr: out.size(dim) must equal indptr.size(-1) - 1");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_b.size(dim) - 1, 0);
    // Allocate uninitialized; the dispatch below fills with
    // `numeric_limits<scalar_t>::max()` before the reduction loop.
    out = at::empty(out_sizes, src_c.options());
  }

  // `arg_out` is always freshly allocated (independent of `optional_out`) and
  // starts at the sentinel `src.size(dim)`. The sentinel both encodes "empty
  // row" for the `masked_fill_` post-step and feeds into the backward's
  // `+1`/`narrow` trick.
  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, indptr_b.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  const int64_t E = src_c.size(dim);
  const int64_t rows_per_slice = indptr_b.size(dim) - 1;
  const int64_t leading = indptr_b.numel() / indptr_b.size(-1);
  const int64_t N = out.size(dim) * leading;
  const int64_t K = out.numel() / N;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_min_csr_cpu", [&] {
        // Init `out` to `numeric_limits::max()` only when the caller did not
        // supply `optional_out` (otherwise the caller's state is the running
        // state, per the `out=` contract).
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::max());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* indptr_data = indptr_b.data_ptr<int64_t>();

        // Parallelize over the flat row index `n ∈ [0, N)`. Each row writes
        // to a disjoint `out` / `arg_out` slot so no atomics are needed. The
        // inner reduction loop is sequential to preserve first-match
        // argindex semantics on tied source values within a row.
        at::parallel_for(
            0, N, at::internal::GRAIN_SIZE, [&](int64_t n_beg, int64_t n_end) {
              for (int64_t n = n_beg; n < n_end; ++n) {
                const int64_t slice = n / rows_per_slice;
                const int64_t row = n % rows_per_slice;
                const int64_t indptr_off = slice * indptr_b.size(-1) + row;
                const int64_t row_start = indptr_data[indptr_off];
                const int64_t row_end = indptr_data[indptr_off + 1];
                const int64_t src_off = slice * E * K;

                if (K == 1) {
                  auto* out_slot = out_data + n;
                  auto* arg_slot = arg_out_data + n;
                  for (int64_t e = row_start; e < row_end; ++e) {
                    const scalar_t v = src_data[src_off + e];
                    if (v < *out_slot) {
                      *out_slot = v;
                      *arg_slot = e;
                    }
                  }
                } else {
                  for (int64_t e = row_start; e < row_end; ++e) {
                    for (int64_t k = 0; k < K; ++k) {
                      const scalar_t v = src_data[src_off + e * K + k];
                      auto* out_slot = out_data + n * K + k;
                      if (v < *out_slot) {
                        *out_slot = v;
                        arg_out_data[n * K + k] = e;
                      }
                    }
                  }
                }
              }
            });
      });

  // Empty rows retain the sentinel in `arg_out` and either keep the
  // `numeric_limits::max()` placeholder (when `optional_out` was not
  // supplied) or the caller's initial value (when it was). For the former,
  // upstream resets the value to `0`; for the latter we leave the caller's
  // value alone so the `out=` contract is honored.
  if (!optional_out.has_value()) {
    out.masked_fill_(arg_out == sentinel, 0);
  }

  return std::make_tuple(out, arg_out);
}

// CPU implementation of `pyg::segment_max_csr`.
//
// Symmetric to `segment_min_csr_kernel`: same (N, E, K) layout with
// `dim = indptr.dim() - 1`, per-row sequential pass maintaining a running
// maximum value and its first-match argindex per row. Two output tensors:
//   * `out`: the per-row maximum value, init to `numeric_limits::lowest()`
//     when `out=None` so the first contributing element wins. Empty rows
//     (no contributing source element) are reset to `0` after the
//     reduction loop (matched against the sentinel in `arg_out`).
//   * `arg_out`: the source position that produced each per-row max, init
//     to the sentinel `src.size(dim)`.
//
// Determinism: strict `>` comparison preserves first-match argindex on ties.
//
// `out=` contract mirrors `segment_min_csr_kernel`: when supplied, the
// caller's buffer is the running state (no lowest-init).
std::tuple<at::Tensor, at::Tensor> segment_max_csr_kernel(
    const at::Tensor& src,
    const at::Tensor& indptr,
    const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= indptr.dim(),
              "segment_max_csr: src.dim() must be >= indptr.dim() "
              "(got src.dim()=",
              src.dim(), ", indptr.dim()=", indptr.dim(), ")");

  const int64_t dim = indptr.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_max_csr: indptr must have at least 1 dimension");

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
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_max_csr: out.size(",
                    i, ") must match src.size(", i, ")");
    }
    TORCH_CHECK(
        src_c.numel() == 0 || out.size(dim) == indptr_b.size(dim) - 1,
        "segment_max_csr: out.size(dim) must equal indptr.size(-1) - 1");
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = std::max<int64_t>(indptr_b.size(dim) - 1, 0);
    out = at::empty(out_sizes, src_c.options());
  }

  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, indptr_b.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  const int64_t E = src_c.size(dim);
  const int64_t rows_per_slice = indptr_b.size(dim) - 1;
  const int64_t leading = indptr_b.numel() / indptr_b.size(-1);
  const int64_t N = out.size(dim) * leading;
  const int64_t K = out.numel() / N;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_max_csr_cpu", [&] {
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::lowest());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* indptr_data = indptr_b.data_ptr<int64_t>();

        at::parallel_for(
            0, N, at::internal::GRAIN_SIZE, [&](int64_t n_beg, int64_t n_end) {
              for (int64_t n = n_beg; n < n_end; ++n) {
                const int64_t slice = n / rows_per_slice;
                const int64_t row = n % rows_per_slice;
                const int64_t indptr_off = slice * indptr_b.size(-1) + row;
                const int64_t row_start = indptr_data[indptr_off];
                const int64_t row_end = indptr_data[indptr_off + 1];
                const int64_t src_off = slice * E * K;

                if (K == 1) {
                  auto* out_slot = out_data + n;
                  auto* arg_slot = arg_out_data + n;
                  for (int64_t e = row_start; e < row_end; ++e) {
                    const scalar_t v = src_data[src_off + e];
                    if (v > *out_slot) {
                      *out_slot = v;
                      *arg_slot = e;
                    }
                  }
                } else {
                  for (int64_t e = row_start; e < row_end; ++e) {
                    for (int64_t k = 0; k < K; ++k) {
                      const scalar_t v = src_data[src_off + e * K + k];
                      auto* out_slot = out_data + n * K + k;
                      if (v > *out_slot) {
                        *out_slot = v;
                        arg_out_data[n * K + k] = e;
                      }
                    }
                  }
                }
              }
            });
      });

  if (!optional_out.has_value()) {
    out.masked_fill_(arg_out == sentinel, 0);
  }

  return std::make_tuple(out, arg_out);
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
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_csr"),
         TORCH_FN(segment_mean_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_csr"),
         TORCH_FN(segment_min_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_csr"),
         TORCH_FN(segment_max_csr_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_csr"), TORCH_FN(gather_csr_kernel));
}

}  // namespace ops
}  // namespace pyg
