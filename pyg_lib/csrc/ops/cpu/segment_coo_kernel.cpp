#include "../segment_coo.h"

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

// CPU implementation of `pyg::segment_sum_coo`.
//
// COO ops fix the reduction axis at `dim = index.dim() - 1` (upstream
// `segment_coo_cpu.cpp:24`). The `index` tensor is sorted-ascending along
// that axis; equal-index runs collapse into a single output bucket.
//
// Layout: `index` is expanded to `src.shape[:index.dim()]` (each leading
// dim of `index` matches `src`). With `dim = index.dim() - 1`:
//   * B = product of leading dims before `dim` (= index.numel() /
//   src.size(dim)),
//   * E = src.size(dim) = index.size(dim),
//   * K = product of trailing dims of `src` past `index.dim()`
//         (= src.numel() / index.numel()),
//   * N = out.size(dim).
//
// `out=` contract: when the caller supplies `optional_out`, we **accumulate**
// into it (no zero-init). Matches upstream and is what makes
// `GatherCOO::backward` efficient.
at::Tensor segment_sum_coo_kernel(const at::Tensor& src,
                                  const at::Tensor& index,
                                  const std::optional<at::Tensor>& optional_out,
                                  std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_sum_coo: src.dim() must be >= index.dim() "
              "(got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  // COO ops fix the reduction axis at the last index dim.
  const int64_t dim = index.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_sum_coo: index must have at least 1 dimension");

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream
  // `segment_coo_cpu.cpp:19-22`). Then force contiguous for the linear scan.
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i)
    sizes[i] = src.size(i);
  auto index_b = index.expand(sizes).contiguous();

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_sum_coo: out.size(",
                    i, ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      out_sizes[dim] = dim_size.value();
    } else if (index_b.numel() == 0) {
      out_sizes[dim] = 0;
    } else {
      // Last index in the (sorted-ascending) last row gives `dim_size - 1`.
      auto tmp = index_b.select(dim, index_b.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      out_sizes[dim] = 1 + *tmp.data_ptr<int64_t>();
    }
    // Zero-init the freshly allocated buffer so the accumulate loop below
    // produces the correct result.
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  const int64_t E = src_c.size(dim);
  // `B` is the product of leading dims of `index` before `dim`. We compute
  // it from `index_b` (post-expand) so it stays well-defined even when the
  // original `index` is 1-D.
  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= index_b.size(i);
  // `K` is the product of trailing dims of `src` past `index.dim()`.
  const int64_t K = src_c.numel() / index_b.numel();
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_sum_coo_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* index_data = index_b.data_ptr<int64_t>();

        // Parallelize over the leading (B) dim only. Each `b` slice writes
        // to a disjoint sub-region of `out`, so no atomics are needed. The
        // inner E loop must remain sequential to correctly accumulate runs
        // of equal indices.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                const int64_t idx_off = b * E;

                if (E == 0)
                  continue;

                // Sequential pass over the sorted-ascending `index` row.
                // Maintain a per-K accumulator that flushes to `out` when
                // the index changes (or at the end of the row).
                int64_t cur_idx = index_data[idx_off];

                if (K == 1) {
                  // Initialize accumulator from current `out` slot so the
                  // `out=` accumulate contract is honored.
                  opmath_t acc =
                      static_cast<opmath_t>(out_data[out_off + cur_idx]);
                  for (int64_t e = 0; e < E; ++e) {
                    acc += static_cast<opmath_t>(src_data[src_off + e]);
                    const bool last = (e == E - 1);
                    const int64_t next_idx =
                        last ? cur_idx : index_data[idx_off + e + 1];
                    if (last || next_idx != cur_idx) {
                      out_data[out_off + cur_idx] = static_cast<scalar_t>(acc);
                      if (!last) {
                        cur_idx = next_idx;
                        acc =
                            static_cast<opmath_t>(out_data[out_off + cur_idx]);
                      }
                    }
                  }
                } else {
                  std::vector<opmath_t> acc(K);
                  for (int64_t k = 0; k < K; ++k)
                    acc[k] = static_cast<opmath_t>(
                        out_data[out_off + cur_idx * K + k]);
                  for (int64_t e = 0; e < E; ++e) {
                    for (int64_t k = 0; k < K; ++k)
                      acc[k] +=
                          static_cast<opmath_t>(src_data[src_off + e * K + k]);
                    const bool last = (e == E - 1);
                    const int64_t next_idx =
                        last ? cur_idx : index_data[idx_off + e + 1];
                    if (last || next_idx != cur_idx) {
                      for (int64_t k = 0; k < K; ++k)
                        out_data[out_off + cur_idx * K + k] =
                            static_cast<scalar_t>(acc[k]);
                      if (!last) {
                        cur_idx = next_idx;
                        for (int64_t k = 0; k < K; ++k)
                          acc[k] = static_cast<opmath_t>(
                              out_data[out_off + cur_idx * K + k]);
                      }
                    }
                  }
                }
              }
            });
      });

  return out;
}

// CPU implementation of `pyg::segment_mean_coo`.
//
// Same layout/shape as `segment_sum_coo_kernel`: sequential pass over the
// sorted-ascending `index`, accumulate runs of equal indices, but ALSO track
// the per-bucket count and divide at the end. The count tensor is allocated
// with shape `index.sizes()` with the last dim replaced by `N` (i.e. without
// the trailing K dims) — this is a flat `B*N` count, matching upstream
// `segment_coo_cpu.cpp:54-56`. Backward gathers this 1-D-per-row count and
// `unsqueeze`s along K, so we keep the storage minimal here.
//
// `out=` contract: upstream MEAN does **not** honor an accumulate contract
// (it overwrites buckets touched by `index`). We match that. Buckets not
// touched by `index` are left at zero (fresh `out`) or untouched (supplied
// `out`). This means the `out=` form of `segment_mean_coo` is rarely useful
// directly, but the kernel still accepts it for API symmetry with sum.
at::Tensor segment_mean_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_mean_coo: src.dim() must be >= index.dim() "
              "(got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  const int64_t dim = index.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_mean_coo: index must have at least 1 dimension");

  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i)
    sizes[i] = src.size(i);
  auto index_b = index.expand(sizes).contiguous();

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_mean_coo: out.size(",
                    i, ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      out_sizes[dim] = dim_size.value();
    } else if (index_b.numel() == 0) {
      out_sizes[dim] = 0;
    } else {
      auto tmp = index_b.select(dim, index_b.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      out_sizes[dim] = 1 + *tmp.data_ptr<int64_t>();
    }
    out = at::zeros(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0) {
    return out;
  }

  const int64_t E = src_c.size(dim);
  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= index_b.size(i);
  const int64_t K = src_c.numel() / index_b.numel();
  const int64_t N = out.size(dim);

  // Count storage: shape `index_b.shape` with last dim replaced by `N`. This
  // is `B*N` flat per the `dim` axis — no trailing K. Allocate as float to
  // match `out`'s scalar type at division time, but use the same dtype as
  // `out` so we can re-use the dispatcher branch.
  auto count_sizes = index_b.sizes().vec();
  count_sizes[dim] = N;
  auto count = at::zeros(count_sizes, src_c.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_mean_coo_cpu", [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* count_data = count.data_ptr<scalar_t>();
        const auto* index_data = index_b.data_ptr<int64_t>();

        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                const int64_t idx_off = b * E;
                const int64_t cnt_off = b * N;

                if (E == 0)
                  continue;

                int64_t cur_idx = index_data[idx_off];
                int64_t run_start = 0;

                if (K == 1) {
                  opmath_t acc = static_cast<opmath_t>(0);
                  for (int64_t e = 0; e < E; ++e) {
                    acc += static_cast<opmath_t>(src_data[src_off + e]);
                    const bool last = (e == E - 1);
                    const int64_t next_idx =
                        last ? cur_idx : index_data[idx_off + e + 1];
                    if (last || next_idx != cur_idx) {
                      out_data[out_off + cur_idx] = static_cast<scalar_t>(acc);
                      count_data[cnt_off + cur_idx] =
                          static_cast<scalar_t>(e + 1 - run_start);
                      if (!last) {
                        cur_idx = next_idx;
                        acc = static_cast<opmath_t>(0);
                        run_start = e + 1;
                      }
                    }
                  }
                } else {
                  std::vector<opmath_t> acc(K, static_cast<opmath_t>(0));
                  for (int64_t e = 0; e < E; ++e) {
                    for (int64_t k = 0; k < K; ++k)
                      acc[k] +=
                          static_cast<opmath_t>(src_data[src_off + e * K + k]);
                    const bool last = (e == E - 1);
                    const int64_t next_idx =
                        last ? cur_idx : index_data[idx_off + e + 1];
                    if (last || next_idx != cur_idx) {
                      for (int64_t k = 0; k < K; ++k)
                        out_data[out_off + cur_idx * K + k] =
                            static_cast<scalar_t>(acc[k]);
                      count_data[cnt_off + cur_idx] =
                          static_cast<scalar_t>(e + 1 - run_start);
                      if (!last) {
                        cur_idx = next_idx;
                        for (int64_t k = 0; k < K; ++k)
                          acc[k] = static_cast<opmath_t>(0);
                        run_start = e + 1;
                      }
                    }
                  }
                }
              }
            });
      });

  // Buckets that received no entries have count == 0; clamp to 1 to avoid
  // division by zero (the corresponding `out` slot is already 0 or
  // user-supplied and untouched).
  count.masked_fill_(count < static_cast<int64_t>(1), 1);

  // Broadcast count along trailing K dims by unsqueezing once per trailing
  // dim of `src` past `index.dim()`.
  auto count_b = count;
  for (int64_t i = 0; i < src_c.dim() - index_b.dim(); ++i)
    count_b = count_b.unsqueeze(-1);
  out.div_(count_b);

  return out;
}

// CPU implementation of `pyg::segment_min_coo`.
//
// Same (B, E, K) layout as `segment_sum_coo_kernel` with `dim = index.dim() -
// 1`. Sequential pass over the sorted-ascending `index`: maintain a running
// minimum value and its first-match argindex per bucket run. Two output
// tensors:
//   * `out`: the per-bucket minimum value, init to `numeric_limits::max()`
//     when `out=None` so the first contributing element wins. Empty
//     buckets (no contributing source element) are reset to `0` after
//     the reduction loop (matched against the sentinel in `arg_out`,
//     mirrors `scatter_min_kernel`).
//   * `arg_out`: the source position that produced each per-bucket min,
//     init to the sentinel `src.size(dim)` (one past the last valid
//     index along `dim`). Has dtype `int64` (matches `index.options()`).
//
// **Determinism:** the inner E loop is sequential per `B` slice and uses
// strict `<` on the comparison, so on ties the kernel preserves
// *first-match* semantics. Parallelism is only over the leading `B` dim
// where slices write to disjoint sub-regions of `out` / `arg_out`.
//
// `out=` contract: when the caller supplies `optional_out`, the running
// state begins from the caller's buffer (no max-init), and the kernel
// **continues** the reduction over that state. The caller is responsible
// for any non-default starting value. This matches upstream.
std::tuple<at::Tensor, at::Tensor> segment_min_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_min_coo: src.dim() must be >= index.dim() "
              "(got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  const int64_t dim = index.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_min_coo: index must have at least 1 dimension");

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream
  // `segment_coo_cpu.cpp:19-22`). Then force contiguous for the linear scan.
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i)
    sizes[i] = src.size(i);
  auto index_b = index.expand(sizes).contiguous();

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_min_coo: out.size(",
                    i, ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      out_sizes[dim] = dim_size.value();
    } else if (index_b.numel() == 0) {
      out_sizes[dim] = 0;
    } else {
      auto tmp = index_b.select(dim, index_b.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      out_sizes[dim] = 1 + *tmp.data_ptr<int64_t>();
    }
    // Allocate uninitialized; the dispatch below fills with
    // `numeric_limits<scalar_t>::max()` before the reduction loop.
    out = at::empty(out_sizes, src_c.options());
  }

  // `arg_out` is always freshly allocated (independent of `optional_out`)
  // and starts at the sentinel `src.size(dim)`. The sentinel both encodes
  // "empty bucket" for the `masked_fill_` post-step and feeds into the
  // backward's `+1`/`narrow` trick.
  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, index_b.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  const int64_t E = src_c.size(dim);
  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= index_b.size(i);
  const int64_t K = src_c.numel() / index_b.numel();
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_min_coo_cpu", [&] {
        // Init `out` to `numeric_limits::max()` only when the caller did
        // not supply `optional_out` (otherwise the caller's state is the
        // running state, per the `out=` contract).
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::max());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* index_data = index_b.data_ptr<int64_t>();

        // Parallelize over the leading (B) dim only. The inner E loop must
        // remain sequential to preserve first-match argindex semantics on
        // tied source values within a bucket run.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                const int64_t idx_off = b * E;

                if (E == 0)
                  continue;

                // Sequential pass over the sorted-ascending `index` row.
                // For each element, look up the current bucket's running
                // min/argindex (already in `out`/`arg_out`) and update
                // when the new value is strictly smaller (first-match).
                for (int64_t e = 0; e < E; ++e) {
                  const int64_t idx = index_data[idx_off + e];
                  if (K == 1) {
                    const scalar_t v = src_data[src_off + e];
                    auto* out_slot = out_data + out_off + idx;
                    if (v < *out_slot) {
                      *out_slot = v;
                      arg_out_data[out_off + idx] = e;
                    }
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const scalar_t v = src_data[src_off + e * K + k];
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

// CPU implementation of `pyg::segment_max_coo`.
//
// Same (B, E, K) layout as `segment_sum_coo_kernel` with `dim = index.dim() -
// 1`. Sequential pass over the sorted-ascending `index`: maintain a running
// maximum value and its first-match argindex per bucket run. Two output
// tensors:
//   * `out`: the per-bucket maximum value, init to `numeric_limits::lowest()`
//     when `out=None` so the first contributing element wins. Empty
//     buckets (no contributing source element) are reset to `0` after
//     the reduction loop (matched against the sentinel in `arg_out`,
//     mirrors `scatter_max_kernel`).
//   * `arg_out`: the source position that produced each per-bucket max,
//     init to the sentinel `src.size(dim)` (one past the last valid
//     index along `dim`). Has dtype `int64` (matches `index.options()`).
//
// **Determinism:** the inner E loop is sequential per `B` slice and uses
// strict `>` on the comparison, so on ties the kernel preserves
// *first-match* semantics. Parallelism is only over the leading `B` dim
// where slices write to disjoint sub-regions of `out` / `arg_out`.
//
// `out=` contract: when the caller supplies `optional_out`, the running
// state begins from the caller's buffer (no lowest-init), and the kernel
// **continues** the reduction over that state. The caller is responsible
// for any non-default starting value. This matches upstream.
std::tuple<at::Tensor, at::Tensor> segment_max_coo_kernel(
    const at::Tensor& src,
    const at::Tensor& index,
    const std::optional<at::Tensor>& optional_out,
    std::optional<int64_t> dim_size) {
  TORCH_CHECK(src.dim() >= index.dim(),
              "segment_max_coo: src.dim() must be >= index.dim() "
              "(got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  const int64_t dim = index.dim() - 1;
  TORCH_CHECK(dim >= 0,
              "segment_max_coo: index must have at least 1 dimension");

  // Broadcast `index` up to `src.shape[:index.dim()]` (upstream
  // `segment_coo_cpu.cpp:19-22`). Then force contiguous for the linear scan.
  auto sizes = index.sizes().vec();
  for (int64_t i = 0; i < index.dim(); ++i)
    sizes[i] = src.size(i);
  auto index_b = index.expand(sizes).contiguous();

  auto src_c = src.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < out.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "segment_max_coo: out.size(",
                    i, ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    if (dim_size.has_value()) {
      out_sizes[dim] = dim_size.value();
    } else if (index_b.numel() == 0) {
      out_sizes[dim] = 0;
    } else {
      auto tmp = index_b.select(dim, index_b.size(dim) - 1);
      tmp = tmp.numel() > 1 ? tmp.max() : tmp;
      out_sizes[dim] = 1 + *tmp.data_ptr<int64_t>();
    }
    // Allocate uninitialized; the dispatch below fills with
    // `numeric_limits<scalar_t>::lowest()` before the reduction loop.
    out = at::empty(out_sizes, src_c.options());
  }

  // `arg_out` is always freshly allocated (independent of `optional_out`)
  // and starts at the sentinel `src.size(dim)`. The sentinel both encodes
  // "empty bucket" for the `masked_fill_` post-step and feeds into the
  // backward's `+1`/`narrow` trick.
  const int64_t sentinel = src_c.size(dim);
  auto arg_out = at::full(out.sizes(), sentinel, index_b.options());

  if (src_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return std::make_tuple(out, arg_out);
  }

  const int64_t E = src_c.size(dim);
  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= index_b.size(i);
  const int64_t K = src_c.numel() / index_b.numel();
  const int64_t N = out.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "segment_max_coo_cpu", [&] {
        // Init `out` to `numeric_limits::lowest()` only when the caller did
        // not supply `optional_out` (otherwise the caller's state is the
        // running state, per the `out=` contract).
        if (!optional_out.has_value()) {
          out.fill_(std::numeric_limits<scalar_t>::lowest());
        }

        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        auto* arg_out_data = arg_out.data_ptr<int64_t>();
        const auto* index_data = index_b.data_ptr<int64_t>();

        // Parallelize over the leading (B) dim only. The inner E loop must
        // remain sequential to preserve first-match argindex semantics on
        // tied source values within a bucket run.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * E * K;
                const int64_t out_off = b * N * K;
                const int64_t idx_off = b * E;

                if (E == 0)
                  continue;

                // Sequential pass over the sorted-ascending `index` row.
                // For each element, look up the current bucket's running
                // max/argindex (already in `out`/`arg_out`) and update
                // when the new value is strictly greater (first-match).
                for (int64_t e = 0; e < E; ++e) {
                  const int64_t idx = index_data[idx_off + e];
                  if (K == 1) {
                    const scalar_t v = src_data[src_off + e];
                    auto* out_slot = out_data + out_off + idx;
                    if (v > *out_slot) {
                      *out_slot = v;
                      arg_out_data[out_off + idx] = e;
                    }
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      const scalar_t v = src_data[src_off + e * K + k];
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

// CPU implementation of `pyg::gather_coo`.
//
// COO ops fix the gather axis at `dim = index.dim() - 1`. Per element:
//
//   out[..., e, ...] = src[..., index[..., e], ...]
//
// Layout (mirrors `segment_sum_coo_kernel`):
//   * B = product of leading dims of `index` before `dim`,
//   * E = index.size(dim) (= out.size(dim)),
//   * N = src.size(dim),
//   * K = product of trailing dims of `src` past `index.dim()`.
//
// `out=` contract: **overwrite** the caller's buffer per element.
at::Tensor gather_coo_kernel(const at::Tensor& src,
                             const at::Tensor& index,
                             const std::optional<at::Tensor>& optional_out) {
  TORCH_CHECK(src.dim() >= index.dim(),
              "gather_coo: src.dim() must be >= index.dim() (got src.dim()=",
              src.dim(), ", index.dim()=", index.dim(), ")");

  const int64_t dim = index.dim() - 1;
  TORCH_CHECK(dim >= 0, "gather_coo: index must have at least 1 dimension");

  // For the leading dims before `dim`, upstream requires src.size(i) ==
  // index.size(i) (no broadcasting). Match that.
  for (int64_t i = 0; i < dim; ++i) {
    TORCH_CHECK(src.size(i) == index.size(i), "gather_coo: src.size(", i,
                ") must match index.size(", i, ")");
  }

  auto src_c = src.contiguous();
  auto index_c = index.contiguous();

  at::Tensor out;
  if (optional_out.has_value()) {
    out = optional_out.value().contiguous();
    for (int64_t i = 0; i < src_c.dim(); ++i) {
      if (i != dim)
        TORCH_CHECK(src_c.size(i) == out.size(i), "gather_coo: out.size(", i,
                    ") must match src.size(", i, ")");
    }
  } else {
    auto out_sizes = src_c.sizes().vec();
    out_sizes[dim] = index_c.size(dim);
    out = at::empty(out_sizes, src_c.options());
  }

  if (src_c.numel() == 0 || index_c.numel() == 0) {
    if (!optional_out.has_value()) {
      out.fill_(0);
    }
    return out;
  }

  const int64_t E = index_c.size(dim);
  int64_t B = 1;
  for (int64_t i = 0; i < dim; ++i)
    B *= index_c.size(i);
  const int64_t K = out.numel() / index_c.numel();
  const int64_t N = src_c.size(dim);

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, src_c.scalar_type(),
      "gather_coo_cpu", [&] {
        const auto* src_data = src_c.data_ptr<scalar_t>();
        auto* out_data = out.data_ptr<scalar_t>();
        const auto* index_data = index_c.data_ptr<int64_t>();

        // Parallelize over the leading (B) dim only. Each `b` slice reads
        // from a disjoint sub-region of `src` and writes to a disjoint
        // sub-region of `out`.
        at::parallel_for(
            0, B, at::internal::GRAIN_SIZE, [&](int64_t b_beg, int64_t b_end) {
              for (int64_t b = b_beg; b < b_end; ++b) {
                const int64_t src_off = b * N * K;
                const int64_t out_off = b * E * K;
                const int64_t idx_off = b * E;
                for (int64_t e = 0; e < E; ++e) {
                  const int64_t idx = index_data[idx_off + e];
                  if (K == 1) {
                    out_data[out_off + e] = src_data[src_off + idx];
                  } else {
                    for (int64_t k = 0; k < K; ++k) {
                      out_data[out_off + e * K + k] =
                          src_data[src_off + idx * K + k];
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
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_sum_coo"),
         TORCH_FN(segment_sum_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_mean_coo"),
         TORCH_FN(segment_mean_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_min_coo"),
         TORCH_FN(segment_min_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_max_coo"),
         TORCH_FN(segment_max_coo_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"), TORCH_FN(gather_coo_kernel));
}

}  // namespace ops
}  // namespace pyg
