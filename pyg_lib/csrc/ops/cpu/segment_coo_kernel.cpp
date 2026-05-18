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
  m.impl(TORCH_SELECTIVE_NAME("pyg::gather_coo"), TORCH_FN(gather_coo_kernel));
}

}  // namespace ops
}  // namespace pyg
