#include <algorithm>
#include <climits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/Parallel.h>
#include <parallel_hashmap/phmap.h>
#include <torch/library.h>

#include "pyg_lib/csrc/config.h"
#include "pyg_lib/csrc/utils/convert.h"

#if WITH_MKL_BLAS()
#include <mkl.h>
#endif

namespace pyg {
namespace ops {

namespace {

#if WITH_MKL_BLAS()

void mkl_blas_gemm_batched(const int* m_array,
                           const int* n_array,
                           const int* k_array,
                           const float* alpha_array,
                           const float** a_array,
                           const int* lda_array,
                           const float** b_array,
                           const int* ldb_array,
                           const float* beta_array,
                           float** c_array,
                           const int* ldc_array,
                           const int group_count,
                           const int* group_size) {
  std::vector<CBLAS_TRANSPOSE> transa(group_count, CblasNoTrans);
  std::vector<CBLAS_TRANSPOSE> transb(group_count, CblasNoTrans);
  const CBLAS_TRANSPOSE* transa_array = transa.data();
  const CBLAS_TRANSPOSE* transb_array = transb.data();
  cblas_sgemm_batch(CblasRowMajor, transa_array, transb_array, m_array, n_array,
                    k_array, alpha_array, a_array, lda_array, b_array,
                    ldb_array, beta_array, c_array, ldc_array, group_count,
                    group_size);
}

void mkl_blas_gemm_batched(const int* m_array,
                           const int* n_array,
                           const int* k_array,
                           const double* alpha_array,
                           const double** a_array,
                           const int* lda_array,
                           const double** b_array,
                           const int* ldb_array,
                           const double* beta_array,
                           double** c_array,
                           const int* ldc_array,
                           const int group_count,
                           const int* group_size) {
  std::vector<CBLAS_TRANSPOSE> transa(group_count, CblasNoTrans);
  std::vector<CBLAS_TRANSPOSE> transb(group_count, CblasNoTrans);
  const CBLAS_TRANSPOSE* transa_array = transa.data();
  const CBLAS_TRANSPOSE* transb_array = transb.data();
  cblas_dgemm_batch(CblasRowMajor, transa_array, transb_array, m_array, n_array,
                    k_array, alpha_array, a_array, lda_array, b_array,
                    ldb_array, beta_array, c_array, ldc_array, group_count,
                    group_size);
}

#else

void mkl_blas_gemm_batched(const int* m_array,
                           const int* n_array,
                           const int* k_array,
                           const float* alpha_array,
                           const float** a_array,
                           const int* lda_array,
                           const float** b_array,
                           const int* ldb_array,
                           const float* beta_array,
                           float** c_array,
                           const int* ldc_array,
                           const int group_count,
                           const int* group_size) {
  TORCH_INTERNAL_ASSERT(false, "MKL BLAS is not supported");
}

void mkl_blas_gemm_batched(const int* m_array,
                           const int* n_array,
                           const int* k_array,
                           const double* alpha_array,
                           const double** a_array,
                           const int* lda_array,
                           const double** b_array,
                           const int* ldb_array,
                           const double* beta_array,
                           double** c_array,
                           const int* ldc_array,
                           const int group_count,
                           const int* group_size) {
  TORCH_INTERNAL_ASSERT(false, "MKL BLAS is not supported");
}

#endif

template <typename scalar_t>
using is_blas_library_type =
    std::integral_constant<bool,
                           std::is_same<scalar_t, float>::value ||
                               std::is_same<scalar_t, double>::value>;

template <typename scalar_t>
bool mkl_path_available() {
  return (WITH_MKL_BLAS() && AT_MKL_ENABLED() &&
          is_blas_library_type<scalar_t>::value);
}

bool mkl_path_possible(const at::IntArrayRef& sizes, int64_t n, int64_t k) {
  const int64_t limit = INT_MAX;
  const bool is_size_invalid =
      n > limit || k > limit ||
      std::any_of(sizes.cbegin(), sizes.cend(),
                  [limit](int64_t m) { return m > limit; });
  return !is_size_invalid;
}

bool mkl_path_possible(const std::vector<at::Tensor>& left,
                       const std::vector<at::Tensor>& right) {
  const int64_t limit = INT_MAX;
  const bool mk_invalid =
      std::any_of(left.cbegin(), left.cend(), [limit](const at::Tensor& t) {
        return t.size(0) > limit || t.size(-1) > limit;
      });
  const bool n_invalid =
      std::any_of(right.cbegin(), right.cend(),
                  [limit](const at::Tensor& t) { return t.size(-1) > limit; });
  const bool is_size_invalid = mk_invalid || n_invalid;
  return !is_size_invalid;
}

template <typename scalar_t>
void parallel_mkl_blas_gemm_batched(const std::vector<int>& ms,
                                    const std::vector<int>& ns,
                                    const std::vector<int>& ks,
                                    const std::vector<scalar_t>& alpha,
                                    const scalar_t** src0_ptrs,
                                    const std::vector<int>& ld_src0,
                                    const scalar_t** src1_ptrs,
                                    const std::vector<int>& ld_src1,
                                    const std::vector<scalar_t>& beta,
                                    scalar_t** dst_ptrs,
                                    const std::vector<int>& ld_dst,
                                    const int group_count,
                                    const std::vector<int>& group_sizes) {
  int64_t work_size = 0;
  for (size_t i = 0; i < group_count; ++i) {
    work_size += ks[i] * group_sizes[i];
  }

  if (group_count > 1) {
    at::parallel_for(0, group_count, 1, [&](size_t beg, size_t end) {
      for (size_t i = beg; i < end; ++i) {
        const auto offset = (i) ? std::accumulate(group_sizes.begin(),
                                                  group_sizes.begin() + i, 0)
                                : 0;
        const scalar_t** src0_ptrs_local = src0_ptrs + offset;
        const scalar_t** src1_ptrs_local = src1_ptrs + offset;
        scalar_t** dst_ptrs_local = dst_ptrs + offset;
        mkl_blas_gemm_batched(&ms[i], &ns[i], &ks[i], &alpha[i],
                              src0_ptrs_local, &ld_src0[i], src1_ptrs_local,
                              &ld_src1[i], &beta[i], dst_ptrs_local, &ld_dst[i],
                              1, &group_sizes[i]);
      }
    });
  } else {
    at::parallel_for(0, group_sizes.front(), 1, [&](size_t beg, size_t end) {
      for (size_t i = beg; i < end; ++i) {
        const scalar_t** src0_ptrs_local = src0_ptrs + i;
        const scalar_t** src1_ptrs_local = src1_ptrs + i;
        scalar_t** dst_ptrs_local = dst_ptrs + i;
        const int bs = 1;
        mkl_blas_gemm_batched(ms.data(), ns.data(), ks.data(), alpha.data(),
                              src0_ptrs_local, ld_src0.data(), src1_ptrs_local,
                              ld_src1.data(), beta.data(), dst_ptrs_local,
                              ld_dst.data(), 1, &bs);
      }
    });
  }
}

void grouped_matmul_out_kernel_at_impl(const std::vector<at::Tensor> input,
                                       const std::vector<at::Tensor> other,
                                       std::vector<at::Tensor> out) {
  for (size_t i = 0; i < out.size(); ++i) {
    at::matmul_out(const_cast<at::Tensor&>(out[i]), input[i], other[i]);
  }
}

void grouped_matmul_out_kernel_mkl_impl(const std::vector<at::Tensor> input,
                                        const std::vector<at::Tensor> other,
                                        std::vector<at::Tensor> out) {
#if WITH_MKL_BLAS()
  // matrix_params<M, N, K>
  using matrix_params = std::tuple<int, int, int>;
  phmap::flat_hash_map<matrix_params, std::vector<size_t>> groups;
  for (size_t i = 0; i < input.size(); ++i) {
    const matrix_params mp = {input[i].size(0), other[i].size(-1),
                              input[i].size(-1)};
    if (groups.count(mp)) {
      groups[mp].push_back(i);
    } else {
      groups.insert({mp, {i}});
    }
  }

  AT_DISPATCH_FLOATING_TYPES(
      input.front().scalar_type(), "grouped_matmul_out_kernel_mkl_impl", [&] {
        const auto group_count = static_cast<int>(groups.size());
        std::vector<scalar_t> alpha(group_count, 1);
        std::vector<scalar_t> beta(group_count, 0);

        std::vector<int> ms(group_count);
        std::vector<int> ns(group_count);
        std::vector<int> ks(group_count);
        std::vector<int> ld_src0(group_count);
        std::vector<int> ld_src1(group_count);
        std::vector<int> ld_dst(group_count);
        std::vector<int> group_sizes(group_count);
        std::vector<scalar_t*> src0;
        std::vector<scalar_t*> src1;
        std::vector<scalar_t*> dst;

        size_t group_idx = 0;
        for (const auto& group_kv : groups) {
          int m;
          int n;
          int k;
          std::tie(m, n, k) = group_kv.first;
          const auto& indices = group_kv.second;

          ms[group_idx] = m;
          ns[group_idx] = n;
          ks[group_idx] = k;
          ld_src0[group_idx] = k;
          ld_src1[group_idx] = n;
          ld_dst[group_idx] = n;
          group_sizes[group_idx] = indices.size();
          ++group_idx;

          for (const auto tensor_idx : indices) {
            src0.push_back(input[tensor_idx].data_ptr<scalar_t>());
            src1.push_back(other[tensor_idx].data_ptr<scalar_t>());
            dst.push_back(out[tensor_idx].data_ptr<scalar_t>());
          }
        }

        auto src0_ptrs = const_cast<const scalar_t**>(src0.data());
        auto src1_ptrs = const_cast<const scalar_t**>(src1.data());
        auto dst_ptrs = dst.data();

#if AT_MKL_SEQUENTIAL()
        // unlikely to happen - requires Torch to be built from source with
        // explicit flag denoting MKL sequential version
        parallel_mkl_blas_gemm_batched(ms, ns, ks, alpha, src0_ptrs, ld_src0,
                                       src1_ptrs, ld_src1, beta, dst_ptrs,
                                       ld_dst, group_count, group_sizes);
#else
        mkl_blas_gemm_batched(ms.data(), ns.data(), ks.data(), alpha.data(),
                              src0_ptrs, ld_src0.data(), src1_ptrs, ld_src1.data(),
                              beta.data(), dst_ptrs, ld_dst.data(), group_count,
                              group_sizes.data());
#endif
      });
#endif
}

std::vector<at::Tensor> grouped_matmul_kernel(const at::TensorList input,
                                              const at::TensorList other) {
  const auto n_matrices = input.size();

  std::vector<at::Tensor> input_contig;
  std::vector<at::Tensor> other_contig;
  std::vector<at::Tensor> out;

  input_contig.reserve(n_matrices);
  other_contig.reserve(n_matrices);
  out.reserve(n_matrices);

  for (size_t i = 0; i < n_matrices; ++i) {
    input_contig.emplace_back(input[i].contiguous());
    other_contig.emplace_back(other[i].contiguous());
    out.emplace_back(input_contig[i].new_empty(
        {input_contig[i].size(0), other_contig[i].size(-1)}));
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input_contig.front().scalar_type(), "grouped_matmul_kernel", [&] {
        if (mkl_path_available<scalar_t>() &&
            mkl_path_possible(input_contig, other_contig)) {
          grouped_matmul_out_kernel_mkl_impl(input_contig, other_contig, out);
        } else {
          grouped_matmul_out_kernel_at_impl(input_contig, other_contig, out);
        }
      });

  return out;
}

struct offset_params {
  int src0_offset;
  int src1_offset;
  int dst_offset;

  offset_params& operator+=(const offset_params& rhs) {
    this->src0_offset += rhs.src0_offset;
    this->src1_offset += rhs.src1_offset;
    this->dst_offset += rhs.dst_offset;
    return *this;
  }
};

void segment_matmul_out_kernel_mkl_impl(const at::Tensor& input,
                                        const at::Tensor& other,
                                        at::Tensor& out,
                                        const at::IntArrayRef& sizes) {
#if WITH_MKL_BLAS()
  const int n = other.size(-1);
  const int k = input.size(-1);
  const int nk = n * k;
  phmap::flat_hash_map<int, std::vector<size_t>> groups;
  std::vector<offset_params> offsets = {{0, 0, 0}};
  offsets.reserve(sizes.size() + 1);
  for (size_t i = 0; i < sizes.size(); ++i) {
    const int m = sizes[i];
    if (groups.count(m)) {
      groups[m].push_back(i);
    } else {
      groups.insert({m, {i}});
    }

    offset_params offset = {m * k, nk, m * n};
    offset += offsets.back();
    offsets.push_back(offset);
  }
  offsets.pop_back();

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "segment_matmul_out_kernel_mkl_impl", [&] {
        const auto group_count = static_cast<int>(groups.size());
        std::vector<scalar_t> alpha(group_count, 1);
        std::vector<scalar_t> beta(group_count, 0);
        std::vector<int> ns(group_count, n);
        std::vector<int> ks(group_count, k);
        std::vector<int> ld_src0(group_count, k);
        std::vector<int> ld_src1(group_count, n);
        std::vector<int> ld_dst(group_count, n);

        std::vector<int> ms(group_count);
        std::vector<int> group_sizes(group_count);
        std::vector<scalar_t*> src0;
        std::vector<scalar_t*> src1;
        std::vector<scalar_t*> dst;

        const auto src0_base_ptr = input.data_ptr<scalar_t>();
        const auto src1_base_ptr = other.data_ptr<scalar_t>();
        const auto dst_base_ptr = out.data_ptr<scalar_t>();

        size_t group_idx = 0;
        for (const auto& group_kv : groups) {
          int m = group_kv.first;
          const auto& indices = group_kv.second;

          ms[group_idx] = m;
          group_sizes[group_idx] = indices.size();
          ++group_idx;

          for (const auto offset_idx : indices) {
            const auto offset = offsets[offset_idx];
            src0.push_back(src0_base_ptr + offset.src0_offset);
            src1.push_back(src1_base_ptr + offset.src1_offset);
            dst.push_back(dst_base_ptr + offset.dst_offset);
          }
        }

        auto src0_ptrs = const_cast<const scalar_t**>(src0.data());
        auto src1_ptrs = const_cast<const scalar_t**>(src1.data());
        auto dst_ptrs = dst.data();

#if AT_MKL_SEQUENTIAL()
        // unlikely to happen - requires Torch to be built from source with
        // explicit flag denoting MKL sequential version
        parallel_mkl_blas_gemm_batched(ms, ns, ks, alpha, src0_ptrs, ld_src0,
                                       src1_ptrs, ld_src1, beta, dst_ptrs,
                                       ld_dst, group_count, group_sizes);
#else
        mkl_blas_gemm_batched(ms.data(), ns.data(), ks.data(), alpha.data(),
                              src0_ptrs, ld_src0.data(), src1_ptrs, ld_src1.data(),
                              beta.data(), dst_ptrs, ld_dst.data(), group_count,
                              group_sizes.data());
#endif
      });
#endif
}

at::Tensor segment_matmul_kernel(const at::Tensor& input,
                                 const at::Tensor& ptr,
                                 const at::Tensor& other) {
  const auto size = pyg::utils::size_from_ptr(ptr).cpu();
  const auto sizes = at::IntArrayRef(size.data_ptr<int64_t>(), size.numel());
  const auto input_contig = input.contiguous();
  const auto other_contig = other.contiguous();
  auto out = input_contig.new_empty({input.size(0), other.size(-1)});

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input_contig.scalar_type(), "segment_matmul_kernel", [&] {
        const auto n = other_contig.size(-1);
        const auto k = input_contig.size(-1);
        if (mkl_path_available<scalar_t>() && mkl_path_possible(sizes, n, k)) {
          segment_matmul_out_kernel_mkl_impl(input_contig, other_contig, out,
                                             sizes);
        } else {
          auto outs = out.split_with_sizes(/*split_size=*/sizes, /*dim=*/0);
          for (auto& out_part : outs) {
            out_part.unsqueeze_(0);
          }
          grouped_matmul_out_kernel_at_impl(
              input_contig.split_with_sizes(/*split_size=*/sizes, /*dim=*/0),
              other_contig.split(/*split_size=*/1, /*dim=*/0), outs);
        }
      });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grouped_matmul"),
         TORCH_FN(grouped_matmul_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::segment_matmul"),
         TORCH_FN(segment_matmul_kernel));
}

}  // namespace ops
}  // namespace pyg
