#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>
#include <map>

enum ReductionType { SUM, MEAN, MUL, DIV, MIN, MAX };

const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"sum", SUM}, {"mean", MEAN}, {"mul", MUL},
    {"div", DIV}, {"min", MIN},   {"max", MAX},
};

#define AT_DISPATCH_REDUCTION_TYPES(reduce, ...) \
  [&] {                                          \
    switch (reduce2REDUCE.at(reduce)) {          \
      case SUM: {                                \
        const ReductionType REDUCE = SUM;        \
        return __VA_ARGS__();                    \
      }                                          \
      case MEAN: {                               \
        const ReductionType REDUCE = MEAN;       \
        return __VA_ARGS__();                    \
      }                                          \
      case MUL: {                                \
        const ReductionType REDUCE = MUL;        \
        return __VA_ARGS__();                    \
      }                                          \
      case DIV: {                                \
        const ReductionType REDUCE = DIV;        \
        return __VA_ARGS__();                    \
      }                                          \
      case MIN: {                                \
        const ReductionType REDUCE = MIN;        \
        return __VA_ARGS__();                    \
      }                                          \
      case MAX: {                                \
        const ReductionType REDUCE = MAX;        \
        return __VA_ARGS__();                    \
      }                                          \
    }                                            \
  }()

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")

template <typename scalar_t, ReductionType REDUCE>
struct Reducer {
  static inline __host__ __device__ scalar_t init() {
    if (REDUCE == MUL || REDUCE == DIV)
      return (scalar_t)1;
    else if (REDUCE == MIN)
      return std::numeric_limits<scalar_t>::max();
    else if (REDUCE == MAX)
      return std::numeric_limits<scalar_t>::lowest();
    else
      return (scalar_t)0;
  }

  static inline __host__ __device__ void update(scalar_t* val,
                                                scalar_t new_val,
                                                int64_t* arg,
                                                int64_t new_arg) {
    if (REDUCE == SUM || REDUCE == MEAN)
      *val = *val + new_val;
    else if (REDUCE == MUL)
      *val = *val * new_val;
    else if (REDUCE == DIV)
      *val = *val / new_val;
    else if ((REDUCE == MIN &&
              static_cast<float>(new_val) < static_cast<float>(*val)) ||
             (REDUCE == MAX &&
              static_cast<float>(new_val) > static_cast<float>(*val))) {
      *val = new_val;
      *arg = new_arg;
    }
  }

  static inline __host__ __device__ void write(scalar_t* address,
                                               scalar_t val,
                                               int64_t* arg_address,
                                               int64_t arg,
                                               int count) {
    if (REDUCE == SUM || REDUCE == MUL || REDUCE == DIV)
      *address = val;
    else if (REDUCE == MEAN)
      *address = val / (scalar_t)(count > 0 ? count : 1);
    else if (REDUCE == MIN || REDUCE == MAX) {
      if (count > 0) {
        *address = val;
        *arg_address = arg;
      } else
        *address = (scalar_t)0;
    }
  }
};

__device__ __inline__ at::Half __shfl_up_sync(const unsigned mask,
                                              const at::Half var,
                                              const unsigned int delta) {
  return __shfl_up_sync(mask, var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_down_sync(const unsigned mask,
                                                const at::Half var,
                                                const unsigned int delta) {
  return __shfl_down_sync(mask, var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_sync(const unsigned mask,
                                           const at::Half var,
                                           const int delta) {
  return __shfl_sync(mask, var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_up(const at::Half var,
                                         const unsigned int delta) {
  return __shfl_up(var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl_down(const at::Half var,
                                           const unsigned int delta) {
  return __shfl_down(var.operator __half(), delta);
}

__device__ __inline__ at::Half __shfl(const at::Half var, const int delta) {
  return __shfl(var.operator __half(), delta);
}

#ifdef USE_ROCM
__device__ __inline__ at::Half __ldg(const at::Half* ptr) {
  return __ldg(reinterpret_cast<const __half*>(ptr));
}
#define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)
#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)
#define SHFL_SYNC(mask, var, delta) __shfl(var, delta)
#else
#define SHFL_UP_SYNC __shfl_up_sync
#define SHFL_DOWN_SYNC __shfl_down_sync
#define SHFL_SYNC __shfl_sync
#endif

namespace pyg {
namespace ops {

namespace {

#define THREADS 256
#define FULL_MASK 0xffffffff

// Paper: Design Principles for Sparse Matrix Multiplication on the GPU
// Code:  https://github.com/owensgroup/merge-spmm
template <typename scalar_t, ReductionType REDUCE, bool HAS_VALUE>
__global__ void spmm_kernel(const int64_t* rowptr_data,
                            const int64_t* col_data,
                            const scalar_t* value_data,
                            const scalar_t* mat_data,
                            scalar_t* out_data,
                            int64_t* arg_out_data,
                            int B,
                            int M,
                            int N,
                            int K) {
  // We ignore blockIdx.y here, because threads
  // across `blockIdx.y` are treated equally.
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int row = thread_idx >> 5;             // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1);  // thread_idx % 32
  int batch_idx = row / M;

  // Compute the column index of `mat` in which the thread is operating.
  int mat_col_idx = lane_idx + (blockIdx.y << 5);

  // Compute the output index (row-major order).
  int out_idx = row * K + mat_col_idx;

  // Helper arrays for warp communication.
  int mat_row, mat_rows[32];
  scalar_t val, vals[HAS_VALUE ? 32 : 1];

  // Do not aggregate/write across the Y-axis (lane_idx < leftover).
  int leftover = K - (blockIdx.y << 5);

  if (batch_idx < B) {
    int row_start = __ldg(rowptr_data + (row % M));
    int row_end = __ldg(rowptr_data + (row % M) + 1);
    int col_idx = row_start + lane_idx;

    scalar_t result = Reducer<scalar_t, REDUCE>::init();
    int64_t arg;

    // Iterate over all `col` indices in parallel within a warp.
    for (int c = row_start; c < row_end; c += 32) {
      if (col_idx < row_end) {
        // Coalesced memory access into `col` and `val`.
        mat_row = __ldg(col_data + col_idx) * K;
        if (HAS_VALUE)
          val = __ldg(value_data + col_idx);
      } else {
        mat_row = -1;
        if (HAS_VALUE)
          val = (scalar_t)0;
      }
      col_idx += 32;

#pragma unroll
      for (int i = 0; i < 32; i++) {
        // Communication between all threads in a warp.
        mat_rows[i] = SHFL_SYNC(FULL_MASK, mat_row, i);
        if (HAS_VALUE)
          vals[i] = SHFL_SYNC(FULL_MASK, val, i);
      }

#pragma unroll
      for (int i = 0; i < 32; i++) {
        if (lane_idx < leftover && mat_rows[i] != -1) {
          // Coalesced memory access into `mat`.
          val = __ldg(mat_data + batch_idx * N * K + mat_rows[i] + mat_col_idx);
          if (HAS_VALUE)
            val = vals[i] * val;
          Reducer<scalar_t, REDUCE>::update(&result, val, &arg, c + i);
        }
      }
    }

    if (lane_idx < leftover) {
      // Coalesced write into `out`.
      Reducer<scalar_t, REDUCE>::write(out_data + out_idx, result,
                                       arg_out_data + out_idx, arg,
                                       row_end - row_start);
    }
  }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> spmm_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    std::optional<torch::Tensor> optional_value,
    torch::Tensor mat,
    std::string reduce) {
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  if (optional_value.has_value())
    CHECK_CUDA(optional_value.value());
  CHECK_CUDA(mat);
  c10::cuda::MaybeSetDevice(rowptr.get_device());

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  std::optional<torch::Tensor> arg_out = std::nullopt;
  int64_t* arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((32 * B * M + THREADS - 1) / THREADS, (K + 31) / 32);

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      if (optional_value.has_value()) {
        auto value_data = optional_value.value().data_ptr<scalar_t>();
        spmm_kernel<scalar_t, REDUCE, true><<<BLOCKS, THREADS, 0, stream>>>(
            rowptr_data, col_data, value_data, mat_data, out_data, arg_out_data,
            B, M, N, K);
      } else {
        spmm_kernel<scalar_t, REDUCE, false><<<BLOCKS, THREADS, 0, stream>>>(
            rowptr_data, col_data, nullptr, mat_data, out_data, arg_out_data, B,
            M, N, K);
      }
    });
  });

  return std::make_tuple(out, arg_out);
}

template <typename scalar_t, ReductionType REDUCE>
__global__ void spmm_value_bw_kernel(const int64_t* row_data,
                                     const int64_t* rowptr_data,
                                     const int64_t* col_data,
                                     const scalar_t* mat_data,
                                     const scalar_t* grad_data,
                                     scalar_t* out_data,
                                     int B,
                                     int M,
                                     int N,
                                     int E,
                                     int K) {
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int index_idx = (thread_idx >> 5);     // thread_idx / 32
  int lane_idx = thread_idx & (32 - 1);  // thread_idx % 32

  if (index_idx < E) {
    int row = __ldg(row_data + index_idx);
    int col = __ldg(col_data + index_idx);

    scalar_t val = (scalar_t)0;
    for (int b = 0; b < B; b++) {
      for (int k = lane_idx; k < K; k += 32) {
        val += mat_data[b * N * K + col * K + k] *
               grad_data[b * M * K + row * K + k];
      }
    }

#pragma unroll
    for (int i = 32 / 2; i > 0; i /= 2) {  // Parallel reduction inside a warp.
      val += SHFL_DOWN_SYNC(FULL_MASK, val, i);
    }

    if (lane_idx == 0) {
      if (REDUCE == MEAN) {
        int row_start = __ldg(rowptr_data + row);
        int row_end = __ldg(rowptr_data + row + 1);
        val /= (scalar_t)max(row_end - row_start, 1);
      }
      out_data[index_idx] = val;
    }
  }
}

torch::Tensor spmm_value_bw_cuda(torch::Tensor row,
                                 torch::Tensor rowptr,
                                 torch::Tensor col,
                                 torch::Tensor mat,
                                 torch::Tensor grad,
                                 std::string reduce) {
  CHECK_CUDA(row);
  CHECK_CUDA(rowptr);
  CHECK_CUDA(col);
  CHECK_CUDA(mat);
  CHECK_CUDA(grad);
  c10::cuda::MaybeSetDevice(row.get_device());

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto BLOCKS = dim3((E * 32 + THREADS - 1) / THREADS);

  auto out = torch::zeros({row.numel()}, grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, mat.scalar_type(), "_", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      spmm_value_bw_kernel<scalar_t, REDUCE><<<BLOCKS, THREADS, 0, stream>>>(
          row_data, rowptr_data, col_data, mat_data, grad_data, out_data, B, M,
          N, E, K);
    });
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm"), TORCH_FN(spmm_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spmm_value_bw"),
         TORCH_FN(spmm_value_bw_cuda));
}

}  // namespace ops
}  // namespace pyg
