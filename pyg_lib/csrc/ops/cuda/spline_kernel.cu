#include "../spline.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

#define THREADS 1024
#define BLOCKS(N) ((N) + THREADS - 1) / THREADS

// Atomic add helpers:
static inline __device__ void atomAdd(float* address, float val) {
  atomicAdd(address, val);
}

#if defined(USE_ROCM) || \
    (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600 || CUDA_VERSION < 8000))
static inline __device__ void atomAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
}
#else
static inline __device__ void atomAdd(double* address, double val) {
  atomicAdd(address, val);
}
#endif

// B-spline basis evaluation (device):
template <typename scalar_t, int64_t degree>
struct Basis {
  static inline __device__ scalar_t forward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 1. - v - k_mod + 2. * v * k_mod;
    } else if (degree == 2) {
      if (k_mod == 0)
        return 0.5 * v * v - v + 0.5;
      else if (k_mod == 1)
        return -v * v + v + 0.5;
      else
        return 0.5 * v * v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (1. - v) * (1. - v) * (1. - v) / 6.;
      else if (k_mod == 1)
        return (3. * v * v * v - 6. * v * v + 4.) / 6.;
      else if (k_mod == 2)
        return (-3. * v * v * v + 3. * v * v + 3. * v + 1.) / 6.;
      else
        return v * v * v / 6.;
    } else {
      return (scalar_t)-1.;
    }
  }

  static inline __device__ scalar_t backward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 2 * k_mod - 1;
    } else if (degree == 2) {
      if (k_mod == 0)
        return v - 1.;
      else if (k_mod == 1)
        return -2. * v + 1.;
      else
        return v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (-v * v + 2. * v - 1.) / 2.;
      else if (k_mod == 1)
        return (3. * v * v - 4. * v) / 2.;
      else if (k_mod == 2)
        return (-3. * v * v + 2. * v + 1.) / 2.;
      else
        return v * v / 2.;
    } else {
      return (scalar_t)-1.;
    }
  }
};

// CUDA version uses non-constexpr DEGREE for template instantiation:
#define AT_DISPATCH_DEGREE_TYPES(degree, ...)     \
  [&] {                                           \
    switch (degree) {                             \
      case 1: {                                   \
        const int64_t DEGREE = 1;                 \
        return __VA_ARGS__();                     \
      }                                           \
      case 2: {                                   \
        const int64_t DEGREE = 2;                 \
        return __VA_ARGS__();                     \
      }                                           \
      case 3: {                                   \
        const int64_t DEGREE = 3;                 \
        return __VA_ARGS__();                     \
      }                                           \
      default:                                    \
        AT_ERROR("Basis degree not implemented"); \
    }                                             \
  }()

// ============================================================================
// Spline Basis Forward Kernel
// ============================================================================

template <typename scalar_t, int64_t degree>
__global__ void spline_basis_fw_cuda_kernel(const scalar_t* pseudo,
                                            const int64_t* kernel_size,
                                            const uint8_t* is_open_spline,
                                            scalar_t* basis,
                                            int64_t* weight_index,
                                            int64_t E,
                                            int64_t D,
                                            int64_t S,
                                            int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / S;
  const int64_t s = thread_idx % S;

  if (thread_idx < numel) {
    int64_t k = s, wi = 0, wi_offset = 1;
    scalar_t b = (scalar_t)1.;

    for (int64_t d = 0; d < D; d++) {
      const int64_t k_mod = k % (degree + 1);
      k /= degree + 1;

      scalar_t v = pseudo[e * D + d];
      v *= kernel_size[d] - degree * is_open_spline[d];

      wi += (((int64_t)v + k_mod) % kernel_size[d]) * wi_offset;
      wi_offset *= kernel_size[d];

      v -= floor(v);
      v = Basis<scalar_t, degree>::forward(v, k_mod);
      b *= v;
    }

    basis[thread_idx] = b;
    weight_index[thread_idx] = wi;
  }
}

std::tuple<at::Tensor, at::Tensor> spline_basis_forward_cuda(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree) {
  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = (int64_t)(powf(degree + 1, D) + 0.5);

  auto basis = at::empty({E, S}, pseudo.options());
  auto weight_index = at::empty({E, S}, kernel_size.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(pseudo.scalar_type(), "spline_basis_fw_cuda", [&] {
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();

    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      spline_basis_fw_cuda_kernel<scalar_t, DEGREE>
          <<<BLOCKS(basis.numel()), THREADS, 0, stream>>>(
              pseudo_data, kernel_size_data, is_open_spline_data, basis_data,
              weight_index_data, E, D, S, basis.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  return std::make_tuple(basis, weight_index);
}

// ============================================================================
// Spline Basis Backward Kernel
// ============================================================================

template <typename scalar_t, int64_t degree>
__global__ void spline_basis_bw_cuda_kernel(const scalar_t* grad_basis,
                                            const scalar_t* pseudo,
                                            const int64_t* kernel_size,
                                            const uint8_t* is_open_spline,
                                            scalar_t* grad_pseudo,
                                            int64_t E,
                                            int64_t D,
                                            int64_t S,
                                            int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / D;
  const int64_t d = thread_idx % D;

  if (thread_idx < numel) {
    scalar_t g = (scalar_t)0., tmp;

    for (int64_t s = 0; s < S; s++) {
      int64_t k_mod = (s / (int64_t)(powf(degree + 1, d) + 0.5)) % (degree + 1);

      scalar_t v = pseudo[e * D + d];
      v *= kernel_size[d] - degree * is_open_spline[d];
      v -= floor(v);
      v = Basis<scalar_t, degree>::backward(v, k_mod);
      tmp = v;

      for (int64_t d_it = 1; d_it < D; d_it++) {
        const int64_t d_new = d_it - (d >= d_it);
        k_mod = (s / (int64_t)(powf(degree + 1, d_new) + 0.5)) % (degree + 1);
        v = pseudo[e * D + d_new];
        v *= kernel_size[d_new] - degree * is_open_spline[d_new];
        v -= floor(v);
        v = Basis<scalar_t, degree>::forward(v, k_mod);
        tmp *= v;
      }
      g += tmp * grad_basis[e * S + s];
    }
    g *= kernel_size[d] - degree * is_open_spline[d];
    grad_pseudo[thread_idx] = g;
  }
}

at::Tensor spline_basis_backward_cuda(const at::Tensor& grad_basis,
                                      const at::Tensor& pseudo,
                                      const at::Tensor& kernel_size,
                                      const at::Tensor& is_open_spline,
                                      int64_t degree) {
  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = grad_basis.size(1);

  auto grad_pseudo = at::empty({E, D}, pseudo.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(pseudo.scalar_type(), "spline_basis_bw_cuda", [&] {
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_pseudo_data = grad_pseudo.data_ptr<scalar_t>();

    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      spline_basis_bw_cuda_kernel<scalar_t, DEGREE>
          <<<BLOCKS(grad_pseudo.numel()), THREADS, 0, stream>>>(
              grad_basis_data, pseudo_data, kernel_size_data,
              is_open_spline_data, grad_pseudo_data, E, D, S,
              grad_pseudo.numel());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  return grad_pseudo;
}

// ============================================================================
// Spline Weighting Forward Kernel
// ============================================================================

template <typename scalar_t>
__global__ void spline_weighting_fw_cuda_kernel(const scalar_t* x,
                                                const scalar_t* weight,
                                                const scalar_t* basis,
                                                const int64_t* weight_index,
                                                scalar_t* out,
                                                int64_t E,
                                                int64_t M_in,
                                                int64_t M_out,
                                                int64_t S,
                                                int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    scalar_t v = (scalar_t)0.;

    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];
      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        scalar_t tmp = weight[wi * M_in * M_out + m_in * M_out + m_out];
        tmp *= b * x[e * M_in + m_in];
        v += tmp;
      }
    }
    out[thread_idx] = v;
  }
}

at::Tensor spline_weighting_forward_cuda(const at::Tensor& x,
                                         const at::Tensor& weight,
                                         const at::Tensor& basis,
                                         const at::Tensor& weight_index) {
  auto E = x.size(0);
  auto M_in = x.size(1);
  auto M_out = weight.size(2);
  auto S = basis.size(1);

  auto out = at::empty({E, M_out}, x.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "spline_weighting_fw_cuda", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto weight_data = weight.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    spline_weighting_fw_cuda_kernel<scalar_t>
        <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
            x_data, weight_data, basis_data, weight_index_data, out_data, E,
            M_in, M_out, S, out.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  });

  return out;
}

// ============================================================================
// Spline Weighting Backward X Kernel
// ============================================================================

template <typename scalar_t>
__global__ void spline_weighting_bw_x_cuda_kernel(const scalar_t* grad_out,
                                                  const scalar_t* weight,
                                                  const scalar_t* basis,
                                                  const int64_t* weight_index,
                                                  scalar_t* grad_x,
                                                  int64_t E,
                                                  int64_t M_in,
                                                  int64_t M_out,
                                                  int64_t S,
                                                  int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_in;
  const int64_t m_in = thread_idx % M_in;

  if (thread_idx < numel) {
    scalar_t v = (scalar_t)0.;

    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_out = 0; m_out < M_out; m_out++) {
        scalar_t tmp = weight[wi * M_out * M_in + m_out * M_in + m_in];
        tmp *= b * grad_out[e * M_out + m_out];
        v += tmp;
      }
    }
    grad_x[thread_idx] = v;
  }
}

at::Tensor spline_weighting_backward_x_cuda(const at::Tensor& grad_out,
                                            const at::Tensor& weight,
                                            const at::Tensor& basis,
                                            const at::Tensor& weight_index) {
  auto E = grad_out.size(0);
  auto M_in = weight.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_x = at::zeros({E, M_in}, grad_out.options());
  auto weight_t =
      weight.transpose(1, 2).contiguous();  // Contiguous memory access.

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "spline_weighting_bw_x_cuda", [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto weight_data = weight_t.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();

        spline_weighting_bw_x_cuda_kernel<scalar_t>
            <<<BLOCKS(grad_x.numel()), THREADS, 0, stream>>>(
                grad_out_data, weight_data, basis_data, weight_index_data,
                grad_x_data, E, M_in, M_out, S, grad_x.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return grad_x;
}

// ============================================================================
// Spline Weighting Backward Weight Kernel
// ============================================================================

template <typename scalar_t>
__global__ void spline_weighting_bw_weight_cuda_kernel(
    const scalar_t* grad_out,
    const scalar_t* x,
    const scalar_t* basis,
    const int64_t* weight_index,
    scalar_t* grad_weight,
    int64_t E,
    int64_t M_in,
    int64_t M_out,
    int64_t S,
    int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    auto g = grad_out[e * M_out + m_out];
    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        auto v = g * b * x[e * M_in + m_in];
        atomAdd(&grad_weight[wi * M_in * M_out + m_in * M_out + m_out], v);
      }
    }
  }
}

at::Tensor spline_weighting_backward_weight_cuda(const at::Tensor& grad_out,
                                                 const at::Tensor& x,
                                                 const at::Tensor& basis,
                                                 const at::Tensor& weight_index,
                                                 int64_t kernel_size) {
  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_weight = at::zeros({kernel_size, M_in, M_out}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "spline_weighting_bw_weight_cuda", [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto x_data = x.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();
        auto grad_weight_data = grad_weight.data_ptr<scalar_t>();

        spline_weighting_bw_weight_cuda_kernel<scalar_t>
            <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
                grad_out_data, x_data, basis_data, weight_index_data,
                grad_weight_data, E, M_in, M_out, S, grad_out.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return grad_weight;
}

// ============================================================================
// Spline Weighting Backward Basis Kernel
// ============================================================================

template <typename scalar_t>
__global__ void spline_weighting_bw_basis_cuda_kernel(
    const scalar_t* grad_out,
    const scalar_t* x,
    const scalar_t* weight,
    const int64_t* weight_index,
    scalar_t* grad_basis,
    int64_t E,
    int64_t M_in,
    int64_t M_out,
    int64_t S,
    int64_t numel) {
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    const scalar_t g = grad_out[e * M_out + m_out];

    for (int64_t s = 0; s < S; s++) {
      scalar_t v = (scalar_t)0.;
      const int64_t wi = weight_index[e * S + s];

      for (int64_t m_in = 0; m_in < M_in; m_in++) {
        const scalar_t w = weight[wi * M_in * M_out + m_in * M_out + m_out];
        v += g * w * x[e * M_in + m_in];
      }
      atomAdd(&grad_basis[e * S + s], v);
    }
  }
}

at::Tensor spline_weighting_backward_basis_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& weight_index) {
  auto E = grad_out.size(0);
  auto M_in = x.size(1);
  auto M_out = grad_out.size(1);
  auto S = weight_index.size(1);

  auto grad_basis = at::zeros({E, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "spline_weighting_bw_basis_cuda", [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto x_data = x.data_ptr<scalar_t>();
        auto weight_data = weight.data_ptr<scalar_t>();
        auto grad_basis_data = grad_basis.data_ptr<scalar_t>();

        spline_weighting_bw_basis_cuda_kernel<scalar_t>
            <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
                grad_out_data, x_data, weight_data, weight_index_data,
                grad_basis_data, E, M_in, M_out, S, grad_out.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });

  return grad_basis;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis"),
         TORCH_FN(spline_basis_forward_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis_backward"),
         TORCH_FN(spline_basis_backward_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting"),
         TORCH_FN(spline_weighting_forward_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_x"),
         TORCH_FN(spline_weighting_backward_x_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_weight"),
         TORCH_FN(spline_weighting_backward_weight_cuda));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_basis"),
         TORCH_FN(spline_weighting_backward_basis_cuda));
}

}  // namespace ops
}  // namespace pyg
