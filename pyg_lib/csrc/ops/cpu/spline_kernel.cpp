#include "../spline.h"

#include <ATen/ATen.h>
#include <torch/library.h>

namespace pyg {
namespace ops {

namespace {

template <typename scalar_t, int64_t degree>
struct Basis {
  static inline scalar_t forward(scalar_t v, int64_t k_mod) {
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

  static inline scalar_t backward(scalar_t v, int64_t k_mod) {
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

#define AT_DISPATCH_DEGREE_TYPES(degree, ...)     \
  [&] {                                           \
    switch (degree) {                             \
      case 1: {                                   \
        static constexpr int64_t DEGREE = 1;      \
        return __VA_ARGS__();                     \
      }                                           \
      case 2: {                                   \
        static constexpr int64_t DEGREE = 2;      \
        return __VA_ARGS__();                     \
      }                                           \
      case 3: {                                   \
        static constexpr int64_t DEGREE = 3;      \
        return __VA_ARGS__();                     \
      }                                           \
      default:                                    \
        AT_ERROR("Basis degree not implemented"); \
    }                                             \
  }()

std::tuple<at::Tensor, at::Tensor> spline_basis_forward_kernel(
    const at::Tensor& pseudo,
    const at::Tensor& kernel_size,
    const at::Tensor& is_open_spline,
    int64_t degree) {
  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = (int64_t)(pow(degree + 1, D) + 0.5);

  auto basis = at::empty({E, S}, pseudo.options());
  auto weight_index = at::empty({E, S}, kernel_size.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, pseudo.scalar_type(), "spline_basis_fw", [&] {
        auto pseudo_data = pseudo.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();

        AT_DISPATCH_DEGREE_TYPES(degree, [&] {
          int64_t k, wi, wi_offset;
          scalar_t b;

          for (int64_t e = 0; e < E; e++) {
            for (int64_t s = 0; s < S; s++) {
              k = s, wi = 0, wi_offset = 1, b = (scalar_t)1.;
              for (int64_t d = 0; d < D; d++) {
                int64_t k_mod = k % (DEGREE + 1);
                k /= DEGREE + 1;

                auto v =
                    pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];
                v *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];

                wi += (((int64_t)v + k_mod) % kernel_size_data[d]) * wi_offset;
                wi_offset *= kernel_size_data[d];

                v -= floor(v);
                v = Basis<scalar_t, DEGREE>::forward(v, k_mod);
                b *= v;
              }
              basis_data[e * S + s] = b;
              weight_index_data[e * S + s] = wi;
            }
          }
        });
      });

  return std::make_tuple(basis, weight_index);
}

at::Tensor spline_basis_backward_kernel(const at::Tensor& grad_basis,
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

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, pseudo.scalar_type(), "spline_basis_bw", [&] {
        auto grad_basis_data = grad_basis.data_ptr<scalar_t>();
        auto pseudo_data = pseudo.data_ptr<scalar_t>();
        auto grad_pseudo_data = grad_pseudo.data_ptr<scalar_t>();

        AT_DISPATCH_DEGREE_TYPES(degree, [&] {
          scalar_t g, tmp;

          for (int64_t e = 0; e < E; e++) {
            for (int64_t d = 0; d < D; d++) {
              g = (scalar_t)0.;
              for (int64_t s = 0; s < S; s++) {
                int64_t k_mod =
                    (s / (int64_t)(pow(DEGREE + 1, d) + 0.5)) % (DEGREE + 1);
                auto v =
                    pseudo_data[e * pseudo.stride(0) + d * pseudo.stride(1)];
                v *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];
                v -= floor(v);
                v = Basis<scalar_t, DEGREE>::backward(v, k_mod);
                tmp = v;

                for (int64_t d_it = 1; d_it < D; d_it++) {
                  int64_t d_new = d_it - (d >= d_it);
                  k_mod = (s / (int64_t)(pow(DEGREE + 1, d_new) + 0.5)) %
                          (DEGREE + 1);
                  v = pseudo_data[e * pseudo.stride(0) +
                                  d_new * pseudo.stride(1)];
                  v *= kernel_size_data[d_new] -
                       DEGREE * is_open_spline_data[d_new];
                  v -= floor(v);
                  v = Basis<scalar_t, DEGREE>::forward(v, k_mod);
                  tmp *= v;
                }
                g += tmp * grad_basis_data[e * grad_basis.stride(0) +
                                           s * grad_basis.stride(1)];
              }
              g *= kernel_size_data[d] - DEGREE * is_open_spline_data[d];
              grad_pseudo_data[e * D + d] = g;
            }
          }
        });
      });

  return grad_pseudo;
}

at::Tensor spline_weighting_forward_kernel(const at::Tensor& x,
                                           const at::Tensor& weight,
                                           const at::Tensor& basis,
                                           const at::Tensor& weight_index) {
  auto E = x.size(0);
  auto M_in = x.size(1);
  auto M_out = weight.size(2);
  auto S = basis.size(1);

  auto out = at::empty({E, M_out}, x.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, x.scalar_type(), "spline_weighting_fw", [&] {
        auto x_data = x.data_ptr<scalar_t>();
        auto weight_data = weight.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();
        auto out_data = out.data_ptr<scalar_t>();

        scalar_t v;

        for (int64_t e = 0; e < E; e++) {
          for (int64_t m_out = 0; m_out < M_out; m_out++) {
            v = 0;
            for (int64_t s = 0; s < S; s++) {
              auto b = basis_data[e * S + s];
              auto wi = weight_index_data[e * S + s];
              for (int64_t m_in = 0; m_in < M_in; m_in++) {
                auto tmp = weight_data[wi * weight.stride(0) +
                                       m_in * weight.stride(1) +
                                       m_out * weight.stride(2)];
                tmp *= b * x_data[e * x.stride(0) + m_in * x.stride(1)];
                v += tmp;
              }
            }
            out_data[e * M_out + m_out] = v;
          }
        }
      });

  return out;
}

at::Tensor spline_weighting_backward_x_kernel(const at::Tensor& grad_out,
                                              const at::Tensor& weight,
                                              const at::Tensor& basis,
                                              const at::Tensor& weight_index) {
  auto E = grad_out.size(0);
  auto M_in = weight.size(1);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);

  auto grad_x = at::zeros({E, M_in}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, grad_out.scalar_type(), "spline_weighting_bw_x",
      [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto weight_data = weight.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();
        auto grad_x_data = grad_x.data_ptr<scalar_t>();

        for (int64_t e = 0; e < E; e++) {
          for (int64_t m_out = 0; m_out < M_out; m_out++) {
            auto g = grad_out_data[e * grad_out.stride(0) +
                                   m_out * grad_out.stride(1)];
            for (int64_t s = 0; s < S; s++) {
              auto b = basis_data[e * S + s];
              auto wi = weight_index_data[e * S + s];
              for (int64_t m_in = 0; m_in < M_in; m_in++) {
                auto w = weight_data[wi * weight.stride(0) +
                                     m_in * weight.stride(1) +
                                     m_out * weight.stride(2)];
                grad_x_data[e * M_in + m_in] += g * b * w;
              }
            }
          }
        }
      });

  return grad_x;
}

at::Tensor spline_weighting_backward_weight_kernel(
    const at::Tensor& grad_out,
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

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, x.scalar_type(), "spline_weighting_bw_weight",
      [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto x_data = x.data_ptr<scalar_t>();
        auto basis_data = basis.data_ptr<scalar_t>();
        auto grad_weight_data = grad_weight.data_ptr<scalar_t>();

        for (int64_t e = 0; e < E; e++) {
          for (int64_t m_out = 0; m_out < M_out; m_out++) {
            auto g = grad_out_data[e * grad_out.stride(0) +
                                   m_out * grad_out.stride(1)];
            for (int64_t s = 0; s < S; s++) {
              auto b = basis_data[e * S + s];
              auto wi = weight_index_data[e * S + s];
              for (int64_t m_in = 0; m_in < M_in; m_in++) {
                auto v = g * b * x_data[e * x.stride(0) + m_in * x.stride(1)];
                grad_weight_data[wi * M_in * M_out + m_in * M_out + m_out] += v;
              }
            }
          }
        }
      });

  return grad_weight;
}

at::Tensor spline_weighting_backward_basis_kernel(
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

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, x.scalar_type(), "spline_weighting_bw_basis",
      [&] {
        auto grad_out_data = grad_out.data_ptr<scalar_t>();
        auto x_data = x.data_ptr<scalar_t>();
        auto weight_data = weight.data_ptr<scalar_t>();
        auto grad_basis_data = grad_basis.data_ptr<scalar_t>();

        for (int64_t e = 0; e < E; e++) {
          for (int64_t m_out = 0; m_out < M_out; m_out++) {
            auto g = grad_out_data[e * grad_out.stride(0) +
                                   m_out * grad_out.stride(1)];
            for (int64_t s = 0; s < S; s++) {
              scalar_t b = 0;
              auto wi = weight_index_data[e * S + s];
              for (int64_t m_in = 0; m_in < M_in; m_in++) {
                auto w = weight_data[wi * weight.stride(0) +
                                     m_in * weight.stride(1) +
                                     m_out * weight.stride(2)];
                w *= x_data[e * x.stride(0) + m_in * x.stride(1)];
                b += w;
              }
              grad_basis_data[e * S + s] += g * b;
            }
          }
        }
      });

  return grad_basis;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis"),
         TORCH_FN(spline_basis_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis_backward"),
         TORCH_FN(spline_basis_backward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting"),
         TORCH_FN(spline_weighting_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_x"),
         TORCH_FN(spline_weighting_backward_x_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_weight"),
         TORCH_FN(spline_weighting_backward_weight_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting_backward_basis"),
         TORCH_FN(spline_weighting_backward_basis_kernel));
}

}  // namespace ops
}  // namespace pyg
