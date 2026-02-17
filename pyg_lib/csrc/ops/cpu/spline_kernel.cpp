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

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_basis"),
         TORCH_FN(spline_basis_forward_kernel));
  m.impl(TORCH_SELECTIVE_NAME("pyg::spline_weighting"),
         TORCH_FN(spline_weighting_forward_kernel));
}

}  // namespace ops
}  // namespace pyg
