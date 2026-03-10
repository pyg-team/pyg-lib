#include "../radius.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include "utils/KDTreeVectorOfVectorsAdaptor.h"
#include "utils/nanoflann.hpp"

namespace pyg {
namespace ops {

namespace {

at::Tensor radius_kernel(const at::Tensor& x,
                         const at::Tensor& y,
                         const std::optional<at::Tensor>& ptr_x,
                         const std::optional<at::Tensor>& ptr_y,
                         double r,
                         int64_t max_num_neighbors,
                         int64_t num_workers,
                         bool ignore_same_index) {
  std::vector<size_t> out_vec;

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "radius_cpu", [&] {
        auto x_data = x.data_ptr<scalar_t>();
        auto y_data = y.data_ptr<scalar_t>();
        typedef std::vector<std::vector<scalar_t>> vec_t;
        nanoflann::SearchParams params;
        params.sorted = false;

        if (!ptr_x.has_value()) {
          vec_t pts(x.size(0));
          for (int64_t i = 0; i < x.size(0); i++) {
            pts[i].resize(x.size(1));
            for (int64_t j = 0; j < x.size(1); j++) {
              pts[i][j] = x_data[i * x.size(1) + j];
            }
          }

          typedef KDTreeVectorOfVectorsAdaptor<vec_t, scalar_t> my_kd_tree_t;
          my_kd_tree_t mat_index(x.size(1), pts, 10);

          for (int64_t i = 0; i < y.size(0); i++) {
            std::vector<std::pair<size_t, scalar_t>> ret_matches;
            size_t num_matches = mat_index.index->radiusSearch(
                y_data + i * y.size(1), r * r, ret_matches, params);

            for (size_t j = 0, count = 0;
                 j < num_matches && count < (size_t)max_num_neighbors; j++) {
              if (!ignore_same_index ||
                  ret_matches[j].first != static_cast<size_t>(i)) {
                out_vec.push_back(ret_matches[j].first);
                out_vec.push_back(i);
                count++;
              }
            }
          }
        } else {
          auto ptr_x_data = ptr_x.value().data_ptr<int64_t>();
          auto ptr_y_data = ptr_y.value().data_ptr<int64_t>();

          for (int64_t b = 0; b < ptr_x.value().size(0) - 1; b++) {
            auto x_start = ptr_x_data[b], x_end = ptr_x_data[b + 1];
            auto y_start = ptr_y_data[b], y_end = ptr_y_data[b + 1];

            if (x_start == x_end || y_start == y_end)
              continue;

            vec_t pts(x_end - x_start);
            for (int64_t i = 0; i < x_end - x_start; i++) {
              pts[i].resize(x.size(1));
              for (int64_t j = 0; j < x.size(1); j++) {
                pts[i][j] = x_data[(i + x_start) * x.size(1) + j];
              }
            }

            typedef KDTreeVectorOfVectorsAdaptor<vec_t, scalar_t> my_kd_tree_t;
            my_kd_tree_t mat_index(x.size(1), pts, 10);

            for (int64_t i = y_start; i < y_end; i++) {
              std::vector<std::pair<size_t, scalar_t>> ret_matches;
              size_t num_matches = mat_index.index->radiusSearch(
                  y_data + i * y.size(1), r * r, ret_matches, params);

              for (size_t j = 0, count = 0;
                   j < num_matches && count < (size_t)max_num_neighbors; j++) {
                if (!ignore_same_index ||
                    x_start + static_cast<int64_t>(ret_matches[j].first) != i) {
                  out_vec.push_back(x_start + ret_matches[j].first);
                  out_vec.push_back(i);
                  count++;
                }
              }
            }
          }
        }
      });

  const int64_t size = out_vec.size() / 2;
  auto out =
      at::from_blob(out_vec.data(), {size, 2}, x.options().dtype(at::kLong));
  return out.t().index_select(0, at::tensor({1, 0})).clone();
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::radius"), TORCH_FN(radius_kernel));
}

}  // namespace ops
}  // namespace pyg
