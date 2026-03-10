#include "../edge_sampler.h"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <unordered_set>
#include <vector>

namespace pyg {
namespace ops {

namespace {

at::Tensor edge_sample_kernel(const at::Tensor& start,
                              const at::Tensor& rowptr,
                              int64_t count,
                              double factor) {
  auto start_data = start.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();

  std::vector<int64_t> e_ids;

  for (int64_t i = 0; i < start.size(0); i++) {
    auto row_start = rowptr_data[start_data[i]];
    auto row_end = rowptr_data[start_data[i] + 1];
    auto num_neighbors = row_end - row_start;

    int64_t size = count;
    if (count < 1)
      size = static_cast<int64_t>(std::ceil(factor * double(num_neighbors)));
    if (size > num_neighbors)
      size = num_neighbors;

    if (size < 0.7 * double(num_neighbors)) {
      std::unordered_set<int64_t> set;
      while (static_cast<int64_t>(set.size()) < size) {
        int64_t sample = std::rand() % num_neighbors;
        set.insert(sample + row_start);
      }
      std::vector<int64_t> v(set.begin(), set.end());
      e_ids.insert(e_ids.end(), v.begin(), v.end());
    } else {
      auto sample = at::randperm(num_neighbors, start.options());
      auto sample_data = sample.data_ptr<int64_t>();
      for (int64_t j = 0; j < size; j++) {
        e_ids.push_back(sample_data[j] + row_start);
      }
    }
  }

  int64_t length = static_cast<int64_t>(e_ids.size());
  return at::from_blob(e_ids.data(), {length}, start.options()).clone();
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::edge_sample"),
         TORCH_FN(edge_sample_kernel));
}

}  // namespace ops
}  // namespace pyg
