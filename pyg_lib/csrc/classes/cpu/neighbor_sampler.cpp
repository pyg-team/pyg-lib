#include <ATen/ATen.h>
#include <torch/library.h>

#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace classes {

namespace {

struct NeighborSampler : torch::CustomClassHolder {
 public:
  NeighborSampler(const at::Tensor& rowptr,
                  const at::Tensor& col,
                  const std::optional<at::Tensor>& edge_weight,
                  const std::optional<at::Tensor>& node_time,
                  const std::optional<at::Tensor>& edge_time)
      : rowptr_(rowptr),
        col_(col),
        edge_weight_(edge_weight),
        node_time_(node_time),
        edge_time_(edge_time){};

  std::tuple<at::Tensor,                 // row
             at::Tensor,                 // col
             at::Tensor,                 // node_id
             std::optional<at::Tensor>,  // edge_id,
             std::optional<at::Tensor>,  // batch,
             std::vector<int64_t>,       // num_sampled_nodes,
             std::vector<int64_t>>       // num_sampled_edges,
  sample(const std::vector<int64_t>& num_neighbors,
         const at::Tensor& seed_node,
         const std::optional<at::Tensor>& seed_time,
         bool disjoint = false,
         std::string temporal_strategy = "uniform",
         bool return_edge_id = true) {
    // TODO
    auto row = at::empty(0);
    auto col = at::empty(0);
    auto node_id = at::empty(0);
    auto edge_id = at::empty(0);
    auto batch = at::empty(0);
    std::vector<int64_t> num_sampled_nodes;
    std::vector<int64_t> num_sampled_edges;
    return std::make_tuple(row, col, node_id, edge_id, batch, num_sampled_nodes,
                           num_sampled_edges);
  }

 private:
  const at::Tensor& rowptr_;
  const at::Tensor& col_;
  const std::optional<at::Tensor>& edge_weight_;
  const std::optional<at::Tensor>& node_time_;
  const std::optional<at::Tensor>& edge_time_;
};

struct HeteroNeighborSampler : torch::CustomClassHolder {
 public:
  HeteroNeighborSampler(
      const std::vector<node_type>& node_types,
      const std::vector<edge_type>& edge_types,
      const c10::Dict<rel_type, at::Tensor>& rowptr,
      const c10::Dict<rel_type, at::Tensor>& col,
      const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_weight,
      const std::optional<c10::Dict<node_type, at::Tensor>>& node_time,
      const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_time)
      : node_types_(node_types),
        edge_types_(edge_types),
        rowptr_(rowptr),
        col_(col),
        edge_weight_(edge_weight),
        node_time_(node_time),
        edge_time_(edge_time){};

  std::tuple<c10::Dict<rel_type, at::Tensor>,                  // row
             c10::Dict<rel_type, at::Tensor>,                  // col
             c10::Dict<node_type, at::Tensor>,                 // node_id
             std::optional<c10::Dict<rel_type, at::Tensor>>,   // edge_id
             std::optional<c10::Dict<node_type, at::Tensor>>,  // batch
             c10::Dict<node_type, std::vector<int64_t>>,  // num_sampled_nodes
             c10::Dict<rel_type, std::vector<int64_t>>>   // num_sampled_edges
  sample(const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
         const c10::Dict<node_type, at::Tensor>& seed_node,
         const std::optional<c10::Dict<node_type, at::Tensor>>& seed_time,
         bool disjoint = false,
         std::string temporal_strategy = "uniform",
         bool return_edge_id = true) {
    // TODO
    c10::Dict<rel_type, at::Tensor> row;
    c10::Dict<rel_type, at::Tensor> col;
    c10::Dict<node_type, at::Tensor> node_id;
    c10::Dict<rel_type, at::Tensor> edge_id;
    c10::Dict<node_type, at::Tensor> batch;
    c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes;
    c10::Dict<rel_type, std::vector<int64_t>> num_sampled_edges;
    return std::make_tuple(row, col, node_id, edge_id, batch, num_sampled_nodes,
                           num_sampled_edges);
  }

 private:
  const std::vector<node_type>& node_types_;
  const std::vector<edge_type>& edge_types_;
  const c10::Dict<rel_type, at::Tensor>& rowptr_;
  const c10::Dict<rel_type, at::Tensor>& col_;
  const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_weight_;
  const std::optional<c10::Dict<node_type, at::Tensor>>& node_time_;
  const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_time_;
};

}  // namespace

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<NeighborSampler>("NeighborSampler")
      .def(torch::init<at::Tensor&, at::Tensor&, std::optional<at::Tensor>,
                       std::optional<at::Tensor>, std::optional<at::Tensor>>())
      .def("sample", &NeighborSampler::sample);

  m.class_<HeteroNeighborSampler>("HeteroNeighborSampler")
      .def(torch::init<std::vector<node_type>, std::vector<edge_type>,
                       c10::Dict<rel_type, at::Tensor>,
                       c10::Dict<rel_type, at::Tensor>,
                       std::optional<c10::Dict<rel_type, at::Tensor>>,
                       std::optional<c10::Dict<node_type, at::Tensor>>,
                       std::optional<c10::Dict<rel_type, at::Tensor>>>())
      .def("sample", &HeteroNeighborSampler::sample);
}

}  // namespace classes
}  // namespace pyg
