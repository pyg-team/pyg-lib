#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <iostream>
#include <stdexcept>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/utils/types.h"
#include "pyg_lib/csrc/sampler/cpu/neighbor_kernel.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"

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
        edge_time_(edge_time) {};

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
        edge_time_(edge_time) {
    for (const auto& kv : rowptr) {
      const at::Tensor& rowptr_v = kv.value();
      TORCH_CHECK(rowptr_v.is_contiguous(), "Non-contiguous 'rowptr'");
    }
    for (const auto& kv : col) {
      const at::Tensor& col_v = kv.value();
      TORCH_CHECK(col_v.is_contiguous(), "Non-contiguous 'col'");
    }
    if (edge_time.has_value()) {
      for (const auto& kv : node_time.value()) {
        const at::Tensor& node_time_v = kv.value();
        TORCH_CHECK(node_time_v.is_contiguous(), "Non-contiguous 'node_time'");
      }
    }
    if (edge_time.has_value()) {
      for (const auto& kv : edge_time.value()) {
        const at::Tensor& time = kv.value();
        TORCH_CHECK(time.is_contiguous(), "Non-contiguous 'edge_time'");
      }
    }
    TORCH_CHECK(!(node_time.has_value() && edge_weight.has_value()),
                "Biased temporal sampling not yet supported");
    TORCH_CHECK(!(edge_time.has_value() && edge_weight.has_value()),
                "Biased temporal sampling not yet supported");

  };

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
    if (this->edge_time_.has_value())
      TORCH_CHECK(seed_time.has_value(), "Seed time needs to be specified");
    if (seed_time.has_value()) {
      for (const auto& kv : seed_time.value()) {
        const at::Tensor& seed_time = kv.value();
        TORCH_CHECK(seed_time.is_contiguous(), "Non-contiguous 'seed_time'");
      }
    }
    c10::Dict<rel_type, at::Tensor> out_row;
    c10::Dict<rel_type, at::Tensor> out_col;
    c10::Dict<node_type, at::Tensor> out_node_id;
    std::optional<c10::Dict<rel_type, at::Tensor>> out_edge_id;
    if (return_edge_id)
      out_edge_id = c10::Dict<rel_type, at::Tensor>();
    else
      out_edge_id = c10::nullopt;
    c10::Dict<node_type, at::Tensor> batch;
    phmap::flat_hash_map<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop;
    c10::Dict<rel_type, std::vector<int64_t>> num_sampled_edges_per_hop;
    typedef std::pair<int64_t, int64_t> pair_int64_t;
    typedef int64_t temporal_t;

    pyg::random::RandintEngine<int64_t> generator;

    phmap::flat_hash_map<node_type, size_t> num_nodes_dict;
    for (const auto& k : this->edge_types_) {
      const auto num_nodes = this->rowptr_.at(to_rel_type(k)).size(0) - 1;
      num_nodes_dict[std::get<0>(k)] = num_nodes;
    }
    // Add node types that only exist in the seed_node
    // TODO why would this happen?
    for (const auto& kv : seed_node) {
      const at::Tensor& seed = kv.value();
      if (num_nodes_dict.count(kv.key()) == 0 && seed.numel() > 0) {
        num_nodes_dict[kv.key()] = seed.max().data_ptr<int64_t>()[0] + 1;
      }
    }

    // Important: Since `disjoint` isn't known at compile time, we just use
    // pairs <int64, int64> for both usecases and fill batch with 0
    // for !disjoint case. This is a bit wasteful but we'll clean this up
    // sometime later. TODO
    phmap::flat_hash_map<node_type, std::vector<pair_int64_t>> sampled_nodes;
    // Mappers to contiguous indices starting at 0
    phmap::flat_hash_map<node_type, pyg::sampler::Mapper<pair_int64_t, int64_t>> mapper_dict;
    // Slices corresponding to a type in the buffer (I think)
    phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    std::vector<int64_t> seed_times;

    // initialize all hashmaps
    for (const auto& k : this->node_types_){
      const auto N = num_nodes_dict.count(k) > 0 ? num_nodes_dict.at(k) : 0;
      sampled_nodes[k]; // Init an empty vector
      num_sampled_nodes_per_hop.insert({k, std::vector<int64_t>(1, 0)});
      mapper_dict.insert({k, pyg::sampler::Mapper<pair_int64_t, int64_t>(N)});
      slice_dict[k] = {0, 0};
    }

    const bool parallel = at::get_num_threads() > 1 && this->edge_types_.size() > 1;
    std::vector<std::vector<edge_type>> threads_edge_types;

    size_t L = 0; // num_layers
    // Split edge types into threads, initialize samplers for each
    for (const auto&k : this->edge_types_){
      L = std::max(L, num_neighbors.at(to_rel_type(k)).size());
      // We'd initialise a sampler for this specific bipartite graph here TODO
      if (parallel) {
	// Each thread is assigned edge types that have the same dst node
        // type. Thanks to this, each thread will operate on a separate mapper
        // and separate sampler.
        bool added = false;
        const auto dst = std::get<2>(k);
        for (auto& e : threads_edge_types) {
          if (std::get<2>(e[0]) == dst) {
            e.push_back(k);
            added = true;
            break;
          }
        }
        if (!added)
          threads_edge_types.push_back({k});
      }
    }
    if (!parallel) {  // One thread handles all edge types.
      threads_edge_types.push_back({this->edge_types_});
    }

    // We fill the buffers with the zero-th layer: seed nodes
    int64_t batch_idx = 0;
    for (const auto& kv : seed_node) {
      const at::Tensor& seed = kv.value();
      slice_dict[kv.key()] = {0, seed.size(0)};
      const auto seed_data = seed.data_ptr<int64_t>();

      if (!disjoint) {
	for(size_t i = 0; i < seed.numel(); ++i){
	  sampled_nodes[kv.key()].push_back({0, seed_data[i]});
          mapper_dict.at(kv.key()).insert({0, seed_data[i]});
	}
      } else {
	// If dealing with disjoint subgraphs, we need to track each batch
	// separately to avoid leakage. Hence we always push pairs
	// <batch_id, node_id>
        auto& curr_sampled_nodes = sampled_nodes.at(kv.key());
        auto& mapper = mapper_dict.at(kv.key());
        for (size_t i = 0; i < seed.numel(); ++i) {
          curr_sampled_nodes.push_back({batch_idx, seed_data[i]});
          mapper.insert({batch_idx, seed_data[i]});
          batch_idx++;
        }
        if (seed_time.has_value()) {
          const at::Tensor& curr_seed_time = seed_time.value().at(kv.key());
          const auto curr_seed_time_data = curr_seed_time.data_ptr<int64_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(curr_seed_time_data[i]);
          }
        } else if (this->node_time_.has_value()) {
          const at::Tensor& time = this->node_time_.value().at(kv.key());
          const auto time_data = time.data_ptr<int64_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }

      num_sampled_nodes_per_hop.at(kv.key())[0] =
          sampled_nodes.at(kv.key()).size();
    }

    // The actual sampling code begins here TODO

    //We rewrite the phmap object into a c10 one for the return value
    c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop_dict;
    for (const auto& k : this->node_types_) {
      num_sampled_nodes_per_hop_dict.insert(
          k, num_sampled_nodes_per_hop.at(k));
    }

    return std::make_tuple(out_row, out_col, out_node_id, out_edge_id, batch, num_sampled_nodes_per_hop_dict,
                           num_sampled_edges_per_hop);
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
