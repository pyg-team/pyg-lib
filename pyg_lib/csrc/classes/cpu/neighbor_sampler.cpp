#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/index_tracker.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace classes {

namespace {

struct NeighborSampler : torch::CustomClassHolder {
 public:
  NeighborSampler(const at::Tensor& rowptr,
                  const at::Tensor& col,
                  const c10::optional<at::Tensor>& edge_weight,
                  const c10::optional<at::Tensor>& node_time,
                  const c10::optional<at::Tensor>& edge_time)
      : rowptr_(rowptr),
        col_(col),
        edge_weight_(edge_weight),
        node_time_(node_time),
        edge_time_(edge_time) {};

  std::tuple<at::Tensor,                 // row
             at::Tensor,                 // col
             at::Tensor,                 // node_id
             c10::optional<at::Tensor>,  // edge_id,
             c10::optional<at::Tensor>,  // batch,
             std::vector<int64_t>,       // num_sampled_nodes,
             std::vector<int64_t>>       // num_sampled_edges,
  sample(const std::vector<int64_t>& num_neighbors,
         const at::Tensor& seed_node,
         const c10::optional<at::Tensor>& seed_time,
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
  const c10::optional<at::Tensor>& edge_weight_;
  const c10::optional<at::Tensor>& node_time_;
  const c10::optional<at::Tensor>& edge_time_;
};

struct HeteroNeighborSampler : torch::CustomClassHolder {
 public:
  typedef std::pair<int64_t, int64_t> pair_int64_t;
  typedef int64_t temporal_t;

  HeteroNeighborSampler(
      const std::vector<node_type> node_types,
      const std::vector<edge_type> edge_types,
      const c10::Dict<rel_type, at::Tensor> rowptr,
      const c10::Dict<rel_type, at::Tensor> col,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_weight,
      const c10::optional<c10::Dict<node_type, at::Tensor>> node_time,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_time)
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

  void uniform_sample(rel_type e_type,
                      const pair_int64_t global_src_node,
                      const int64_t local_src_node,
                      const int64_t count,
                      pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
                      pyg::random::RandintEngine<int64_t>& generator,
                      std::vector<pair_int64_t>& out_global_dst_nodes,
                      bool return_edge_id = true) {
    auto rowptr_v = rowptr_.at(e_type).data_ptr<int64_t>();
    const auto row_start = rowptr_v[std::get<1>(global_src_node)];
    const auto row_end = rowptr_v[std::get<1>(global_src_node) + 1];
    if ((row_end - row_start == 0) || (count == 0))
      return;
    _sample(e_type, global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes, return_edge_id);
  }

  void node_temporal_sample(
      rel_type e_type,
      std::string temporal_strategy,
      const pair_int64_t global_src_node,
      const int64_t local_src_node,
      const int64_t count,
      const temporal_t seed_time,
      const temporal_t* time,
      pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
      pyg::random::RandintEngine<int64_t>& generator,
      std::vector<pair_int64_t>& out_global_dst_nodes,
      bool return_edge_id = true) {
    auto row_start =
        rowptr_.at(e_type).data_ptr<int64_t>()[std::get<1>(global_src_node)];
    auto row_end = rowptr_.at(e_type)
                       .data_ptr<int64_t>()[std::get<1>(global_src_node) + 1];
    auto col = col_.at(e_type).data_ptr<int64_t>();

    if ((row_end - row_start == 0) || (count == 0))
      return;

    // Find new `row_end` such that all neighbors fulfill temporal constraints:
    auto it = std::upper_bound(
        col + row_start, col + row_end, seed_time,
        [&](const int64_t& a, const int64_t& b) { return a < time[b]; });
    row_end = it - col;

    if (temporal_strategy == "last" && count >= 0) {
      row_start = std::max(row_start, (int64_t)(row_end - count));
    }

    if (row_end - row_start > 1) {
      TORCH_CHECK(time[col[row_start]] <= time[col[row_end - 1]],
                  "Found invalid non-sorted temporal neighborhood");
    }

    _sample(e_type, global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes, return_edge_id);
  }

  std::tuple<c10::Dict<rel_type, at::Tensor>,                  // row
             c10::Dict<rel_type, at::Tensor>,                  // col
             c10::Dict<node_type, at::Tensor>,                 // node_id
             c10::optional<c10::Dict<rel_type, at::Tensor>>,   // edge_id
             c10::optional<c10::Dict<node_type, at::Tensor>>,  // batch
             c10::Dict<node_type, std::vector<int64_t>>,  // num_sampled_nodes
             c10::Dict<rel_type, std::vector<int64_t>>>   // num_sampled_edges
  sample(const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
         const c10::Dict<node_type, at::Tensor>& seed_node,
         const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time,
         bool disjoint = false,
         std::string temporal_strategy = "uniform",
         bool return_edge_id = true) {
    if (edge_time_.has_value())
      TORCH_CHECK(seed_time.has_value(), "Seed time needs to be specified");
    if (seed_time.has_value()) {
      for (const auto& kv : seed_time.value()) {
        const at::Tensor& seed_time = kv.value();
        TORCH_CHECK(seed_time.is_contiguous(), "Non-contiguous 'seed_time'");
      }
    }
    TORCH_CHECK(temporal_strategy == "uniform" || temporal_strategy == "last",
                "No valid temporal strategy found");

    pyg::random::RandintEngine<int64_t> generator;

    phmap::flat_hash_map<node_type, size_t> num_nodes_dict;
    for (const auto& k : edge_types_) {
      const auto num_nodes = rowptr_.at(to_rel_type(k)).size(0) - 1;
      num_nodes_dict[std::get<0>(k)] = num_nodes;
    }
    // Add node types that only exist in the seed_node
    // This is a fallback logic for empty graphs
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
    phmap::flat_hash_map<node_type, pyg::sampler::Mapper<pair_int64_t, int64_t>>
        mapper_dict;
    // Slices corresponding to a type in the buffer row (I think)
    phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    std::vector<int64_t> seed_times;

    // initialize all hashmaps that store intermediate/output data
    init_placeholders();
    for (const auto& k : node_types_) {
      const auto N = num_nodes_dict.count(k) > 0 ? num_nodes_dict.at(k) : 0;
      sampled_nodes[k];  // Init an empty vector
      mapper_dict.insert({k, pyg::sampler::Mapper<pair_int64_t, int64_t>(N)});
      slice_dict[k] = {0, 0};
    }

    size_t L = 0;  // num_layers
    // Split edge types into threads
    for (const auto& k : edge_types_) {
      num_sampled_edges_per_hop_.insert(
          {to_rel_type(k), std::vector<int64_t>(1, 0)});
      L = std::max(L, num_neighbors.at(to_rel_type(k)).size());
    }

    // We fill the buffers with the zero-th layer: seed nodes
    int64_t batch_idx = 0;
    for (const auto& kv : seed_node) {
      const at::Tensor& seed = kv.value();
      slice_dict[kv.key()] = {0, seed.size(0)};
      const auto seed_data = seed.data_ptr<int64_t>();

      if (!disjoint) {
        for (size_t i = 0; i < seed.numel(); ++i) {
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
          const auto curr_seed_time_data =
              curr_seed_time.data_ptr<temporal_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(curr_seed_time_data[i]);
          }
        } else if (node_time_.has_value()) {
          const at::Tensor& time = node_time_.value().at(kv.key());
          const auto time_data = time.data_ptr<int64_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }

      num_sampled_nodes_per_hop_.at(kv.key())[0] =
          sampled_nodes.at(kv.key()).size();
      for (auto& node : sampled_nodes.at(kv.key())) {
        sampled_batch_.at(kv.key()).push_back(node.first);
        sampled_node_ids_.at(kv.key()).push_back(node.second);
      }
    }

    // The actual sampling code begins here
    for (size_t ell = 0; ell < L; ++ell) {
      for (const auto& k : edge_types_) {
        const auto src = std::get<0>(k);
        const auto dst = std::get<2>(k);
        const auto exp_count = num_neighbors.at(to_rel_type(k))[ell];
        auto& src_sampled_nodes = sampled_nodes.at(src);
        auto& dst_mapper = mapper_dict.at(dst);
        size_t begin, end;
        std::tie(begin, end) = slice_dict.at(src);
        num_sampled_edges_per_hop_[to_rel_type(k)].push_back(0);
        // We skip weighted/biased edges and edge-temporal sampling for
        // now
        if ((!node_time_.has_value() ||
             !node_time_.value().contains(dst)) &&
            (!edge_time_.has_value() ||
             !edge_time_.value().contains(to_rel_type(k)))) {
          for (size_t i = begin; i < end; ++i)
            uniform_sample(
                /*e_type=*/to_rel_type(k),
                /*global_src_node=*/src_sampled_nodes[i],
                /*local_src_node=*/i,
                /*count=*/exp_count,
                /*dst_mapper=*/dst_mapper,
                /*generator=*/generator,
                /*out_global_dst_nodes=*/sampled_nodes.at(dst),
                /*return_edge_id=*/return_edge_id);
        } else {
          // Node-level temporal sampling:
          const at::Tensor& dst_time = node_time_.value().at(dst);
          const auto dst_time_data = dst_time.data_ptr<temporal_t>();
          for (size_t i = begin; i < end; ++i) {
            const auto batch_idx = src_sampled_nodes[i].first;
            node_temporal_sample(
                /*e_type=*/to_rel_type(k),
                /*temporal_strategy=*/temporal_strategy,
                /*global_src_node=*/src_sampled_nodes[i],
                /*local_src_node=*/i,
                /*count=*/exp_count,
                /*seed_time=*/seed_times[batch_idx],
                /*time=*/dst_time_data,
                /*dst_mapper=*/dst_mapper,
                /*generator=*/generator,
                /*out_global_dst_nodes=*/sampled_nodes.at(dst),
                /*return_edge_id=*/return_edge_id);
          }
        }
      }
      
      // Update which slice of the sampled_nodes_dict[k] belongs to which hop
      for (const auto& k : node_types_) {
        slice_dict[k] = {slice_dict.at(k).second, sampled_nodes.at(k).size()};
        num_sampled_nodes_per_hop_.at(k).push_back(slice_dict.at(k).second -
                                                   slice_dict.at(k).first);
        for (int64_t i = slice_dict.at(k).first; i < slice_dict.at(k).second;
             i++) {
          auto node = sampled_nodes.at(k)[i];
          sampled_batch_.at(k).push_back(node.first);
          sampled_node_ids_.at(k).push_back(node.second);
        }
        // TODO: This is probably the part where I should implement some fancy
        // logic for rebalancing
      }
    }

    // We rewrite phmap objects into c10 ones for the return value
    c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop_dict;
    c10::Dict<node_type, at::Tensor> out_node_id;
    c10::Dict<node_type, at::Tensor> batch;
    for (const auto& k : node_types_) {
      out_node_id.insert(k, pyg::utils::from_vector(sampled_node_ids_.at(k)));
      num_sampled_nodes_per_hop_dict.insert(k,
                                            num_sampled_nodes_per_hop_.at(k));
      batch.insert(k, pyg::utils::from_vector(sampled_batch_.at(k)));
    }

    c10::Dict<rel_type, std::vector<int64_t>> num_sampled_edges_per_hop_dict;
    c10::Dict<rel_type, at::Tensor> out_row;
    c10::Dict<rel_type, at::Tensor> out_col;
    c10::optional<c10::Dict<rel_type, at::Tensor>> out_edge_id;
    if (return_edge_id)
      out_edge_id = c10::Dict<rel_type, at::Tensor>();
    else
      out_edge_id = c10::nullopt;
    for (const auto& k : edge_types_) {
      auto k_rel = to_rel_type(k);
      num_sampled_edges_per_hop_dict.insert(k_rel,
                                            num_sampled_edges_per_hop_[k_rel]);
      const auto row = pyg::utils::from_vector(sampled_rows_[k_rel]);
      out_row.insert(k_rel, row);
      const auto col = pyg::utils::from_vector(sampled_cols_[k_rel]);
      out_col.insert(k_rel, col);
      if (return_edge_id) {
        const auto edge_id = pyg::utils::from_vector(sampled_edge_ids_[k_rel]);
        out_edge_id.value().insert(k_rel, edge_id);
      }
    }
    clear_placeholders();
    return std::make_tuple(out_row, out_col, out_node_id, out_edge_id, batch,
                           num_sampled_nodes_per_hop_dict,
                           num_sampled_edges_per_hop_dict);
  }

 private:
  void clear_placeholders() {
    num_sampled_nodes_per_hop_.clear();
    num_sampled_edges_per_hop_.clear();
    sampled_cols_.clear();
    sampled_rows_.clear();
    sampled_batch_.clear();
    sampled_node_ids_.clear();
    sampled_edge_ids_.clear();
  }

  void init_placeholders() {
    for (const auto& k : node_types_) {
      num_sampled_nodes_per_hop_.insert({k, std::vector<int64_t>(1, 0)});
      sampled_batch_.insert({k, std::vector<int64_t>()});
      sampled_node_ids_.insert({k, std::vector<int64_t>()});
    }
    for (const auto& k : edge_types_) {
      auto k_rel = to_rel_type(k);
      num_sampled_edges_per_hop_.insert({k_rel, std::vector<int64_t>(1, 0)});
      sampled_cols_.insert({k_rel, std::vector<int64_t>()});
      sampled_rows_.insert({k_rel, std::vector<int64_t>()});
      sampled_edge_ids_.insert({k_rel, std::vector<int64_t>()});
    }
  }

  void _sample(rel_type e_type,
               const pair_int64_t global_src_node,
               const int64_t local_src_node,
               const int64_t row_start,
               const int64_t row_end,
               const int64_t count,
               pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
               pyg::random::RandintEngine<int64_t>& generator,
               std::vector<pair_int64_t>& out_global_dst_nodes,
               bool return_edge_id = true) {
    const auto population = row_end - row_start;
    // For now we'll assume that we're not sampling with replacement since
    // that's the default
    // Case 1: Sample the full neighborhood:
    if (count < 0 || count >= population) {
      for (int64_t edge_id = row_start; edge_id < row_end; ++edge_id)
        add_edge(e_type, edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, return_edge_id);
    }  // We skip Case 2: sample with replacement
    // Case 3: Sample without replacement:
    else {
      auto index_tracker = pyg::sampler::IndexTracker<int64_t>(population);
      for (auto i = population - count; i < population; ++i) {
        auto rnd = generator(0, i + 1);
        if (!index_tracker.try_insert(rnd)) {
          rnd = i;
          index_tracker.insert(i);
        }
        const auto edge_id = row_start + rnd;
        add_edge(e_type, edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes, return_edge_id);
      }
    }
  }

  inline void add_edge(rel_type e_type,
                       const int64_t edge_id,
                       const pair_int64_t global_src_node,
                       const int64_t local_src_node,
                       pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
                       std::vector<pair_int64_t>& out_global_dst_nodes,
                       bool return_edge_id = true) {
    const auto global_dst_node_value =
        col_.at(e_type).data_ptr<int64_t>()[edge_id];
    const auto global_dst_node =
        std::make_pair(std::get<0>(global_src_node), global_dst_node_value);

    // There was special handling for a distributed setting which we ignore
    // for now

    const auto res = dst_mapper.insert(global_dst_node);
    if (res.second)  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node);
    // handle the global vars below
    num_sampled_edges_per_hop_[e_type]
                              [num_sampled_edges_per_hop_[e_type].size() - 1]++;
    sampled_cols_[e_type].push_back(res.first);
    sampled_rows_[e_type].push_back(local_src_node);
    if (return_edge_id)
      sampled_edge_ids_.at(e_type).push_back(edge_id);
  }

  const std::vector<node_type> node_types_;
  const std::vector<edge_type> edge_types_;
  const c10::Dict<rel_type, at::Tensor> rowptr_;
  const c10::Dict<rel_type, at::Tensor> col_;
  const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_weight_;
  const c10::optional<c10::Dict<node_type, at::Tensor>> node_time_;
  const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_time_;
  phmap::flat_hash_map<node_type, std::vector<int64_t>>
      num_sampled_nodes_per_hop_;
  phmap::flat_hash_map<node_type, std::vector<int64_t>> sampled_batch_;
  phmap::flat_hash_map<node_type, std::vector<int64_t>> sampled_node_ids_;
  phmap::flat_hash_map<rel_type, std::vector<int64_t>>
      num_sampled_edges_per_hop_;
  phmap::flat_hash_map<rel_type, std::vector<int64_t>> sampled_cols_;
  phmap::flat_hash_map<rel_type, std::vector<int64_t>> sampled_rows_;
  phmap::flat_hash_map<rel_type, std::vector<int64_t>> sampled_edge_ids_;
};

}  // namespace

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<NeighborSampler>("NeighborSampler")
      .def(torch::init<at::Tensor&, at::Tensor&, c10::optional<at::Tensor>,
                       c10::optional<at::Tensor>, c10::optional<at::Tensor>>())
      .def("sample", &NeighborSampler::sample);

  m.class_<HeteroNeighborSampler>("HeteroNeighborSampler")
      .def(torch::init<std::vector<node_type>, std::vector<edge_type>,
                       c10::Dict<rel_type, at::Tensor>,
                       c10::Dict<rel_type, at::Tensor>,
                       c10::optional<c10::Dict<rel_type, at::Tensor>>,
                       c10::optional<c10::Dict<node_type, at::Tensor>>,
                       c10::optional<c10::Dict<rel_type, at::Tensor>>>())
      .def("sample", &HeteroNeighborSampler::sample);
}

}  // namespace classes
}  // namespace pyg
