#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <algorithm>

#include "pyg_lib/csrc/classes/cpu/neighbor_sampler.h"
#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/index_tracker.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace classes {

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

MetapathTracker::MetapathTracker(
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
    const std::vector<node_type>& seed_node_types)
    : edge_types_(edge_types), num_neighbors_(num_neighbors) {
  // node_type -> list of metapath_id
  phmap::flat_hash_map<node_type, std::vector<int64_t>> sampled_metapaths;
  n_metapaths_ = 0;
  for (const auto& node_t : seed_node_types) {
    seed_metapaths_.insert({node_t, n_metapaths_});
    sampled_metapaths.insert({node_t, std::vector<int64_t>(1, n_metapaths_++)});
  }
  int64_t L = 0;
  for (const auto& kv : num_neighbors)
    if (kv.value().size() > L)
      L = kv.value().size();
  for (int64_t i = 0; i < L; i++) {
    phmap::flat_hash_map<node_type, std::vector<int64_t>> source_metapaths(
        sampled_metapaths);
    sampled_metapaths.clear();
    for (const auto& edge_t : edge_types) {
      auto rel_t = to_rel_type(edge_t);
      auto src_node_t = std::get<0>(edge_t);
      auto dst_node_t = std::get<2>(edge_t);
      if (source_metapaths.find(src_node_t) == source_metapaths.end())
        continue;
      for (const auto& metapath : source_metapaths[src_node_t]) {
        if (sampled_metapaths.find(dst_node_t) == sampled_metapaths.end())
          sampled_metapaths[dst_node_t];  // init
        int64_t new_metapath_id = n_metapaths_++;
        sampled_metapaths[dst_node_t].push_back(new_metapath_id);
        metapath_tree_[rel_t][metapath] = new_metapath_id;
      }
    }
  }
}

int64_t MetapathTracker::get_neighbor_metapath(const int64_t& metapath_id,
                                               const rel_type& edge) {
  return metapath_tree_[edge][metapath_id];
}

int64_t MetapathTracker::get_sample_size(const int64_t& batch_id,
                                         const int64_t& src_metapath_id,
                                         const edge_type& edge) {
  auto rel = to_rel_type(edge);
  auto dst_metapath_id = get_neighbor_metapath(src_metapath_id, rel);
  return expected_sample_size_[batch_id][dst_metapath_id];
}

void MetapathTracker::report_sample_size(const int64_t& batch_id,
                                         const int64_t& metapath_id,
                                         const int64_t n_sampled) {
  if (reported_sample_size_.find(batch_id) == reported_sample_size_.end())
    reported_sample_size_[batch_id];  // init
  if (reported_sample_size_[batch_id].find(metapath_id) ==
      reported_sample_size_[batch_id].end())
    reported_sample_size_[batch_id][metapath_id] = 0;
  reported_sample_size_[batch_id][metapath_id] += n_sampled;
}

int64_t MetapathTracker::get_reported_sample_size(const int64_t& batch_id,
                                                  const int64_t& metapath_id) {
  if (reported_sample_size_.find(batch_id) == reported_sample_size_.end())
    return 0;
  if (reported_sample_size_[batch_id].find(metapath_id) ==
      reported_sample_size_[batch_id].end())
    return 0;
  return reported_sample_size_[batch_id][metapath_id];
}

int64_t MetapathTracker::init_batch(const int64_t& batch_id,
                                    const node_type& node_t,
                                    const int64_t batch_size) {
  auto seed_metapath = seed_metapaths_[node_t];
  reported_sample_size_[batch_id][seed_metapath] = batch_size;
  expected_sample_size_[batch_id][seed_metapath] = batch_size;
  _init_expected_sample_size(seed_metapath, batch_id, 0);
  return seed_metapath;
}

void MetapathTracker::_init_expected_sample_size(const int64_t src_metapath,
                                                 const int64_t batch_id,
                                                 const int64_t hop) {
  for (auto& kv : metapath_tree_) {
    if (kv.second.find(src_metapath) == kv.second.end())
      continue;
    const auto& dst_metapath = kv.second[src_metapath];
    const auto& num_neigh_v = num_neighbors_.at(kv.first);
    int64_t multiplier = 0;
    if (num_neigh_v.size() > hop)
      multiplier = num_neigh_v[hop];
    if (multiplier > 0) {
      expected_sample_size_[batch_id][dst_metapath] =
          multiplier * expected_sample_size_[batch_id][src_metapath];
      _init_expected_sample_size(dst_metapath, batch_id, hop + 1);
    }
  }
}

HeteroNeighborSampler::HeteroNeighborSampler(
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

void HeteroNeighborSampler::uniform_sample(
    rel_type e_type,
    const triplet_int64_t global_src_node,
    const int64_t local_src_node,
    const int64_t count,
    pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
    pyg::random::RandintEngine<int64_t>& generator,
    std::vector<triplet_int64_t>& out_global_dst_nodes,
    MetapathTracker& metapath_tracker,
    bool return_edge_id = true) {
  auto rowptr_v = rowptr_.at(e_type).data_ptr<int64_t>();
  const auto row_start = rowptr_v[std::get<1>(global_src_node)];
  const auto row_end = rowptr_v[std::get<1>(global_src_node) + 1];
  if ((row_end - row_start == 0) || (count == 0))
    return;
  _sample(e_type, global_src_node, local_src_node, row_start, row_end, count,
          dst_mapper, generator, out_global_dst_nodes, metapath_tracker,
          return_edge_id);
}

void HeteroNeighborSampler::node_temporal_sample(
    rel_type e_type,
    std::string temporal_strategy,
    const triplet_int64_t global_src_node,
    const int64_t local_src_node,
    const int64_t count,
    const temporal_t seed_time,
    const temporal_t* time,
    pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
    pyg::random::RandintEngine<int64_t>& generator,
    std::vector<triplet_int64_t>& out_global_dst_nodes,
    MetapathTracker& metapath_tracker,
    bool return_edge_id = true) {
  auto row_start =
      rowptr_.at(e_type).data_ptr<int64_t>()[std::get<1>(global_src_node)];
  auto row_end = row_start + find_num_neighbors_temporal(
                                 e_type, global_src_node, seed_time, time);
  if ((row_end - row_start == 0) || (count == 0))
    return;

  if (temporal_strategy == "last" && count >= 0) {
    row_start = std::max(row_start, (int64_t)(row_end - count));
  }
  if (row_end - row_start == 0)
    return;
  _sample(e_type, global_src_node, local_src_node, row_start, row_end, count,
          dst_mapper, generator, out_global_dst_nodes, metapath_tracker,
          return_edge_id);
}

int64_t HeteroNeighborSampler::find_num_neighbors_temporal(
    rel_type e_type,
    const triplet_int64_t global_src_node,
    const temporal_t seed_time,
    const temporal_t* time) {
  auto row_start =
      rowptr_.at(e_type).data_ptr<int64_t>()[std::get<1>(global_src_node)];
  auto row_end =
      rowptr_.at(e_type).data_ptr<int64_t>()[std::get<1>(global_src_node) + 1];
  auto col = col_.at(e_type).data_ptr<int64_t>();

  if (row_end - row_start == 0)
    return 0;

  // Find new `row_end` such that all neighbors fulfill temporal constraints:
  auto it = std::upper_bound(
      col + row_start, col + row_end, seed_time,
      [&](const int64_t& a, const int64_t& b) { return a < time[b]; });

  row_end = it - col;
  if (row_end - row_start > 1) {
    TORCH_CHECK(time[col[row_start]] <= time[col[row_end - 1]],
                "Found invalid non-sorted temporal neighborhood");
  }
  return row_end - row_start;
}

int64_t HeteroNeighborSampler::find_num_neighbors(
    rel_type e_type,
    const triplet_int64_t global_src_node) {
  auto rowptr_v = rowptr_.at(e_type).data_ptr<int64_t>();
  const auto row_start = rowptr_v[std::get<1>(global_src_node)];
  const auto row_end = rowptr_v[std::get<1>(global_src_node) + 1];
  return row_end - row_start;
}

std::tuple<c10::Dict<rel_type, at::Tensor>,                  // row
           c10::Dict<rel_type, at::Tensor>,                  // col
           c10::Dict<node_type, at::Tensor>,                 // node_id
           c10::optional<c10::Dict<rel_type, at::Tensor>>,   // edge_id
           c10::optional<c10::Dict<node_type, at::Tensor>>,  // batch
           c10::Dict<node_type, std::vector<int64_t>>,  // num_sampled_nodes
           c10::Dict<rel_type, std::vector<int64_t>>>   // num_sampled_edges
HeteroNeighborSampler::sample(
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
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
  // tuples <int64, int64, int64> for both usecases and fill batch with 0
  // for !disjoint case. This is a bit wasteful but we'll clean this up
  // sometime later. TODO
  // Each triplet is (batch_id, node_id, metapath_id)
  phmap::flat_hash_map<node_type, std::vector<triplet_int64_t>> sampled_nodes;
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

  std::vector<node_type> seed_node_types;
  for (const auto& kv : seed_node)
    seed_node_types.push_back(kv.key());
  MetapathTracker metapath_tracker(edge_types_, num_neighbors, seed_node_types);

  // We fill the buffers with the zero-th layer: seed nodes
  int64_t batch_idx = 0;
  for (const auto& kv : seed_node) {
    const at::Tensor& seed = kv.value();
    slice_dict[kv.key()] = {0, seed.size(0)};
    const auto seed_data = seed.data_ptr<int64_t>();

    if (!disjoint) {
      auto metapath_id = metapath_tracker.init_batch(0, kv.key(), seed.numel());
      for (size_t i = 0; i < seed.numel(); ++i) {
        sampled_nodes[kv.key()].push_back({0, seed_data[i], metapath_id});
        mapper_dict.at(kv.key()).insert({0, seed_data[i]});
      }
    } else {
      // If dealing with disjoint subgraphs, we need to track each batch
      // separately to avoid leakage. Hence we always push pairs
      // <batch_id, node_id>
      auto& curr_sampled_nodes = sampled_nodes.at(kv.key());
      auto& mapper = mapper_dict.at(kv.key());
      for (size_t i = 0; i < seed.numel(); ++i) {
        auto metapath_id = metapath_tracker.init_batch(batch_idx, kv.key(), 1);
        curr_sampled_nodes.push_back({batch_idx, seed_data[i], metapath_id});
        mapper.insert({batch_idx, seed_data[i]});
        batch_idx++;
      }
      if (seed_time.has_value()) {
        const at::Tensor& curr_seed_time = seed_time.value().at(kv.key());
        const auto curr_seed_time_data = curr_seed_time.data_ptr<temporal_t>();
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
      sampled_batch_.at(kv.key()).push_back(std::get<0>(node));
      sampled_node_ids_.at(kv.key()).push_back(std::get<1>(node));
    }
  }

  // The actual sampling code begins here
  for (size_t ell = 0; ell < L; ++ell) {
    for (const auto& k : edge_types_) {
      // inner loop for edge type k: src->dst
      const auto src = std::get<0>(k);
      const auto dst = std::get<2>(k);
      auto& src_sampled_nodes = sampled_nodes.at(src);
      auto& dst_mapper = mapper_dict.at(dst);
      size_t begin, end;
      std::tie(begin, end) = slice_dict.at(src);
      num_sampled_edges_per_hop_[to_rel_type(k)].push_back(0);

      // Track occurrences of each batch to be on hold of balancing
      phmap::flat_hash_map<int64_t, int64_t> batch_total_count;
      phmap::flat_hash_map<int64_t, int64_t> batch_processed_count;
      for (size_t i = begin; i < end; ++i) {
        auto batch = std::get<0>(src_sampled_nodes[i]);
        if (batch_total_count.find(batch) == batch_total_count.end()) {
          batch_total_count[batch] = 0;
          batch_processed_count[batch] = 0;
        }
        batch_total_count[batch]++;
      }
      // We skip weighted/biased edges and edge-temporal sampling for now
      bool is_static =
          ((!node_time_.has_value() || !node_time_.value().contains(dst)) &&
           (!edge_time_.has_value() ||
            !edge_time_.value().contains(to_rel_type(k))));
      // Whenever we undersample nodes in a batch due to the lack of
      // neighbors, we allow oversampling neighbors of later nodes
      // of the same batch. To ensure that we batch to maximum capacity
      // we sample in the order of the growing number of neighbors.
      std::vector<pair_int64_t> node_order;
      for (auto i = begin; i < end; i++) {
        int64_t num_neighbors;
        if (is_static)
          num_neighbors =
              find_num_neighbors(to_rel_type(k), src_sampled_nodes[i]);
        else
          num_neighbors = find_num_neighbors_temporal(
              to_rel_type(k), src_sampled_nodes[i], seed_times[batch_idx],
              node_time_.value().at(dst).data_ptr<temporal_t>());
        node_order.push_back({num_neighbors, i});
      }
      std::sort(node_order.begin(), node_order.end());
      if (is_static) {
        for (auto& nd : node_order) {
          int64_t i = nd.second;
          const auto batch_idx = std::get<0>(src_sampled_nodes[i]);
          const auto expected_total = metapath_tracker.get_sample_size(
              batch_idx, std::get<2>(src_sampled_nodes[i]), k);
          const auto dst_metapath_id = metapath_tracker.get_neighbor_metapath(
              std::get<2>(src_sampled_nodes[i]), to_rel_type(k));
          const auto reported_total = metapath_tracker.get_reported_sample_size(
              batch_idx, dst_metapath_id);
          const auto remaining_batches =
              batch_total_count[batch_idx] - batch_processed_count[batch_idx];
          int64_t sample_size =
              (expected_total - reported_total) / remaining_batches;
          uniform_sample(
              /*e_type=*/to_rel_type(k),
              /*global_src_node=*/src_sampled_nodes[i],
              /*local_src_node=*/i,
              /*count=*/sample_size,
              /*dst_mapper=*/dst_mapper,
              /*generator=*/generator,
              /*out_global_dst_nodes=*/sampled_nodes.at(dst),
              /*metapath_tracker=*/metapath_tracker,
              /*return_edge_id=*/return_edge_id);
          batch_processed_count[batch_idx]++;
        }
      } else {
        // Node-level temporal sampling:
        const at::Tensor& dst_time = node_time_.value().at(dst);
        const auto dst_time_data = dst_time.data_ptr<temporal_t>();
        for (auto& nd : node_order) {
          int64_t i = nd.second;
          const auto batch_idx = std::get<0>(src_sampled_nodes[i]);
          const auto expected_total = metapath_tracker.get_sample_size(
              batch_idx, std::get<2>(src_sampled_nodes[i]), k);
          const auto dst_metapath_id = metapath_tracker.get_neighbor_metapath(
              std::get<2>(src_sampled_nodes[i]), to_rel_type(k));
          const auto reported_total = metapath_tracker.get_reported_sample_size(
              batch_idx, dst_metapath_id);
          const auto remaining_batches =
              batch_total_count[batch_idx] - batch_processed_count[batch_idx];
          int64_t sample_size =
              (expected_total - reported_total) / remaining_batches;
          node_temporal_sample(
              /*e_type=*/to_rel_type(k),
              /*temporal_strategy=*/temporal_strategy,
              /*global_src_node=*/src_sampled_nodes[i],
              /*local_src_node=*/i,
              /*count=*/sample_size,
              /*seed_time=*/seed_times[batch_idx],
              /*time=*/dst_time_data,
              /*dst_mapper=*/dst_mapper,
              /*generator=*/generator,
              /*out_global_dst_nodes=*/sampled_nodes.at(dst),
              /*metapath_tracker=*/metapath_tracker,
              /*return_edge_id=*/return_edge_id);
          batch_processed_count[batch_idx]++;
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
        sampled_batch_.at(k).push_back(std::get<0>(node));
        sampled_node_ids_.at(k).push_back(std::get<1>(node));
      }
    }
  }

  // We rewrite phmap objects into c10 ones for the return value
  c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop_dict;
  c10::Dict<node_type, at::Tensor> out_node_id;
  c10::Dict<node_type, at::Tensor> batch;
  for (const auto& k : node_types_) {
    out_node_id.insert(k, pyg::utils::from_vector(sampled_node_ids_.at(k)));
    num_sampled_nodes_per_hop_dict.insert(k, num_sampled_nodes_per_hop_.at(k));
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

void HeteroNeighborSampler::clear_placeholders() {
  num_sampled_nodes_per_hop_.clear();
  num_sampled_edges_per_hop_.clear();
  sampled_cols_.clear();
  sampled_rows_.clear();
  sampled_batch_.clear();
  sampled_node_ids_.clear();
  sampled_edge_ids_.clear();
}

void HeteroNeighborSampler::init_placeholders() {
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

void HeteroNeighborSampler::_sample(
    rel_type e_type,
    const triplet_int64_t global_src_node,
    const int64_t local_src_node,
    const int64_t row_start,
    const int64_t row_end,
    const int64_t count,
    pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
    pyg::random::RandintEngine<int64_t>& generator,
    std::vector<triplet_int64_t>& out_global_dst_nodes,
    MetapathTracker& metapath_tracker,
    bool return_edge_id = true) {
  const auto population = row_end - row_start;
  // For now we'll assume that we're not sampling with replacement since
  // that's the default
  // Case 1: Sample the full neighborhood:
  if (count < 0 || count >= population) {
    for (int64_t edge_id = row_start; edge_id < row_end; ++edge_id)
      add_edge(e_type, edge_id, global_src_node, local_src_node, dst_mapper,
               out_global_dst_nodes, metapath_tracker, return_edge_id);
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
               out_global_dst_nodes, metapath_tracker, return_edge_id);
    }
  }
}

inline void HeteroNeighborSampler::add_edge(
    rel_type e_type,
    const int64_t edge_id,
    const triplet_int64_t global_src_node,
    const int64_t local_src_node,
    pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
    std::vector<triplet_int64_t>& out_global_dst_nodes,
    MetapathTracker& metapath_tracker,
    bool return_edge_id = true) {
  const auto global_dst_node_value =
      col_.at(e_type).data_ptr<int64_t>()[edge_id];

  auto dst_metapath_id = metapath_tracker.get_neighbor_metapath(
      std::get<2>(global_src_node), e_type);
  const auto global_dst_node = std::make_tuple(
      std::get<0>(global_src_node), global_dst_node_value, dst_metapath_id);

  // There was special handling for a distributed setting which we ignore
  // for now

  const auto res = dst_mapper.insert(std::make_pair(
      std::get<0>(global_dst_node), std::get<1>(global_dst_node)));
  if (res.second)  // not yet sampled.
    out_global_dst_nodes.push_back(global_dst_node);
  // handle the global vars below
  metapath_tracker.report_sample_size(std::get<0>(global_src_node),
                                      dst_metapath_id, 1);
  num_sampled_edges_per_hop_[e_type]
                            [num_sampled_edges_per_hop_[e_type].size() - 1]++;
  sampled_cols_[e_type].push_back(res.first);
  sampled_rows_[e_type].push_back(local_src_node);
  if (return_edge_id)
    sampled_edge_ids_.at(e_type).push_back(edge_id);
}

}  // namespace classes

TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<pyg::classes::NeighborSampler>("NeighborSampler")
      .def(torch::init<at::Tensor&, at::Tensor&, c10::optional<at::Tensor>,
                       c10::optional<at::Tensor>, c10::optional<at::Tensor>>())
      .def("sample", &pyg::classes::NeighborSampler::sample);

  m.class_<pyg::classes::HeteroNeighborSampler>("HeteroNeighborSampler")
      .def(torch::init<std::vector<node_type>, std::vector<edge_type>,
                       c10::Dict<rel_type, at::Tensor>,
                       c10::Dict<rel_type, at::Tensor>,
                       c10::optional<c10::Dict<rel_type, at::Tensor>>,
                       c10::optional<c10::Dict<node_type, at::Tensor>>,
                       c10::optional<c10::Dict<rel_type, at::Tensor>>>())
      .def("sample", &pyg::classes::HeteroNeighborSampler::sample);

}  // namespace classes
}  // namespace pyg
