#pragma once

#include <ATen/ATen.h>
#include <parallel_hashmap/phmap.h>
#include <torch/library.h>

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace classes {

struct MetapathTracker : torch::CustomClassHolder {
  /* This is a helper class for NeighborSampler. It pre-computes all possible
   * metapaths and how many of each we are expected to sample if we always
   * sample the full number specified in `num_neighbors`. It can then be used
   * to track the actual number of sampled edges of each type.
   * */
 public:
  MetapathTracker(
      const std::vector<edge_type>& edge_types,
      const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
      const std::vector<node_type>& seed_node_types);

  int64_t get_neighbor_metapath(const int64_t& metapath_id,
                                const rel_type& edge);
  int64_t get_sample_size(const int64_t& batch_id,
                          const int64_t& src_metapath_id,
                          const edge_type& edge);
  void report_sample_size(const int64_t& batch_id,
                          const int64_t& metapath_id,
                          const int64_t n_sampled);
  int64_t get_reported_sample_size(const int64_t& batch_id,
                                   const int64_t& metapath_id);
  int64_t init_batch(const int64_t& batch_id,
                     const node_type& node_t,
                     const int64_t batch_size);
  void _init_expected_sample_size(const int64_t src_metapath,
                                  const int64_t batch_id,
                                  const int64_t hop);

  phmap::flat_hash_map<int64_t, phmap::flat_hash_map<int64_t, int64_t>>
      reported_sample_size_;

 private:
  std::vector<edge_type> edge_types_;
  c10::Dict<rel_type, std::vector<int64_t>> num_neighbors_;
  int64_t n_metapaths_;
  phmap::flat_hash_set<int64_t> batch_ids_;
  phmap::flat_hash_map<node_type, int64_t> seed_metapaths_;
  phmap::flat_hash_map<rel_type, phmap::flat_hash_map<int64_t, int64_t>>
      metapath_tree_;
  phmap::flat_hash_map<int64_t, phmap::flat_hash_map<int64_t, int64_t>>
      expected_sample_size_;
};

struct HeteroNeighborSampler : torch::CustomClassHolder {
  using pair_int64_t = std::pair<int64_t, int64_t>;
  using triplet_int64_t = std::tuple<int64_t, int64_t, int64_t>;
  using temporal_t = int64_t;

  HeteroNeighborSampler(
      const std::vector<node_type> node_types,
      const std::vector<edge_type> edge_types,
      const c10::Dict<rel_type, at::Tensor> rowptr,
      const c10::Dict<rel_type, at::Tensor> col,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_weight,
      const c10::optional<c10::Dict<node_type, at::Tensor>> node_time,
      const c10::optional<c10::Dict<rel_type, at::Tensor>> edge_time);

  std::tuple<c10::Dict<rel_type, at::Tensor>,
             c10::Dict<rel_type, at::Tensor>,
             c10::Dict<node_type, at::Tensor>,
             c10::optional<c10::Dict<rel_type, at::Tensor>>,
             c10::optional<c10::Dict<node_type, at::Tensor>>,
             c10::Dict<node_type, std::vector<int64_t>>,
             c10::Dict<rel_type, std::vector<int64_t>>>
  sample(const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors,
         const c10::Dict<node_type, at::Tensor>& seed_node,
         const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time,
         bool disjoint,
         std::string temporal_strategy,
         bool return_edge_id);

  void uniform_sample(rel_type e_type,
                      const triplet_int64_t global_src_node,
                      const int64_t local_src_node,
                      const int64_t count,
                      pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
                      pyg::random::RandintEngine<int64_t>& generator,
                      std::vector<triplet_int64_t>& out_global_dst_nodes,
                      MetapathTracker& metapath_tracker,
                      bool return_edge_id);

  void node_temporal_sample(
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
      bool return_edge_id);

  int64_t find_num_neighbors_temporal(rel_type e_type,
                                      const triplet_int64_t global_src_node,
                                      const temporal_t seed_time,
                                      const temporal_t* time);

  int64_t find_num_neighbors(rel_type e_type,
                             const triplet_int64_t global_src_node);

 private:
  void clear_placeholders();
  void init_placeholders();
  void _sample(rel_type e_type,
               const triplet_int64_t global_src_node,
               const int64_t local_src_node,
               const int64_t row_start,
               const int64_t row_end,
               const int64_t count,
               pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
               pyg::random::RandintEngine<int64_t>& generator,
               std::vector<triplet_int64_t>& out_global_dst_nodes,
               MetapathTracker& metapath_tracker,
               bool return_edge_id);
  inline void add_edge(rel_type e_type,
                       const int64_t edge_id,
                       const triplet_int64_t global_src_node,
                       const int64_t local_src_node,
                       pyg::sampler::Mapper<pair_int64_t, int64_t>& dst_mapper,
                       std::vector<triplet_int64_t>& out_global_dst_nodes,
                       MetapathTracker& metapath_tracker,
                       bool return_edge_id);

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
}  // namespace classes
}  // namespace pyg
