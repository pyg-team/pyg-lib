#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "parallel_hashmap/phmap.h"

#include "pyg_lib/csrc/random/cpu/rand_engine.h"
#include "pyg_lib/csrc/sampler/cpu/index_tracker.h"
#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/sampler/cpu/neighbor_kernel.h"
#include "pyg_lib/csrc/sampler/subgraph.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

#include <chrono>

namespace pyg {
namespace sampler {

bool bidir_sampling_opt1 {false}; // option1 adds reverse links as links are sampled and added (so in add function)
bool bidir_sampling_opt2 {true};  // option2 adds reverse links at the end of sampling (in get_sampled_edges)
bool u_ordering {true}; // true: unique links via re-ordering, false via hashing function
bool get_durations {false};
namespace {

// Helper classes for bipartite neighbor sampling //////////////////////////////

// `node_t` is either a scalar or a pair of scalars (example_id, node_id):

template <typename node_t,
          typename scalar_t,
          typename temporal_t,
          bool replace,
          bool save_edges,
          bool save_edge_ids>
class NeighborSampler {
 typedef std::tuple<std::pair<scalar_t, scalar_t>, scalar_t> link;// added to support bidirectional sampling
 public:
  NeighborSampler(const scalar_t* rowptr,
                  const scalar_t* col,
                  const std::string temporal_strategy)
      : rowptr_(rowptr), col_(col), temporal_strategy_(temporal_strategy) {
    TORCH_CHECK(temporal_strategy == "uniform" || temporal_strategy == "last",
                "No valid temporal strategy found");
  }

  void uniform_sample(const node_t global_src_node,
                      const scalar_t local_src_node,
                      const int64_t count,
                      pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                      pyg::random::RandintEngine<scalar_t>& generator,
                      std::vector<node_t>& out_global_dst_nodes) {
    const auto row_start = rowptr_[to_scalar_t(global_src_node)];
    const auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];
    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes);
  }

  void temporal_sample(const node_t global_src_node,
                       const scalar_t local_src_node,
                       const int64_t count,
                       const temporal_t seed_time,
                       const temporal_t* time,
                       pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                       pyg::random::RandintEngine<scalar_t>& generator,
                       std::vector<node_t>& out_global_dst_nodes) {
    auto row_start = rowptr_[to_scalar_t(global_src_node)];
    auto row_end = rowptr_[to_scalar_t(global_src_node) + 1];

    // Find new `row_end` such that all neighbors fulfill temporal constraints:
    auto it = std::upper_bound(
        col_ + row_start, col_ + row_end, seed_time,
        [&](const scalar_t& a, const scalar_t& b) { return a < time[b]; });
    row_end = it - col_;

    if (temporal_strategy_ == "last" && count >= 0) {
      row_start = std::max(row_start, (scalar_t)(row_end - count));
    }

    if (row_end - row_start > 1) {
      TORCH_CHECK(time[col_[row_start]] <= time[col_[row_end - 1]],
                  "Found invalid non-sorted temporal neighborhood");
    }

    _sample(global_src_node, local_src_node, row_start, row_end, count,
            dst_mapper, generator, out_global_dst_nodes);
  }

  std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
  get_sampled_edges(bool csc = false) {
    TORCH_CHECK(save_edges, "No edges have been stored")
    if (bidir_sampling_opt2){
      //1 first duplicate rows and cols concatenating cols to rows, and rows to cols, like in to_bidirectional
      if (save_edges) {
        _mutuallyExtendVectors(sampled_rows_, sampled_cols_ );
      }
      //2 if edge_id is not none, use the same edge_id for both (not clear why but it is like this in pyg)
      // and duplicate edge_id vector 
      if (save_edge_ids){
        _extendVector(sampled_edge_ids_, sampled_edge_ids_);
      }
      //3 What above covers what it is done in utils.py:to_bidirectional. 
      // Now, do all is done in coalesce.py. See first how much it takes to reorder
       // The following call _removeDuplicateLinks() reorders links AND removes duplicates. 
       // Note that does that on a link structure and
       // does not use a smart sequence and then a mask as in coalesce.py. 
       // Also it does not suport any reduction function 
       // It just keeps the first linkID found for that link.
       //_removeDuplicateLinks();
    }
    const auto row = pyg::utils::from_vector(sampled_rows_);
    const auto col = pyg::utils::from_vector(sampled_cols_);
    c10::optional<at::Tensor> edge_id = c10::nullopt;
    if (save_edge_ids) {
      edge_id = pyg::utils::from_vector(sampled_edge_ids_);
    }
    // if (bidir_sampling_opt2){..} could place here, with the idea of using torch C++API to use sort and scatter
    if (!csc) {
      return std::make_tuple(row, col, edge_id);
    } else {
      return std::make_tuple(col, row, edge_id);
    }
  }

  void removeDuplicateLinks(){
    // set time init in case of logging
    auto startRD = std::chrono::high_resolution_clock::now();

    _removeDuplicateLinks();

    if (get_durations){
      auto stopRD = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopRD - startRD);
      std::cout << "removeDuplicateLinks duration: " << duration.count() << std::endl;
    }
  }

 private:
  inline scalar_t to_scalar_t(const scalar_t& node) { return node; }
  inline scalar_t to_scalar_t(const std::pair<scalar_t, scalar_t>& node) {
    return std::get<1>(node);
  }

  inline scalar_t to_node_t(const scalar_t& node, const scalar_t& ref) {
    return node;
  }
  inline std::pair<scalar_t, scalar_t> to_node_t(
      const scalar_t& node,
      const std::pair<scalar_t, scalar_t>& ref) {
    return {std::get<0>(ref), node};
  }

  void _sample(const node_t global_src_node,
               const scalar_t local_src_node,
               const scalar_t row_start,
               const scalar_t row_end,
               const int64_t count,
               pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
               pyg::random::RandintEngine<scalar_t>& generator,
               std::vector<node_t>& out_global_dst_nodes) {
    if (count == 0)
      return;

    const auto population = row_end - row_start;

    if (population == 0)
      return;

    // Case 1: Sample the full neighborhood:
    if (count < 0 || (!replace && count >= population)) {
      for (scalar_t edge_id = row_start; edge_id < row_end; ++edge_id) {
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }

    // Case 2: Sample with replacement:
    else if (replace) {
      for (size_t i = 0; i < count; ++i) {
        const auto edge_id = generator(row_start, row_end);
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }

    // Case 3: Sample without replacement:
    else {
      auto index_tracker = IndexTracker<scalar_t>(population);
      for (size_t i = population - count; i < population; ++i) {
        auto rnd = generator(0, i + 1);
        if (!index_tracker.try_insert(rnd)) {
          rnd = i;
          index_tracker.insert(i);
        }
        const auto edge_id = row_start + rnd;
        add(edge_id, global_src_node, local_src_node, dst_mapper,
            out_global_dst_nodes);
      }
    }
  }
  
  inline void add(const scalar_t edge_id,
                  const node_t global_src_node,
                  const scalar_t local_src_node,
                  pyg::sampler::Mapper<node_t, scalar_t>& dst_mapper,
                  std::vector<node_t>& out_global_dst_nodes) {
    const auto global_dst_node_value = col_[edge_id];
    const auto global_dst_node =
        to_node_t(global_dst_node_value, global_src_node);
    const auto res = dst_mapper.insert(global_dst_node);

    if (res.second) {  // not yet sampled.
      out_global_dst_nodes.push_back(global_dst_node);
    }
    if (save_edges) {
      num_sampled_edges_per_hop[num_sampled_edges_per_hop.size() - 1]++;
      sampled_rows_.push_back(local_src_node);
      sampled_cols_.push_back(res.first);
      if (pyg::sampler::bidir_sampling_opt1) {//adding reverse link 
          sampled_rows_.push_back(res.first);
          sampled_cols_.push_back(local_src_node); 
      }
      if (save_edge_ids) {
        sampled_edge_ids_.push_back(edge_id);
        if (pyg::sampler::bidir_sampling_opt1){ //re-use same edge-id for reverse link
          sampled_edge_ids_.push_back(edge_id);
        }
      }
    }
  }

//  use to check link existence as it gets added: to slow approach
//  inline int checkLinkExistence(const std::vector<scalar_t>& source, 
//                                 const std::vector<scalar_t>& destination, 
//                                 const std::tuple<scalar_t, scalar_t>& newLink) {
//     scalar_t newSource = std::get<0>(newLink);
//     scalar_t newDestination = std::get<1>(newLink);

//     for (int i = 0; i < source.size(); ++i) {
//         if (source[i] == newSource && destination[i] == newDestination) {
//             return 1;  // Link already exists
//         }
//     }
//     return 0;  // Link does not exist
//  }

  void _mutuallyExtendVectors(std::vector<scalar_t>& avector, std::vector<scalar_t>& bvector) {
    std::vector ttemp(avector.begin(), avector.end()); //maybe useless
    avector.insert(avector.end(), bvector.begin(), bvector.end());
    bvector.insert(bvector.end(), ttemp.begin(), ttemp.end());
  }

  void _extendVector(std::vector<scalar_t>& avector, const std::vector<scalar_t>& bvector) {
    avector.insert(avector.end(), bvector.begin(), bvector.end());
  }

  void _removeDuplicateLinks(){
      if (u_ordering){
        _removeDuplicateLinks_ordering_func();
      }else{
        _removeDuplicateLinks_hashing_func();
      }
  }

// finding unique entries with hashing function
  struct pair_hash {
      template <class T1, class T2>
      std::size_t operator () (const std::pair<T1, T2>& p) const {
          auto h1 = std::hash<T1>{}(p.first);
          auto h2 = std::hash<T2>{}(p.second);
          return h1 ^ h2;
      }
  };

  void _removeDuplicateLinks_hashing_func() {
      std::unordered_map<std::pair<scalar_t, scalar_t>, scalar_t, pair_hash> linkMap;
      std::vector<scalar_t> uniqueSource;
      std::vector<scalar_t> uniqueDestination;
      std::vector<scalar_t> uniqueLinkIDs;
      uniqueSource.reserve(sampled_rows_.size());
      uniqueDestination.reserve(sampled_rows_.size());
      uniqueLinkIDs.reserve(sampled_rows_.size());

      // std::cout << "sampled_rows_.size is: " << sampled_rows_.size() << std::endl;
      // auto startHSH = std::chrono::high_resolution_clock::now();

      for (int i = 0; i < sampled_rows_.size(); ++i) {
        std::pair<scalar_t, scalar_t> link = std::make_pair(sampled_rows_[i], sampled_cols_[i]);
        if (linkMap.count(link) == 0) { /*this is the call to hash function with () operator*/
          linkMap[link] = i;
          uniqueSource.push_back(sampled_rows_[i]);
          uniqueDestination.push_back(sampled_cols_[i]);
          uniqueLinkIDs.push_back(sampled_edge_ids_[i]);
        }
      }   
      sampled_rows_.assign(uniqueSource.begin(), uniqueSource.end());
      sampled_cols_.assign(uniqueDestination.begin(), uniqueDestination.end());
      sampled_edge_ids_.assign(uniqueLinkIDs.begin(), uniqueLinkIDs.end());   

      // auto stopHSH = std::chrono::high_resolution_clock::now();
      // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopHSH - startHSH);
      // std::cout << "Time taken by generating unique link vectors using hasing function : " << duration.count() << std::endl; 
      return; 
  }

// finding unique entries ordering first
  // comparator function for sorting based on the first node of the link
  inline static bool sortByFirst(const link& a, const link& b) {
      return (std::get<0>(a)).first < (std::get<0>(b)).first;
  }

  // comparator for sorting based on the second node of the link
  inline static bool sortBySecond(const link& a, const link& b) {
      return (std::get<0>(a)).second < (std::get<0>(b)).second;
  }

  // sort an array of links
  void sortLinks(std::vector<link>& arr, bool sortByFirstElement) {
      std::vector<link> initialVector = arr;
      if (sortByFirstElement) {
          std::sort(arr.begin(), arr.end(), sortByFirst);
      } else {
          std::sort(arr.begin(), arr.end(), sortBySecond);
      }
      return;
  }

  // remove duplicates from the vector
  std::vector<link> __removeDuplicates(const std::vector<link>& links) {
      std::vector<link> result;
      if (links.empty()) {
          return result;
      }
      result.push_back(links[0]);
      for (size_t i = 1; i < links.size(); i++) {
          const auto& currentLink = links[i];
          const auto& prevLink = links[i - 1];
          if (std::get<0>(currentLink) != std::get<0>(prevLink)) {
              result.push_back(currentLink);
          }
      }
      return result;
  }

  void _removeDuplicateLinks_ordering_func() {
      std::vector<scalar_t> uniqueSource;
      std::vector<scalar_t> uniqueDestination;
      std::vector<scalar_t> uniqueLinkIDs;
      uniqueSource.reserve(sampled_rows_.size());
      uniqueDestination.reserve(sampled_rows_.size());
      uniqueLinkIDs.reserve(sampled_rows_.size());
  
      // set initial time for logging 
      auto startHSH = std::chrono::high_resolution_clock::now();

      std::vector<link> links;
      links.reserve(sampled_rows_.size());
      for (size_t i = 0; i < sampled_rows_.size(); ++i) {
          links.emplace_back(std::make_pair(sampled_rows_[i], sampled_cols_[i]), sampled_edge_ids_[i]);
      }
      if (get_durations){
        auto stopHSH = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopHSH - startHSH);
        std::cout << "Time taken by PREPARATION links structure: " << duration.count() << std::endl; 
      }

      if (get_durations){
        startHSH = std::chrono::high_resolution_clock::now();
      }
      // Sort by first element and get the initial vector ordered
      sortLinks(links, true);
      if (get_durations){
        auto stopHSH = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopHSH - startHSH);
        std::cout << "Time taken by REORDERING links structure: " << duration.count() << std::endl;
      }

      if (get_durations){
        startHSH = std::chrono::high_resolution_clock::now();
      }
      //remove duplicate links
      auto noduplicates =__removeDuplicates(links);
      if (get_durations){
        auto stopHSH = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopHSH - startHSH);
        std::cout << "Time taken by REMOVING duplicates from links structure: " << duration.count() << std::endl;
      }

      if (get_durations){
        startHSH = std::chrono::high_resolution_clock::now();
      }
      // Copy the data from the links vector to the NeighborSampler fields
      for (size_t i = 0; i < noduplicates.size(); ++i) {
          sampled_rows_[i] = std::get<0>(noduplicates[i]).first;
          sampled_cols_[i] = std::get<0>(noduplicates[i]).second;
          sampled_edge_ids_[i] = std::get<1>(noduplicates[i]);
          //std::cout << "s,d,l: " << source[i] <<"-" << destination[i] << "-"<< linkIDs[i] << std::endl;
      }
      if (get_durations){
        auto stopHSH = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopHSH - startHSH);
        std::cout << "Time taken by copying back data to NeighboSampler fields: " << duration.count() << std::endl; 
      }
      return; 
  }

  const scalar_t* rowptr_;
  const scalar_t* col_;
  const std::string temporal_strategy_;
  std::vector<scalar_t> sampled_rows_;
  std::vector<scalar_t> sampled_cols_;
  std::vector<scalar_t> sampled_edge_ids_;

 public:
  std::vector<int64_t> num_sampled_edges_per_hop;
};


// Homogeneous neighbor sampling ///////////////////////////////////////////////

template <bool replace, bool directed, bool disjoint, bool return_edge_id>
std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
sample(const at::Tensor& rowptr,
       const at::Tensor& col,
       const at::Tensor& seed,
       const std::vector<int64_t>& num_neighbors,
       const c10::optional<at::Tensor>& time,
       const c10::optional<at::Tensor>& seed_time,
       const bool csc,
       const std::string temporal_strategy) {
  TORCH_CHECK(!time.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");

  TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  if (time.has_value()) {
    TORCH_CHECK(time.value().is_contiguous(), "Non-contiguous 'time'");
  }
  if (seed_time.has_value()) {
    TORCH_CHECK(seed_time.value().is_contiguous(),
                "Non-contiguous 'seed_time'");
  }

  at::Tensor out_row, out_col, out_node_id;
  c10::optional<at::Tensor> out_edge_id = c10::nullopt;
  std::vector<int64_t> num_sampled_nodes_per_hop;
  std::vector<int64_t> num_sampled_edges_per_hop;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    // TODO(zeyuan): Do not force int64_t for time type.
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id>
        NeighborSamplerImpl;

    pyg::random::RandintEngine<scalar_t> generator;

    std::vector<node_t> sampled_nodes;
    auto mapper = Mapper<node_t, scalar_t>(/*num_nodes=*/rowptr.size(0) - 1);
    auto sampler =
        NeighborSamplerImpl(rowptr.data_ptr<scalar_t>(),
                            col.data_ptr<scalar_t>(), temporal_strategy);
    std::vector<temporal_t> seed_times;

    const auto seed_data = seed.data_ptr<scalar_t>();
    if constexpr (!disjoint) {
      sampled_nodes = pyg::utils::to_vector<scalar_t>(seed);
      mapper.fill(seed);
    } else {
      for (size_t i = 0; i < seed.numel(); ++i) {
        sampled_nodes.push_back({i, seed_data[i]});
        mapper.insert({i, seed_data[i]});
      }
      if (seed_time.has_value()) {
        const auto seed_time_data = seed_time.value().data_ptr<temporal_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          seed_times.push_back(seed_time_data[i]);
        }
      } else if (time.has_value()) {
        const auto time_data = time.value().data_ptr<temporal_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          seed_times.push_back(time_data[seed_data[i]]);
        }
      }
    }

    num_sampled_nodes_per_hop.push_back(seed.numel());

    size_t begin = 0, end = seed.size(0);
    auto startSMP = std::chrono::high_resolution_clock::now();

    for (size_t ell = 0; ell < num_neighbors.size(); ++ell) {
      const auto count = num_neighbors[ell];
      sampler.num_sampled_edges_per_hop.push_back(0);
      if (!time.has_value()) {
        for (size_t i = begin; i < end; ++i) {
          sampler.uniform_sample(/*global_src_node=*/sampled_nodes[i],
                                 /*local_src_node=*/i, count, mapper, generator,
                                 /*out_global_dst_nodes=*/sampled_nodes);
        }
      } else if constexpr (!std::is_scalar<node_t>::value) {  // Temporal:
        const auto time_data = time.value().data_ptr<temporal_t>();
        for (size_t i = begin; i < end; ++i) {
          const auto batch_idx = sampled_nodes[i].first;
          sampler.temporal_sample(/*global_src_node=*/sampled_nodes[i],
                                  /*local_src_node=*/i, count,
                                  seed_times[batch_idx], time_data, mapper,
                                  generator,
                                  /*out_global_dst_nodes=*/sampled_nodes);
        }
      }
      begin = end, end = sampled_nodes.size();
      num_sampled_nodes_per_hop.push_back(end - begin);
    }
    if (bidir_sampling_opt1){
            sampler.removeDuplicateLinks();
    }
    if (get_durations){
      auto stopSMP = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stopSMP - startSMP);
      std::cout << "Whole Uniform sampling duration: " << duration.count() << std::endl;
    }
    out_node_id = pyg::utils::from_vector(sampled_nodes);
    TORCH_CHECK(directed, "Undirected subgraphs not yet supported");
    if (directed) {
      std::tie(out_row, out_col, out_edge_id) = sampler.get_sampled_edges(csc); /* AZ here std::vector ==> torch.tensor*/
    } else {
      TORCH_CHECK(!disjoint, "Disjoint subgraphs not yet supported");
    }

    num_sampled_edges_per_hop = sampler.num_sampled_edges_per_hop;
  });
  
  return std::make_tuple(out_row, out_col, out_node_id, out_edge_id,
                         num_sampled_nodes_per_hop, num_sampled_edges_per_hop);
}

// Heterogeneous neighbor sampling /////////////////////////////////////////////

template <bool replace, bool directed, bool disjoint, bool return_edge_id>
std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
sample(const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
       const c10::Dict<rel_type, at::Tensor>& col_dict,
       const c10::Dict<node_type, at::Tensor>& seed_dict,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
       const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const bool csc,
       const std::string temporal_strategy) {
  TORCH_CHECK(!time_dict.has_value() || disjoint,
              "Temporal sampling needs to create disjoint subgraphs");

  for (const auto& kv : rowptr_dict) {
    const at::Tensor& rowptr = kv.value();
    TORCH_CHECK(rowptr.is_contiguous(), "Non-contiguous 'rowptr'");
  }
  for (const auto& kv : col_dict) {
    const at::Tensor& col = kv.value();
    TORCH_CHECK(col.is_contiguous(), "Non-contiguous 'col'");
  }
  for (const auto& kv : seed_dict) {
    const at::Tensor& seed = kv.value();
    TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  }
  if (time_dict.has_value()) {
    for (const auto& kv : time_dict.value()) {
      const at::Tensor& time = kv.value();
      TORCH_CHECK(time.is_contiguous(), "Non-contiguous 'time'");
    }
  }
  if (seed_time_dict.has_value()) {
    for (const auto& kv : seed_time_dict.value()) {
      const at::Tensor& seed_time = kv.value();
      TORCH_CHECK(seed_time.is_contiguous(), "Non-contiguous 'seed_time'");
    }
  }

  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;
  c10::Dict<node_type, at::Tensor> out_node_id_dict;
  c10::optional<c10::Dict<node_type, at::Tensor>> out_edge_id_dict;
  if (return_edge_id) {
    out_edge_id_dict = c10::Dict<rel_type, at::Tensor>();
  } else {
    out_edge_id_dict = c10::nullopt;
  }
  std::unordered_map<node_type, std::vector<int64_t>>
      num_sampled_nodes_per_hop_map;
  c10::Dict<node_type, std::vector<int64_t>> num_sampled_nodes_per_hop_dict;
  c10::Dict<rel_type, std::vector<int64_t>> num_sampled_edges_per_hop_dict;

  const auto scalar_type = seed_dict.begin()->value().scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "hetero_sample_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;
    typedef int64_t temporal_t;
    typedef NeighborSampler<node_t, scalar_t, temporal_t, replace, directed,
                            return_edge_id>
        NeighborSamplerImpl;

    pyg::random::RandintEngine<scalar_t> generator;

    phmap::flat_hash_map<node_type, size_t> num_nodes_dict;
    for (const auto& k : edge_types) {
      const auto num_nodes = rowptr_dict.at(to_rel_type(k)).size(0) - 1;
      num_nodes_dict[!csc ? std::get<0>(k) : std::get<2>(k)] = num_nodes;
    }
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      if (num_nodes_dict.count(kv.key()) == 0 && seed.numel() > 0) {
        num_nodes_dict[kv.key()] = seed.max().data_ptr<scalar_t>()[0] + 1;
      }
    }

    size_t L = 0;  // num_layers.
    phmap::flat_hash_map<node_type, std::vector<node_t>> sampled_nodes_dict;
    phmap::flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
    phmap::flat_hash_map<edge_type, NeighborSamplerImpl> sampler_dict;
    phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    std::vector<scalar_t> seed_times;

    for (const auto& k : node_types) {
      const auto N = num_nodes_dict.count(k) > 0 ? num_nodes_dict.at(k) : 0;
      sampled_nodes_dict[k];  // Initialize empty vector.
      num_sampled_nodes_per_hop_map.insert({k, std::vector<int64_t>(1, 0)});
      mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
      slice_dict[k] = {0, 0};
    }

    const bool parallel = at::get_num_threads() > 1 && edge_types.size() > 1;
    std::vector<std::vector<edge_type>> threads_edge_types;

    for (const auto& k : edge_types) {
      L = std::max(L, num_neighbors_dict.at(to_rel_type(k)).size());
      sampler_dict.insert(
          {k, NeighborSamplerImpl(
                  rowptr_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  col_dict.at(to_rel_type(k)).data_ptr<scalar_t>(),
                  temporal_strategy)});

      if (parallel) {
        // Each thread is assigned edge types that have the same dst node type.
        // Thanks to this, each thread will operate on a separate mapper and
        // separate sampler.
        bool added = false;
        const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
        for (auto& e : threads_edge_types) {
          if ((!csc ? std::get<2>(e[0]) : std::get<0>(e[0])) == dst) {
            e.push_back(k);
            added = true;
            break;
          }
        }
        if (!added)
          threads_edge_types.push_back({k});
      }
    }
    if (!parallel) {  // If not parallel then one thread handles all edge types.
      threads_edge_types.push_back({edge_types});
    }

    scalar_t batch_idx = 0;
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      slice_dict[kv.key()] = {0, seed.size(0)};

      if constexpr (!disjoint) {
        sampled_nodes_dict[kv.key()] = pyg::utils::to_vector<scalar_t>(seed);
        mapper_dict.at(kv.key()).fill(seed);
      } else {
        auto& sampled_nodes = sampled_nodes_dict.at(kv.key());
        auto& mapper = mapper_dict.at(kv.key());
        const auto seed_data = seed.data_ptr<scalar_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          sampled_nodes.push_back({batch_idx, seed_data[i]});
          mapper.insert({batch_idx, seed_data[i]});
          batch_idx++;
        }
        if (seed_time_dict.has_value()) {
          const at::Tensor& seed_time = seed_time_dict.value().at(kv.key());
          const auto seed_time_data = seed_time.data_ptr<scalar_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(seed_time_data[i]);
          }
        } else if (time_dict.has_value()) {
          const at::Tensor& time = time_dict.value().at(kv.key());
          const auto time_data = time.data_ptr<scalar_t>();
          seed_times.reserve(seed_times.size() + seed.numel());
          for (size_t i = 0; i < seed.numel(); ++i) {
            seed_times.push_back(time_data[seed_data[i]]);
          }
        }
      }

      num_sampled_nodes_per_hop_map.at(kv.key())[0] =
          sampled_nodes_dict.at(kv.key()).size();
    }

    size_t begin, end;
    for (size_t ell = 0; ell < L; ++ell) {
      phmap::flat_hash_map<node_type, std::vector<node_t>>
          dst_sampled_nodes_dict;
      if (parallel) {
        for (const auto& k : threads_edge_types) {  // Intialize empty vectors.
          dst_sampled_nodes_dict[!csc ? std::get<2>(k[0]) : std::get<0>(k[0])];
        }
      }
      at::parallel_for(
          0, threads_edge_types.size(), 1, [&](size_t _s, size_t _e) {
            for (auto j = _s; j < _e; j++) {
              for (const auto& k : threads_edge_types[j]) {
                const auto src = !csc ? std::get<0>(k) : std::get<2>(k);
                const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);
                const auto count = num_neighbors_dict.at(to_rel_type(k))[ell];
                auto& src_sampled_nodes = sampled_nodes_dict.at(src);
                auto& dst_sampled_nodes = parallel
                                              ? dst_sampled_nodes_dict.at(dst)
                                              : sampled_nodes_dict.at(dst);
                auto& dst_mapper = mapper_dict.at(dst);
                auto& sampler = sampler_dict.at(k);
                size_t begin, end;
                std::tie(begin, end) = slice_dict.at(src);

                sampler.num_sampled_edges_per_hop.push_back(0);

                if (!time_dict.has_value() ||
                    !time_dict.value().contains(dst)) {
                  for (size_t i = begin; i < end; ++i) {
                    sampler.uniform_sample(
                        /*global_src_node=*/src_sampled_nodes[i],
                        /*local_src_node=*/i, count, dst_mapper, generator,
                        dst_sampled_nodes);
                  }
                } else if constexpr (!std::is_scalar<
                                         node_t>::value) {  // Temporal:
                  const at::Tensor& dst_time = time_dict.value().at(dst);
                  const auto dst_time_data = dst_time.data_ptr<temporal_t>();
                  for (size_t i = begin; i < end; ++i) {
                    const auto batch_idx = src_sampled_nodes[i].first;
                    sampler.temporal_sample(
                        /*global_src_node=*/src_sampled_nodes[i],
                        /*local_src_node=*/i, count, seed_times[batch_idx],
                        dst_time_data, dst_mapper, generator,
                        dst_sampled_nodes);
                  }
                }
              }
            }
          });

      if (parallel) {
        for (auto& dst_sampled : dst_sampled_nodes_dict) {
          std::copy(dst_sampled.second.begin(), dst_sampled.second.end(),
                    std::back_inserter(sampled_nodes_dict[dst_sampled.first]));
        }
      }
      for (const auto& k : node_types) {
        slice_dict[k] = {slice_dict.at(k).second,
                         sampled_nodes_dict.at(k).size()};
        num_sampled_nodes_per_hop_map.at(k).push_back(slice_dict.at(k).second -
                                                      slice_dict.at(k).first);
      }
    }

    for (const auto& k : node_types) {
      out_node_id_dict.insert(
          k, pyg::utils::from_vector(sampled_nodes_dict.at(k)));
      num_sampled_nodes_per_hop_dict.insert(
          k, num_sampled_nodes_per_hop_map.at(k));
    }

    TORCH_CHECK(directed, "Undirected heterogeneous graphs not yet supported");
    if (directed) {
      for (const auto& k : edge_types) {
        const auto edges = sampler_dict.at(k).get_sampled_edges(csc);
        out_row_dict.insert(to_rel_type(k), std::get<0>(edges));
        out_col_dict.insert(to_rel_type(k), std::get<1>(edges));
        num_sampled_edges_per_hop_dict.insert(
            to_rel_type(k), sampler_dict.at(k).num_sampled_edges_per_hop);
        if (return_edge_id) {
          out_edge_id_dict.value().insert(to_rel_type(k),
                                          std::get<2>(edges).value());
        }
      }
    }
  });

  return std::make_tuple(out_row_dict, out_col_dict, out_node_id_dict,
                         out_edge_id_dict, num_sampled_nodes_per_hop_dict,
                         num_sampled_edges_per_hop_dict);
}

// Dispatcher //////////////////////////////////////////////////////////////////

#define DISPATCH_SAMPLE(replace, directed, disjount, return_edge_id, ...) \
  if (replace && directed && disjoint && return_edge_id)                  \
    return sample<true, true, true, true>(__VA_ARGS__);                   \
  if (replace && directed && disjoint && !return_edge_id)                 \
    return sample<true, true, true, false>(__VA_ARGS__);                  \
  if (replace && directed && !disjoint && return_edge_id)                 \
    return sample<true, true, false, true>(__VA_ARGS__);                  \
  if (replace && directed && !disjoint && !return_edge_id)                \
    return sample<true, true, false, false>(__VA_ARGS__);                 \
  if (replace && !directed && disjoint && return_edge_id)                 \
    return sample<true, false, true, true>(__VA_ARGS__);                  \
  if (replace && !directed && disjoint && !return_edge_id)                \
    return sample<true, false, true, false>(__VA_ARGS__);                 \
  if (replace && !directed && !disjoint && return_edge_id)                \
    return sample<true, false, false, true>(__VA_ARGS__);                 \
  if (replace && !directed && !disjoint && !return_edge_id)               \
    return sample<true, false, false, false>(__VA_ARGS__);                \
  if (!replace && directed && disjoint && return_edge_id)                 \
    return sample<false, true, true, true>(__VA_ARGS__);                  \
  if (!replace && directed && disjoint && !return_edge_id)                \
    return sample<false, true, true, false>(__VA_ARGS__);                 \
  if (!replace && directed && !disjoint && return_edge_id)                \
    return sample<false, true, false, true>(__VA_ARGS__);                 \
  if (!replace && directed && !disjoint && !return_edge_id)               \
    return sample<false, true, false, false>(__VA_ARGS__);                \
  if (!replace && !directed && disjoint && return_edge_id)                \
    return sample<false, false, true, true>(__VA_ARGS__);                 \
  if (!replace && !directed && disjoint && !return_edge_id)               \
    return sample<false, false, true, false>(__VA_ARGS__);                \
  if (!replace && !directed && !disjoint && return_edge_id)               \
    return sample<false, false, false, true>(__VA_ARGS__);                \
  if (!replace && !directed && !disjoint && !return_edge_id)              \
    return sample<false, false, false, false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           c10::optional<at::Tensor>,
           std::vector<int64_t>,
           std::vector<int64_t>>
neighbor_sample_kernel(const at::Tensor& rowptr,
                       const at::Tensor& col,
                       const at::Tensor& seed,
                       const std::vector<int64_t>& num_neighbors,
                       const c10::optional<at::Tensor>& time,
                       const c10::optional<at::Tensor>& seed_time,
                       bool csc,
                       bool replace,
                       bool directed,
                       bool disjoint,
                       std::string temporal_strategy,
                       bool return_edge_id) {
  DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, rowptr, col,
                  seed, num_neighbors, time, seed_time, csc, temporal_strategy);
}

std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           c10::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
hetero_neighbor_sample_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<rel_type, at::Tensor>& rowptr_dict,
    const c10::Dict<rel_type, at::Tensor>& col_dict,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& time_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
    bool csc,
    bool replace,
    bool directed,
    bool disjoint,
    std::string temporal_strategy,
    bool return_edge_id) {
  DISPATCH_SAMPLE(replace, directed, disjoint, return_edge_id, node_types,
                  edge_types, rowptr_dict, col_dict, seed_dict,
                  num_neighbors_dict, time_dict, seed_time_dict, csc,
                  temporal_strategy);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::neighbor_sample"),
         TORCH_FN(neighbor_sample_kernel));
}

// We use `BackendSelect` as a fallback to the dispatcher logic as automatic
// dispatching of dictionaries is not yet supported by PyTorch.
// See: pytorch/aten/src/ATen/templates/RegisterBackendSelect.cpp.
TORCH_LIBRARY_IMPL(pyg, BackendSelect, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::hetero_neighbor_sample"),
         TORCH_FN(hetero_neighbor_sample_kernel));
}

}  // namespace sampler
}  // namespace pyg
