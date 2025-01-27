#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <parallel_hashmap/phmap.h>
#include <torch/library.h>

#include "pyg_lib/csrc/sampler/cpu/mapper.h"
#include "pyg_lib/csrc/utils/cpu/convert.h"
#include "pyg_lib/csrc/utils/types.h"

namespace pyg {
namespace sampler {

namespace {

template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor> get_sampled_edges(
    std::vector<scalar_t> sampled_rows,
    std::vector<scalar_t> sampled_cols,
    const bool csc = false) {
  const auto row = pyg::utils::from_vector(sampled_rows);
  const auto col = pyg::utils::from_vector(sampled_cols);

  if (!csc) {
    return std::make_tuple(row, col);
  } else {
    return std::make_tuple(col, row);
  }
}

template <bool disjoint>
std::tuple<at::Tensor, at::Tensor> relabel(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_duplicates,
    const std::vector<int64_t>& num_sampled_neighbors_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    const bool csc) {
  if (disjoint) {
    TORCH_CHECK(batch.has_value(),
                "Batch needs to be specified to create disjoint subgraphs");
    TORCH_CHECK(batch.value().is_contiguous(), "Non-contiguous 'batch'");
    TORCH_CHECK(batch.value().numel() == sampled_nodes_with_duplicates.numel(),
                "Each node must belong to a subgraph");
  }
  TORCH_CHECK(seed.is_contiguous(), "Non-contiguous 'seed'");
  TORCH_CHECK(sampled_nodes_with_duplicates.is_contiguous(),
              "Non-contiguous 'sampled_nodes_with_duplicates'");

  at::Tensor out_row, out_col;

  AT_DISPATCH_INTEGRAL_TYPES(seed.scalar_type(), "relabel_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

    const auto sampled_nodes_data =
        sampled_nodes_with_duplicates.data_ptr<scalar_t>();
    const auto batch_data =
        !disjoint ? nullptr : batch.value().data_ptr<scalar_t>();

    std::vector<scalar_t> sampled_rows;
    std::vector<scalar_t> sampled_cols;
    auto mapper = Mapper<node_t, scalar_t>(num_nodes);

    const auto seed_data = seed.data_ptr<scalar_t>();
    if constexpr (!disjoint) {
      mapper.fill(seed);
    } else {
      for (size_t i = 0; i < seed.numel(); ++i) {
        mapper.insert({i, seed_data[i]});
      }
    }
    size_t begin = 0, end = 0;
    for (auto i = 0; i < num_sampled_neighbors_per_node.size(); i++) {
      end += num_sampled_neighbors_per_node[i];

      for (auto j = begin; j < end; j++) {
        std::pair<scalar_t, bool> res;
        if constexpr (!disjoint)
          res = mapper.insert(sampled_nodes_data[j]);
        else
          res = mapper.insert({batch_data[j], sampled_nodes_data[j]});
        sampled_rows.push_back(i);
        sampled_cols.push_back(res.first);
      }

      begin = end;
    }

    std::tie(out_row, out_col) =
        get_sampled_edges<scalar_t>(sampled_rows, sampled_cols, csc);
  });

  return std::make_tuple(out_row, out_col);
}

template <bool disjoint>
std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
relabel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_duplicates_dict,
    const c10::Dict<rel_type, std::vector<std::vector<int64_t>>>&
        num_sampled_neighbors_per_node_dict,
    const c10::Dict<node_type, int64_t>& num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    const bool csc) {
  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;

  auto scalar_type = seed_dict.begin()->value().scalar_type();
  AT_DISPATCH_INTEGRAL_TYPES(scalar_type, "hetero_relabel_kernel", [&] {
    typedef std::pair<scalar_t, scalar_t> pair_scalar_t;
    typedef std::conditional_t<!disjoint, scalar_t, pair_scalar_t> node_t;

    phmap::flat_hash_map<node_type, scalar_t*> sampled_nodes_data_dict;
    phmap::flat_hash_map<node_type, scalar_t*> batch_data_dict;
    phmap::flat_hash_map<edge_type, std::vector<scalar_t>> sampled_rows_dict;
    phmap::flat_hash_map<edge_type, std::vector<scalar_t>> sampled_cols_dict;
    // `srcs_slice_dict` defines the number of src nodes for each edge type in
    // a given layer in the form of a range. Local src nodes (`sampled_rows`)
    // will be created on its basis, so for a given edge type the ranges will
    // not be repeated, and the starting value of the next layer will be the
    // end value from the previous layer.
    phmap::flat_hash_map<edge_type, std::pair<size_t, size_t>> srcs_slice_dict;

    phmap::flat_hash_map<node_type, Mapper<node_t, scalar_t>> mapper_dict;
    phmap::flat_hash_map<node_type, std::pair<size_t, size_t>> slice_dict;
    phmap::flat_hash_map<node_type, int64_t> srcs_offset_dict;

    const bool parallel = at::get_num_threads() > 1 && edge_types.size() > 1;
    std::vector<std::vector<edge_type>> threads_edge_types;

    for (const auto& k : edge_types) {
      // Initialize empty vectors.
      sampled_rows_dict[k];
      sampled_cols_dict[k];

      // `num_sampled_neighbors_per_node_dict` is a dictionary where for
      // each edge type it contains information about how many neighbors every
      // src node has sampled. These values are saved in a separate vector for
      // each layer.
      size_t num_src_nodes =
          num_sampled_neighbors_per_node_dict.at(to_rel_type(k))[0].size();
      srcs_slice_dict[k] = {0, num_src_nodes};

      if (parallel) {
        // Each thread is assigned edge types that have the same dst node
        // type. Thanks to this, each thread will operate on a separate
        // mapper and separate sampler.
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
    if (!parallel) {
      // If not parallel then one thread handles all edge types.
      threads_edge_types.push_back({edge_types});
    }

    for (const auto& k : node_types) {
      sampled_nodes_data_dict.insert(
          {k, sampled_nodes_with_duplicates_dict.at(k).data_ptr<scalar_t>()});
      int64_t N = num_nodes_dict.at(k);
      mapper_dict.insert({k, Mapper<node_t, scalar_t>(N)});
      slice_dict[k] = {0, 0};
      srcs_offset_dict[k] = 0;
      if constexpr (disjoint) {
        batch_data_dict.insert(
            {k, batch_dict.value().at(k).data_ptr<scalar_t>()});
      }
    }
    scalar_t batch_idx = 0;
    for (const auto& kv : seed_dict) {
      const at::Tensor& seed = kv.value();
      if constexpr (!disjoint) {
        mapper_dict.at(kv.key()).fill(seed);
      } else {
        auto& mapper = mapper_dict.at(kv.key());
        const auto seed_data = seed.data_ptr<scalar_t>();
        for (size_t i = 0; i < seed.numel(); ++i) {
          mapper.insert({batch_idx, seed_data[i]});
          batch_idx++;
        }
      }
    }

    size_t num_layers =
        num_sampled_neighbors_per_node_dict.at(to_rel_type(edge_types[0]))
            .size();
    // Iterate over the layers
    for (auto ell = 0; ell < num_layers; ++ell) {
      at::parallel_for(
          0, threads_edge_types.size(), 1, [&](size_t _s, size_t _e) {
            for (auto t = _s; t < _e; t++) {
              for (const auto& k : threads_edge_types[t]) {
                const auto dst = !csc ? std::get<2>(k) : std::get<0>(k);

                auto [src_begin, src_end] = srcs_slice_dict.at(k);

                for (auto i = src_begin; i < src_end; i++) {
                  auto& dst_mapper = mapper_dict.at(dst);
                  auto& dst_sampled_nodes_data =
                      sampled_nodes_data_dict.at(dst);

                  // For each edge type `slice_dict` defines the number of
                  // nodes sampled by a src node `i` in the form of a range.
                  // The indices in the given range point to global dst nodes
                  // from `dst_sampled_nodes_data`.
                  slice_dict.at(dst).second +=
                      num_sampled_neighbors_per_node_dict.at(
                          to_rel_type(k))[ell][i - src_begin];
                  auto [begin, end] = slice_dict.at(dst);

                  for (auto j = begin; j < end; j++) {
                    std::pair<scalar_t, bool> res;
                    if constexpr (!disjoint) {
                      res = dst_mapper.insert(dst_sampled_nodes_data[j]);
                    } else {
                      res = dst_mapper.insert({batch_data_dict.at(dst)[j],
                                               dst_sampled_nodes_data[j]});
                    }
                    sampled_rows_dict.at(k).push_back(i);
                    sampled_cols_dict.at(k).push_back(res.first);
                  }
                  slice_dict.at(dst).first = end;
                }
              }
            }
          });

      // Get local src nodes ranges for the next layer
      if (ell < num_layers - 1) {
        for (const auto& k : edge_types) {
          // Edges with the same src node types will have the same src node
          // offsets.
          const auto src = !csc ? std::get<0>(k) : std::get<2>(k);
          if (srcs_offset_dict[src] < srcs_slice_dict.at(k).second) {
            srcs_offset_dict[src] = srcs_slice_dict.at(k).second;
          }
        }
        for (const auto& k : edge_types) {
          const auto src = !csc ? std::get<0>(k) : std::get<2>(k);
          srcs_slice_dict[k] = {
              srcs_offset_dict.at(src),
              srcs_offset_dict.at(src) + num_sampled_neighbors_per_node_dict
                                             .at(to_rel_type(k))[ell + 1]
                                             .size()};
        }
      }
    }

    for (const auto& k : edge_types) {
      const auto edges = get_sampled_edges<scalar_t>(
          sampled_rows_dict.at(k), sampled_cols_dict.at(k), csc);
      out_row_dict.insert(to_rel_type(k), std::get<0>(edges));
      out_col_dict.insert(to_rel_type(k), std::get<1>(edges));
    }
  });

  return std::make_tuple(out_row_dict, out_col_dict);
}

#define DISPATCH_RELABEL(disjoint, ...) \
  if (disjoint)                         \
    return relabel<true>(__VA_ARGS__);  \
  if (!disjoint)                        \
    return relabel<false>(__VA_ARGS__);

}  // namespace

std::tuple<at::Tensor, at::Tensor> relabel_neighborhood_kernel(
    const at::Tensor& seed,
    const at::Tensor& sampled_nodes_with_duplicates,
    const std::vector<int64_t>& num_sampled_neighbors_per_node,
    const int64_t num_nodes,
    const c10::optional<at::Tensor>& batch,
    bool csc,
    bool disjoint) {
  DISPATCH_RELABEL(disjoint, seed, sampled_nodes_with_duplicates,
                   num_sampled_neighbors_per_node, num_nodes, batch, csc);
}

std::tuple<c10::Dict<rel_type, at::Tensor>, c10::Dict<rel_type, at::Tensor>>
hetero_relabel_neighborhood_kernel(
    const std::vector<node_type>& node_types,
    const std::vector<edge_type>& edge_types,
    const c10::Dict<node_type, at::Tensor>& seed_dict,
    const c10::Dict<node_type, at::Tensor>& sampled_nodes_with_duplicates_dict,
    const c10::Dict<rel_type, std::vector<std::vector<int64_t>>>&
        num_sampled_neighbors_per_node_dict,
    const c10::Dict<node_type, int64_t>& num_nodes_dict,
    const c10::optional<c10::Dict<node_type, at::Tensor>>& batch_dict,
    bool csc,
    bool disjoint) {
  c10::Dict<rel_type, at::Tensor> out_row_dict, out_col_dict;
  DISPATCH_RELABEL(disjoint, node_types, edge_types, seed_dict,
                   sampled_nodes_with_duplicates_dict,
                   num_sampled_neighbors_per_node_dict, num_nodes_dict,
                   batch_dict, csc);
}

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::relabel_neighborhood"),
         TORCH_FN(relabel_neighborhood_kernel));
}

TORCH_LIBRARY_IMPL(pyg, BackendSelect, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::hetero_relabel_neighborhood"),
         TORCH_FN(hetero_relabel_neighborhood_kernel));
}

}  // namespace sampler
}  // namespace pyg
