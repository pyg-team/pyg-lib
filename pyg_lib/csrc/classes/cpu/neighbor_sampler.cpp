#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

namespace pyg {
namespace classes {

namespace {

#define DISPATCH_CASE_KEY(...)                         \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_KEY(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_KEY(__VA_ARGS__))

template <typename node_t,
	  typename scalar_t,
	  typename temporal_t,
    typename tensor_t,
	  bool replace,
	  bool save_edges,
	  bool save_edge_ids,
	  bool distributed>
struct NeighborSampleImplBase {
  virtual ~NeighborSampleBase(const tensor_t rowptr,
                              const tensor_t col,
                              const tensor_t seed,
                              const std:optional<tensor_t>& node_time) = 0;
};

struct NeighborSampleImpl :: NeighborSampleImplBase {
  NeighborSample(const at::Tensor& rowptr,
                 const at::Tensor& col,
                 const at::Tensor& seed,
                 const std:optional<at::Tensor>& node_time):
          rowptr(rowptr), col(col), seed(seed), node_time(node_time);

  std::tuple<at::Tensor,
             at::Tensor,
             at::Tensor,
             std::optional<at::Tensor>,
             std::vector<int64_t>,
             std::vector<int64_t>>
  sample(const std::vector<int64_t>& num_neighbors,
              const std::optional<at::Tensor>& node_time,
              const std::optional<at::Tensor>& edge_time,
              const std::optional<at::Tensor>& seed_time,
              const std::optional<at::Tensor>& edge_weight,
              bool csc,
              std::string temporal_strategy){
    //Bring over code from Homogeneous neighbor sampling in neighbor_kernel.cpp
    return std::make_tuple(/*stuff*/);
  }
};

struct HeteroNeighborSampleImpl :: NeighborSampleImplBase {
  HeteroNeighborSampleImpl(const c10::Dict<rel_type, at::Tensor>& rowptr,
                 const c10::Dict<rel_type, at::Tensor>& col,
                 const c10::Dict<rel_type, at::Tensor>& seed,
                 const std:optional<c10::Dict<rel_type, at::Tensor>>& node_time):
          rowptr(rowptr), col(col), seed(seed), node_time(node_time);

  std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           std::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
   sample(const std::vector<node_type>& node_types,
       const std::vector<edge_type>& edge_types,
       const c10::Dict<rel_type, std::vector<int64_t>>& num_neighbors_dict,
       const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_time_dict,
       const std::optional<c10::Dict<node_type, at::Tensor>>& seed_time_dict,
       const std::optional<c10::Dict<rel_type, at::Tensor>>& edge_weight_dict,
       const bool csc,
       const std::string temporal_strategy){
    //Bring over code from Heterogeneous neighbor sampling in neighbor_kernel.cpp
    return std::make_tuple(/*stuff*/)
   }
};


struct HeteroNeighborSample : torch::CustomClassHolder {
 public:
  NeighborSample(/*params*/){
    // Do the dispatch where you instantiate the correct neighbor sampler
  }
  //You should probably use templated types here
  std::tuple<c10::Dict<rel_type, at::Tensor>,
           c10::Dict<rel_type, at::Tensor>,
           c10::Dict<node_type, at::Tensor>,
           std::optional<c10::Dict<rel_type, at::Tensor>>,
           c10::Dict<node_type, std::vector<int64_t>>,
           c10::Dict<rel_type, std::vector<int64_t>>>
  sample(/*params*/){return /*stuff*/;}
};

}  // namespace
TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.class_<NeighborSample>("NeighborSample")
      .def(torch::init<at::Tensor&, int64_t>()) // fix args
      .def("sample", &NeighborSample::sample);
}


}  // namespace classes
}  // namespace pyg
