#include <ATen/ATen.h>
#include <torch/library.h>

#include <metis.h>

namespace pyg {
namespace partition {

namespace {

at::Tensor metis_kernel(const at::Tensor& rowptr,
                        const at::Tensor& col,
                        int64_t num_partitions,
                        const c10::optional<at::Tensor>& node_weight,
                        const c10::optional<at::Tensor>& edge_weight,
                        bool recursive) {
  int64_t nvtxs = rowptr.numel() - 1;
  int64_t ncon = 1;
  auto* xadj = rowptr.data_ptr<int64_t>();
  auto* adjncy = col.data_ptr<int64_t>();

  int64_t* vwgt = NULL;
  if (node_weight.has_value())
    vwgt = node_weight.value().data_ptr<int64_t>();

  int64_t* adjwgt = NULL;
  if (edge_weight.has_value())
    adjwgt = edge_weight.value().data_ptr<int64_t>();

  int64_t objval = -1;
  auto part = at::empty({nvtxs}, rowptr.options());
  auto part_data = part.data_ptr<int64_t>();

  if (recursive) {
    METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                             &num_partitions, NULL, NULL, NULL, &objval,
                             part_data);
  } else {
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                        &num_partitions, NULL, NULL, NULL, &objval, part_data);
  }

  return part;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::metis"), TORCH_FN(metis_kernel));
}

}  // namespace partition
}  // namespace pyg
