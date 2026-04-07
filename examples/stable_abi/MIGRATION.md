# pyg-lib PyTorch Stable ABI Migration Plan

## Motivation

Today, every pyg-lib release must be rebuilt against every PyTorch version because the C++ extension links against unstable ATen/c10 internals. PyTorch 2.10+ provides a **stable C++ ABI** so extensions compiled once remain binary compatible across PyTorch versions. Migrating pyg-lib to the stable ABI would let a single wheel work on multiple PyTorch versions, dramatically simplifying releases and CI.

This document captures (a) what the stable ABI provides, (b) which pyg-lib ops can adopt it today vs. need refactoring vs. are blocked, and (c) a phased migration plan.

## Proof of Concept

A working minimal example lives in this directory (`examples/stable_abi/`). It ports `pyg::cuda_version` to the stable ABI and validates binary compatibility. Tested working with the same `_C.abi3.so` against PyTorch 2.10.0+cpu, 2.11.0+cu129 (build version), and 2.12.0.dev (nightly).

Build and test:

```
cd examples/stable_abi && pip install -e . --no-build-isolation && python -c "import torch; torch.ops.load_library('./_C.abi3.so'); print(torch.ops.pyg.cuda_version())"
```

## What the Stable ABI Provides

### `torch::stable::Tensor`

A `shared_ptr<AtenTensorOpaque>` wrapper. Provides metadata accessors (`dim`, `numel`, `sizes`, `strides`, `size(i)`, `stride(i)`, `scalar_type`, `device`, `layout`, `is_contiguous`, `is_cuda`, `is_cpu`, `defined`, `storage_offset`, `element_size`, `get_device_index`), typed data pointers (`mutable_data_ptr<T>()`, `const_data_ptr<T>()`), and `set_requires_grad`. Notably **no member methods for tensor operations** — `contiguous()`, `view()`, `to()`, etc. are free functions in the `torch::stable::` namespace.

### `torch::stable::ops` — ~30 free functions

| Category                | Functions                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| **Creation**            | `empty`, `empty_like`, `new_empty`, `new_zeros`, `full`, `from_blob`, `clone`                |
| **Reshape/View**        | `view`, `reshape`, `flatten`, `unsqueeze`, `squeeze`, `transpose`, `select`, `narrow`, `pad` |
| **In-place**            | `fill_`, `zero_`, `copy_`                                                                    |
| **Compute**             | `matmul`, `sum`, `sum_out`, `subtract`, `amax`                                               |
| **Convert**             | `to`, `contiguous`                                                                           |
| **Parallelism** (2.10+) | `parallel_for`, `get_num_threads`                                                            |

### `torch::headeronly`

- `ScalarType`, `DeviceType`, `Layout`, `MemoryFormat` enums
- `STD_TORCH_CHECK(...)` (throws `std::runtime_error`, replaces `TORCH_CHECK`)
- `THO_DISPATCH_*` and `THO_DISPATCH_V2` macros — covers `AT_FLOATING_TYPES`, `AT_INTEGRAL_TYPES`, `AT_ALL_TYPES`, `AT_FLOAT8_TYPES`, etc.
- `HeaderOnlyArrayRef<T>` (slimmed-down `ArrayRef`)

### `torch::stable::accelerator`

`DeviceGuard` (RAII), `Stream`, `getCurrentStream(device_index)`. CUDA streams obtained via the C shim: `aoti_torch_get_current_cuda_stream(device_idx, &stream_ptr)`.

### Op registration

```cpp
STABLE_TORCH_LIBRARY_FRAGMENT(pyg, m) {
  m.def("my_op(Tensor a) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(pyg, CPU, m) {
  m.impl("my_op", TORCH_BOX(&my_op_cpu));
}
```

`TORCH_BOX(&fn)` wraps a normal C++ function into a boxed kernel that the stable dispatcher can call. Supported parameter types: `bool`, `int64_t`, `double`, `torch::stable::Tensor`, `ScalarType`, `Layout`, `Device`, `MemoryFormat`, `std::string`, `std::vector<T>`, `std::optional<T>`.

### Escape hatch

Any registered ATen op can be invoked by name via `torch_call_dispatcher("aten::op_name", "overload", stack, TORCH_ABI_VERSION)`, building the `StableIValue*` stack manually with `torch::stable::detail::from<T>`.

### What is NOT available

- `c10::SymInt` / `SymFloat` / `SymBool` — no symbolic shapes, no `torch.compile` dynamic shape support
- `c10::Dict` / `c10::List` of arbitrary types
- `torch::autograd::Function` — no custom C++ backward
- `torch::class_` / `CustomClassHolder` — no TorchScript custom classes
- `at::TensorOptions`, `at::TensorIterator`, `at::Scalar`, `at::Generator`
- `at::TensorArg` / `at::checkAll*`
- `c10::Dispatcher::singleton().findSchemaOrThrow()` typed redispatch
- `at::cuda::*` C++ headers (only the C shim is available)
- pybind11 / direct CPython object access

## pyg-lib Op Feasibility

### GREEN — Migrate immediately

These ops use only CPU code, plain types, no custom autograd, and dispatch macros covered by `THO_DISPATCH_*`.

| Op                           | Schema                                                                                                                | Notes                                    |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| `pyg::cuda_version`          | `() -> int`                                                                                                           | No tensors. POC done.                    |
| `pyg::index_sort`            | `(Tensor, int? max) -> (Tensor, Tensor)`                                                                              | CPU-only, integral dispatch              |
| `pyg::edge_sample`           | `(Tensor start, Tensor rowptr, int count, float factor) -> Tensor`                                                    | CPU-only, no dispatch macros             |
| `pyg::subgraph`              | `(Tensor rowptr, Tensor col, Tensor nodes, bool return_edge_id) -> (Tensor, Tensor, Tensor?)`                         | CPU-only                                 |
| `pyg::merge_sampler_outputs` | nested tensor/int lists                                                                                               | CPU-only                                 |
| `pyg::relabel_neighborhood`  | `(Tensor, Tensor, int[], int, Tensor?, bool, bool) -> (Tensor, Tensor)`                                               | CPU-only                                 |
| `pyg::metis`                 | `(Tensor rowptr, Tensor col, int num_partitions, Tensor? node_weight, Tensor? edge_weight, bool recursive) -> Tensor` | CPU-only, calls METIS C library directly |

### YELLOW — Migrate with refactoring

Three refactor categories cover most yellow ops:

**A) Move autograd backward to Python**

Five forward ops have `torch::autograd::Function` C++ subclasses in `pyg_lib/csrc/ops/autograd/*.cpp`:

- `grouped_matmul`, `segment_matmul` (`autograd/matmul_kernel.cpp`)
- `softmax_csr` (`autograd/softmax_kernel.cpp`)
- `sampled_op` (`autograd/sampled_kernel.cpp`)
- `spline_basis`, `spline_weighting` (`autograd/spline_kernel.cpp`)

The forward kernels stay in C++; only the registration to the `Autograd` dispatch key moves to Python `torch.autograd.Function` wrappers.

**B) Replace `at::cuda::getCurrentCUDAStream()` with the C shim**

Mechanical replacement:

```cpp
// Before
auto stream = at::cuda::getCurrentCUDAStream();

// After
void* stream_ptr = nullptr;
TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
auto stream = static_cast<cudaStream_t>(stream_ptr);
```

Affects: `matmul`, `sampled_op`, `fps`, `knn`, `nearest`, `radius`, `grid_cluster`, `graclus_cluster`, `random_walk`.

**C) Replace dispatch wrapper boilerplate**

Affects nearly every op:

- `at::TensorArg` / `at::checkAll*` → `STD_TORCH_CHECK`
- `c10::Dispatcher::singleton().findSchemaOrThrow(...).typed<...>().call(...)` → direct call to the kernel function (with `STABLE_TORCH_LIBRARY_IMPL` handling backend dispatch)

**Special yellow cases:**

| Op                                  | Extra blocker                                                                   | Workaround                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `grouped_matmul` / `segment_matmul` | `at::globalContext().float32MatmulPrecision()` not in stable ABI                | Hardcode TF32 policy or accept as Python-side argument            |
| `grouped_matmul` / `segment_matmul` | `at::IntArrayRef` constructed from `data_ptr<int64_t>()` for `split_with_sizes` | Pass `std::vector<int64_t>` from Python or split tensors manually |
| `graclus_cluster` CUDA              | Uses `__device__` global symbols + `cudaMemcpyFromSymbol`                       | Refactor to remove global symbols                                 |
| `sampled_op` autograd               | Uses internal `at::index_select_backward`                                       | Reimplement as `zeros + scatter_add_` from Python                 |

### RED — Hard-blocked

| Op / Class                           | Blocker                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `pyg::hetero_neighbor_sample`        | Schema uses `Dict(str, Tensor)`, `Dict(str, int[])` — `c10::Dict` not in stable ABI |
| `pyg::hetero_relabel_neighborhood`   | Same `c10::Dict` blocker                                                            |
| `pyg::CPUHashMap` (class)            | `torch::class_` / `CustomClassHolder` / `def_pickle` — no stable equivalent         |
| `pyg::CUDAHashMap` (class)           | `torch::class_` blocker + `cuco::static_map`                                        |
| `pyg::NeighborSampler` (class)       | `torch::class_` blocker                                                             |
| `pyg::HeteroNeighborSampler` (class) | `torch::class_` blocker + `c10::Dict`                                               |

These would have to either stay on the unstable ABI indefinitely, or be reimplemented as Python wrappers around stable-ABI tensor ops (changing the public schema, breaking TorchScript compatibility).

## Top Blockers Summary

| Blocker                                        | Affected ops                        | Severity                       |
| ---------------------------------------------- | ----------------------------------- | ------------------------------ |
| `torch::class_` (TorchScript classes)          | 4 classes                           | Hard block — no workaround     |
| `c10::Dict` in op schema                       | 2 hetero ops + 1 class              | Hard block — no workaround     |
| `torch::autograd::Function`                    | 5 forward ops with autograd kernels | Refactor — move to Python      |
| `at::cuda::*` headers                          | 9 CUDA kernels                      | Refactor — replace with C shim |
| `at::TensorArg` / `checkAll*`                  | ~20 dispatch wrappers               | Mechanical refactor            |
| `c10::Dispatcher::findSchemaOrThrow`           | ~20 dispatch wrappers               | Mechanical refactor            |
| `at::globalContext().float32MatmulPrecision()` | matmul ops CUDA                     | Hardcode or pass from Python   |
| `at::IntArrayRef` from data_ptr                | matmul ops                          | Pass `std::vector` from Python |

## Architectural Approach

Follow vLLM's pattern: **build a second extension `libpyg_stable.so` alongside `libpyg.so`**.

- Both extensions register ops under the `pyg` namespace
- Python loads both at import time; callers see no difference
- Migrated ops live in `libpyg_stable.so`; not-yet-migrated and red-blocked ops stay in `libpyg.so`
- Build flags isolated: `libpyg_stable.so` uses `-DTORCH_TARGET_VERSION=0x020a000000000000` and `-DPy_LIMITED_API=0x03090000`; `libpyg.so` keeps current flags
- Each migrated op is a small, reviewable PR
- Existing build never breaks; can roll back individual ops trivially

vLLM's prep work that we will likely also need:

- Move all C++ meta/fake functions to Python (`@torch.library.register_fake`) — stable ABI has no SymInt
- Move all autograd backwards to Python (`torch.library.register_autograd`)

## Recommended Migration Phases

| Phase                    | Scope                                                                                             | Validates                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **0**                    | `cuda_version` (POC, done)                                                                        | Build infrastructure                                              |
| **1**                    | `index_sort`, `edge_sample`, `subgraph`, `merge_sampler_outputs`, `relabel_neighborhood`, `metis` | CPU-only easy wins; tensor marshalling, list types                |
| **2**                    | `neighbor_sample`, `dist_neighbor_sample`                                                         | Larger CPU-only ops; complex schemas with optionals               |
| **3**                    | `random_walk`, `nearest`, `radius`, `grid_cluster`, `knn`, `fps`                                  | CPU+CUDA stream shim pattern                                      |
| **4**                    | `softmax_csr`, `spline_basis`, `spline_weighting`                                                 | Autograd-to-Python pattern                                        |
| **5**                    | `grouped_matmul`, `segment_matmul`, `sampled_op`, `graclus_cluster`                               | Hardest cases — CUTLASS, matmul precision, `__device__` symbols   |
| **Stays in `libpyg.so`** | `hetero_*` ops, all `torch::class_` classes                                                       | Until PyTorch ships `c10::Dict` and `torch::class_` in stable ABI |

The single biggest leverage point is **Phase 4** (move C++ autograd to Python) — once that pattern is established, ~6 ops move from yellow toward green.

## Build/Compile Reference

Stable ABI compile flags (per vLLM and the official PyTorch reference):

```
-DPy_LIMITED_API=0x03090000
-DTORCH_TARGET_VERSION=0x020a000000000000   # PyTorch 2.10 minimum
-DUSE_CUDA                                  # nvcc only, for CUDA stream shim
```

CMake equivalent (vLLM-style):

```cmake
Python_add_library(${MOD_NAME} MODULE USE_SABI 3 WITH_SOABI ${SOURCES})
target_compile_definitions(${MOD_NAME} PRIVATE
    TORCH_TARGET_VERSION=0x020A000000000000ULL)
```

`TORCH_TARGET_VERSION` format: `[MAJ 1 byte][MIN 1 byte][PATCH 1 byte][ABI TAG 5 bytes]`. PyTorch 2.10 = `0x020a000000000000`.

## References

- Official docs: https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html
- Stable C++ API reference: https://docs.pytorch.org/cppdocs/stable.html
- Reference implementation: https://github.com/pytorch/extension-cpp (`extension_cpp_stable/`)
- vLLM tracking issue: https://github.com/vllm-project/vllm/issues/26946
- vLLM stable ABI source tree: https://github.com/vllm-project/vllm/tree/main/csrc/libtorch_stable
