# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [0.5.0] - 2023-MM-DD
### Added
- Added PyTorch 2.7 support ([#442](https://github.com/pyg-team/pyg-lib/pull/442))
- Added CUDA >= 12.6 support ([#431](https://github.com/pyg-team/pyg-lib/pull/431))
- Added PyTorch 2.5 support ([#360](https://github.com/pyg-team/pyg-lib/pull/338))
- Added PyTorch 2.4 support ([#338](https://github.com/pyg-team/pyg-lib/pull/338))
- Added PyTorch 2.3 support ([#322](https://github.com/pyg-team/pyg-lib/pull/322))
- Added Windows support ([#315](https://github.com/pyg-team/pyg-lib/pull/315))
- Added macOS Apple Silicon support ([#310](https://github.com/pyg-team/pyg-lib/pull/310))
### Changed
### Removed
- Dropped Python 3.8 support ([#356](https://github.com/pyg-team/pyg-lib/pull/356))
- Removed linking to Python ([#462](https://github.com/pyg-team/pyg-lib/pull/462))

## [0.4.0] - 2024-02-07
### Added
- Added PyTorch 2.2 support ([#294](https://github.com/pyg-team/pyg-lib/pull/294))
- Added `softmax_csr` implementation ([#264](https://github.com/pyg-team/pyg-lib/pull/264), [#282](https://github.com/pyg-team/pyg-lib/pull/282))
- Added support for edge-level sampling ([#280](https://github.com/pyg-team/pyg-lib/pull/280))
- Added support for `bfloat16` data type in `segment_matmul` and `grouped_matmul` (CPU only) ([#272](https://github.com/pyg-team/pyg-lib/pull/272))
### Changed
- Dropped the MKL code path when sampling neighbors with `replace=False` since it does not correctly prevent duplicates ([#275](https://github.com/pyg-team/pyg-lib/pull/275))
- Added `--biased` parameter to run benchmarks for biased sampling ([#267](https://github.com/pyg-team/pyg-lib/pull/267))
- Improved speed of biased sampling ([#270](https://github.com/pyg-team/pyg-lib/pull/270))
- Fixed `grouped_matmul` when tensors are not contiguous ([#290](https://github.com/pyg-team/pyg-lib/pull/290))
### Removed

## [0.3.0] - 2023-10-11
### Added
- Added PyTorch 2.1 support ([#256](https://github.com/pyg-team/pyg-lib/pull/256))
- Added low-level support for distributed neighborhood sampling ([#246](https://github.com/pyg-team/pyg-lib/pull/246), [#252](https://github.com/pyg-team/pyg-lib/pull/252), [#253](https://github.com/pyg-team/pyg-lib/pull/253), [#254](https://github.com/pyg-team/pyg-lib/pull/254))
- Added support for homogeneous and heterogeneous biased neighborhood sampling ([#247](https://github.com/pyg-team/pyg-lib/pull/247), [#251](https://github.com/pyg-team/pyg-lib/pull/251))
- Added dispatch for XPU device in `index_sort` ([#243](https://github.com/pyg-team/pyg-lib/pull/243))
- Added `metis` partitioning ([#229](https://github.com/pyg-team/pyg-lib/pull/229))
- Enable `hetero_neighbor_samplee` to work in parallel ([#211](https://github.com/pyg-team/pyg-lib/pull/211))
### Changed
- Fixed vector-based mapping issue in `Mapping` ([#244](https://github.com/pyg-team/pyg-lib/pull/244))
- Fixed performance issues reported by Coverity Tool ([#240](https://github.com/pyg-team/pyg-lib/pull/240))
- Updated `cutlass` version for speed boosts in `segment_matmul` and `grouped_matmul` ([#235](https://github.com/pyg-team/pyg-lib/pull/235))
- Drop nested tensor wrapper for `grouped_matmul` implementation ([#226](https://github.com/pyg-team/pyg-lib/pull/226))
- Added `generate_range_of_ints` function (it uses MKL library in order to generate ints) to RandintEngine class ([#222](https://github.com/pyg-team/pyg-lib/pull/222))
- Fixed TorchScript support in `grouped_matmul` ([#220](https://github.com/pyg-team/pyg-lib/pull/220))
### Removed

## [0.2.0] - 2023-03-22
### Added
- Added PyTorch 2.0 support ([#214](https://github.com/pyg-team/pyg-lib/pull/214))
- `neighbor_sample` routines now also return information about the number of sampled nodes/edges per layer ([#197](https://github.com/pyg-team/pyg-lib/pull/197))
- Added `index_sort` implementation ([#181](https://github.com/pyg-team/pyg-lib/pull/181), [#192](https://github.com/pyg-team/pyg-lib/pull/192))
- Added `triton>=2.0` support ([#171](https://github.com/pyg-team/pyg-lib/pull/171))
- Added `bias` term to `grouped_matmul` and `segment_matmul` ([#161](https://github.com/pyg-team/pyg-lib/pull/161))
- Added `sampled_op` implementation ([#156](https://github.com/pyg-team/pyg-lib/pull/156), [#159](https://github.com/pyg-team/pyg-lib/pull/159), [#160](https://github.com/pyg-team/pyg-lib/pull/160))
### Changed
- Improved `[segment|grouped]_matmul` GPU implementation by reducing launch overheads ([#213](https://github.com/pyg-team/pyg-lib/pull/213))
- Sample the nodes with the same timestamp as seed nodes ([#187](https://github.com/pyg-team/pyg-lib/pull/187))
- Added `write-csv` (saves benchmark results as csv file) and `libraries` (determines which libraries will be used in benchmark) parameters ([#167](https://github.com/pyg-team/pyg-lib/pull/167))
- Enable benchmarking of neighbor sampler on temporal graphs ([#165](https://github.com/pyg-team/pyg-lib/pull/165))
- Improved `[segment|grouped]_matmul` CPU implementation via `at::matmul_out` and MKL BLAS `gemm_batch` ([#146](https://github.com/pyg-team/pyg-lib/pull/146), [#172](https://github.com/pyg-team/pyg-lib/pull/172))
### Removed

## [0.1.0] - 2022-11-28
### Added
- Added PyTorch 1.13 support ([#145](https://github.com/pyg-team/pyg-lib/pull/145))
- Added native PyTorch support for `grouped_matmul` ([#137](https://github.com/pyg-team/pyg-lib/pull/137))
- Added `fused_scatter_reduce` operation for multiple reductions ([#141](https://github.com/pyg-team/pyg-lib/pull/141), [#142](https://github.com/pyg-team/pyg-lib/pull/142))
- Added `triton` dependency ([#133](https://github.com/pyg-team/pyg-lib/pull/133), [#134](https://github.com/pyg-team/pyg-lib/pull/134))
- Enable `pytest` testing ([#132](https://github.com/pyg-team/pyg-lib/pull/132))
- Added C++-based autograd and TorchScript support for `segment_matmul` ([#120](https://github.com/pyg-team/pyg-lib/pull/120), [#122](https://github.com/pyg-team/pyg-lib/pull/122))
- Allow overriding `time` for seed nodes via `seed_time` in `neighbor_sample` ([#118](https://github.com/pyg-team/pyg-lib/pull/118))
- Added `[segment|grouped]_matmul` CPU implementation ([#111](https://github.com/pyg-team/pyg-lib/pull/111))
- Added `temporal_strategy` option to `neighbor_sample` ([#114](https://github.com/pyg-team/pyg-lib/pull/114))
- Added benchmarking tool (Google Benchmark) along with `pyg::sampler::Mapper` benchmark example ([#101](https://github.com/pyg-team/pyg-lib/pull/101))
- Added CSC mode to `pyg::sampler::neighbor_sample` and `pyg::sampler::hetero_neighbor_sample` ([#95](https://github.com/pyg-team/pyg-lib/pull/95), [#96](https://github.com/pyg-team/pyg-lib/pull/96))
- Speed up `pyg::sampler::neighbor_sample` via `IndexTracker` implementation ([#84](https://github.com/pyg-team/pyg-lib/pull/84))
- Added `pyg::sampler::hetero_neighbor_sample` implementation ([#90](https://github.com/pyg-team/pyg-lib/pull/90), [#92](https://github.com/pyg-team/pyg-lib/pull/92), [#94](https://github.com/pyg-team/pyg-lib/pull/94), [#97](https://github.com/pyg-team/pyg-lib/pull/97), [#98](https://github.com/pyg-team/pyg-lib/pull/98), [#99](https://github.com/pyg-team/pyg-lib/pull/99), [#102](https://github.com/pyg-team/pyg-lib/pull/102), [#110](https://github.com/pyg-team/pyg-lib/pull/110))
- Added `pyg::utils::to_vector` implementation ([#88](https://github.com/pyg-team/pyg-lib/pull/88))
- Added support for PyTorch 1.12 ([#57](https://github.com/pyg-team/pyg-lib/pull/57), [#58](https://github.com/pyg-team/pyg-lib/pull/58))
- Added `grouped_matmul` and `segment_matmul` CUDA implementations via `cutlass` ([#51](https://github.com/pyg-team/pyg-lib/pull/51), [#56](https://github.com/pyg-team/pyg-lib/pull/56), [#61](https://github.com/pyg-team/pyg-lib/pull/61), [#64](https://github.com/pyg-team/pyg-lib/pull/64), [#69](https://github.com/pyg-team/pyg-lib/pull/69), [#73](https://github.com/pyg-team/pyg-lib/pull/73), [#123](https://github.com/pyg-team/pyg-lib/pull/123))
- Added `pyg::sampler::neighbor_sample` implementation ([#54](https://github.com/pyg-team/pyg-lib/pull/54), [#76](https://github.com/pyg-team/pyg-lib/pull/76), [#77](https://github.com/pyg-team/pyg-lib/pull/77), [#78](https://github.com/pyg-team/pyg-lib/pull/78), [#80](https://github.com/pyg-team/pyg-lib/pull/80), [#81](https://github.com/pyg-team/pyg-lib/pull/81)), [#85](https://github.com/pyg-team/pyg-lib/pull/85), [#86](https://github.com/pyg-team/pyg-lib/pull/86), [#87](https://github.com/pyg-team/pyg-lib/pull/87), [#89](https://github.com/pyg-team/pyg-lib/pull/89))
- Added `pyg::sampler::Mapper` utility for mapping global to local node indices ([#45](https://github.com/pyg-team/pyg-lib/pull/45), [#83](https://github.com/pyg-team/pyg-lib/pull/83))
- Added benchmark script ([#45](https://github.com/pyg-team/pyg-lib/pull/45), [#79](https://github.com/pyg-team/pyg-lib/pull/79), [#82](https://github.com/pyg-team/pyg-lib/pull/82), [#91](https://github.com/pyg-team/pyg-lib/pull/91), [#93](https://github.com/pyg-team/pyg-lib/pull/93), [#106](https://github.com/pyg-team/pyg-lib/pull/106))
- Added download script for benchmark data ([#44](https://github.com/pyg-team/pyg-lib/pull/44))
- Added `biased sampling` utils ([#38](https://github.com/pyg-team/pyg-lib/pull/38))
- Added `CHANGELOG.md` ([#39](https://github.com/pyg-team/pyg-lib/pull/39))
- Added `pyg.subgraph()` ([#31](https://github.com/pyg-team/pyg-lib/pull/31))
- Added nightly builds ([#28](https://github.com/pyg-team/pyg-lib/pull/28), [#36](https://github.com/pyg-team/pyg-lib/pull/36))
- Added `rand` CPU engine ([#26](https://github.com/pyg-team/pyg-lib/pull/26), [#29](https://github.com/pyg-team/pyg-lib/pull/29), [#32](https://github.com/pyg-team/pyg-lib/pull/32), [#33](https://github.com/pyg-team/pyg-lib/pull/33))
- Added `pyg.random_walk()` ([#21](https://github.com/pyg-team/pyg-lib/pull/21), [#24](https://github.com/pyg-team/pyg-lib/pull/24), [#25](https://github.com/pyg-team/pyg-lib/pull/25))
- Added documentation via `readthedocs` ([#19](https://github.com/pyg-team/pyg-lib/pull/19), [#20](https://github.com/pyg-team/pyg-lib/pull/29))
- Added code coverage report ([#15](https://github.com/pyg-team/pyg-lib/pull/15), [#16](https://github.com/pyg-team/pyg-lib/pull/16), [#17](https://github.com/pyg-team/pyg-lib/pull/17), [#18](https://github.com/pyg-team/pyg-lib/pull/18))
- Added `CMakeExtension` support ([#14](https://github.com/pyg-team/pyg-lib/pull/14))
- Added test suite via `gtest` ([#13](https://github.com/pyg-team/pyg-lib/pull/13))
- Added `clang-format` linting via `pre-commit` ([#12](https://github.com/pyg-team/pyg-lib/pull/12))
- Added `CMake` support ([#5](https://github.com/pyg-team/pyg-lib/pull/5))
- Added `pyg.cuda_version()` ([#4](https://github.com/pyg-team/pyg-lib/pull/4))
### Changed
- Allow different types for graph and timestamp data ([#143](https://github.com/pyg-team/pyg-lib/pull/143))
- Fixed dispatcher in `hetero_neighbor_sample` ([#125](https://github.com/pyg-team/pyg-lib/pull/125))
- Require sorted neighborhoods according to time in temporal sampling ([#108](https://github.com/pyg-team/pyg-lib/pull/108))
- Only sample neighbors with a strictly earlier timestamp than the seed node ([#104](https://github.com/pyg-team/pyg-lib/pull/104))
- Prevent absolute paths in wheel ([#75](https://github.com/pyg-team/pyg-lib/pull/75))
- Improved installation instructions ([#68](https://github.com/pyg-team/pyg-lib/pull/68))
- Replaced std::unordered_map with a faster phmap::flat_hash_map ([#65](https://github.com/pyg-team/pyg-lib/pull/65))
- Fixed versions of `checkout` and `setup-python` in CI ([#52](https://github.com/pyg-team/pytorch_geometric/pull/52))
- Make use of the `pyg_sphinx_theme` documentation template ([#47](https://github.com/pyg-team/pyg-lib/pull/47))
- Auto-compute number of threads and blocks in CUDA kernels ([#41](https://github.com/pyg-team/pyg-lib/pull/41))
- Optional return types in `pyg.subgraph()` ([#40](https://github.com/pyg-team/pyg-lib/pull/40))
- Absolute headers ([#30](https://github.com/pyg-team/pyg-lib/pull/30))
- Use `at::equal` rather than `at::all` in tests ([#37](https://github.com/pyg-team/pyg-lib/pull/37))
- Build `*.so` extension on Mac instead of `*.dylib`([#107](https://github.com/pyg-team/pyg-lib/pull/107))
