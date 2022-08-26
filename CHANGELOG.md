# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- Added support for PyTorch 1.12 ([#57](https://github.com/pyg-team/pyg-lib/pull/57), [#58](https://github.com/pyg-team/pyg-lib/pull/58))
- Added `grouped_matmul` and `segment_matmul` CUDA implementations via `cutlass` ([#51](https://github.com/pyg-team/pyg-lib/pull/51), [#56](https://github.com/pyg-team/pyg-lib/pull/56), [#61](https://github.com/pyg-team/pyg-lib/pull/61), [#64](https://github.com/pyg-team/pyg-lib/pull/64), [#69](https://github.com/pyg-team/pyg-lib/pull/69))
- Added `pyg::sampler::neighbor_sample` implementation ([#54](https://github.com/pyg-team/pyg-lib/pull/54), [#76](https://github.com/pyg-team/pyg-lib/pull/76))
- Added `pyg::sampler::Mapper` utility for mapping global to local node indices ([#45](https://github.com/pyg-team/pyg-lib/pull/45)))
- Added benchmark script ([#45](https://github.com/pyg-team/pyg-lib/pull/45))
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
- Prevent absolute paths in wheel ([#75](https://github.com/pyg-team/pyg-lib/pull/75))
- Improved installation instructions ([#68](https://github.com/pyg-team/pyg-lib/pull/68))
- Replaced std::unordered_map with a faster phmap::flat_hash_map ([#65](https://github.com/pyg-team/pyg-lib/pull/65))
- Fixed versions of `checkout` and `setup-python` in CI ([#52](https://github.com/pyg-team/pytorch_geometric/pull/52))
- Make use of the `pyg_sphinx_theme` documentation template ([#47](https://github.com/pyg-team/pyg-lib/pull/47))
- Auto-compute number of threads and blocks in CUDA kernels ([#41](https://github.com/pyg-team/pyg-lib/pull/41))
- Optional return types in `pyg.subgraph()` ([#40](https://github.com/pyg-team/pyg-lib/pull/40))
- Absolute headers ([#30](https://github.com/pyg-team/pyg-lib/pull/30))
- Use `at::equal` rather than `at::all` in tests ([#37](https://github.com/pyg-team/pyg-lib/pull/37))
### Removed
