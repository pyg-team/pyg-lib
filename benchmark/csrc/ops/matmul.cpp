#include <algorithm>
#include <random>
#include <vector>

#include <ATen/ATen.h>
#include <benchmark/benchmark.h>

#include <pyg_lib/csrc/ops/matmul.h>
#include <pyg_lib/csrc/utils/cpu/convert.h>

template <typename scalar_t>
void BenchmarkSegmentMatmul(benchmark::State& state) {
  const int64_t mn = state.range(0);
  const int64_t k = state.range(1);
  const auto equal_chunks = static_cast<bool>(state.range(2));
  const auto options =
      at::TensorOptions().dtype(c10::CppTypeToScalarType<scalar_t>::value);

  std::vector<int64_t> ptr_vec = {0};
  if (equal_chunks) {
    const int64_t chunk_size = 8;
    for (int64_t i = 0; i < mn / chunk_size; ++i) {
      ptr_vec.push_back((i + 1) * chunk_size);
    }
  } else {
    const int64_t min_chunk_size = 2;
    const int64_t max_chunk_size = 8;
    std::mt19937 gen;
    std::uniform_int_distribution<int64_t> distrib(min_chunk_size,
                                                   max_chunk_size);
    while (ptr_vec.back() < mn - max_chunk_size) {
      ptr_vec.push_back(ptr_vec.back() + distrib(gen));
    }
    ptr_vec.push_back(mn);
  }
  const int64_t b = ptr_vec.size() - 1;

  const auto src0 = at::randn({mn, k}, options);
  const auto src1 = at::randn({b, k, mn}, options);
  const auto ptr = pyg::utils::from_vector(ptr_vec, true);

  for (auto _ : state) {
    const auto out = pyg::ops::segment_matmul(src0, ptr, src1);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BenchmarkSegmentMatmul<float>)
    ->ArgsProduct({benchmark::CreateRange(64, 1024, 2),
                   benchmark::CreateRange(16, 256, 2),
                   {0, 1}})
    ->ArgNames({"M,N", "K", "EqualChunks"})
    ->UseRealTime();
