#include <vector>

#include <benchmark/benchmark.h>

#include <pyg_lib/csrc/random/cpu/rand_engine.h>

constexpr int64_t beg = 0;
constexpr int64_t end = 1 << 15;

void BenchmarkRandEngineWithMKL(benchmark::State& state) {
  const int64_t count = state.range(0);
  pyg::random::RandintEngine<int64_t> generator;

  for (auto _ : state) {
    const auto out =
        std::move(generator.generate_range_of_ints(beg, end, count));
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BenchmarkRandEngineWithMKL)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 12);

void BenchmarkRandEngineWithPrefetching(benchmark::State& state) {
  const int64_t count = state.range(0);
  pyg::random::RandintEngine<int64_t> generator;

  for (auto _ : state) {
    for (int64_t i = 0; i < count; ++i) {
      const auto out = generator(beg, end);
      benchmark::DoNotOptimize(out);
    }
  }
}
BENCHMARK(BenchmarkRandEngineWithPrefetching)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 12);
