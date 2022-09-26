#include <algorithm>
#include <random>
#include <type_traits>
#include <vector>

#include <benchmark/benchmark.h>

#include "pyg_lib/csrc/sampler/cpu/mapper.h"

template <typename node_t, typename scalar_t>
class BenchmarkMapperCreationAndInsertion : public benchmark::Fixture {
  static_assert(std::is_integral<node_t>::value &&
                    std::is_integral<scalar_t>::value,
                "Integral type required for both node_t and scalar_t");

 protected:
  void SetUp(const benchmark::State& state) override {
    // deterministic draw, default seed is 5489u
    std::mt19937 gen;
    // fill nodes with whole range two times and shuffle it
    // this will allow some indices to occur more than once
    const auto num_nodes = state.range(0);
    nodes_.resize(num_nodes * 2);
    std::iota(nodes_.begin(), nodes_.begin() + num_nodes, 0);
    std::iota(nodes_.begin() + num_nodes, nodes_.end(), 0);
    std::shuffle(nodes_.begin(), nodes_.end(), gen);
  }

  void TearDown(const benchmark::State& state) override { nodes_.clear(); }

  void PerformTest(size_t num_nodes, size_t num_entries, size_t samples) {
    insertion_fail_counter_ = 0;
    pyg::sampler::Mapper<node_t, scalar_t> mapper(num_nodes, num_entries);
    for (size_t idx = 0; idx < samples; ++idx) {
      const auto result = mapper.insert(nodes_[idx]);
      insertion_fail_counter_ += !result.second;
    }
  }

  void Loop(benchmark::State& state) {
    const auto num_nodes = state.range(0);
    const auto num_entries = state.range(1);
    const auto samples = num_entries > 0 ? num_entries : num_nodes;
    for (auto _ : state) {
      PerformTest(num_nodes, num_entries, samples);
    }
    state.SetComplexityN(samples);
    state.counters["Insertion Fail Rate [%]"] =
        100.0 * insertion_fail_counter_ / samples;
  }

  std::vector<node_t> nodes_{};
  int64_t insertion_fail_counter_{0};
};

static void duplicated_range(benchmark::internal::Benchmark* benchmark) {
  const auto range = benchmark::CreateDenseRange(1024E3, 2048E3, 128E3);
  for (const auto value : range) {
    benchmark->Args(std::vector<int64_t>(2, value));
  }
}

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkMapperCreationAndInsertion,
                            WithFlatHashMap,
                            int64_t,
                            int64_t)
(benchmark::State& state) {
  Loop(state);
}
BENCHMARK_REGISTER_F(BenchmarkMapperCreationAndInsertion, WithFlatHashMap)
    ->ArgsProduct({benchmark::CreateDenseRange(1024E3, 2048E3, 128E3), {-1}})
    ->ArgNames({"num_nodes", "num_entries"})
    ->Complexity(benchmark::oN);

BENCHMARK_TEMPLATE_DEFINE_F(BenchmarkMapperCreationAndInsertion,
                            WithVector,
                            int64_t,
                            int64_t)
(benchmark::State& state) {
  Loop(state);
}
BENCHMARK_REGISTER_F(BenchmarkMapperCreationAndInsertion, WithVector)
    ->Apply(duplicated_range)
    ->ArgNames({"num_nodes", "num_entries"})
    ->Complexity(benchmark::oN);
