/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE_radix_sort file in the LICENSE directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <utility>

#if !defined(_OPENMP)

namespace pyg {
namespace ops {

bool inline is_radix_sort_available() {
  return false;
}

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(K* inp_key_buf,
                                      V* inp_value_buf,
                                      K* tmp_key_buf,
                                      V* tmp_value_buf,
                                      int64_t elements_count,
                                      int64_t max_value) {
  TORCH_CHECK(
      false,
      "radix_sort_parallel: pyg-lib is not compiled with OpenMP support");
}

}  // namespace ops
}  // namespace pyg

#else

#include <c10/util/llvmMathExtras.h>
#include <omp.h>

namespace pyg {
namespace ops {

namespace {

// Copied from fbgemm implementation available here:
// https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/src/cpu_utils.cpp
//
// `radix_sort_parallel` is only available when pyg-lib is compiled with
// OpenMP, since the algorithm requires sync between omp threads, which can not
// be perfectly mapped to `at::parallel_for` at the current stage.

// histogram size per thread
constexpr int RDX_HIST_SIZE = 256;

template <typename K, typename V>
void radix_sort_kernel(K* input_keys,
                       V* input_values,
                       K* output_keys,
                       V* output_values,
                       int elements_count,
                       int* histogram,
                       int* histogram_ps,
                       int pass) {
  int tid = omp_get_thread_num();
  int nthreads = omp_get_num_threads();
  int elements_count_4 = elements_count / 4 * 4;

  int* local_histogram = &histogram[RDX_HIST_SIZE * tid];
  int* local_histogram_ps = &histogram_ps[RDX_HIST_SIZE * tid];

  // Step 1: compute histogram
  for (int i = 0; i < RDX_HIST_SIZE; i++) {
    local_histogram[i] = 0;
  }

#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    local_histogram[(key_1 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_2 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_3 >> (pass * 8)) & 0xFF]++;
    local_histogram[(key_4 >> (pass * 8)) & 0xFF]++;
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      local_histogram[(key >> (pass * 8)) & 0xFF]++;
    }
  }
#pragma omp barrier
  // Step 2: prefix sum
  if (tid == 0) {
    int sum = 0, prev_sum = 0;
    for (int bins = 0; bins < RDX_HIST_SIZE; bins++) {
      for (int t = 0; t < nthreads; t++) {
        sum += histogram[t * RDX_HIST_SIZE + bins];
        histogram_ps[t * RDX_HIST_SIZE + bins] = prev_sum;
        prev_sum = sum;
      }
    }
    histogram_ps[RDX_HIST_SIZE * nthreads] = prev_sum;
    TORCH_CHECK(prev_sum == elements_count);
  }
#pragma omp barrier

  // Step 3: scatter
#pragma omp for schedule(static)
  for (int64_t i = 0; i < elements_count_4; i += 4) {
    K key_1 = input_keys[i];
    K key_2 = input_keys[i + 1];
    K key_3 = input_keys[i + 2];
    K key_4 = input_keys[i + 3];

    int bin_1 = (key_1 >> (pass * 8)) & 0xFF;
    int bin_2 = (key_2 >> (pass * 8)) & 0xFF;
    int bin_3 = (key_3 >> (pass * 8)) & 0xFF;
    int bin_4 = (key_4 >> (pass * 8)) & 0xFF;

    int pos;
    pos = local_histogram_ps[bin_1]++;
    output_keys[pos] = key_1;
    output_values[pos] = input_values[i];
    pos = local_histogram_ps[bin_2]++;
    output_keys[pos] = key_2;
    output_values[pos] = input_values[i + 1];
    pos = local_histogram_ps[bin_3]++;
    output_keys[pos] = key_3;
    output_values[pos] = input_values[i + 2];
    pos = local_histogram_ps[bin_4]++;
    output_keys[pos] = key_4;
    output_values[pos] = input_values[i + 3];
  }
  if (tid == (nthreads - 1)) {
    for (int64_t i = elements_count_4; i < elements_count; ++i) {
      K key = input_keys[i];
      int pos = local_histogram_ps[(key >> (pass * 8)) & 0xFF]++;
      output_keys[pos] = key;
      output_values[pos] = input_values[i];
    }
  }
}

}  // namespace

bool inline is_radix_sort_available() {
  return true;
}

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(K* inp_key_buf,
                                      V* inp_value_buf,
                                      K* tmp_key_buf,
                                      V* tmp_value_buf,
                                      int64_t elements_count,
                                      int64_t max_value) {
  int maxthreads = omp_get_max_threads();
  std::unique_ptr<int[]> histogram_tmp(new int[RDX_HIST_SIZE * maxthreads]);
  std::unique_ptr<int[]> histogram_ps_tmp(
      new int[RDX_HIST_SIZE * maxthreads + 1]);
  int* histogram = histogram_tmp.get();
  int* histogram_ps = histogram_ps_tmp.get();
  if (max_value == 0) {
    return std::make_pair(inp_key_buf, inp_value_buf);
  }

  // __builtin_clz is not portable
  int num_bits =
      sizeof(K) * 8 - c10::llvm::countLeadingZeros(
                          static_cast<std::make_unsigned_t<K> >(max_value));
  unsigned int num_passes = (num_bits + 7) / 8;

#pragma omp parallel
  {
    K* input_keys = inp_key_buf;
    V* input_values = inp_value_buf;
    K* output_keys = tmp_key_buf;
    V* output_values = tmp_value_buf;

    for (unsigned int pass = 0; pass < num_passes; pass++) {
      radix_sort_kernel(input_keys, input_values, output_keys, output_values,
                        elements_count, histogram, histogram_ps, pass);

      std::swap(input_keys, output_keys);
      std::swap(input_values, output_values);
#pragma omp barrier
    }
  }
  return (num_passes % 2 == 0 ? std::make_pair(inp_key_buf, inp_value_buf)
                              : std::make_pair(tmp_key_buf, tmp_value_buf));
}

}  // namespace ops
}  // namespace pyg

#endif
