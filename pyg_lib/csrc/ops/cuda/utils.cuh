#pragma once

// Explicit non-template comparison/min/max functions to avoid NVCC ambiguous
// operator overload errors from c10::SymInt (error #3343-D) on CUDA 13+.
// These provide unambiguous overload resolution for float/double in kernels.

__device__ __forceinline__ bool scalar_gt(float a, float b) {
  return a > b;
}
__device__ __forceinline__ bool scalar_gt(double a, double b) {
  return a > b;
}
__device__ __forceinline__ bool scalar_lt(float a, float b) {
  return a < b;
}
__device__ __forceinline__ bool scalar_lt(double a, double b) {
  return a < b;
}
__device__ __forceinline__ bool scalar_ge(float a, float b) {
  return a >= b;
}
__device__ __forceinline__ bool scalar_ge(double a, double b) {
  return a >= b;
}
__device__ __forceinline__ float scalar_min(float a, float b) {
  return fminf(a, b);
}
__device__ __forceinline__ double scalar_min(double a, double b) {
  return fmin(a, b);
}
__device__ __forceinline__ float scalar_max(float a, float b) {
  return fmaxf(a, b);
}
__device__ __forceinline__ double scalar_max(double a, double b) {
  return fmax(a, b);
}

// Binary search for batch index given ptr boundaries.
__forceinline__ __device__ int64_t get_example_idx(int64_t idx,
                                                   const int64_t* ptr,
                                                   int64_t num_examples) {
  for (int64_t i = 0; i < num_examples; i++) {
    if (ptr[i + 1] > idx)
      return i;
  }
  return num_examples - 1;
}
