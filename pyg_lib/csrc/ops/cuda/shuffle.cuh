#pragma once

// Warp-shuffle helpers for `pyg::segment_*` CUDA kernels.
//
// Port of upstream `pytorch_scatter/csrc/cuda/utils.cuh:16-46`. CUDA's built-in
// `__shfl_up_sync` / `__shfl_down_sync` intrinsics are templated over the
// arithmetic dtypes (int, float, double, ...) but do **not** natively support
// `at::Half` / `at::BFloat16`. We provide thin overloads that bit-cast the
// half-precision value through the underlying `unsigned short` storage so the
// compiler can resolve the shuffle to the native 32-bit `__shfl_*_sync`
// variant.
//
// The `SHFL_UP_SYNC` / `SHFL_DOWN_SYNC` macros expand to the `_sync` variants
// on CUDA and to the deprecated unsync `__shfl_*` builtins on ROCm (where the
// `_sync` family is not available). Mirrors the upstream `#ifdef USE_ROCM`
// split.

#include <ATen/ATen.h>
#include <cuda_runtime.h>

// `warp_mask_t`: 32-bit on CUDA, 64-bit on ROCm. The mask is the first
// argument of `__shfl_*_sync` and selects which lanes of the warp participate.
// On CUDA we always use the full mask `0xffffffff`; on ROCm the wavefront is
// 64-wide so the mask is 64-bit.
#ifdef USE_ROCM
using warp_mask_t = unsigned long long;
#else
using warp_mask_t = unsigned int;
#endif

// `at::Half` shuffle overloads. CUDA's `__shfl_*_sync` does not have a native
// overload for `__half` on older toolkits (the 32-bit-wide `__shfl_xor_sync`
// for `__half` was added in CUDA 9, but the `_up`/`_down` variants still want
// an integral or floating-point lane value). We bit-cast through `unsigned
// short` — `at::Half::x` is the IEEE 754 binary16 bit pattern stored as
// `uint16_t` — which lets the compiler resolve to the 32-bit `__shfl_sync`
// instruction (widened/narrowed automatically).
//
// The four-argument overload (mask, value, delta, width) is what the
// `SHFL_UP_SYNC` / `SHFL_DOWN_SYNC` macros expand to from `__shfl_*_sync`
// (which itself has an optional fourth `width` parameter defaulting to
// `warpSize = 32`). We provide a 3-arg overload that forwards to the 4-arg one
// with the default width.
__device__ __inline__ at::Half __shfl_up_sync(const warp_mask_t mask,
                                              const at::Half var,
                                              const unsigned int delta,
                                              const int width) {
  // `at::Half::x` is `unsigned short`; pass it through `__shfl_up_sync(...,
  // unsigned int, ...)`. The implicit widening to `unsigned int` is lossless.
  at::Half ret;
  ret.x = static_cast<unsigned short>(
      __shfl_up_sync(mask, static_cast<unsigned int>(var.x), delta, width));
  return ret;
}

__device__ __inline__ at::Half __shfl_up_sync(const warp_mask_t mask,
                                              const at::Half var,
                                              const unsigned int delta) {
  return __shfl_up_sync(mask, var, delta, warpSize);
}

__device__ __inline__ at::Half __shfl_down_sync(const warp_mask_t mask,
                                                const at::Half var,
                                                const unsigned int delta,
                                                const int width) {
  at::Half ret;
  ret.x = static_cast<unsigned short>(
      __shfl_down_sync(mask, static_cast<unsigned int>(var.x), delta, width));
  return ret;
}

__device__ __inline__ at::Half __shfl_down_sync(const warp_mask_t mask,
                                                const at::Half var,
                                                const unsigned int delta) {
  return __shfl_down_sync(mask, var, delta, warpSize);
}

// `at::BFloat16` shuffle overloads. Mirror the `at::Half` overloads above.
// `at::BFloat16::x` is also `uint16_t`, holding the BF16 bit pattern.
__device__ __inline__ at::BFloat16 __shfl_up_sync(const warp_mask_t mask,
                                                  const at::BFloat16 var,
                                                  const unsigned int delta,
                                                  const int width) {
  at::BFloat16 ret;
  ret.x = static_cast<unsigned short>(
      __shfl_up_sync(mask, static_cast<unsigned int>(var.x), delta, width));
  return ret;
}

__device__ __inline__ at::BFloat16 __shfl_up_sync(const warp_mask_t mask,
                                                  const at::BFloat16 var,
                                                  const unsigned int delta) {
  return __shfl_up_sync(mask, var, delta, warpSize);
}

__device__ __inline__ at::BFloat16 __shfl_down_sync(const warp_mask_t mask,
                                                    const at::BFloat16 var,
                                                    const unsigned int delta,
                                                    const int width) {
  at::BFloat16 ret;
  ret.x = static_cast<unsigned short>(
      __shfl_down_sync(mask, static_cast<unsigned int>(var.x), delta, width));
  return ret;
}

__device__ __inline__ at::BFloat16 __shfl_down_sync(const warp_mask_t mask,
                                                    const at::BFloat16 var,
                                                    const unsigned int delta) {
  return __shfl_down_sync(mask, var, delta, warpSize);
}

// ROCm fallback uses the unsync `__shfl_up` / `__shfl_down` instead.
#ifdef USE_ROCM
__device__ __inline__ at::Half __shfl_up(const at::Half var,
                                         const unsigned int delta) {
  at::Half ret;
  ret.x = static_cast<unsigned short>(
      __shfl_up(static_cast<unsigned int>(var.x), delta));
  return ret;
}
__device__ __inline__ at::Half __shfl_down(const at::Half var,
                                           const unsigned int delta) {
  at::Half ret;
  ret.x = static_cast<unsigned short>(
      __shfl_down(static_cast<unsigned int>(var.x), delta));
  return ret;
}
__device__ __inline__ at::BFloat16 __shfl_up(const at::BFloat16 var,
                                             const unsigned int delta) {
  at::BFloat16 ret;
  ret.x = static_cast<unsigned short>(
      __shfl_up(static_cast<unsigned int>(var.x), delta));
  return ret;
}
__device__ __inline__ at::BFloat16 __shfl_down(const at::BFloat16 var,
                                               const unsigned int delta) {
  at::BFloat16 ret;
  ret.x = static_cast<unsigned short>(
      __shfl_down(static_cast<unsigned int>(var.x), delta));
  return ret;
}
#define SHFL_UP_SYNC(mask, var, delta) __shfl_up(var, delta)
#define SHFL_DOWN_SYNC(mask, var, delta) __shfl_down(var, delta)
#else
#define SHFL_UP_SYNC __shfl_up_sync
#define SHFL_DOWN_SYNC __shfl_down_sync
#endif
