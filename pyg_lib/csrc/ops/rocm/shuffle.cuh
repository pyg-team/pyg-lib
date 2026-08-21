#pragma once

// Warp-shuffle helpers for `pyg::segment_*` ROCm/HIP kernels.
//
// ROCm uses the unsync `__shfl_up` / `__shfl_down` builtins (the `_sync`
// family is not available), so the public macros `SHFL_UP_SYNC` and
// `SHFL_DOWN_SYNC` expand to the unsync variants. Half-precision overloads
// for the unsync builtins are provided below; the CUDA `_sync` overloads are
// omitted on ROCm because the base `__shfl_*_sync` intrinsics do not exist.

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
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
// `warp_mask_t`: 32-bit on CUDA, 64-bit on ROCm.
using warp_mask_t = unsigned int;

__device__ __inline__ at::Half __shfl_up_sync(const warp_mask_t mask,
                                              const at::Half var,
                                              const unsigned int delta,
                                              const int width) {
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

#define SHFL_UP_SYNC __shfl_up_sync
#define SHFL_DOWN_SYNC __shfl_down_sync
#endif
