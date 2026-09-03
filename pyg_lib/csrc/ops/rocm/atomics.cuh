#pragma once

// Atomic helpers for `pyg::scatter_*` and `pyg::segment_*` ROCm/HIP kernels.
//
// Port of the CUDA `atomics.cuh` to HIP. The CAS-loop strategy is identical:
// rewind the address to the nearest aligned storage word, then CAS the full
// word while updating only the bytes/halfword we care about. HIP provides the
// same `atomicCAS`/`atomicAdd` intrinsics as CUDA for 32/64-bit types, and the
// bit-casting builtins (`__float_as_int`, `__double_as_longlong`, ...) are also
// available under hipcc.
//
// The overloads live in the global namespace so they participate in overload
// resolution alongside HIP's built-in `atomicAdd(int*, int)`,
// `atomicAdd(float*, float)`, etc.

#include <type_traits>

#include <ATen/ATen.h>
#include <hip/hip_runtime.h>

// `atomicAdd(at::Half*, at::Half)` — CAS-loop on the enclosing 32-bit word.
static inline __device__ at::Half atomicAdd(at::Half* address, at::Half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);

  at::Half ret;
  ret.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return ret;
}

// `atomicAdd(at::BFloat16*, at::BFloat16)` — same CAS-loop pattern.
static inline __device__ at::BFloat16 atomicAdd(at::BFloat16* address,
                                                at::BFloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    at::BFloat16 hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);

  at::BFloat16 ret;
  ret.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return ret;
}

// Narrow-integer CAS-loop helpers (1-byte and 2-byte).
namespace pyg_atomics_detail {

template <typename scalar_t>
static inline __device__ scalar_t atomicAdd1B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 3));
  const uint32_t shift = ((size_t)address & 3) * 8;
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t sum;

  do {
    assumed = old;
    sum = (uint32_t)((scalar_t)((old >> shift) & 0xff) + val) & 0xff;
    old = (assumed & ~(0x000000ff << shift)) | (sum << shift);
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  return (scalar_t)((old >> shift) & 0xff);
}

template <typename scalar_t>
static inline __device__ scalar_t atomicAdd2B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 2));
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t sum;
  uint32_t newval;

  do {
    assumed = old;
    sum = (uint32_t)((size_t)address & 2 ? (scalar_t)(old >> 16) + val
                                         : (scalar_t)(old & 0xffff) + val) &
          0xffff;
    newval = (size_t)address & 2 ? (old & 0xffff) | (sum << 16)
                                 : (old & 0xffff0000) | sum;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
}

}  // namespace pyg_atomics_detail

static inline __device__ uint8_t atomicAdd(uint8_t* address, uint8_t val) {
  return pyg_atomics_detail::atomicAdd1B<uint8_t>(address, val);
}
static inline __device__ int8_t atomicAdd(int8_t* address, int8_t val) {
  return pyg_atomics_detail::atomicAdd1B<int8_t>(address, val);
}
static inline __device__ int16_t atomicAdd(int16_t* address, int16_t val) {
  return pyg_atomics_detail::atomicAdd2B<int16_t>(address, val);
}

// `atomicAdd(int64_t*, int64_t)` — HIP provides `atomicAdd` for
// `unsigned long long int`, so we reinterpret the address. Signed overflow
// wraps around on 2's-complement, matching CPU behavior.
static inline __device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  return (int64_t)atomicAdd((unsigned long long int*)address,
                            (unsigned long long int)val);
}

// -----------------------------------------------------------------------------
// `atomicMul` — CAS-loop multiply for every dtype in
// `AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16, ...)`.

namespace pyg_atomics_detail {

// 1-byte CAS-loop multiply (uint8_t / int8_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMul1B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 3));
  const uint32_t shift = ((size_t)address & 3) * 8;
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t prod;

  do {
    assumed = old;
    prod = (uint32_t)((scalar_t)((old >> shift) & 0xff) * val) & 0xff;
    old = (assumed & ~(0x000000ff << shift)) | (prod << shift);
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  return (scalar_t)((old >> shift) & 0xff);
}

// 2-byte CAS-loop multiply (int16_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMul2B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 2));
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t prod;
  uint32_t newval;

  do {
    assumed = old;
    prod = (uint32_t)((size_t)address & 2 ? (scalar_t)(old >> 16) * val
                                          : (scalar_t)(old & 0xffff) * val) &
           0xffff;
    newval = (size_t)address & 2 ? (old & 0xffff) | (prod << 16)
                                 : (old & 0xffff0000) | prod;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
}

// 2-byte CAS-loop multiply for `at::Half` / `at::BFloat16`. Preserves the IEEE
// bit pattern via `scalar_t::x` and multiplies via the operator overload.
template <typename scalar_t>
static inline __device__ scalar_t atomicMulHalf(scalar_t* address,
                                                scalar_t val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    scalar_t hprod;
    hprod.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hprod = hprod * val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hprod.x << 16)
                              : (old & 0xffff0000) | hprod.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);

  scalar_t ret;
  ret.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return ret;
}

}  // namespace pyg_atomics_detail

// Integer overloads.
static inline __device__ uint8_t atomicMul(uint8_t* address, uint8_t val) {
  return pyg_atomics_detail::atomicMul1B<uint8_t>(address, val);
}
static inline __device__ int8_t atomicMul(int8_t* address, int8_t val) {
  return pyg_atomics_detail::atomicMul1B<int8_t>(address, val);
}
static inline __device__ int16_t atomicMul(int16_t* address, int16_t val) {
  return pyg_atomics_detail::atomicMul2B<int16_t>(address, val);
}
static inline __device__ int32_t atomicMul(int32_t* address, int32_t val) {
  uint32_t* address_as_ui = (uint32_t*)address;
  uint32_t old = *address_as_ui;
  uint32_t assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ui, assumed, (uint32_t)((int32_t)assumed * val));
  } while (assumed != old);
  return (int32_t)old;
}
static inline __device__ int64_t atomicMul(int64_t* address, int64_t val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    (unsigned long long int)((int64_t)assumed * val));
  } while (assumed != old);
  return (int64_t)old;
}

// Floating-point overloads.
static inline __device__ at::Half atomicMul(at::Half* address, at::Half val) {
  return pyg_atomics_detail::atomicMulHalf<at::Half>(address, val);
}
static inline __device__ at::BFloat16 atomicMul(at::BFloat16* address,
                                                at::BFloat16 val) {
  return pyg_atomics_detail::atomicMulHalf<at::BFloat16>(address, val);
}
static inline __device__ float atomicMul(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}
static inline __device__ double atomicMul(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val * __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// -----------------------------------------------------------------------------
// `atomicMin` — CAS-loop min for every dtype in
// `AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16, ...)`.

namespace pyg_atomics_detail {

// 1-byte CAS-loop min (uint8_t / int8_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMin1B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 3));
  const uint32_t shift = ((size_t)address & 3) * 8;
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t result;

  do {
    assumed = old;
    const scalar_t cur = (scalar_t)((old >> shift) & 0xff);
    const scalar_t new_val = val < cur ? val : cur;
    result = (assumed & ~(0x000000ff << shift)) |
             (((uint32_t)new_val & 0xff) << shift);
    old = atomicCAS(address_as_ui, assumed, result);
  } while (assumed != old);
  return (scalar_t)((old >> shift) & 0xff);
}

// 2-byte CAS-loop min (int16_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMin2B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 2));
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t newval;

  do {
    assumed = old;
    const scalar_t cur =
        (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
    const scalar_t new_val = val < cur ? val : cur;
    const uint32_t lo = (uint32_t)new_val & 0xffff;
    newval = (size_t)address & 2 ? (old & 0xffff) | (lo << 16)
                                 : (old & 0xffff0000) | lo;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
}

// 2-byte CAS-loop min for `at::Half` / `at::BFloat16`. Compare as float to
// avoid operator-overload ambiguity.
template <typename scalar_t>
static inline __device__ scalar_t atomicMinHalf(scalar_t* address,
                                                scalar_t val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  const float val_f = static_cast<float>(val);

  do {
    assumed = old;
    scalar_t cur;
    cur.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    const scalar_t chosen = val_f < static_cast<float>(cur) ? val : cur;
    old = (size_t)address & 2 ? (old & 0xffff) | ((unsigned int)chosen.x << 16)
                              : (old & 0xffff0000) | (unsigned int)chosen.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);

  scalar_t ret;
  ret.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return ret;
}

}  // namespace pyg_atomics_detail

// Integer overloads.
static inline __device__ uint8_t atomicMin(uint8_t* address, uint8_t val) {
  return pyg_atomics_detail::atomicMin1B<uint8_t>(address, val);
}
static inline __device__ int8_t atomicMin(int8_t* address, int8_t val) {
  return pyg_atomics_detail::atomicMin1B<int8_t>(address, val);
}
static inline __device__ int16_t atomicMin(int16_t* address, int16_t val) {
  return pyg_atomics_detail::atomicMin2B<int16_t>(address, val);
}
// `int32_t` (`int`) — HIP's native `atomicMin(int*, int)` participates in
// overload resolution; skip the wrapper to avoid redefinition.
//
// `int64_t` is `long int` on Linux x86_64 (LP64); HIP's native
// `atomicMin(long long int*, long long int)` covers a *different* type there,
// so we provide our own CAS loop. On Windows MSVC (LLP64), `int64_t` *is*
// `long long int`, so the SFINAE constraint avoids redefinition.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, int64_t> &&
                                      !std::is_same_v<T, long long>>>
static inline __device__ T atomicMin(T* address, T val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    const T cur = (T)assumed;
    const T new_val = val < cur ? val : cur;
    old = atomicCAS(address_as_ull, assumed, (unsigned long long int)new_val);
  } while (assumed != old);
  return (T)old;
}

// Floating-point overloads.
static inline __device__ at::Half atomicMin(at::Half* address, at::Half val) {
  return pyg_atomics_detail::atomicMinHalf<at::Half>(address, val);
}
static inline __device__ at::BFloat16 atomicMin(at::BFloat16* address,
                                                at::BFloat16 val) {
  return pyg_atomics_detail::atomicMinHalf<at::BFloat16>(address, val);
}
#ifndef __HIP_PLATFORM_AMD__
static inline __device__ float atomicMin(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed;

  do {
    assumed = old;
    const float cur = __int_as_float(assumed);
    const float new_val = val < cur ? val : cur;
    old = atomicCAS(address_as_i, assumed, __float_as_int(new_val));
  } while (assumed != old);
  return __int_as_float(old);
}
static inline __device__ double atomicMin(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    const double cur = __longlong_as_double(assumed);
    const double new_val = val < cur ? val : cur;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif  // __HIP_PLATFORM_AMD__

// -----------------------------------------------------------------------------
// `atomicMax` — symmetric to `atomicMin`.

namespace pyg_atomics_detail {

// 1-byte CAS-loop max (uint8_t / int8_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMax1B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 3));
  const uint32_t shift = ((size_t)address & 3) * 8;
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t result;

  do {
    assumed = old;
    const scalar_t cur = (scalar_t)((old >> shift) & 0xff);
    const scalar_t new_val = val > cur ? val : cur;
    result = (assumed & ~(0x000000ff << shift)) |
             (((uint32_t)new_val & 0xff) << shift);
    old = atomicCAS(address_as_ui, assumed, result);
  } while (assumed != old);
  return (scalar_t)((old >> shift) & 0xff);
}

// 2-byte CAS-loop max (int16_t).
template <typename scalar_t>
static inline __device__ scalar_t atomicMax2B(scalar_t* address, scalar_t val) {
  uint32_t* address_as_ui = (uint32_t*)((char*)address - ((size_t)address & 2));
  uint32_t old = *address_as_ui;
  uint32_t assumed;
  uint32_t newval;

  do {
    assumed = old;
    const scalar_t cur =
        (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
    const scalar_t new_val = val > cur ? val : cur;
    const uint32_t lo = (uint32_t)new_val & 0xffff;
    newval = (size_t)address & 2 ? (old & 0xffff) | (lo << 16)
                                 : (old & 0xffff0000) | lo;
    old = atomicCAS(address_as_ui, assumed, newval);
  } while (assumed != old);
  return (size_t)address & 2 ? (scalar_t)(old >> 16) : (scalar_t)(old & 0xffff);
}

// 2-byte CAS-loop max for `at::Half` / `at::BFloat16`.
template <typename scalar_t>
static inline __device__ scalar_t atomicMaxHalf(scalar_t* address,
                                                scalar_t val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  const float val_f = static_cast<float>(val);

  do {
    assumed = old;
    scalar_t cur;
    cur.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    const scalar_t chosen = val_f > static_cast<float>(cur) ? val : cur;
    old = (size_t)address & 2 ? (old & 0xffff) | ((unsigned int)chosen.x << 16)
                              : (old & 0xffff0000) | (unsigned int)chosen.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);

  scalar_t ret;
  ret.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return ret;
}

}  // namespace pyg_atomics_detail

// Integer overloads.
static inline __device__ uint8_t atomicMax(uint8_t* address, uint8_t val) {
  return pyg_atomics_detail::atomicMax1B<uint8_t>(address, val);
}
static inline __device__ int8_t atomicMax(int8_t* address, int8_t val) {
  return pyg_atomics_detail::atomicMax1B<int8_t>(address, val);
}
static inline __device__ int16_t atomicMax(int16_t* address, int16_t val) {
  return pyg_atomics_detail::atomicMax2B<int16_t>(address, val);
}
// `int32_t` (`int`) — native HIP `atomicMax(int*, int)` already participates.
//
// `int64_t` CAS-loop wrapper for LP64 where `int64_t` is `long`.
template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, int64_t> &&
                                      !std::is_same_v<T, long long>>>
static inline __device__ T atomicMax(T* address, T val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    const T cur = (T)assumed;
    const T new_val = val > cur ? val : cur;
    old = atomicCAS(address_as_ull, assumed, (unsigned long long int)new_val);
  } while (assumed != old);
  return (T)old;
}

// Floating-point overloads.
static inline __device__ at::Half atomicMax(at::Half* address, at::Half val) {
  return pyg_atomics_detail::atomicMaxHalf<at::Half>(address, val);
}
static inline __device__ at::BFloat16 atomicMax(at::BFloat16* address,
                                                at::BFloat16 val) {
  return pyg_atomics_detail::atomicMaxHalf<at::BFloat16>(address, val);
}
#ifndef __HIP_PLATFORM_AMD__
static inline __device__ float atomicMax(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i;
  int assumed;

  do {
    assumed = old;
    const float cur = __int_as_float(assumed);
    const float new_val = val > cur ? val : cur;
    old = atomicCAS(address_as_i, assumed, __float_as_int(new_val));
  } while (assumed != old);
  return __int_as_float(old);
}
static inline __device__ double atomicMax(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;

  do {
    assumed = old;
    const double cur = __longlong_as_double(assumed);
    const double new_val = val > cur ? val : cur;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif  // __HIP_PLATFORM_AMD__
