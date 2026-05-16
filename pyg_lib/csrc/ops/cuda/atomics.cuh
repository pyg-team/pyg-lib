#pragma once

// Atomic helpers for `pyg::scatter_*` and `pyg::segment_*` CUDA kernels.
//
// This file is grown commit-by-commit as the migration of pytorch_scatter
// kernels lands. Commit 1 (`scatter_sum`) introduces:
//
//   * 16-bit CAS-loop `atomicAdd` for `at::Half` and `at::BFloat16` (the
//     primary helpers — CUDA's native `__half`/`__nv_bfloat16` `atomicAdd`
//     intrinsics are gated on sm_70+/sm_80+ respectively, while the CAS-loop
//     works on every architecture pyg-lib targets).
//   * Narrow-int CAS-loop `atomicAdd` for `uint8_t`, `int8_t`, `int16_t`,
//     plus a `int64_t` wrapper that forwards to CUDA's `unsigned long long`
//     intrinsic. These are mandatory for the
//     `AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16, ...)` instantiation set —
//     without them the kernel fails to link for narrow integer dtypes.
//
// Pattern mirrors `pytorch_scatter/csrc/cuda/atomics.cuh`: rewind the address
// to the nearest 32-bit-aligned word, then CAS the full word while updating
// only the bytes/halfword we care about. Selector `(size_t)address & 2` picks
// high (offset 2) vs. low (offset 0) halfword; `(size_t)address & 3` picks
// the byte position.
//
// The overloads are defined in the global namespace so they participate in
// overload resolution alongside the built-in CUDA `atomicAdd(int*, int)`,
// `atomicAdd(float*, float)`, etc. Defining them inside `pyg::ops` would
// shadow the built-ins for any unqualified call site inside that namespace.

#include <ATen/ATen.h>
#include <cuda_runtime.h>

// `atomicAdd(at::Half*, at::Half)` — CAS-loop on the enclosing 32-bit word.
//
// Notes:
//   * `address - ((size_t)address & 2)` rewinds to the 4-byte-aligned word.
//   * `at::Half::x` is `unsigned short` (the IEEE 754 binary16 bit pattern).
//   * `hsum = old_half + val` uses `at::Half`'s `operator+`, which itself
//     promotes to `float` for the actual addition — this matches the upstream
//     pytorch_scatter behavior and keeps precision sane for accumulation.
//   * Returns the previous value at `address`, mirroring CUDA's built-in
//     `atomicAdd` return convention.
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

// Narrow-integer CAS-loop helpers (1-byte and 2-byte). Mirrors the
// 1/2-byte specializations of `AtomicAddIntegerImpl` in
// `pytorch_scatter/csrc/cuda/atomics.cuh:6-41`.

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

// `atomicAdd(int64_t*, int64_t)` — CUDA's native `atomicAdd` is defined for
// `unsigned long long int`, so we reinterpret the address. Signed overflow
// wraps around on 2's-complement, which is what we want for sum accumulation
// (and matches CPU behavior for the same dtype).
static inline __device__ int64_t atomicAdd(int64_t* address, int64_t val) {
  return (int64_t)atomicAdd((unsigned long long int*)address,
                            (unsigned long long int)val);
}
