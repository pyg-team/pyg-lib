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
// Commit 2 (`scatter_mul`) adds:
//
//   * CAS-loop `atomicMul` for every dtype in
//     `AT_DISPATCH_ALL_TYPES_AND2(Half, BFloat16, ...)`. CUDA does not provide
//     a native `atomicMul` for any dtype, so all overloads CAS on the
//     enclosing 32-bit or 64-bit storage word. See the comment block above
//     the overloads for the per-dtype storage strategy.
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

// -----------------------------------------------------------------------------
// `atomicMul` — Commit 2 (`scatter_mul`).
//
// CUDA does not natively provide `atomicMul` for any dtype, so every overload
// below is a CAS-loop on the enclosing 32-bit or 64-bit storage word.
//
// Storage strategy per dtype (mirrors `pytorch_scatter/csrc/cuda/atomics.cuh`'s
// `AtomicMulIntegerImpl<scalar, size>` / `AtomicMulDecimalImpl<scalar, size>`
// specializations):
//   * 1-byte ints (uint8_t / int8_t)            -> CAS on 32-bit word, byte
//                                                  selector `(addr & 3) * 8`.
//   * 2-byte ints + at::Half + at::BFloat16     -> CAS on 32-bit word,
//                                                  halfword selector `addr &
//                                                  2`.
//   * int32_t                                   -> CAS on 32-bit word.
//   * float                                     -> CAS on 32-bit word via
//                                                  `__float_as_int` /
//                                                  `__int_as_float`.
//   * int64_t                                   -> CAS on 64-bit word
//                                                  (`unsigned long long`).
//   * double                                    -> CAS on 64-bit word via
//                                                  `__double_as_longlong` /
//                                                  `__longlong_as_double`.
//
// As with the `atomicAdd` overloads above, these live in the global namespace
// so they participate in overload resolution alongside the CUDA built-ins.

namespace pyg_atomics_detail {

// 1-byte CAS-loop multiply (uint8_t / int8_t). Selector chooses which byte of
// the 32-bit word we're updating; the other three bytes are preserved.
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

// 2-byte CAS-loop multiply (int16_t / at::Half / at::BFloat16).
// `at::Half` / `at::BFloat16` use this template via `scalar_t::operator*`,
// which (like `operator+` in the `atomicAdd` overloads) promotes to `float`
// for the actual multiply — matches upstream pytorch_scatter precision.
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

// 2-byte CAS-loop multiply for `at::Half` / `at::BFloat16`. The
// integer-flavoured template above would clobber the IEEE 754 bit pattern
// because it casts the half-word to/from `scalar_t` directly. Instead, we
// stash the bits into `scalar_t::x` (the underlying `unsigned short`),
// multiply via the operator overload (which promotes to float), then write
// the result's bit pattern back.
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
