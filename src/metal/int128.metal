/*
 * Bitcoin Puzzle Solver - 128-bit integer helpers for Metal.
 *
 * This file provides foundational 128-bit signed and unsigned integer
 * operations used by the field arithmetic implementations. This version uses
 * a struct of two 64-bit integers to emulate 128-bit types.
 *
 * Original source:
 * - _libsecp256k1/src/int128_struct.h
 * - _libsecp256k1/src/int128_struct_impl.h
 */

#include <metal_stdlib>

using namespace metal;

// Emulated 128-bit unsigned integer type.
struct uint128_t {
    ulong lo;
    ulong hi;
};

// Emulated 128-bit signed integer type.
using int128_t = uint128_t;


// --- Core 64x64 -> 128 multiplication ---

/* Emulate 64x64->128 unsigned multiplication using 32x32->64 multiplications. */
static inline ulong umul128(ulong a, ulong b, thread ulong* hi) {
    ulong ll = (ulong)(uint)a * (uint)b;
    ulong lh = (uint)a * (b >> 32);
    ulong hl = (a >> 32) * (uint)b;
    ulong hh = (a >> 32) * (b >> 32);
    ulong mid34 = (ll >> 32) + (uint)lh + (uint)hl;
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid34 >> 32);
    return (mid34 << 32) + (uint)ll;
}

/* Emulate 64x64->128 signed multiplication. */
static inline long mul128(long a, long b, thread long* hi) {
    ulong ll = (ulong)(uint)a * (uint)b;
    long lh = (uint)a * (b >> 32);
    long hl = (a >> 32) * (uint)b;
    long hh = (a >> 32) * (b >> 32);
    ulong mid34 = (ll >> 32) + (uint)lh + (uint)hl;
    *hi = hh + (lh >> 32) + (hl >> 32) + (mid34 >> 32);
    return (long)((mid34 << 32) + (uint)ll);
}


// --- 128-bit Unsigned Integer Helpers ---

static inline void u128_mul(thread uint128_t &r, ulong a, ulong b) {
   r.lo = umul128(a, b, &r.hi);
}

static inline void u128_accum_mul(thread uint128_t &r, ulong a, ulong b) {
   ulong lo, hi;
   lo = umul128(a, b, &hi);
   r.lo += lo;
   r.hi += hi + (r.lo < lo);
}

static inline void u128_accum_u64(thread uint128_t &r, ulong a) {
   r.lo += a;
   r.hi += (r.lo < a);
}

static inline void u128_rshift(thread uint128_t &r, uint n) {
    if (n >= 64) {
        r.lo = r.hi >> (n - 64);
        r.hi = 0;
    } else if (n > 0) {
        r.lo = (r.hi << (64 - n)) | (r.lo >> n);
        r.hi >>= n;
    }
}

static inline ulong u128_to_u64(const thread uint128_t &a) {
   return a.lo;
}

static inline void i128_sub_u64(thread int128_t &r, ulong b) {
    ulong old_lo = r.lo;
    r.lo -= b;
    r.hi -= (r.lo > old_lo);
}


// --- 128-bit Signed Integer Helpers ---

static inline void i128_mul(thread int128_t &r, long a, long b) {
   long hi;
   r.lo = (ulong)mul128(a, b, &hi);
   r.hi = (ulong)hi;
}

static inline void i128_accum_mul(thread int128_t &r, long a, long b) {
   long hi;
   ulong lo = (ulong)mul128(a, b, &hi);
   r.lo += lo;
   hi += (r.lo < lo);
   r.hi += hi;
}

static inline void i128_rshift(thread int128_t &r, uint n) {
    if (n >= 64) {
        r.lo = (ulong)((long)r.hi >> (n - 64));
        r.hi = (ulong)((long)r.hi >> 63);
    } else if (n > 0) {
        r.lo = (r.hi << (64 - n)) | (r.lo >> n);
        r.hi = (ulong)((long)r.hi >> n);
    }
}

static inline ulong i128_to_u64(const thread int128_t &a) {
   return a.lo;
}

static inline long i128_to_i64(const thread int128_t &a) {
   return (long)a.lo;
}
