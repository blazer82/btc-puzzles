/*
 * Bitcoin Puzzle Solver - Metal port of secp256k1 field arithmetic.
 *
 * This file corresponds to Stage 1 of the GPU implementation plan. It provides
 * foundational 256-bit big-integer arithmetic operations. The logic is a
 * direct port of the 5x52-bit limb representation from libsecp256k1.
 *
 * Original source:
 * - _libsecp256k1/src/field_5x52.h
 * - _libsecp256k1/src/field_5x52_int128_impl.h
 * - _libsecp256k1/src/int128_native_impl.h
 */

#include <metal_stdlib>

using namespace metal;

// A 128-bit unsigned integer type, available in Metal.
using uint128_t = metal::uint128_t;

/*
 * This field implementation represents a value as 5 ulong limbs in base 2^52.
 * It is a direct port of the `secp256k1_fe` struct.
 */
struct fe {
    ulong n[5];
};

// --- 128-bit Unsigned Integer Helpers ---
// Ported from _libsecp256k1/src/int128_native_impl.h

static inline void u128_mul(thread uint128_t &r, ulong a, ulong b) {
    r = (uint128_t)a * b;
}

static inline void u128_accum_mul(thread uint128_t &r, ulong a, ulong b) {
    r += (uint128_t)a * b;
}

static inline void u128_accum_u64(thread uint128_t &r, ulong a) {
    r += a;
}

static inline void u128_rshift(thread uint128_t &r, uint n) {
    r >>= n;
}

static inline ulong u128_to_u64(const thread uint128_t &a) {
    return (ulong)a;
}

// --- Field Element Arithmetic ---

/*
 * Limb-wise addition for 256-bit integers (r = a + b).
 * This is a basic big-integer operation; it does not perform modular reduction.
 */
static inline void fe_add(thread fe &r, const thread fe &a, const thread fe &b) {
    r.n[0] = a.n[0] + b.n[0];
    r.n[1] = a.n[1] + b.n[1];
    r.n[2] = a.n[2] + b.n[2];
    r.n[3] = a.n[3] + b.n[3];
    r.n[4] = a.n[4] + b.n[4];
}

/*
 * Limb-wise subtraction for 256-bit integers (r = a - b).
 * This is a basic big-integer operation; it does not perform modular reduction.
 */
static inline void fe_sub(thread fe &r, const thread fe &a, const thread fe &b) {
    r.n[0] = a.n[0] - b.n[0];
    r.n[1] = a.n[1] - b.n[1];
    r.n[2] = a.n[2] - b.n[2];
    r.n[3] = a.n[3] - b.n[3];
    r.n[4] = a.n[4] - b.n[4];
}

/*
 * 256-bit integer multiplication.
 * Ported from `secp256k1_fe_mul_inner` in _libsecp256k1/src/field_5x52_int128_impl.h
 */
static inline void fe_mul(thread fe &r, const thread fe &a_in, const thread fe &b_in) {
    uint128_t c, d;
    ulong t3, t4, tx, u0;
    ulong a0 = a_in.n[0], a1 = a_in.n[1], a2 = a_in.n[2], a3 = a_in.n[3], a4 = a_in.n[4];
    constant ulong M = 0xFFFFFFFFFFFFF;
    constant ulong R = 0x1000003D10ULL;

    u128_mul(d, a0, b_in.n[3]);
    u128_accum_mul(d, a1, b_in.n[2]);
    u128_accum_mul(d, a2, b_in.n[1]);
    u128_accum_mul(d, a3, b_in.n[0]);

    u128_mul(c, a4, b_in.n[4]);

    u128_accum_mul(d, R, u128_to_u64(c));
    u128_rshift(c, 64);

    t3 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    u128_accum_mul(d, a0, b_in.n[4]);
    u128_accum_mul(d, a1, b_in.n[3]);
    u128_accum_mul(d, a2, b_in.n[2]);
    u128_accum_mul(d, a3, b_in.n[1]);
    u128_accum_mul(d, a4, b_in.n[0]);

    u128_accum_mul(d, R << 12, u128_to_u64(c));

    t4 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    tx = (t4 >> 48);
    t4 &= (M >> 4);

    u128_mul(c, a0, b_in.n[0]);

    u128_accum_mul(d, a1, b_in.n[4]);
    u128_accum_mul(d, a2, b_in.n[3]);
    u128_accum_mul(d, a3, b_in.n[2]);
    u128_accum_mul(d, a4, b_in.n[1]);

    u0 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    u0 = (u0 << 4) | tx;

    u128_accum_mul(c, u0, R >> 4);

    r.n[0] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    u128_accum_mul(c, a0, b_in.n[1]);
    u128_accum_mul(c, a1, b_in.n[0]);

    u128_accum_mul(d, a2, b_in.n[4]);
    u128_accum_mul(d, a3, b_in.n[3]);
    u128_accum_mul(d, a4, b_in.n[2]);

    u128_accum_mul(c, u128_to_u64(d) & M, R);
    u128_rshift(d, 52);

    r.n[1] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    u128_accum_mul(c, a0, b_in.n[2]);
    u128_accum_mul(c, a1, b_in.n[1]);
    u128_accum_mul(c, a2, b_in.n[0]);

    u128_accum_mul(d, a3, b_in.n[4]);
    u128_accum_mul(d, a4, b_in.n[3]);

    u128_accum_mul(c, R, u128_to_u64(d));
    u128_rshift(d, 64);

    r.n[2] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    u128_accum_mul(c, R << 12, u128_to_u64(d));
    u128_accum_u64(c, t3);

    r.n[3] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    r.n[4] = u128_to_u64(c) + t4;
}

/*
 * 256-bit integer squaring.
 * Ported from `secp256k1_fe_sqr_inner` in _libsecp256k1/src/field_5x52_int128_impl.h
 */
static inline void fe_sqr(thread fe &r, const thread fe &a_in) {
    uint128_t c, d;
    ulong a0 = a_in.n[0], a1 = a_in.n[1], a2 = a_in.n[2], a3 = a_in.n[3], a4 = a_in.n[4];
    ulong t3, t4, tx, u0;
    constant ulong M = 0xFFFFFFFFFFFFFULL;
    constant ulong R = 0x1000003D10ULL;

    u128_mul(d, a0 * 2, a3);
    u128_accum_mul(d, a1 * 2, a2);

    u128_mul(c, a4, a4);

    u128_accum_mul(d, R, u128_to_u64(c));
    u128_rshift(c, 64);

    t3 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    a4 *= 2;
    u128_accum_mul(d, a0, a4);
    u128_accum_mul(d, a1 * 2, a3);
    u128_accum_mul(d, a2, a2);

    u128_accum_mul(d, R << 12, u128_to_u64(c));

    t4 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    tx = (t4 >> 48);
    t4 &= (M >> 4);

    u128_mul(c, a0, a0);

    u128_accum_mul(d, a1, a4);
    u128_accum_mul(d, a2 * 2, a3);

    u0 = u128_to_u64(d) & M;
    u128_rshift(d, 52);

    u0 = (u0 << 4) | tx;

    u128_accum_mul(c, u0, R >> 4);

    r.n[0] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    a0 *= 2;
    u128_accum_mul(c, a0, a1);

    u128_accum_mul(d, a2, a4);
    u128_accum_mul(d, a3, a3);

    u128_accum_mul(c, u128_to_u64(d) & M, R);
    u128_rshift(d, 52);

    r.n[1] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    u128_accum_mul(c, a0, a2);
    u128_accum_mul(c, a1, a1);

    u128_accum_mul(d, a3, a4);

    u128_accum_mul(c, R, u128_to_u64(d));
    u128_rshift(d, 64);

    r.n[2] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    u128_accum_mul(c, R << 12, u128_to_u64(d));
    u128_accum_u64(c, t3);

    r.n[3] = u128_to_u64(c) & M;
    u128_rshift(c, 52);

    r.n[4] = u128_to_u64(c) + t4;
}
