/*
 * Bitcoin Puzzle Solver - Basic field arithmetic for Metal.
 *
 * This file corresponds to Stage 1 of the GPU implementation plan. It provides
 * foundational 256-bit big-integer arithmetic operations. The logic is a
 * direct port of the 5x52-bit limb representation from libsecp256k1.
 *
 * Original source:
 * - _libsecp256k1/src/field_5x52.h
 * - _libsecp256k1/src/field_5x52_int128_impl.h
 */

#include <metal_stdlib>
#include "int128.metal"

using namespace metal;

/*
 * This field implementation represents a value as 5 ulong limbs in base 2^52.
 * It is a direct port of the `secp256k1_fe` struct.
 */
struct fe {
    ulong n[5];
};

// --- Field Element Constants ---
constant fe fe_one = {{1, 0, 0, 0, 0}};

// --- Basic Field Element Operations ---

/* Set a field element to a small integer. */
static inline void fe_set_int(thread fe &r, int a) {
    r.n[0] = a;
    r.n[1] = 0;
    r.n[2] = 0;
    r.n[3] = 0;
    r.n[4] = 0;
}

/*
 * Limb-wise addition for 256-bit integers (r += a).
 * This is a basic big-integer operation; it does not perform modular reduction.
 */
static inline void fe_add(thread fe &r, const thread fe &a) {
    r.n[0] += a.n[0];
    r.n[1] += a.n[1];
    r.n[2] += a.n[2];
    r.n[3] += a.n[3];
    r.n[4] += a.n[4];
}

/*
 * Limb-wise subtraction for 256-bit integers (r -= a), with borrow propagation.
 * This is a basic big-integer operation; it does not perform modular reduction.
 */
static inline void fe_sub(thread fe &r, const thread fe &a) {
    int128_t t;

    t.lo = r.n[0]; t.hi = 0; i128_sub_u64(t, a.n[0]); r.n[0] = t.lo;
    t.lo = r.n[1]; t.hi = (long)t.hi >> 63; i128_sub_u64(t, a.n[1]); r.n[1] = t.lo;
    t.lo = r.n[2]; t.hi = (long)t.hi >> 63; i128_sub_u64(t, a.n[2]); r.n[2] = t.lo;
    t.lo = r.n[3]; t.hi = (long)t.hi >> 63; i128_sub_u64(t, a.n[3]); r.n[3] = t.lo;
    t.lo = r.n[4]; t.hi = (long)t.hi >> 63; i128_sub_u64(t, a.n[4]); r.n[4] = t.lo;
}

/*
 * 256-bit integer multiplication.
 * Ported from `secp256k1_fe_mul_inner` in _libsecp256k1/src/field_5x52_int128_impl.h
 */
static inline void fe_mul(thread fe &r, const thread fe &a_in, const thread fe &b_in) {
    uint128_t c, d;
    ulong t3, t4, tx, u0;
    ulong a0 = a_in.n[0], a1 = a_in.n[1], a2 = a_in.n[2], a3 = a_in.n[3], a4 = a_in.n[4];
    const ulong M = 0xFFFFFFFFFFFFF;
    const ulong R = 0x1000003D10ULL;

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
    const ulong M = 0xFFFFFFFFFFFFFULL;
    const ulong R = 0x1000003D10ULL;

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

/* Multiply a field element by a small integer. */
static inline void fe_mul_int(thread fe &r, int a) {
    r.n[0] *= a;
    r.n[1] *= a;
    r.n[2] *= a;
    r.n[3] *= a;
    r.n[4] *= a;
}

/* Halve a field element. Ported from `secp256k1_fe_impl_half`. */
static inline void fe_half(thread fe &r) {
    ulong t0 = r.n[0], t1 = r.n[1], t2 = r.n[2], t3 = r.n[3], t4 = r.n[4];
    const ulong one = 1;
    ulong mask = -(t0 & one) >> 12;

    t0 += 0xFFFFEFFFFFC2FUL & mask;
    t1 += mask;
    t2 += mask;
    t3 += mask;
    t4 += mask >> 4;

    r.n[0] = (t0 >> 1) + ((t1 & one) << 51);
    r.n[1] = (t1 >> 1) + ((t2 & one) << 51);
    r.n[2] = (t2 >> 1) + ((t3 & one) << 51);
    r.n[3] = (t3 >> 1) + ((t4 & one) << 51);
    r.n[4] = (t4 >> 1);
}

/* Conditionally move a field element. */
static inline void fe_cmov(thread fe &r, const thread fe &a, bool flag) {
    r.n[0] = select(r.n[0], a.n[0], flag);
    r.n[1] = select(r.n[1], a.n[1], flag);
    r.n[2] = select(r.n[2], a.n[2], flag);
    r.n[3] = select(r.n[3], a.n[3], flag);
    r.n[4] = select(r.n[4], a.n[4], flag);
}
