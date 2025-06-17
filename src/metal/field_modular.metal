/*
 * Bitcoin Puzzle Solver - Modular field arithmetic for Metal.
 *
 * This file corresponds to Stage 2 of the GPU implementation plan. It implements
 * modular reduction and inversion, building on the big-integer arithmetic from Stage 1.
 *
 * Original source:
 * - _libsecp256k1/src/field_5x52_impl.h
 * - _libsecp256k1/src/modinv64_impl.h
 */

#include <metal_stdlib>
#include "int128.metal"
#include "field_basic.metal"

using namespace metal;

// --- Modular Inverse Implementation (using Bernstein-Yang) ---
// Ported from _libsecp256k1/src/modinv64_impl.h

/* A signed 62-bit limb representation of integers. */
struct modinv64_signed62 {
    long v[5];
};

/* Data type for transition matrices. */
struct modinv64_trans2x2 {
    long u, v, q, r;
};

struct modinv64_modinfo {
    modinv64_signed62 modulus;
    ulong modulus_inv62;
};

// secp256k1 modulus information for modinv64
constant modinv64_modinfo const_modinfo_fe = {
    /* modulus = */ {{ -0x1000003D1L, 0, 0, 0, 256 }},
    /* modulus_inv62 = */ 0x27C7F6E22DDACACFL
};

/* Take as input a signed62 number in range (-2*modulus,modulus), and add a multiple of the modulus
 * to it to bring it to range [0,modulus). If sign < 0, the input will also be negated in the
 * process. The input must have limbs in range (-2^62,2^62). The output will have limbs in range
 * [0,2^62). */
static void modinv64_normalize_62(thread modinv64_signed62 &r_in, long sign, const constant modinv64_modinfo& modinfo) {
    const long M62 = (long)(0xFFFFFFFFFFFFFFFFUL >> 2);
    long r0 = r_in.v[0], r1 = r_in.v[1], r2 = r_in.v[2], r3 = r_in.v[3], r4 = r_in.v[4];
    volatile long cond_add, cond_negate;

    cond_add = r4 >> 63;
    r0 += modinfo.modulus.v[0] & cond_add;
    r1 += modinfo.modulus.v[1] & cond_add;
    r2 += modinfo.modulus.v[2] & cond_add;
    r3 += modinfo.modulus.v[3] & cond_add;
    r4 += modinfo.modulus.v[4] & cond_add;

    cond_negate = sign >> 63;
    r0 = (r0 ^ cond_negate) - cond_negate;
    r1 = (r1 ^ cond_negate) - cond_negate;
    r2 = (r2 ^ cond_negate) - cond_negate;
    r3 = (r3 ^ cond_negate) - cond_negate;
    r4 = (r4 ^ cond_negate) - cond_negate;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    cond_add = r4 >> 63;
    r0 += modinfo.modulus.v[0] & cond_add;
    r1 += modinfo.modulus.v[1] & cond_add;
    r2 += modinfo.modulus.v[2] & cond_add;
    r3 += modinfo.modulus.v[3] & cond_add;
    r4 += modinfo.modulus.v[4] & cond_add;

    r1 += r0 >> 62; r0 &= M62;
    r2 += r1 >> 62; r1 &= M62;
    r3 += r2 >> 62; r2 &= M62;
    r4 += r3 >> 62; r3 &= M62;

    r_in.v[0] = r0;
    r_in.v[1] = r1;
    r_in.v[2] = r2;
    r_in.v[3] = r3;
    r_in.v[4] = r4;
}

static long modinv64_divsteps_59(long zeta, ulong f0, ulong g0, thread modinv64_trans2x2 &t) {
    ulong u = 8, v = 0, q = 0, r = 8;
    volatile ulong c1, c2;
    ulong mask1, mask2, f = f0, g = g0, x, y, z;

    for (int i = 3; i < 62; ++i) {
        c1 = zeta >> 63;
        mask1 = c1;
        c2 = g & 1;
        mask2 = -c2;

        x = (f ^ mask1) - mask1;
        y = (u ^ mask1) - mask1;
        z = (v ^ mask1) - mask1;

        g += x & mask2;
        q += y & mask2;
        r += z & mask2;

        mask1 &= mask2;
        zeta = (zeta ^ mask1) - 1;

        f += g & mask1;
        u += q & mask1;
        v += r & mask1;

        g >>= 1;
        u <<= 1;
        v <<= 1;
    }
    t.u = (long)u;
    t.v = (long)v;
    t.q = (long)q;
    t.r = (long)r;
    return zeta;
}

static void modinv64_update_de_62(thread modinv64_signed62 &d, thread modinv64_signed62 &e, const thread modinv64_trans2x2 &t, const constant modinv64_modinfo& modinfo) {
    const ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
    const long d0 = d.v[0], d1 = d.v[1], d2 = d.v[2], d3 = d.v[3], d4 = d.v[4];
    const long e0 = e.v[0], e1 = e.v[1], e2 = e.v[2], e3 = e.v[3], e4 = e.v[4];
    const long u = t.u, v = t.v, q = t.q, r = t.r;
    long md, me, sd, se;
    int128_t cd, ce;

    sd = d4 >> 63;
    se = e4 >> 63;
    md = (u & sd) + (v & se);
    me = (q & sd) + (r & se);

    i128_mul(cd, u, d0);
    i128_accum_mul(cd, v, e0);
    i128_mul(ce, q, d0);
    i128_accum_mul(ce, r, e0);

    md -= (modinfo.modulus_inv62 * i128_to_u64(cd) + md) & M62;
    me -= (modinfo.modulus_inv62 * i128_to_u64(ce) + me) & M62;

    i128_accum_mul(cd, modinfo.modulus.v[0], md);
    i128_accum_mul(ce, modinfo.modulus.v[0], me);
    // The original code verifies that the bottom 62 bits are zero before shifting
    i128_rshift(cd, 62);
    i128_rshift(ce, 62);

    i128_accum_mul(cd, u, d1);
    i128_accum_mul(cd, v, e1);
    i128_accum_mul(ce, q, d1);
    i128_accum_mul(ce, r, e1);
    if (modinfo.modulus.v[1] != 0) {
        i128_accum_mul(cd, modinfo.modulus.v[1], md);
        i128_accum_mul(ce, modinfo.modulus.v[1], me);
    }
    d.v[0] = i128_to_u64(cd) & M62; i128_rshift(cd, 62);
    e.v[0] = i128_to_u64(ce) & M62; i128_rshift(ce, 62);

    i128_accum_mul(cd, u, d2);
    i128_accum_mul(cd, v, e2);
    i128_accum_mul(ce, q, d2);
    i128_accum_mul(ce, r, e2);
    if (modinfo.modulus.v[2] != 0) {
        i128_accum_mul(cd, modinfo.modulus.v[2], md);
        i128_accum_mul(ce, modinfo.modulus.v[2], me);
    }
    d.v[1] = i128_to_u64(cd) & M62; i128_rshift(cd, 62);
    e.v[1] = i128_to_u64(ce) & M62; i128_rshift(ce, 62);

    i128_accum_mul(cd, u, d3);
    i128_accum_mul(cd, v, e3);
    i128_accum_mul(ce, q, d3);
    i128_accum_mul(ce, r, e3);
    if (modinfo.modulus.v[3] != 0) {
        i128_accum_mul(cd, modinfo.modulus.v[3], md);
        i128_accum_mul(ce, modinfo.modulus.v[3], me);
    }
    d.v[2] = i128_to_u64(cd) & M62; i128_rshift(cd, 62);
    e.v[2] = i128_to_u64(ce) & M62; i128_rshift(ce, 62);

    i128_accum_mul(cd, u, d4);
    i128_accum_mul(cd, v, e4);
    i128_accum_mul(ce, q, d4);
    i128_accum_mul(ce, r, e4);
    i128_accum_mul(cd, modinfo.modulus.v[4], md);
    i128_accum_mul(ce, modinfo.modulus.v[4], me);
    d.v[3] = i128_to_u64(cd) & M62; i128_rshift(cd, 62);
    e.v[3] = i128_to_u64(ce) & M62; i128_rshift(ce, 62);

    d.v[4] = i128_to_i64(cd);
    e.v[4] = i128_to_i64(ce);
}

static void modinv64_update_fg_62(thread modinv64_signed62 &f, thread modinv64_signed62 &g, const thread modinv64_trans2x2 &t) {
    const ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
    const long f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    const long g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];
    const long u = t.u, v = t.v, q = t.q, r = t.r;
    int128_t cf, cg;

    i128_mul(cf, u, f0);
    i128_accum_mul(cf, v, g0);
    i128_mul(cg, q, f0);
    i128_accum_mul(cg, r, g0);
    i128_rshift(cf, 62);
    i128_rshift(cg, 62);

    i128_accum_mul(cf, u, f1);
    i128_accum_mul(cf, v, g1);
    i128_accum_mul(cg, q, f1);
    i128_accum_mul(cg, r, g1);
    f.v[0] = i128_to_u64(cf) & M62; i128_rshift(cf, 62);
    g.v[0] = i128_to_u64(cg) & M62; i128_rshift(cg, 62);

    i128_accum_mul(cf, u, f2);
    i128_accum_mul(cf, v, g2);
    i128_accum_mul(cg, q, f2);
    i128_accum_mul(cg, r, g2);
    f.v[1] = i128_to_u64(cf) & M62; i128_rshift(cf, 62);
    g.v[1] = i128_to_u64(cg) & M62; i128_rshift(cg, 62);

    i128_accum_mul(cf, u, f3);
    i128_accum_mul(cf, v, g3);
    i128_accum_mul(cg, q, f3);
    i128_accum_mul(cg, r, g3);
    f.v[2] = i128_to_u64(cf) & M62; i128_rshift(cf, 62);
    g.v[2] = i128_to_u64(cg) & M62; i128_rshift(cg, 62);

    i128_accum_mul(cf, u, f4);
    i128_accum_mul(cf, v, g4);
    i128_accum_mul(cg, q, f4);
    i128_accum_mul(cg, r, g4);
    f.v[3] = i128_to_u64(cf) & M62; i128_rshift(cf, 62);
    g.v[3] = i128_to_u64(cg) & M62; i128_rshift(cg, 62);

    f.v[4] = i128_to_i64(cf);
    g.v[4] = i128_to_i64(cg);
}

static void modinv64_inv(thread modinv64_signed62 &x, const constant modinv64_modinfo& modinfo) {
    modinv64_signed62 d = {{0, 0, 0, 0, 0}};
    modinv64_signed62 e = {{1, 0, 0, 0, 0}};
    modinv64_signed62 f = modinfo.modulus;
    modinv64_signed62 g = x;
    long zeta = -1;

    for (int i = 0; i < 10; ++i) {
        modinv64_trans2x2 t;
        zeta = modinv64_divsteps_59(zeta, f.v[0], g.v[0], t);
        modinv64_update_de_62(d, e, t, modinfo);
        modinv64_update_fg_62(f, g, t);
    }

    modinv64_normalize_62(d, f.v[4], modinfo);
    x = d;
}

// --- Field Element Conversion and Normalization ---
// Ported from _libsecp256k1/src/field_5x52_impl.h

static void fe_to_signed62(thread modinv64_signed62 &r, const thread fe &a) {
    const ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
    const ulong a0 = a.n[0], a1 = a.n[1], a2 = a.n[2], a3 = a.n[3], a4 = a.n[4];

    r.v[0] = (a0       | a1 << 52) & M62;
    r.v[1] = (a1 >> 10 | a2 << 42) & M62;
    r.v[2] = (a2 >> 20 | a3 << 32) & M62;
    r.v[3] = (a3 >> 30 | a4 << 22) & M62;
    r.v[4] =  a4 >> 40;
}

static void fe_from_signed62(thread fe &r, const thread modinv64_signed62 &a) {
    const ulong M52 = 0xFFFFFFFFFFFFFFFFUL >> 12;
    const ulong a0 = a.v[0], a1 = a.v[1], a2 = a.v[2], a3 = a.v[3], a4 = a.v[4];

    r.n[0] =  a0                   & M52;
    r.n[1] = (a0 >> 52 | a1 << 10) & M52;
    r.n[2] = (a1 >> 42 | a2 << 20) & M52;
    r.n[3] = (a2 >> 32 | a3 << 30) & M52;
    r.n[4] = (a3 >> 22 | a4 << 40);
}

/*
 * Reduce a field element to its normalized form.
 * Ported from `secp256k1_fe_impl_normalize` in _libsecp256k1/src/field_5x52_impl.h
 */
static inline void fe_normalize(thread fe &r) {
    ulong t0 = r.n[0], t1 = r.n[1], t2 = r.n[2], t3 = r.n[3], t4 = r.n[4];
    ulong m;
    ulong x;
    const ulong M52 = 0xFFFFFFFFFFFFFUL;

    x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFULL;

    t0 += x * 0x1000003D1UL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52; m = t1;
    t3 += (t2 >> 52); t2 &= M52; m &= t2;
    t4 += (t3 >> 52); t3 &= M52; m &= t3;

    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFULL) & (m == M52) & (t0 >= 0xFFFFEFFFFFC2FULL));

    t0 += x * 0x1000003D1UL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    t4 &= 0x0FFFFFFFFFFFFULL;

    r.n[0] = t0; r.n[1] = t1; r.n[2] = t2; r.n[3] = t3; r.n[4] = t4;
}

/*
 * Compute the modular inverse of a field element.
 * Ported from `secp256k1_fe_impl_inv` in _libsecp256k1/src/field_5x52_impl.h
 */
static inline void fe_inv(thread fe &r, const thread fe &x) {
    fe tmp = x;
    modinv64_signed62 s;

    fe_normalize(tmp);
    fe_to_signed62(s, tmp);
    modinv64_inv(s, const_modinfo_fe);
    fe_from_signed62(r, s);
}

/* Negate a field element with specified magnitude. Ported from secp256k1_fe_impl_negate_unchecked. */
static inline void fe_negate(thread fe &r, const thread fe &a, int m) {
    const ulong M52 = 0xFFFFFFFFFFFFFUL;
    r.n[0] = 0xFFFFEFFFFFC2FUL * 2 * (m + 1) - a.n[0];
    r.n[1] = M52 * 2 * (m + 1) - a.n[1];
    r.n[2] = M52 * 2 * (m + 1) - a.n[2];
    r.n[3] = M52 * 2 * (m + 1) - a.n[3];
    r.n[4] = 0x0FFFFFFFFFFFFULL * 2 * (m + 1) - a.n[4];
}

/* Check if a field element normalizes to zero. Ported from secp256k1_fe_impl_normalizes_to_zero. */
static inline bool fe_normalizes_to_zero(const thread fe &r) {
    ulong t0 = r.n[0], t1 = r.n[1], t2 = r.n[2], t3 = r.n[3], t4 = r.n[4];
    const ulong M52 = 0xFFFFFFFFFFFFFUL;

    // First, we need to normalize the high bits
    ulong x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFUL;

    // Apply the first reduction step
    t0 += x * 0x1000003D1UL;

    // Propagate carries
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Check if we need another reduction step
    bool needs_extra_reduction = (t4 >> 48) || 
                                ((t4 == 0x0FFFFFFFFFFFFUL) && 
                                 (t3 == M52) && 
                                 (t2 == M52) && 
                                 (t1 == M52) && 
                                 (t0 >= 0xFFFFEFFFFFC2FUL));

    // Apply second reduction if needed
    if (needs_extra_reduction) {
        t0 += 0x1000003D1UL;
        t1 += (t0 >> 52); t0 &= M52;
        t2 += (t1 >> 52); t1 &= M52;
        t3 += (t2 >> 52); t2 &= M52;
        t4 += (t3 >> 52); t3 &= M52;
        t4 &= 0x0FFFFFFFFFFFFUL;
    }

    // Check if this is the modulus or zero
    // The modulus is: { 0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF }
    bool is_modulus = (t0 == 0xFFFFEFFFFFC2FUL) && 
                      (t1 == M52) && 
                      (t2 == M52) && 
                      (t3 == M52) && 
                      (t4 == 0x0FFFFFFFFFFFFUL);
                      
    // Return true if all limbs are zero or if this is the modulus
    return ((t0 | t1 | t2 | t3 | t4) == 0) || is_modulus;
}

/* Create a copy of a field element and subtract another from it. */
static inline fe fe_sub_const(const thread fe &a, const thread fe &b) {
    fe result = a;
    fe_sub(result, b);
    return result;
}
