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
 * Limb-wise subtraction for 256-bit integers (r -= a).
 * This is a basic big-integer operation; it does not perform modular reduction.
 */
static inline void fe_sub(thread fe &r, const thread fe &a) {
    r.n[0] -= a.n[0];
    r.n[1] -= a.n[1];
    r.n[2] -= a.n[2];
    r.n[3] -= a.n[3];
    r.n[4] -= a.n[4];
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

//
// --- Stage 3: Elliptic Curve (Group) Operations ---
//
// This section implements the group operations for secp256k1, such as
// point addition and doubling. It uses the field arithmetic from Stage 2.
//
// Original source:
// - _libsecp256k1/src/group_impl.h
//

/* A group element in affine coordinates. */
struct ge {
    fe x;
    fe y;
    bool infinity;
};

/* A group element in Jacobian coordinates. */
struct gej {
    fe x;
    fe y;
    fe z;
    bool infinity;
};

// --- Field Element Constants ---
constant fe fe_one = {{1, 0, 0, 0, 0}};

// --- Additional Field Element Helpers ---

/* Set a field element to a small integer. */
static inline void fe_set_int(thread fe &r, int a) {
    r.n[0] = a;
    r.n[1] = 0;
    r.n[2] = 0;
    r.n[3] = 0;
    r.n[4] = 0;
}

/* Negate a field element. Note: this is a simplified, non-optimized version. */
static inline void fe_negate(thread fe &r, const thread fe &a) {
    fe zero = {{0,0,0,0,0}};
    r = zero;
    fe_sub(r, a);
    fe_normalize(r);
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
    constant ulong one = 1;
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

/* Check if a field element normalizes to zero. */
static inline bool fe_normalizes_to_zero(thread fe &r) {
    ulong t0 = r.n[0], t1 = r.n[1], t2 = r.n[2], t3 = r.n[3], t4 = r.n[4];
    ulong z0, z1;
    constant ulong M52 = 0xFFFFFFFFFFFFFUL;

    ulong x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFUL;

    t0 += x * 0x1000003D1UL;
    t1 += (t0 >> 52); t0 &= M52; z0  = t0; z1  = t0 ^ 0x1000003D0UL;
    t2 += (t1 >> 52); t1 &= M52; z0 |= t1; z1 &= t1;
    t3 += (t2 >> 52); t2 &= M52; z0 |= t2; z1 &= t2;
    t4 += (t3 >> 52); t3 &= M52; z0 |= t3; z1 &= t3;
                                z0 |= t4; z1 &= t4 ^ 0xF000000000000UL;

    return (z0 == 0) | (z1 == M52);
}

// --- Group Operation Functions ---

/* Set a Jacobian group element to the point at infinity. */
static inline void gej_set_infinity(thread gej &r) {
    r.infinity = true;
    fe_set_int(r.x, 0);
    fe_set_int(r.y, 0);
    fe_set_int(r.z, 0);
}

/* Double a Jacobian group element. Ported from `secp256k1_gej_double`. */
static inline void gej_double(thread gej &r, const thread gej &a) {
    r.infinity = a.infinity;
    if (r.infinity) {
        return;
    }

    fe l, s, t;

    fe_mul(r.z, a.z, a.y);
    fe_sqr(s, a.y);
    fe_sqr(l, a.x);
    fe_mul_int(l, 3);
    fe_half(l);
    fe_negate(t, s);
    fe_mul(t, t, a.x);
    fe_sqr(r.x, l);
    fe_add(r.x, t);
    fe_add(r.x, t);
    fe_sqr(s, s);
    fe t_prime = r.x;
    fe_add(t_prime, t);
    fe_mul(r.y, t_prime, l);
    fe_add(r.y, s);
    fe temp_y = r.y;
    fe_negate(r.y, temp_y);
}

/* Add a Jacobian group element to an affine one. Ported from `secp256k1_gej_add_ge`. */
static inline void gej_add_ge(thread gej &r, const thread gej &a, const thread ge &b) {
    if (a.infinity) {
        r.infinity = b.infinity;
        if (!b.infinity) {
            r.x = b.x;
            r.y = b.y;
            r.z = fe_one;
        }
        return;
    }
    if (b.infinity) {
        r = a;
        return;
    }

    fe zz, u1, u2, s1, s2, t, tt, m, n, q, rr;
    fe m_alt, rr_alt;
    bool degenerate;

    fe_sqr(zz, a.z);
    u1 = a.x;
    fe_mul(u2, b.x, zz);
    s1 = a.y;
    fe_mul(s2, b.y, zz);
    fe_mul(s2, s2, a.z);
    t = u1; fe_add(t, u2);
    m = s1; fe_add(m, s2);
    fe_sqr(rr, t);
    fe_negate(m_alt, u2);
    fe_mul(tt, u1, m_alt);
    fe_add(rr, tt);

    degenerate = fe_normalizes_to_zero(m);

    rr_alt = s1;
    fe_mul_int(rr_alt, 2);
    m_alt = u1;
    fe_sub(m_alt, u2);

    fe_cmov(rr_alt, rr, !degenerate);
    fe_cmov(m_alt, m, !degenerate);

    fe_sqr(n, m_alt);
    q = t; fe_negate(q, q);
    fe_mul(q, q, n);

    fe_sqr(n, n);
    fe_cmov(n, m, degenerate);
    fe_sqr(t, rr_alt);
    fe_mul(r.z, a.z, m_alt);
    fe_add(t, q);
    r.x = t;
    fe_mul_int(t, 2);
    fe_add(t, q);
    fe_mul(t, t, rr_alt);
    fe_add(t, n);
    fe_negate(r.y, t);
    fe_half(r.y);

    fe_cmov(r.x, b.x, a.infinity);
    fe_cmov(r.y, b.y, a.infinity);
    fe_cmov(r.z, fe_one, a.infinity);

    r.infinity = fe_normalizes_to_zero(r.z);
}

//
// --- Stage 2: Modular (Field) Arithmetic ---
//
// This section implements modular reduction and inversion, building on the
// big-integer arithmetic from Stage 1.
//
// Original source:
// - _libsecp256k1/src/field_5x52_impl.h
// - _libsecp256k1/src/modinv64_impl.h
//

// --- 128-bit Signed Integer Helpers ---
// Ported from _libsecp256k1/src/int128_native_impl.h

// A 128-bit signed integer type, available in Metal.
using int128_t = metal::int128_t;

static inline void i128_mul(thread int128_t &r, long a, long b) {
    r = (int128_t)a * b;
}

static inline void i128_accum_mul(thread int128_t &r, long a, long b) {
    r += (int128_t)a * b;
}

static inline void i128_rshift(thread int128_t &r, uint n) {
    r >>= n;
}

static inline ulong i128_to_u64(const thread int128_t &a) {
    return (ulong)a;
}

static inline long i128_to_i64(const thread int128_t &a) {
    return (long)a;
}

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
static void modinv64_normalize_62(thread modinv64_signed62 &r_in, long sign, constant modinv64_modinfo& modinfo) {
    constant long M62 = (long)(0xFFFFFFFFFFFFFFFFUL >> 2);
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

static void modinv64_update_de_62(thread modinv64_signed62 &d, thread modinv64_signed62 &e, const thread modinv64_trans2x2 &t, constant modinv64_modinfo& modinfo) {
    constant ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
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
    constant ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
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

static void modinv64_inv(thread modinv64_signed62 &x, constant modinv64_modinfo& modinfo) {
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
    constant ulong M62 = 0xFFFFFFFFFFFFFFFFUL >> 2;
    const ulong a0 = a.n[0], a1 = a.n[1], a2 = a.n[2], a3 = a.n[3], a4 = a.n[4];

    r.v[0] = (a0       | a1 << 52) & M62;
    r.v[1] = (a1 >> 10 | a2 << 42) & M62;
    r.v[2] = (a2 >> 20 | a3 << 32) & M62;
    r.v[3] = (a3 >> 30 | a4 << 22) & M62;
    r.v[4] =  a4 >> 40;
}

static void fe_from_signed62(thread fe &r, const thread modinv64_signed62 &a) {
    constant ulong M52 = 0xFFFFFFFFFFFFFFFFUL >> 12;
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
    constant ulong M52 = 0xFFFFFFFFFFFFFUL;

    x = t4 >> 48;
    t4 &= 0x0FFFFFFFFFFFFUL;

    t0 += x * 0x1000003D1UL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52; m = t1;
    t3 += (t2 >> 52); t2 &= M52; m &= t2;
    t4 += (t3 >> 52); t3 &= M52; m &= t3;

    x = (t4 >> 48) | ((t4 == 0x0FFFFFFFFFFFFUL) & (m == M52) & (t0 >= 0xFFFFEFFFFFC2FUL));

    t0 += x * 0x1000003D1UL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    t4 &= 0x0FFFFFFFFFFFFUL;

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
