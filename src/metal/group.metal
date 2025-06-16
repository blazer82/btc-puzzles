/*
 * Bitcoin Puzzle Solver - Elliptic curve group operations for Metal.
 *
 * This file corresponds to Stage 3 of the GPU implementation plan. It implements
 * the group operations for secp256k1, such as point addition and doubling.
 * It uses the field arithmetic from Stages 1 and 2.
 *
 * Original source:
 * - _libsecp256k1/src/group_impl.h
 */

#include <metal_stdlib>
#include "field_basic.metal"
#include "field_modular.metal"

using namespace metal;

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
    fe_negate_raw(t, s);
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
    fe_negate_raw(r.y, temp_y);
}

/* Set a Jacobian group element from an affine one. */
static inline void gej_set_ge(thread gej &r, const thread ge &a) {
    r.infinity = a.infinity;
    r.x = a.x;
    r.y = a.y;
    fe_set_int(r.z, 1);
}

/* Conditionally move a Jacobian group element. */
static inline void gej_cmov(thread gej &r, const thread gej &a, bool flag) {
    fe_cmov(r.x, a.x, flag);
    fe_cmov(r.y, a.y, flag);
    fe_cmov(r.z, a.z, flag);
    r.infinity = select(r.infinity, a.infinity, flag);
}

/* Add a Jacobian group element to an affine one. Ported from `secp256k1_gej_add_ge`. */
static inline void gej_add_ge(thread gej &r, const thread gej &a, const thread ge &b) {
    // This function is a port of a constant-time implementation. It avoids branching
    // on the inputs by computing the addition and then using cmov to select the
    // correct result if one of the inputs was infinity.

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
    fe_negate_raw(m_alt, u2);
    fe_mul(tt, u1, m_alt);
    fe_add(rr, tt);

    degenerate = fe_normalizes_to_zero(m);

    rr_alt = s1;
    fe_mul_int(rr_alt, 2);
    // In the C source, m_alt (which holds -u2) is reused by adding u1.
    // We replicate that here to keep the logic identical.
    fe_add(m_alt, u1); // m_alt is now u1 - u2

    fe_cmov(rr_alt, rr, !degenerate);
    fe_cmov(m_alt, m, !degenerate);

    fe_sqr(n, m_alt);
    q = t; fe_negate_raw(q, q);
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
    fe_negate_raw(r.y, t);
    fe_half(r.y);

    // The calculated infinity flag is for the case a = -b.
    r.infinity = fe_normalizes_to_zero(r.z);

    // Now, handle cases where a or b are infinity using cmov.
    gej r_calc = r;

    // If a is infinity, result is b.
    gej b_gej;
    gej_set_ge(b_gej, b);
    gej_cmov(r_calc, b_gej, a.infinity);

    // If b is infinity, result is a.
    // This happens after the 'a is infinity' check, so if both are
    // infinity, the result is 'a' (which is infinity), which is correct.
    gej a_copy = a;
    gej_cmov(r_calc, a_copy, b.infinity);

    r = r_calc;
}
