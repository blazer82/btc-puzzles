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
        gej_set_infinity(r);
        return;
    }

    fe l, s, t;

    fe_mul(r.z, a.z, a.y);
    fe_sqr(s, a.y);
    fe_sqr(l, a.x);
    fe_mul_int(l, 3);
    fe_half(l);
    fe_negate(t, s, 1);
    fe_mul(t, t, a.x);
    fe_sqr(r.x, l);
    fe_add(r.x, t);
    fe_add(r.x, t);
    fe_sqr(s, s);
    fe_add(t, r.x);
    fe_mul(r.y, t, l);
    fe_add(r.y, s);
    // Final negation, input magnitude is 2, output is 3.
    fe_negate(r.y, r.y, 2);
}

/* Set a Jacobian group element from an affine one. */
static inline void gej_set_ge(thread gej &r, const thread ge &a) {
    r.infinity = a.infinity;
    r.x = a.x;
    r.y = a.y;
    fe_set_int(r.z, 1);
}

/* Set an affine group element from a Jacobian one. */
static inline void ge_set_gej(thread ge &r, const thread gej &a) {
    if (a.infinity) {
        r.infinity = true;
        fe_set_int(r.x, 0);
        fe_set_int(r.y, 0);
        return;
    }
    r.infinity = false;

    fe z_inv, z_inv_sqr, z_inv_cubed;
    fe_inv(z_inv, a.z);
    fe_sqr(z_inv_sqr, z_inv);
    fe_mul(z_inv_cubed, z_inv_sqr, z_inv);
    
    fe_mul(r.x, a.x, z_inv_sqr);
    fe_mul(r.y, a.y, z_inv_cubed);

    fe_normalize(r.x);
    fe_normalize(r.y);
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
    // Handle infinity cases first
    if (a.infinity) {
        if (b.infinity) {
            gej_set_infinity(r);
            return;
        }
        gej_set_ge(r, b);
        return;
    }
    
    if (b.infinity) {
        r = a;
        return;
    }

    // This function is a port of a constant-time implementation for the non-infinity cases
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
    fe_negate(m_alt, u2, 1);
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
    // In the C code, t has magnitude GEJ_X_M+1=17. The output q has mag 18.
    q = t; fe_negate(q, q, 17);
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
    // In the C code, t has magnitude GEJ_Y_M+2=18. The resulting r.y has mag 19.
    fe_negate(r.y, t, 18);
    fe_half(r.y);

    // The calculated infinity flag is for the case a = -b.
    r.infinity = fe_normalizes_to_zero(r.z);
}
