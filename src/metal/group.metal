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
    ulong infinity;
};

/* A group element in Jacobian coordinates. */
struct gej {
    fe x;
    fe y;
    fe z;
    ulong infinity;
};

// --- Group Operation Functions ---

/* Set a Jacobian group element to the point at infinity. */
static inline void gej_set_infinity(thread gej &r) {
    r.infinity = 1;
    fe_set_int(r.x, 0);
    fe_set_int(r.y, 0);
    fe_set_int(r.z, 0);
}

/* Double a Jacobian group element. Ported from `secp256k1_gej_double`. */
static inline void gej_double(thread gej &r, const thread gej &a) {
    /* Operations: 3 mul, 4 sqr, 8 add/half/mul_int/negate */
    fe l, s, t;

    r.infinity = a.infinity;

    /* Formula used:
     * L = (3/2) * X1^2
     * S = Y1^2
     * T = -X1*S
     * X3 = L^2 + 2*T
     * Y3 = -(L*(X3 + T) + S^2)
     * Z3 = Y1*Z1
     */

    fe_mul(r.z, a.z, a.y);           /* Z3 = Y1*Z1 (1) */
    fe_sqr(s, a.y);                  /* S = Y1^2 (1) */
    fe_sqr(l, a.x);                  /* L = X1^2 (1) */
    fe_mul_int(l, 3);                /* L = 3*X1^2 (3) */
    fe_half(l);                      /* L = 3/2*X1^2 (2) */
    fe_negate(t, s, 1);              /* T = -S (2) */
    fe_mul(t, t, a.x);               /* T = -X1*S (1) */
    fe_sqr(r.x, l);                  /* X3 = L^2 (1) */
    fe_add(r.x, t);                  /* X3 = L^2 + T (2) */
    fe_add(r.x, t);                  /* X3 = L^2 + 2*T (3) */
    fe_sqr(s, s);                    /* S' = S^2 (1) */
    fe_add(t, r.x);                  /* T' = X3 + T (4) */
    fe_mul(r.y, t, l);               /* Y3 = L*(X3 + T) (1) */
    fe_add(r.y, s);                  /* Y3 = L*(X3 + T) + S^2 (2) */
    fe_negate(r.y, r.y, 2);          /* Y3 = -(L*(X3 + T) + S^2) (3) */
}

/* Set a Jacobian group element from an affine one. */
static inline void gej_set_ge(thread gej &r, const thread ge &a) {
    r.infinity = a.infinity;
    if (r.infinity != 0) {
        gej_set_infinity(r);
        return;
    }
    r.x = a.x;
    r.y = a.y;
    fe_set_int(r.z, 1);
}

/* Set an affine group element from a Jacobian one. */
static inline void ge_set_gej(thread ge &r, const thread gej &a) {
    if (a.infinity != 0) {
        r.infinity = 1;
        fe_set_int(r.x, 0);
        fe_set_int(r.y, 0);
        return;
    }
    r.infinity = 0;

    // Check if z is zero (which would indicate infinity)
    fe zero;
    fe_set_int(zero, 0);
    bool z_is_zero = true;
    for (int i = 0; i < 5; i++) {
        if (a.z.n[i] != 0) {
            z_is_zero = false;
            break;
        }
    }
    
    if (z_is_zero) {
        r.infinity = 1;
        fe_set_int(r.x, 0);
        fe_set_int(r.y, 0);
        return;
    }

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
    r.infinity = flag ? a.infinity : r.infinity;
}

/* Add a Jacobian group element to an affine one. Ported from `secp256k1_gej_add_ge`. */
static inline void gej_add_ge(thread gej &r, const thread gej &a, const thread ge &b) {
    // Handle infinity cases first
    if (a.infinity != 0) {
        if (b.infinity != 0) {
            gej_set_infinity(r);
            return;
        }
        gej_set_ge(r, b);
        return;
    }
    
    if (b.infinity != 0) {
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

    // First check if m_alt (u1-u2) is zero, which means x-coordinates match
    bool x_equal = fe_normalizes_to_zero(m_alt);
    
    // If x-coordinates match, check if y-coordinates are opposites
    bool y_opposite = false;
    if (x_equal) {
        // If y-coordinates are opposites, s1 + s2 will be 0.
        fe y_sum = s1;
        fe_add(y_sum, s2);
        y_opposite = fe_normalizes_to_zero(y_sum);
    }

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

    // Set infinity flag if points are inverses (same x, opposite y) or if z normalizes to zero
    r.infinity = ((x_equal && y_opposite) || fe_normalizes_to_zero(r.z)) ? 1 : 0;
}
