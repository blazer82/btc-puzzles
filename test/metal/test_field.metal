#include <metal_stdlib>
#include "../../src/metal/field.metal"

using namespace metal;

/*
 * A simple kernel to test the fe_mul function. It takes two input field
 * elements and writes their product to an output buffer.
 */
kernel void test_fe_mul(
    device const fe* a [[buffer(0)]],
    device const fe* b [[buffer(1)]],
    device fe* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // Only run for the first thread.
    if (gid == 0) {
        fe a_thread = a[0];
        fe b_thread = b[0];
        fe r_thread;
        fe_mul(r_thread, a_thread, b_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_normalize.
 */
kernel void test_fe_normalize(
    device const fe* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = a[0];
        fe_normalize(r_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_sqr.
 */
kernel void test_fe_sqr(
    device const fe* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe a_thread = a[0];
        fe r_thread;
        fe_sqr(r_thread, a_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test ge_set_gej (convert Jacobian to affine).
 */
kernel void test_ge_set_gej(
    device const gej* a [[buffer(0)]],
    device ge* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej a_thread = a[0];
        ge r_thread;
        ge_set_gej(r_thread, a_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_inv.
 */
kernel void test_fe_inv(
    device const fe* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe a_thread = a[0];
        fe r_thread;
        fe_inv(r_thread, a_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test gej_double (Jacobian point doubling).
 */
kernel void test_gej_double(
    device const gej* a [[buffer(0)]],
    device gej* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej a_thread = a[0];
        gej r_thread;
        gej_double(r_thread, a_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test gej_add_ge (add affine point to Jacobian point).
 */
kernel void test_gej_add_ge(
    device const gej* a [[buffer(0)]],
    device const ge* b  [[buffer(1)]],
    device gej* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej a_thread = a[0];
        ge b_thread = b[0];
        gej r_thread;
        gej_add_ge(r_thread, a_thread, b_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test gej_set_infinity.
 */
kernel void test_gej_set_infinity(
    device gej* r [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej r_thread;
        gej_set_infinity(r_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_add.
 */
kernel void test_fe_add(
    device const fe* a [[buffer(0)]],
    device const fe* b [[buffer(1)]],
    device fe* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = a[0];
        fe b_thread = b[0];
        fe_add(r_thread, b_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_sub.
 */
kernel void test_fe_sub(
    device const fe* a [[buffer(0)]],
    device const fe* b [[buffer(1)]],
    device fe* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = a[0];
        fe b_thread = b[0];
        fe_sub(r_thread, b_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_mul_int.
 */
kernel void test_fe_mul_int(
    device const fe* a [[buffer(0)]],
    device const int* factor [[buffer(1)]],
    device fe* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = a[0];
        fe_mul_int(r_thread, factor[0]);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_negate.
 */
kernel void test_fe_negate(
    device const fe* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe a_thread = a[0];
        fe r_thread;
        fe_negate(r_thread, a_thread, 1);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_half.
 */
kernel void test_fe_half(
    device const fe* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = a[0];
        fe_half(r_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_cmov.
 */
kernel void test_fe_cmov(
    device const fe* r_in [[buffer(0)]],
    device const fe* a    [[buffer(1)]],
    device const bool* flag [[buffer(2)]],
    device fe* r_out      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread = r_in[0];
        fe a_thread = a[0];
        fe_cmov(r_thread, a_thread, flag[0]);
        r_out[0] = r_thread;
    }
}

/*
 * Kernel to test fe_set_int.
 */
kernel void test_fe_set_int(
    device const int* a [[buffer(0)]],
    device fe* r       [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe r_thread;
        fe_set_int(r_thread, a[0]);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test fe_normalizes_to_zero.
 */
kernel void test_fe_normalizes_to_zero(
    device const fe* a [[buffer(0)]],
    device uint* result [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        fe a_thread = a[0];
        result[0] = fe_normalizes_to_zero(a_thread) ? 1 : 0;
    }
}

/*
 * Kernel to test individual steps of gej_double for debugging.
 * This kernel performs the same operations as gej_double but allows
 * us to inspect intermediate values.
 */
kernel void test_gej_double_debug(
    device const gej* a [[buffer(0)]],
    device fe* debug_values [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej a_thread = a[0];
        fe l, s, t;
        
        // Step 1: S = Y^2
        fe_sqr(s, a_thread.y);
        debug_values[0] = s;
        
        // Step 2: L = X^2
        fe_sqr(l, a_thread.x);
        debug_values[1] = l;
        
        // Step 3: L = 3*L
        fe_mul_int(l, 3);
        debug_values[2] = l;
        
        // Step 4: L = L/2
        fe_half(l);
        debug_values[3] = l;
        
        // Step 5: T = -S
        fe_negate(t, s, 1);
        debug_values[4] = t;
        
        // Step 6: T = T * X
        fe_mul(t, t, a_thread.x);
        debug_values[5] = t;
        
        // Step 7: Z_new = Y * Z
        fe z_new;
        fe_mul(z_new, a_thread.z, a_thread.y);
        debug_values[6] = z_new;
    }
}

/*
 * Kernel to test multiple point operations in sequence.
 * This tests a common pattern: double a point and add another point to it.
 */
kernel void test_point_double_and_add(
    device const gej* p [[buffer(0)]],
    device const ge* q [[buffer(1)]],
    device gej* r [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej p_thread = p[0];
        ge q_thread = q[0];
        gej r_thread;
        
        // Double p
        gej_double(r_thread, p_thread);
        
        // Add q to the result
        gej_add_ge(r_thread, r_thread, q_thread);
        
        r[0] = r_thread;
    }
}

/*
 * Kernel to test gej_cmov.
 */
kernel void test_gej_cmov(
    device const gej* r_in [[buffer(0)]],
    device const gej* a    [[buffer(1)]],
    device const bool* flag [[buffer(2)]],
    device gej* r_out      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej r_thread = r_in[0];
        gej a_thread = a[0];
        gej_cmov(r_thread, a_thread, flag[0]);
        r_out[0] = r_thread;
    }
}

/*
 * Kernel to test gej_set_ge.
 */
kernel void test_gej_set_ge(
    device const ge* a [[buffer(0)]],
    device gej* r      [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        ge a_thread = a[0];
        gej r_thread;
        gej_set_ge(r_thread, a_thread);
        r[0] = r_thread;
    }
}

/*
 * Kernel to test a sequence of operations that would be used in scalar multiplication.
 * This tests the pattern: P = 2*P + Q
 */
kernel void test_scalar_mul_step(
    device const gej* p [[buffer(0)]],
    device const ge* q [[buffer(1)]],
    device gej* r       [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej p_thread = p[0];
        ge q_thread = q[0];
        gej doubled_p;
        gej r_thread;

        // Double p
        gej_double(doubled_p, p_thread);
        
        // Add q
        gej_add_ge(r_thread, doubled_p, q_thread);
        
        // Write back the result
        r[0] = r_thread;
    }
}

/*
 * Kernel to test edge cases in point addition.
 * This tests adding a point to itself, which should give the same result as doubling.
 */
kernel void test_add_point_to_itself(
    device const gej* p [[buffer(0)]],
    device gej* r_add [[buffer(1)]],
    device gej* r_double [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0) {
        gej p_thread = p[0];
        ge p_affine;
        
        // Convert p to affine
        ge_set_gej(p_affine, p_thread);
        
        // Add p to itself using gej_add_ge
        gej p_jac_from_affine;
        gej_set_ge(p_jac_from_affine, p_affine);
        gej r_add_thread;
        gej_add_ge(r_add_thread, p_jac_from_affine, p_affine);
        
        // Double p using gej_double
        gej r_double_thread;
        gej_double(r_double_thread, p_thread);
        
        // Write results
        r_add[0] = r_add_thread;
        r_double[0] = r_double_thread;
    }
}
