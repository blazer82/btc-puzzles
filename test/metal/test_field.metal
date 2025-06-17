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
