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
        fe_mul(r[0], a[0], b[0]);
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
        r[0] = a[0];
        fe_normalize(r[0]);
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
        fe_sqr(r[0], a[0]);
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
        fe_inv(r[0], a[0]);
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
        gej_double(r[0], a[0]);
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
        gej_add_ge(r[0], a[0], b[0]);
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
        gej_set_infinity(r[0]);
    }
}
