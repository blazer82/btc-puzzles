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