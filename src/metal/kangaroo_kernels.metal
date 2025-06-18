/*
 * Bitcoin Puzzle Solver - Kangaroo hopping kernels for Metal.
 *
 * This file contains the high-level kernels for executing the Pollard's
 * Kangaroo algorithm on the GPU.
 */

#include <metal_stdlib>
#include "field.metal"

using namespace metal;

/*
 * Executes one hop for a batch of kangaroos in parallel.
 *
 * Each thread in the grid corresponds to a single kangaroo. It reads the
 * kangaroo's current state (point and distance), determines the next hop
 * based on the point's x-coordinate, updates the state, and writes it back
 * to the buffers.
 *
 * This kernel modifies the kangaroos and distances buffers in-place.
 */
kernel void batch_hop_kangaroos(
    device gej* kangaroos [[buffer(0)]],
    device ulong* distances [[buffer(1)]],
    device const ge* precomputed_hops [[buffer(2)]],
    device const uint* num_precomputed_hops [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Get current state for this thread's kangaroo
    gej p = kangaroos[gid];
    ulong d = distances[gid];

    // Convert to affine to get the true x-coordinate for hop selection.
    ge p_aff;
    ge_set_gej(p_aff, p);

    // Select hop based on the first limb of the AFFINE X coordinate.
    // This matches the logic of the working CPU implementation.
    uint hop_index = p_aff.x.n[0] % num_precomputed_hops[0];

    // Get hop data
    ge hop_point = precomputed_hops[hop_index];
    ulong hop_distance = 1UL << hop_index;

    // Update state by adding the affine hop point to our Jacobian point
    gej_add_ge(p, p, hop_point);
    d += hop_distance;

    // Write back results
    kangaroos[gid] = p;
    distances[gid] = d;
}

/*
 * Converts a batch of Jacobian points to affine points.
 *
 * Each thread reads one Jacobian point, converts it to affine coordinates
 * using the formula (X, Y, Z) -> (X/Z^2, Y/Z^3), and writes the result
 * to the output buffer.
 */
kernel void batch_ge_set_gej(
    device const gej* jacobian_points [[buffer(0)]],
    device ge* affine_points [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    gej p_jac = jacobian_points[gid];
    ge p_aff;
    ge_set_gej(p_aff, p_jac);
    affine_points[gid] = p_aff;
}
