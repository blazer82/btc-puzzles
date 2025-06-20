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
    constant const ge* precomputed_hops [[buffer(2)]],
    constant const uint* num_precomputed_hops [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Get current state for this thread's kangaroo
    gej current_p = kangaroos[gid];
    ulong d = distances[gid];

    // Convert to affine to get the true x-coordinate for hop selection.
    ge p_aff;
    ge_set_gej(p_aff, current_p);

    // Select hop using hash-based selection to ensure strong path differentiation.
    // We'll use a simple hash function that combines the x-coordinate with the kangaroo ID.
    // This mirrors the CPU's SHA-256 approach but uses a simpler hash suitable for GPU.
    ulong x_coord = p_aff.x.n[0];  // Use lowest limb as representative coordinate
    ulong combined = x_coord ^ (ulong(gid) << 32);  // XOR with shifted ID for mixing
    // Apply additional mixing to improve distribution
    combined ^= combined >> 16;
    combined *= 0x85ebca6b;  // Multiply by a large prime for better distribution
    combined ^= combined >> 13;
    uint hop_index = uint(combined) % num_precomputed_hops[0];

    // Get hop data
    ge hop_point = precomputed_hops[hop_index];
    ulong hop_distance = 1UL << hop_index;

    // Update state by adding the affine hop point to our Jacobian point
    // Use a temporary variable to avoid potential aliasing issues
    gej next_p;
    gej_add_ge(next_p, current_p, hop_point);
    d += hop_distance;

    // Write back results
    kangaroos[gid] = next_p;
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
