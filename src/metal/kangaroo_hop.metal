/*
 * Bitcoin Puzzle Solver - Final Kangaroo Hop Kernel for Metal.
 *
 * This file corresponds to Stage 4 of the GPU implementation plan. It assembles
 * all the cryptographic primitives from the previous stages into a single
 * compute kernel that performs one parallel hop for all kangaroos.
 */

#include <metal_stdlib>
#include "field.metal"

using namespace metal;

// --- Data Structures for Kernel I/O ---

/*
 * Structure to hold the result of a distinguished point find.
 * This is what the GPU kernel writes back to the CPU.
 */
struct DPResult {
    ge point;
    ulong distance;
};

// --- Main Kangaroo Hop Kernel ---

/*
 * Performs one parallel hop for a grid of kangaroos.
 * Each thread in the grid corresponds to a single kangaroo.
 */
kernel void kangaroo_step(
    // Input/Output: Kangaroo state buffers
    device ge *current_points     [[buffer(0)]],
    device ulong *distances        [[buffer(1)]],

    // Input: Read-only data
    constant ge *hop_points       [[buffer(2)]],
    constant uint &dp_threshold   [[buffer(3)]],

    // Output: For distinguished points
    device atomic_uint *dp_counter [[buffer(4)]],
    device DPResult *dp_output_buffer [[buffer(5)]],

    // Thread identifier
    uint gid [[thread_position_in_grid]]
) {
    // 1. Load current state for this kangaroo (thread).
    ge current_point = current_points[gid];
    ulong current_distance = distances[gid];

    // 2. Select which hop to take based on the current point's x-coordinate.
    // This is a port of hop_strategy.select_hop_index.
    // We use the lowest 52 bits of the x-coordinate as a pseudo-random value.
    uint hop_index = current_point.x.n[0] % 16;

    // 3. Perform the elliptic curve hop: new_point = current_point + hop_point.
    // Fetch the pre-computed hop point.
    ge hop_point = hop_points[hop_index];

    // Convert current point to Jacobian coordinates for efficient addition.
    gej current_point_j;
    gej_set_ge(current_point_j, current_point);

    // Perform the addition.
    gej_add_ge(current_point_j, current_point_j, hop_point);

    // Convert the result back to affine coordinates.
    ge new_point;
    ge_set_gej(new_point, current_point_j);

    // 4. Update distance.
    ulong hop_distance = 1UL << hop_index;
    ulong new_distance = current_distance + hop_distance;

    // 5. Check if the new point is distinguished.
    // This is a port of distinguished_points.is_distinguished.
    // The mask is for the lower `dp_threshold` bits.
    ulong dp_mask = (dp_threshold > 0) ? ((1UL << dp_threshold) - 1) : 0;
    bool is_dp = (dp_threshold == 0) || ((new_point.x.n[0] & dp_mask) == 0);

    // 6. Write back the new state for the next iteration.
    current_points[gid] = new_point;
    distances[gid] = new_distance;

    // 7. If it's a distinguished point, report it back to the CPU.
    if (is_dp && !new_point.infinity) {
        // Atomically get a unique index to write the result to.
        uint result_idx = atomic_fetch_add_explicit(dp_counter, 1, memory_order_relaxed);

        // Write the results to the output buffer.
        dp_output_buffer[result_idx].point = new_point;
        dp_output_buffer[result_idx].distance = new_distance;
    }
}
