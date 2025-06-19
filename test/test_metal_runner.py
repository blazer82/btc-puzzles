import pytest
import numpy as np
from coincurve.keys import PublicKey

import cryptography_utils as crypto
import hop_strategy as hs
from metal_runner import MetalRunner
import metal_utils as mu
from test_metal_field import MetalTestHelper  # For test utilities


@pytest.mark.gpu
class TestMetalRunner:
    @pytest.fixture(scope="class")
    def metal_runner(self):
        """Fixture to provide a MetalRunner instance, skipping if not available."""
        try:
            return MetalRunner()
        except RuntimeError as e:
            pytest.skip(f"Skipping GPU test: {e}")

    @pytest.fixture(scope="class")
    def metal_field_tester(self):
        """Fixture to provide a MetalTestHelper instance."""
        try:
            return MetalTestHelper("test/metal/test_field.metal")
        except Exception as e:
            pytest.skip(f"Skipping GPU test: {e}")

    def test_initialization(self, metal_runner):
        """Tests that the MetalRunner initializes without errors."""
        assert metal_runner.device is not None
        assert metal_runner.command_queue is not None

    def test_batch_hop_kernel(self, metal_runner, metal_field_tester):
        """
        Tests the 'batch_hop_kangaroos' kernel.
        It verifies that a single kangaroo starting at G hops to the correct
        next point and updates its distance correctly.
        """
        # 1. Compile the kernel library
        metal_runner.compile_library(
            "kangaroo_kernels", "src/metal/kangaroo_kernels.metal")

        # 2. Prepare inputs
        g = crypto.get_generator_point()
        precomputed_hops_list = hs.generate_precomputed_hops()
        num_hops = len(precomputed_hops_list)

        # Buffers for the kernel
        kangaroos_np = np.array(
            [mu.point_to_gej_struct(g)], dtype=np.uint64).reshape(1, 16)
        distances_np = np.array([0], dtype=np.uint64)
        precomputed_hops_np = np.array(
            [mu.point_to_ge_struct(p) for p in precomputed_hops_list], dtype=np.uint64
        )
        num_hops_np = np.array([num_hops], dtype=np.uint32)

        # 3. Calculate expected result on CPU using the full x-coordinate
        full_x_coord = crypto.get_x_coordinate_int(g)
        expected_hop_index = hs.select_hop_index(full_x_coord, num_hops)
        expected_hop_distance = 2**expected_hop_index
        expected_next_point = crypto.point_add(
            g, precomputed_hops_list[expected_hop_index])

        # 4. Run the kernel
        metal_runner.run_kernel(
            "kangaroo_kernels",
            "batch_hop_kangaroos",
            num_threads=1,
            buffers_np=[kangaroos_np, distances_np,
                        precomputed_hops_np, num_hops_np],
        )

        # 5. Verify results
        assert distances_np[0] == expected_hop_distance
        gpu_gej_result = kangaroos_np[0]
        gpu_affine_struct = metal_field_tester.run_kernel(
            "test_ge_set_gej", [gpu_gej_result], 11 * 8
        )
        expected_affine_struct = mu.point_to_ge_struct(expected_next_point)
        assert np.array_equal(gpu_affine_struct, expected_affine_struct)

