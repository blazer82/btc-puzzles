"""
Tests for the GpuExecutor class, which manages Metal kernel execution.
"""
import pytest
import numpy as np

from gpu_executor import GpuExecutor
import cryptography_utils as crypto
import hop_strategy as hs


# --- Fixtures ---

@pytest.fixture(scope="module")
def hop_points():
    """Provides the 16 pre-computed hop points for all tests in this module."""
    return hs.generate_precomputed_hops()


@pytest.fixture
def executor(hop_points):
    """Provides a standard GpuExecutor instance with a high DP threshold."""
    # A high threshold prevents accidental DP finds during most tests.
    return GpuExecutor(hop_points=hop_points, dp_threshold=30)


# --- Test Class ---

@pytest.mark.gpu
class TestGpuExecutor:
    """
    Test suite for GpuExecutor. Requires a Metal-compatible GPU.
    """

    def test_int_fe_conversion_roundtrip(self, executor):
        """
        Tests that converting a Python integer to a Metal `fe` struct and
        back results in the original value.
        """
        original_int = 0x1234567890ABCDEF1234567890ABCDEF
        fe_struct = executor._fe_from_int(original_int)
        result_int = executor._int_from_fe(fe_struct)
        assert result_int == original_int

    def test_pubkey_ge_conversion_roundtrip(self, executor):
        """
        Tests that converting a coincurve PublicKey to a Metal `ge` struct
        and back results in the original public key.
        """
        original_pubkey = crypto.get_generator_point()
        ge_struct = executor._ge_from_pubkey(original_pubkey)
        result_pubkey = executor._pubkey_from_ge(ge_struct)
        assert result_pubkey.format() == original_pubkey.format()

    def test_create_state_buffers(self, executor):
        """
        Tests that state buffers are created with the correct size and initial values.
        """
        num_walkers = 10
        initial_points = [crypto.get_generator_point()] * num_walkers
        executor.create_state_buffers(initial_points)

        assert executor.num_walkers == num_walkers
        assert executor.points_buffer is not None
        assert executor.distances_buffer is not None
        assert executor.dp_counter_buffer is not None
        assert executor.dp_output_buffer is not None

        # Verify initial distances are all zero
        distances_buffer = executor.distances_buffer.contents().as_buffer(executor.distances_buffer.length())
        distances_np = np.frombuffer(distances_buffer, dtype=np.uint64)
        assert np.all(distances_np == 0)

    def test_execute_step_modifies_state(self, executor):
        """
        Tests that executing a kernel step modifies the points and distances
        in the GPU buffers.
        """
        # 1. Setup initial state
        g = crypto.get_generator_point()
        executor.create_state_buffers([g])

        # 2. Capture initial state from buffers by copying to a bytes object
        initial_points_bytes = bytes(executor.points_buffer.contents().as_buffer(executor.points_buffer.length()))
        initial_distances_bytes = bytes(executor.distances_buffer.contents().as_buffer(executor.distances_buffer.length()))

        # 3. Execute one step
        executor.execute_step()

        # 4. Capture final state from buffers
        final_points_bytes = bytes(executor.points_buffer.contents().as_buffer(executor.points_buffer.length()))
        final_distances_bytes = bytes(executor.distances_buffer.contents().as_buffer(executor.distances_buffer.length()))

        # 5. Assert that state has changed
        assert initial_points_bytes != final_points_bytes
        assert initial_distances_bytes != final_distances_bytes

    def test_dp_reporting_and_reset(self, hop_points):
        """
        Tests the full distinguished point reporting and reset cycle.
        """
        # 1. Setup an executor that will always find a DP
        executor = GpuExecutor(hop_points=hop_points, dp_threshold=0)
        g = crypto.get_generator_point()
        executor.create_state_buffers([g])

        # 2. Execute one step
        executor.execute_step()

        # 3. Check for the distinguished point
        dps = executor.get_distinguished_points()
        assert len(dps) == 1

        # Verify the reported distance is correct
        # hop_index = x % 16. For G, x is known.
        g_x, _ = g.point()
        hop_index = g_x % 16
        expected_distance = 2**hop_index
        reported_point, reported_distance = dps[0]

        assert reported_distance == expected_distance
        assert reported_point.format() != g.format() # Point should have moved

        # 4. Reset the counter
        executor.reset_dp_counter()

        # 5. Verify no DPs are reported now
        dps_after_reset = executor.get_distinguished_points()
        assert len(dps_after_reset) == 0
