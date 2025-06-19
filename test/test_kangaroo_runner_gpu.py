import pytest
import numpy as np

import cryptography_utils as crypto
from kangaroo_runner_gpu import KangarooRunnerGPU
from kangaroo_runner_cpu import KangarooRunnerCPU
import metal_utils as mu


@pytest.fixture
def simple_puzzle():
    """
    Provides a simple puzzle definition where the private key is 3.
    The search range is small, [2, 4], ensuring the midpoint strategy
    starts nearby.
    """
    target_privkey = 3
    pubkey_hex = crypto.scalar_multiply(
        target_privkey).format(compressed=True).hex()
    return {
        "puzzle_number": 3,
        "public_key": pubkey_hex,
        "range_start": "0x2",
        "range_end": "0x4",
    }


@pytest.fixture
def puzzle_5_def():
    """Loads puzzle #5 definition."""
    import config_manager as cm
    # Assumes puzzles.json is in the root, and tests are run from root.
    return cm.load_puzzle_definition(5, 'puzzles.json')


@pytest.fixture
def puzzle_35_def():
    """Loads puzzle #35 definition."""
    import config_manager as cm
    return cm.load_puzzle_definition(35, 'puzzles.json')


@pytest.fixture
def fast_profile():
    """
    Provides a solver profile designed for fast testing.
    - A small number of walkers.
    - A low DP threshold (0) makes every point distinguished, guaranteeing
      frequent trap checks and a quick collision.
    """
    return {
        "num_walkers": "2",
        "distinguished_point_threshold": "2",
        "start_point_strategy": "midpoint",
    }


@pytest.mark.gpu
class TestKangarooRunner:
    def test_initialization(self, simple_puzzle, fast_profile):
        """Tests that the runner initializes correctly with GPU data structures."""
        try:
            runner = KangarooRunnerGPU(simple_puzzle, fast_profile)
        except RuntimeError as e:
            pytest.skip(f"Skipping GPU test: {e}")

        num_walkers = int(fast_profile['num_walkers'])

        # Check herd array shapes
        assert runner.tame_kangaroos_np.shape == (num_walkers, 16)
        assert runner.wild_kangaroos_np.shape == (num_walkers, 16)
        assert runner.tame_distances_np.shape == (num_walkers,)
        assert runner.wild_distances_np.shape == (num_walkers,)

        # Check tame herd starting key
        expected_start_key = (2 + 4) // 2
        assert runner.start_key_tame == expected_start_key

        # Check that warm-up hops differentiate kangaroos
        # With 2 walkers, walker 0 does 0 warm-up hops, walker 1 does 1.
        assert runner.tame_distances_np[0] == 0
        assert runner.wild_distances_np[0] == 0
        assert runner.tame_distances_np[1] > 0
        assert runner.wild_distances_np[1] > 0

        # Check that the points for walker 1 are different from walker 0
        assert not np.array_equal(
            runner.tame_kangaroos_np[0], runner.tame_kangaroos_np[1])
        assert not np.array_equal(
            runner.wild_kangaroos_np[0], runner.wild_kangaroos_np[1])

        # Check total hops after warm-up
        expected_hops = 2 * sum(range(num_walkers))
        assert runner.get_total_hops_performed() == expected_hops

    def test_solve_simple_puzzle_e2e(self, simple_puzzle, fast_profile):
        """
        Performs an end-to-end test, solving a simple puzzle.
        This verifies that the entire GPU-based workflow (hop, convert, check)
        is functioning correctly and can find a known private key.
        """
        try:
            runner = KangarooRunnerGPU(simple_puzzle, fast_profile)
        except RuntimeError as e:
            pytest.skip(f"Skipping GPU test: {e}")

        solution = None
        max_steps = 1000  # Safety break to prevent infinite loops in tests
        for _ in range(max_steps):
            solution = runner.step()
            if solution is not None:
                break

        assert solution is not None, "Solver failed to find a solution within max_steps"

        # The private key for the puzzle is 3
        expected_solution = 3
        assert solution == expected_solution

    def test_solve_puzzle_5_e2e(self, puzzle_5_def, fast_profile):
        """
        Performs an end-to-end test, solving puzzle #5.
        This verifies that the GPU workflow can solve a puzzle from the set.
        """
        try:
            runner = KangarooRunnerGPU(puzzle_5_def, fast_profile)
        except RuntimeError as e:
            pytest.skip(f"Skipping GPU test: {e}")

        solution = None
        # With a low DP threshold, this should be found very quickly.
        max_steps = 10000
        for i in range(max_steps):
            solution = runner.step()
            if solution is not None:
                break

        assert solution is not None, f"Solver failed to find a solution for puzzle #5 within {max_steps} steps"

        # The private key for puzzle #5 is 21.
        expected_solution = 21
        assert solution == expected_solution


    def test_gpu_vs_cpu_multi_hop_by_hop(self, puzzle_35_def):
        """
        Compares the state of multiple kangaroos in parallel between GPU and CPU.
        This tests for any cross-thread interference during a kernel run.
        """
        num_walkers_to_test = 2
        profile = {
            "num_walkers": str(num_walkers_to_test),
            "distinguished_point_threshold": "0",
            "start_point_strategy": "midpoint",
        }

        try:
            gpu_runner = KangarooRunnerGPU(puzzle_35_def, profile)
        except (RuntimeError, Exception) as e:
            pytest.skip(f"Skipping GPU test: {e}")

        cpu_runner = KangarooRunnerCPU(puzzle_35_def, profile)
        cpu_tame_kangaroos = cpu_runner.tame_kangaroos
        cpu_wild_kangaroos = cpu_runner.wild_kangaroos

        num_hops_to_test = 1000
        for i in range(num_hops_to_test):
            # 1. Advance CPU kangaroos (both herds)
            for k in cpu_tame_kangaroos:
                k.hop(cpu_runner.precomputed_hops)
            for k in cpu_wild_kangaroos:
                k.hop(cpu_runner.precomputed_hops)

            # 2. Advance GPU kangaroos (both herds) in single kernel calls
            gpu_runner.metal_runner.run_kernel(
                "kangaroo_kernels", "batch_hop_kangaroos", num_walkers_to_test,
                [gpu_runner.tame_kangaroos_np, gpu_runner.tame_distances_np,
                 gpu_runner.precomputed_hops_np, gpu_runner.num_hops_np]
            )
            gpu_runner.metal_runner.run_kernel(
                "kangaroo_kernels", "batch_hop_kangaroos", num_walkers_to_test,
                [gpu_runner.wild_kangaroos_np, gpu_runner.wild_distances_np,
                 gpu_runner.precomputed_hops_np, gpu_runner.num_hops_np]
            )

            # 3. Compare states for both herds
            tame_affine_np = np.zeros(
                (num_walkers_to_test, 11), dtype=np.uint64)
            wild_affine_np = np.zeros(
                (num_walkers_to_test, 11), dtype=np.uint64)
            gpu_runner.metal_runner.run_kernel(
                "kangaroo_kernels", "batch_ge_set_gej", num_walkers_to_test,
                [gpu_runner.tame_kangaroos_np, tame_affine_np]
            )
            gpu_runner.metal_runner.run_kernel(
                "kangaroo_kernels", "batch_ge_set_gej", num_walkers_to_test,
                [gpu_runner.wild_kangaroos_np, wild_affine_np]
            )

            for k_idx in range(num_walkers_to_test):
                # Compare tame kangaroos
                cpu_point, cpu_dist = cpu_tame_kangaroos[k_idx].get_state()
                gpu_dist = gpu_runner.tame_distances_np[k_idx]
                gpu_point_xy = mu.ge_struct_to_point(tame_affine_np[k_idx])

                assert cpu_dist == gpu_dist, f"Tame distance mismatch on hop {i+1} for k {k_idx}"
                assert cpu_point.point(
                ) == gpu_point_xy, f"Tame point mismatch on hop {i+1} for k {k_idx}"

                # Compare wild kangaroos
                cpu_point, cpu_dist = cpu_wild_kangaroos[k_idx].get_state()
                gpu_dist = gpu_runner.wild_distances_np[k_idx]
                gpu_point_xy = mu.ge_struct_to_point(wild_affine_np[k_idx])

                assert cpu_dist == gpu_dist, f"Wild distance mismatch on hop {i+1} for k {k_idx}"
                assert cpu_point.point(
                ) == gpu_point_xy, f"Wild point mismatch on hop {i+1} for k {k_idx}"
