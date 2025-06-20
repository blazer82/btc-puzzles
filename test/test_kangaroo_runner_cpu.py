import pytest

import cryptography_utils as crypto
from kangaroo_runner_cpu import KangarooRunnerCPU


@pytest.fixture
def puzzle_def():
    """
    Provides a puzzle definition for testing.
    The puzzle is to find the private key 3, in a tiny range [2, 3].
    The tame herd will start from the midpoint key, 2.
    The wild herd will start from the public key for 3.
    """
    target_privkey = 3
    pubkey_hex = crypto.scalar_multiply(target_privkey).format(compressed=True).hex()
    return {
        "puzzle_number": 5,
        "public_key": pubkey_hex,
        "range_start": "0x2",
        "range_end": "0x3",
    }


@pytest.fixture
def profile_config():
    """Provides a solver profile for testing."""
    return {
        "num_walkers": "2",
        "distinguished_point_threshold": "20",  # High enough to avoid random hits
    }


class TestKangarooRunner:
    def test_initialization(self, puzzle_def, profile_config):
        """Tests that the runner initializes correctly."""
        runner = KangarooRunnerCPU(puzzle_def, profile_config)
        num_walkers = int(profile_config['num_walkers'])

        # Check herd sizes
        assert len(runner.tame_kangaroos) == num_walkers
        assert len(runner.wild_kangaroos) == num_walkers

        # Check DP threshold
        assert runner.dp_threshold == int(profile_config['distinguished_point_threshold'])

        # Check tame herd starting key
        expected_start_key = (2 + 3) // 2
        assert runner.start_key_tame == expected_start_key

        # Check that kangaroos are initialized with unique IDs and zero distance
        assert runner.tame_kangaroos[0].id == 0
        assert runner.tame_kangaroos[1].id == 1
        assert runner.tame_kangaroos[0].distance == 0
        assert runner.tame_kangaroos[1].distance == 0
        assert runner.wild_kangaroos[0].distance == 0
        assert runner.wild_kangaroos[1].distance == 0

        # Check total hops is zero after initialization (no warm-up)
        assert runner.get_total_hops_performed() == 0

    def test_step_updates_total_hops(self, puzzle_def, profile_config):
        """Tests that a single step correctly updates the total hop count."""
        runner = KangarooRunnerCPU(puzzle_def, profile_config)
        initial_hops = runner.get_total_hops_performed()
        assert initial_hops == 0  # No warm-up hops
        num_walkers = int(profile_config['num_walkers'])

        runner.step()

        # Hops should increase by the total number of kangaroos
        expected_new_hops = initial_hops + (2 * num_walkers)
        assert runner.get_total_hops_performed() == expected_new_hops

    def test_collision_tame_finds_wild_in_trap(self, puzzle_def, profile_config, monkeypatch):
        """
        Tests a collision where a tame kangaroo lands on a distinguished point
        that is already in the wild herd's trap.
        """
        # Set a very high DP threshold to prevent accidental finds
        profile_config['distinguished_point_threshold'] = '256'
        runner = KangarooRunnerCPU(puzzle_def, profile_config)

        # 1. Create a fake collision point and add it to the wild trap
        collision_point = crypto.scalar_multiply(12345)
        collision_point_xy = collision_point.point()
        wild_dist = 500
        runner.wild_trap.add_point(collision_point_xy, wild_dist)

        # 2. Force a tame kangaroo to be at this exact position
        tame_k = runner.tame_kangaroos[0]
        tame_dist = 12345 - runner.start_key_tame
        tame_k.current_point = collision_point
        tame_k.distance = tame_dist

        # 3. Patch `is_distinguished` to fire only for our collision point
        collision_x = crypto.get_x_coordinate_int(collision_point)
        monkeypatch.setattr(
            'kangaroo_runner_cpu.dp.is_distinguished',
            lambda x, threshold: x == collision_x
        )

        # 4. Patch `hop` to do nothing, so our manual setup isn't disturbed
        monkeypatch.setattr('kangaroo_runner_cpu.Kangaroo.hop', lambda self, precomputed_hops: None)

        # 5. Execute the step and check for the correct solution
        solution = runner.step()
        assert solution is not None

        n = crypto.get_curve_order_n()
        expected_solution = (runner.start_key_tame + tame_dist - wild_dist) % n
        assert solution == expected_solution

    def test_collision_wild_finds_tame_in_trap(self, puzzle_def, profile_config, monkeypatch):
        """
        Tests a collision where a wild kangaroo lands on a distinguished point
        that is already in the tame herd's trap.
        """
        profile_config['distinguished_point_threshold'] = '256'
        runner = KangarooRunnerCPU(puzzle_def, profile_config)

        # 1. Create a fake collision point and add it to the tame trap
        collision_point = crypto.scalar_multiply(54321)
        collision_point_xy = collision_point.point()
        tame_dist = 54321 - runner.start_key_tame
        runner.tame_trap.add_point(collision_point_xy, tame_dist)

        # 2. Force a wild kangaroo to be at this exact position
        wild_k = runner.wild_kangaroos[0]
        wild_dist = 999  # Arbitrary distance
        wild_k.current_point = collision_point
        wild_k.distance = wild_dist

        # 3. Patch `is_distinguished` to fire only for our collision point
        collision_x = crypto.get_x_coordinate_int(collision_point)
        monkeypatch.setattr(
            'kangaroo_runner_cpu.dp.is_distinguished',
            lambda x, threshold: x == collision_x
        )

        # 4. Patch `hop` to do nothing
        monkeypatch.setattr('kangaroo_runner_cpu.Kangaroo.hop', lambda self, precomputed_hops: None)

        # 5. Execute the step and check for the correct solution
        solution = runner.step()
        assert solution is not None

        n = crypto.get_curve_order_n()
        expected_solution = (runner.start_key_tame + tame_dist - wild_dist) % n
        assert solution == expected_solution

    def test_no_collision_adds_points_to_traps(self, puzzle_def, profile_config, monkeypatch):
        """
        Tests that when distinguished points are found but no collision occurs,
        they are correctly added to their respective traps.
        """
        profile_config['distinguished_point_threshold'] = '256'
        runner = KangarooRunnerCPU(puzzle_def, profile_config)

        # 1. Define two points that will become distinguished
        dp_tame_point = crypto.scalar_multiply(100)
        dp_wild_point = crypto.scalar_multiply(200)

        # 2. Force one tame and one wild kangaroo to be at these positions
        tame_k = runner.tame_kangaroos[0]
        tame_dist = 100 - runner.start_key_tame
        tame_k.current_point = dp_tame_point
        tame_k.distance = tame_dist

        wild_k = runner.wild_kangaroos[0]
        wild_dist = 150  # Arbitrary
        wild_k.current_point = dp_wild_point
        wild_k.distance = wild_dist

        # 3. Patch `is_distinguished` to fire for both points
        dp_tame_x = crypto.get_x_coordinate_int(dp_tame_point)
        dp_wild_x = crypto.get_x_coordinate_int(dp_wild_point)
        monkeypatch.setattr(
            'kangaroo_runner_cpu.dp.is_distinguished',
            lambda x, threshold: x in (dp_tame_x, dp_wild_x)
        )

        # 4. Patch `hop` to do nothing
        monkeypatch.setattr('kangaroo_runner_cpu.Kangaroo.hop', lambda self, precomputed_hops: None)

        # 5. Execute the step
        solution = runner.step()

        # 6. Verify no solution was found
        assert solution is None

        # 7. Verify the points were added to the correct traps
        assert runner.tame_trap.get_point(dp_tame_point.point()) == tame_dist
        assert runner.wild_trap.get_point(dp_wild_point.point()) == wild_dist

        # 8. Verify traps don't contain points from the other herd
        assert runner.wild_trap.get_point(dp_tame_point.point()) is None
        assert runner.tame_trap.get_point(dp_wild_point.point()) is None

    def test_initialization_random_start(self, puzzle_def, profile_config, monkeypatch):
        """Tests initialization with a random start point strategy."""
        profile_config['start_point_strategy'] = 'random'

        # Mock random.randint to return a predictable value (the end of the range)
        range_end = int(puzzle_def['range_end'], 16)
        monkeypatch.setattr(
            'kangaroo_runner_cpu.random.randint',
            lambda start, end: range_end
        )

        runner = KangarooRunnerCPU(puzzle_def, profile_config)
        assert runner.start_key_tame == range_end

    def test_initialization_invalid_strategy(self, puzzle_def, profile_config):
        """Tests that an invalid start point strategy raises an error."""
        profile_config['start_point_strategy'] = 'invalid_strategy'
        with pytest.raises(ValueError, match="Unknown start_point_strategy"):
            KangarooRunnerCPU(puzzle_def, profile_config)
            
    def test_solve_puzzle_5_e2e(self):
        """
        Performs an end-to-end test, solving puzzle #5.
        This verifies that the CPU workflow can solve a puzzle from the set.
        """
        # Load puzzle #5 definition directly
        puzzle_5_def = {
            "puzzle_number": 5,
            "public_key": "02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5",
            "range_start": "0x10",
            "range_end": "0x20",
        }
        
        # Use a fast profile with a low DP threshold to ensure quick collision
        fast_profile = {
            "num_walkers": "2",
            "distinguished_point_threshold": "2",  # Every point is distinguished
            "start_point_strategy": "midpoint",  # Midpoint for deterministic start
        }
        
        runner = KangarooRunnerCPU(puzzle_5_def, fast_profile)
        
        solution = None
        # With a low DP threshold, this should be found very quickly
        max_steps = 10000
        for i in range(max_steps):
            solution = runner.step()
            if solution is not None:
                break
        
        assert solution is not None, f"Solver failed to find a solution for puzzle #5 within {max_steps} steps"
        
        # The private key for puzzle #5 is 21
        expected_solution = 21
        assert solution == expected_solution
