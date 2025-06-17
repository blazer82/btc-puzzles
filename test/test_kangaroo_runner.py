import pytest
from unittest.mock import MagicMock, patch

import cryptography_utils as crypto
from kangaroo_runner import KangarooRunner


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


@patch('kangaroo_runner.GpuExecutor')
class TestKangarooRunner:
    def test_initialization(self, mock_gpu_executor, puzzle_def, profile_config):
        """Tests that the runner initializes correctly and configures executors."""
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        runner = KangarooRunner(puzzle_def, profile_config)

        # Check that two GpuExecutor instances were created
        assert mock_gpu_executor.call_count == 2

        # Check DP threshold
        assert runner.dp_threshold == int(profile_config['distinguished_point_threshold'])

        # Check tame herd starting key
        expected_start_key = (2 + 3) // 2
        assert runner.start_key_tame == expected_start_key

        # Check that state buffers were created on both executors
        runner.tame_executor.create_state_buffers.assert_called_once()
        runner.wild_executor.create_state_buffers.assert_called_once()

        # Check initial hop count is zero (no warm-up hops)
        assert runner.get_total_hops_performed() == 0

    def test_step_updates_total_hops(self, mock_gpu_executor, puzzle_def, profile_config):
        """Tests that a single step correctly updates the total hop count."""
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        runner = KangarooRunner(puzzle_def, profile_config)
        initial_hops = runner.get_total_hops_performed()
        num_walkers = int(profile_config['num_walkers'])
        runner.tame_executor.num_walkers = num_walkers

        # Mock the return of get_distinguished_points to be empty
        runner.tame_executor.get_distinguished_points.return_value = []
        runner.wild_executor.get_distinguished_points.return_value = []

        runner.step()

        # Hops should increase by the total number of kangaroos
        expected_new_hops = initial_hops + (2 * num_walkers)
        assert runner.get_total_hops_performed() == expected_new_hops

        # Check that executors were stepped
        runner.tame_executor.execute_step.assert_called_once()
        runner.wild_executor.execute_step.assert_called_once()

    def test_collision_tame_finds_wild_in_trap(self, mock_gpu_executor, puzzle_def, profile_config):
        """
        Tests a collision where a tame kangaroo lands on a distinguished point
        that is already in the wild herd's trap.
        """
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        runner = KangarooRunner(puzzle_def, profile_config)
        runner.tame_executor.num_walkers = int(profile_config['num_walkers'])

        # 1. Create a fake collision point and add it to the wild trap
        collision_point_pubkey = crypto.scalar_multiply(12345)
        collision_point_xy = collision_point_pubkey.point()
        wild_dist = 500
        runner.wild_trap.add_point(collision_point_xy, wild_dist)

        # 2. Mock the tame executor to report this point as a new DP
        tame_dist = 12345 - runner.start_key_tame
        runner.tame_executor.get_distinguished_points.return_value = [
            (collision_point_pubkey, tame_dist)
        ]
        runner.wild_executor.get_distinguished_points.return_value = []

        # 3. Execute the step and check for the correct solution
        solution = runner.step()
        assert solution is not None

        n = crypto.get_curve_order_n()
        expected_solution = (runner.start_key_tame + tame_dist - wild_dist) % n
        assert solution == expected_solution

    def test_collision_wild_finds_tame_in_trap(self, mock_gpu_executor, puzzle_def, profile_config):
        """
        Tests a collision where a wild kangaroo lands on a distinguished point
        that is already in the tame herd's trap.
        """
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        runner = KangarooRunner(puzzle_def, profile_config)
        runner.tame_executor.num_walkers = int(profile_config['num_walkers'])

        # 1. Create a fake collision point and add it to the tame trap
        collision_point_pubkey = crypto.scalar_multiply(54321)
        collision_point_xy = collision_point_pubkey.point()
        tame_dist = 54321 - runner.start_key_tame
        runner.tame_trap.add_point(collision_point_xy, tame_dist)

        # 2. Mock the wild executor to report this point as a new DP
        wild_dist = 999  # Arbitrary distance
        runner.wild_executor.get_distinguished_points.return_value = [
            (collision_point_pubkey, wild_dist)
        ]
        runner.tame_executor.get_distinguished_points.return_value = []

        # 3. Execute the step and check for the correct solution
        solution = runner.step()
        assert solution is not None

        n = crypto.get_curve_order_n()
        expected_solution = (runner.start_key_tame + tame_dist - wild_dist) % n
        assert solution == expected_solution

    def test_no_collision_adds_points_to_traps(self, mock_gpu_executor, puzzle_def, profile_config):
        """
        Tests that when distinguished points are found but no collision occurs,
        they are correctly added to their respective traps.
        """
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        runner = KangarooRunner(puzzle_def, profile_config)
        runner.tame_executor.num_walkers = int(profile_config['num_walkers'])

        # 1. Define two points that will be "found" by the executors
        dp_tame_pubkey = crypto.scalar_multiply(100)
        dp_wild_pubkey = crypto.scalar_multiply(200)
        tame_dist = 100 - runner.start_key_tame
        wild_dist = 150  # Arbitrary

        # 2. Mock executors to return these points
        runner.tame_executor.get_distinguished_points.return_value = [(dp_tame_pubkey, tame_dist)]
        runner.wild_executor.get_distinguished_points.return_value = [(dp_wild_pubkey, wild_dist)]

        # 3. Execute the step
        solution = runner.step()

        # 4. Verify no solution was found
        assert solution is None

        # 5. Verify the points were added to the correct traps
        assert runner.tame_trap.get_point(dp_tame_pubkey.point()) == tame_dist
        assert runner.wild_trap.get_point(dp_wild_pubkey.point()) == wild_dist

        # 6. Verify traps don't contain points from the other herd
        assert runner.wild_trap.get_point(dp_tame_pubkey.point()) is None
        assert runner.tame_trap.get_point(dp_wild_pubkey.point()) is None

    def test_initialization_random_start(self, mock_gpu_executor, puzzle_def, profile_config, monkeypatch):
        """Tests initialization with a random start point strategy."""
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        profile_config['start_point_strategy'] = 'random'

        # Mock random.randint to return a predictable value (the end of the range)
        range_end = int(puzzle_def['range_end'], 16)
        monkeypatch.setattr(
            'kangaroo_runner.random.randint',
            lambda start, end: range_end
        )

        runner = KangarooRunner(puzzle_def, profile_config)
        assert runner.start_key_tame == range_end

    def test_initialization_invalid_strategy(self, mock_gpu_executor, puzzle_def, profile_config):
        """Tests that an invalid start point strategy raises an error."""
        mock_gpu_executor.side_effect = [MagicMock(), MagicMock()]
        profile_config['start_point_strategy'] = 'invalid_strategy'
        with pytest.raises(ValueError, match="Unknown start_point_strategy"):
            KangarooRunner(puzzle_def, profile_config)
