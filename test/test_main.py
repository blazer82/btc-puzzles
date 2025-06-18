import pytest
from unittest.mock import MagicMock, patch

import main as main_module


@pytest.fixture
def puzzle_def():
    """Provides a sample puzzle definition."""
    return {
        "puzzle_number": 5,
        "public_key": "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "range_start": "0x2",
        "range_end": "0x3",
    }


@pytest.fixture
def profile_config():
    """Provides a sample solver profile."""
    return {
        "num_walkers": "2",
        "distinguished_point_threshold": "20",
    }


class TestMain:
    @patch('main.KangarooRunnerGPU')
    @patch('main.KangarooRunnerCPU')
    @patch('main.cm.load_profile')
    @patch('main.cm.load_puzzle_definition')
    @patch('main.argparse.ArgumentParser')
    @patch('main.time.time')
    def test_main_success_flow(
        self,
        mock_time,
        mock_arg_parser,
        mock_load_puzzle,
        mock_load_profile,
        mock_kangaroo_runner_cpu,
        mock_kangaroo_runner_gpu,
        capsys,
        puzzle_def,
        profile_config,
    ):
        """
        Tests the entire successful execution flow of the main script:
        - Argument parsing
        - Configuration loading
        - Runner initialization
        - Main loop with progress reporting
        - Solution finding and printing
        """
        # --- Setup Mocks ---
        # 1. Argument Parser
        mock_args = MagicMock()
        mock_args.puzzle = 5
        mock_args.profile = 'verify'
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # 2. Config Manager
        mock_load_puzzle.return_value = puzzle_def
        profile_config['runner_type'] = 'cpu'  # Ensure we're testing the CPU path
        mock_load_profile.return_value = profile_config

        # 3. KangarooRunner
        mock_runner_instance = MagicMock()
        solution_key = 12345
        mock_runner_instance.step.side_effect = [None, None, solution_key]
        # Calls: 1. init (0), 2. progress report (200), 3. final report (300)
        mock_runner_instance.get_total_hops_performed.side_effect = [0, 200, 300]
        mock_kangaroo_runner_cpu.return_value = mock_runner_instance

        # 4. Time
        # Progress report is every 10s. A report should trigger after the 2nd step.
        # Time elapsed for report: 1010.3 - 1000.0 = 10.3s
        # HPS = (200-0) / 10.3 = 19.4 -> formats to "19"
        time_sequence = [1000.0, 1005.0, 1010.3, 1015.0, 1015.0]
        mock_time.side_effect = time_sequence

        # --- Run main ---
        main_module.main()

        # --- Assertions ---
        captured = capsys.readouterr()

        # Check initialization messages
        assert "Loading puzzle #5 and profile 'verify'..." in captured.out
        assert "Initializing KangarooRunner..." in captured.out
        assert "Initialization complete. Starting search..." in captured.out

        # Check progress report (triggered at time 1010.3)
        assert "Runtime: 10.30s" in captured.out
        assert "Hops: 200" in captured.out
        assert "19 H/s" in captured.out

        # Check solution output
        assert "✨ Collision found! ✨" in captured.out
        assert f"Private Key (dec): {solution_key}" in captured.out
        assert f"Private Key (hex): {hex(solution_key)}" in captured.out
        assert "Total search time: 15.00 seconds" in captured.out
        assert "Total hops performed: 300" in captured.out

        # Verify mocks were called correctly
        mock_load_puzzle.assert_called_once_with(5, 'puzzles.json')
        mock_load_profile.assert_called_once_with('verify', 'profiles')
        mock_kangaroo_runner_cpu.assert_called_once_with(puzzle_def, profile_config)
        mock_kangaroo_runner_gpu.assert_not_called()
        assert mock_runner_instance.step.call_count == 3

    @patch('main.KangarooRunnerGPU')
    @patch('main.KangarooRunnerCPU')
    @patch('main.cm.load_profile')
    @patch('main.cm.load_puzzle_definition')
    @patch('main.argparse.ArgumentParser')
    @patch('main.time.time')
    def test_main_gpu_flow(
        self,
        mock_time,
        mock_arg_parser,
        mock_load_puzzle,
        mock_load_profile,
        mock_kangaroo_runner_cpu,
        mock_kangaroo_runner_gpu,
        capsys,
        puzzle_def,
        profile_config,
    ):
        """
        Tests the execution flow using the GPU runner.
        """
        # --- Setup Mocks ---
        # 1. Argument Parser
        mock_args = MagicMock()
        mock_args.puzzle = 5
        mock_args.profile = 'solve'
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # 2. Config Manager
        mock_load_puzzle.return_value = puzzle_def
        profile_config['runner_type'] = 'gpu'  # Use GPU runner
        mock_load_profile.return_value = profile_config

        # 3. KangarooRunner
        mock_runner_instance = MagicMock()
        solution_key = 12345
        mock_runner_instance.step.side_effect = [None, solution_key]
        mock_runner_instance.get_total_hops_performed.side_effect = [0, 500, 500]
        mock_kangaroo_runner_gpu.return_value = mock_runner_instance

        # 4. Time
        # Need 4 time values: start time, step time, progress report time, and final time for total_runtime
        mock_time.side_effect = [1000.0, 1005.0, 1010.0, 1015.0]

        # --- Run main ---
        main_module.main()

        # --- Assertions ---
        captured = capsys.readouterr()
        assert "Using GPU runner" in captured.out
        assert "✨ Collision found! ✨" in captured.out

        # Verify mocks were called correctly
        mock_kangaroo_runner_gpu.assert_called_once_with(puzzle_def, profile_config)
        mock_kangaroo_runner_cpu.assert_not_called()
        assert mock_runner_instance.step.call_count == 2

    @patch('main.cm.load_puzzle_definition')
    @patch('main.argparse.ArgumentParser')
    def test_main_config_error(self, mock_arg_parser, mock_load_puzzle, capsys):
        """
        Tests that the script handles a configuration loading error gracefully.
        """
        # --- Setup Mocks ---
        # 1. Argument Parser
        mock_args = MagicMock()
        mock_args.puzzle = 99
        mock_args.profile = 'nonexistent'
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # 2. Config Manager to raise an error
        error_message = "Puzzle number 99 not found"
        mock_load_puzzle.side_effect = ValueError(error_message)

        # --- Run main ---
        main_module.main()

        # --- Assertions ---
        captured = capsys.readouterr()
        assert f"Error loading configuration: {error_message}" in captured.out
        assert "Collision found" not in captured.out
