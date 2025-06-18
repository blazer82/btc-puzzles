"""
The application's entry point. Parses command-line arguments, orchestrates
the KangarooRunner, manages the main loop, and provides user feedback.
"""
import argparse
import math
import time
from typing import Optional

import config_manager as cm
from kangaroo_runner_cpu import KangarooRunnerCPU
from kangaroo_runner_gpu import KangarooRunnerGPU

# --- Constants ---
# Assumes the script is run from the root of the repository
PUZZLES_FILE = 'puzzles.json'
PROFILES_DIR = 'profiles'
PROGRESS_INTERVAL_SECONDS = 10  # Report progress every 10 seconds


def main():
    """
    Main application entry point.
    - Parses command-line arguments.
    - Loads puzzle and profile configurations.
    - Initializes and runs the KangarooRunnerCPU.
    - Reports progress and prints the final solution.
    """
    parser = argparse.ArgumentParser(description="Bitcoin Puzzle Solver using Pollard's Kangaroo Algorithm.")
    parser.add_argument('--puzzle', type=int, required=True, help='The puzzle number to solve.')
    parser.add_argument('--profile', type=str, required=True, help='The solver profile to use (e.g., "verify").')
    args = parser.parse_args()

    print(f"Loading puzzle #{args.puzzle} and profile '{args.profile}'...")
    try:
        puzzle_def = cm.load_puzzle_definition(args.puzzle, PUZZLES_FILE)
        profile_config = cm.load_profile(args.profile, PROFILES_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        return

    print("Initializing KangarooRunner...")
    runner_type = profile_config.get('runner_type', 'cpu').lower()
    
    try:
        if runner_type == 'gpu':
            runner = KangarooRunnerGPU(puzzle_def, profile_config)
            print("Using GPU runner")
        else:
            runner = KangarooRunnerCPU(puzzle_def, profile_config)
            print("Using CPU runner")
        print("Initialization complete. Starting search...")
    except Exception as e:
        print(f"Error initializing runner: {e}")
        return

    # --- Progress Reporting Setup ---
    last_report_time = time.time()
    last_hops_done = runner.get_total_hops_performed()
    start_time = last_report_time

    # --- ETA Calculation Setup ---
    range_start = int(puzzle_def['range_start'], 16)
    range_end = int(puzzle_def['range_end'], 16)
    search_space_size = range_end - range_start
    # Expected hops is sqrt of the search space size for Pollard's Kangaroo
    expected_total_hops = math.sqrt(search_space_size)

    solution: Optional[int] = None
    while solution is None:
        solution = runner.step()

        current_time = time.time()
        if current_time - last_report_time >= PROGRESS_INTERVAL_SECONDS:
            total_hops_done = runner.get_total_hops_performed()
            time_elapsed_interval = current_time - last_report_time
            hops_in_interval = total_hops_done - last_hops_done

            hps = hops_in_interval / time_elapsed_interval if time_elapsed_interval > 0 else 0

            # --- ETA String Formatting ---
            time_remaining_str = "N/A"
            if hps > 0:
                remaining_hops = expected_total_hops - total_hops_done
                if remaining_hops > 0:
                    time_remaining_sec = remaining_hops / hps
                    m, s = divmod(time_remaining_sec, 60)
                    h, m = divmod(m, 60)
                    d, h = divmod(h, 24)
                    time_remaining_str = f"{int(d)}d {int(h)}h {int(m)}m {int(s)}s"
                else:
                    time_remaining_str = "Approaching..."

            # Get trap sizes if the runner supports it
            trap_info = ""
            if hasattr(runner, 'get_trap_sizes'):
                tame_size, wild_size = runner.get_trap_sizes()
                trap_info = f" | DPs: {tame_size:,}/{wild_size:,}"

            total_runtime = current_time - start_time
            print(
                f"Runtime: {total_runtime:.2f}s | "
                f"Hops: {total_hops_done:,} ({hps:,.0f} H/s){trap_info} | "
                f"ETA: {time_remaining_str}"
            )

            last_report_time = current_time
            last_hops_done = total_hops_done

    # --- Solution Found ---
    total_runtime = time.time() - start_time
    print("\n" + "="*50)
    print(" ✨ Collision found! ✨")
    print(f"  Puzzle Number: {puzzle_def['puzzle_number']}")
    print(f"  Public Key:    {puzzle_def['public_key']}")
    print(f"  Private Key (dec): {solution}")
    print(f"  Private Key (hex): {hex(solution)}")
    print("="*50)
    print(f"Total search time: {total_runtime:.2f} seconds")
    print(f"Total hops performed: {runner.get_total_hops_performed():,}")


if __name__ == "__main__":
    main()
