# System Architecture

This document outlines the architecture for the Bitcoin puzzle solver. The design separates configuration, core logic, and application management to allow for flexibility in solving both easy puzzles for verification and hard puzzles for serious attempts.

## 1. Configuration Layer

This layer is responsible for managing all parameters for a given solver run. This allows for easy switching between different puzzles and performance profiles without changing the core code.

### Components:

*   **Puzzle Definitions (`puzzles.json`)**: A structured data file (e.g., JSON) containing the specific parameters for each Bitcoin puzzle.
    *   `puzzle_number`: The official number of the puzzle (e.g., 65).
    *   `public_key`: The target public key to find the private key for.
    *   `range_start`, `range_end`: The search space for the private key, typically `2^(n-1)` to `2^n - 1`.

*   **Solver Profiles (`profiles/`)**: A directory of configuration files (in INI format) that define the solver's behavior. Each profile will contain a `[Solver]` section with the following keys:
    *   `num_walkers`: The number of kangaroos to run in parallel on the GPU.
    *   `distinguished_point_threshold`: The number of trailing zero bits required for a point's x-coordinate to be considered "distinguished".
    *   **`verify.ini`**: A profile for quick verification of easy puzzles. It would specify a small `num_walkers` and a low `distinguished_point_threshold` to get a result quickly.
    *   **`solve.ini`**: A profile for serious, long-running attempts on hard puzzles. It would configure a very large `num_walkers` to maximize GPU throughput and a high `distinguished_point_threshold` to minimize CPU overhead.

## 2. Core Solver Engine

This is the heart of the application, containing the implementation of Pollard's Kangaroo Algorithm. It is designed to be a self-contained, high-performance module focused exclusively on the cryptographic computation.

### Components:

*   **`KangarooRunner` Class**: The main class that encapsulates the state and logic of the algorithm.
    *   **Initialization**: Takes a puzzle definition and a solver profile as input. It initializes all necessary data structures (kangaroo positions, distances) as tensors directly on the Apple Silicon GPU (MPS device). The tame kangaroos start at the public key point corresponding to the midpoint of the puzzle's key range (`(range_start + range_end) // 2`), while the wild kangaroos start at the target public key. The midpoint private key scalar is stored separately for the final solution calculation. To ensure walkers follow unique paths, they are differentiated with a "warm-up" phase: each walker `i` performs `i` initial hops before the main search begins.
    *   **`step()` Method**: The performance-critical method. It executes a single, parallel "hop" for all kangaroos simultaneously on the GPU. The logic for determining the next hop is contained within this GPU-accelerated operation.
        *   **Hop Selection**: The next hop is chosen deterministically based on the kangaroo's current position. The x-coordinate of the current point is used with a modulo operator (`point.x % num_hops`) to select a hop from a predefined set.
        *   **Hop Set**: The set of possible hops is defined by a fixed, hard-coded list of 16 power-of-two distances: `2^0, 2^1, ..., 2^15`. The corresponding elliptic curve points for these distances (`2^i * G`) are pre-computed and stored on the GPU for efficient lookup. This set does not need to be configured.
    *   **Distinguished Point Handling**: Manages the "traps" for distinguished points. A point is considered "distinguished" if its x-coordinate has a number of trailing zero bits equal to or greater than the `distinguished_point_threshold` set in the solver profile. When a kangaroo lands on such a point, its position and distance are efficiently transferred to the CPU.
        *   **Trap Data Structure**: Two separate Python dictionaries (`tame_trap`, `wild_trap`) are used to store distinguished points for each herd. The dictionary key is the point's `(x, y)` coordinate tuple, and the value is its total distance. This provides `O(1)` average time complexity for lookups. If multiple kangaroos from the same herd land on the same distinguished point, it is acceptable for the last one to overwrite the entry in the trap.
        *   **Collision Logic**: When a batch of new distinguished points is found, they are processed in two phases. First, for each new point, the *opposite* herd's trap is checked for a collision. If no collisions are found, the new points are then added to their respective traps. This ensures correct handling of multiple points found in the same parallel step. Upon collision, the final private key is calculated as `(start_key_tame + distance_tame - distance_wild) mod n`, where `n` is the order of the curve.

## 3. Application & State Management Layer

This layer serves as the user-facing entry point. It orchestrates the solver engine, handles user input, and manages the application's lifecycle, especially for long-running tasks.

### Components:

*   **Main Script (`main.py`)**: The primary executable.
    *   **Argument Parsing**: Uses a library like `argparse` to handle command-line arguments (e.g., `--puzzle 65`, `--profile solve`, `--resume`).
    *   **Orchestration**: Instantiates the `KangarooRunner` with the selected configuration and runs its `step()` method in a loop.
    *   **Progress Reporting**: Prints statistics to the console at a fixed time interval (e.g., every 30 seconds) to provide consistent user feedback. It will show metrics like hops per second, distinguished points found, and an estimated time to completion. The ETA is based on the total work done versus the theoretical work required.
        *   **Hops Per Second (HPS)**: The aggregate rate of all walkers, calculated as `(num_walkers * steps_in_interval) / time_in_interval`.
        *   **ETA Formula**: `time_remaining = (expected_total_hops - total_hops_done) / hps`, where `expected_total_hops` is `sqrt(puzzle_range_size)` and `total_hops_done` is the cumulative number of steps taken by all walkers.

*   **State Persistence**: A critical feature for solving hard puzzles that may take days or weeks.
    *   **`save_state()` / `load_state()`**: Methods within the `KangarooRunner` to serialize the complete state of the solver (all kangaroo positions and distances) to a file. The state file will be named dynamically based on the puzzle and profile (e.g., `puzzle_65_solve.state`) to avoid conflicts between runs.
    *   **Signal Handling**: The main script will trap system signals (e.g., `Ctrl+C`) to trigger a graceful shutdown, ensuring `save_state()` is called before exiting.
    *   **Resumption**: If the `--resume` flag is provided, the application will use `load_state()` to restore the solver to its last saved state and continue the search seamlessly.
