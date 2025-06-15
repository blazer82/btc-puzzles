# Bitcoin Puzzle Solver PoC: Unit-Testable Components

This document outlines the modular breakdown for the Proof of Concept (PoC) implementation of the Bitcoin Puzzle Solver using Pollard's Kangaroo algorithm. The focus is on creating small, independent, and unit-testable components to ensure robustness and facilitate development.

1. cryptography_utils.py

- Responsibility: Encapsulate all direct interactions with the coincurve library and raw elliptic curve mathematics specific to secp256k1.
- Key Functions/Classes:
  - get_generator_point() -> coincurve.keys.PublicKey: Returns the secp256k1 generator point G.
  - get_curve_order_n() -> int: Returns the order of the curve n.
  - scalar_multiply(scalar: int, point: coincurve.keys.PublicKey = None) -> coincurve.keys.PublicKey: Performs scalar multiplication of a point (defaults to G).
  - point_add(point1: coincurve.keys.PublicKey, point2: coincurve.keys.PublicKey) -> coincurve.keys.PublicKey: Performs elliptic curve point addition.
  - point_to_bytes(point: coincurve.keys.PublicKey) -> bytes: Converts a PublicKey object to its compressed byte representation.
  - point_from_bytes(data: bytes) -> coincurve.keys.PublicKey: Converts compressed bytes back to a PublicKey object.
  - get_x_coordinate_int(point: coincurve.keys.PublicKey) -> int: Extracts the x-coordinate of a point as an integer.

2. hop_strategy.py

- Responsibility: Define the deterministic logic for how kangaroos select their next hop based on their current position.
- Key Functions/Classes:
  - generate_precomputed_hops(num_hops_power_of_2: int) -> List[coincurve.keys.PublicKey]: Calculates and returns a list of pre-computed hop points (e.g., 2^0 \cdot G, 2^1 \cdot G, \dots). Leverages cryptography_utils.
  - select_hop_index(point_x_coordinate: int, num_precomputed_hops: int) -> int: Determines the index of the next hop using the x-coordinate modulo the number of available hops.

3. distinguished_points.py

- Responsibility: Implement the criteria for identifying "distinguished points" and manage the storage (PointTrap) for these points.
- Key Functions/Classes:
  - is_distinguished(point_x_coordinate: int, threshold: int) -> bool: Checks if an x-coordinate meets the trailing zero bit criterion.
  - PointTrap: A class to store distinguished points (key: (x_coord_int, y_coord_int) tuple, value: accumulated distance).
    - add_point(point_xy_tuple: Tuple[int, int], distance: int): Adds or updates a point in the trap.
    - get_point(point_xy_tuple: Tuple[int, int]) -> Optional[int]: Retrieves the distance for a given point.

4. kangaroo.py

- Responsibility: Model a single kangaroo walker, managing its state (current position, accumulated distance) and its hopping behavior.
- Key Functions/Classes:
  - Kangaroo Class:
    - **init**(initial_point: coincurve.keys.PublicKey, is_tame: bool, initial_warmup_hops: int = 0): Initializes the kangaroo, performing optional warm-up hops.
    - hop(precomputed_hops: List[coincurve.keys.PublicKey]) -> None: Advances the kangaroo by one hop, updating its position and distance.
    - get_state() -> Tuple[coincurve.keys.PublicKey, int]: Returns the current point and accumulated distance.
    - get_xy_tuple() -> Tuple[int, int]: Provides the x,y coordinates as a tuple for use as a PointTrap key.

5. kangaroo_runner.py

- Responsibility: The core solver engine. Orchestrates multiple kangaroos, manages the PointTrap instances for tame and wild herds, detects collisions, and calculates the final private key.
- Key Functions/Classes:
  - KangarooRunner Class:
    - **init**(puzzle_def: Dict, profile_config: Dict): Sets up the initial state for all tame and wild kangaroos, pre-computes hops, and initializes traps.
    - step() -> Optional[int]: Executes one parallel hop for all kangaroos. Collects new distinguished points, checks for collisions across herds, and adds points to traps. Returns the solved private key upon collision.
    - get_total_hops_performed() -> int: Provides the cumulative number of hops for progress reporting.

6. config_manager.py

- Responsibility: Handle loading of puzzle definitions from JSON and solver profiles from INI files.
- Key Functions/Classes:
  - load_puzzle_definition(puzzle_number: int, puzzles_file_path: str) -> Dict: Loads a specific puzzle's parameters.
  - load_profile(profile_name: str, profiles_dir_path: str) -> Dict: Loads a specific solver profile's parameters.

7. main.py

- Responsibility: The application's entry point. Parses command-line arguments, orchestrates the KangarooRunner, manages the main loop, and provides user feedback.
- Key Functions/Classes:
  - main():
    - Parses arguments (--puzzle, --profile).
    - Loads configuration using config_manager.
    - Instantiates KangarooRunner.
    - Runs the main loop, calling runner.step() until a solution is found.
    - Implements basic progress reporting (Hops Per Second, Estimated Time of Arrival).
    - Prints the final private key upon success.
