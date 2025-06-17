"""
The core solver engine. Orchestrates the GpuExecutor to run the Pollard's
Kangaroo algorithm on the GPU. It manages the PointTrap instances for tame and
wild herds, detects collisions, and calculates the final private key.
"""
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

import cryptography_utils as crypto
import distinguished_points as dp
import hop_strategy as hs
from gpu_executor import GpuExecutor


class KangarooRunner:
    """
    Orchestrates the Pollard's Kangaroo search using a GPU executor.

    This class manages the two herds of kangaroos (tame and wild) by
    orchestrating the GpuExecutor, which runs the search in parallel on the
    GPU. It handles the distinguished point traps and collision detection.
    """

    def __init__(self, puzzle_def: Dict, profile_config: Dict):
        """
        Initializes the KangarooRunner and the GPU executors.

        Sets up the initial state for all tame and wild kangaroos and passes
        them to the GpuExecutor. This includes pre-computing hops and
        initializing traps.

        Args:
            puzzle_def (Dict): The parameters for the specific puzzle.
            profile_config (Dict): The parameters for the solver's behavior.
        """
        # Load configurations
        self.puzzle_def = puzzle_def
        self.profile_config = profile_config
        self.dp_threshold = int(self.profile_config['distinguished_point_threshold'])
        num_walkers = int(self.profile_config['num_walkers'])
        start_point_strategy = self.profile_config.get('start_point_strategy', 'midpoint')

        # Pre-compute hops
        self.precomputed_hops = hs.generate_precomputed_hops()

        # Initialize GPU executors for each herd
        self.tame_executor = GpuExecutor(hop_points=self.precomputed_hops, dp_threshold=self.dp_threshold)
        self.wild_executor = GpuExecutor(hop_points=self.precomputed_hops, dp_threshold=self.dp_threshold)

        # Initialize traps
        self.tame_trap = dp.PointTrap()
        self.wild_trap = dp.PointTrap()

        # Setup tame herd initial state
        range_start = int(self.puzzle_def['range_start'], 16)
        range_end = int(self.puzzle_def['range_end'], 16)

        if start_point_strategy == 'midpoint':
            self.start_key_tame = (range_start + range_end) // 2
        elif start_point_strategy == 'random':
            self.start_key_tame = random.randint(range_start, range_end)
        else:
            raise ValueError(f"Unknown start_point_strategy: {start_point_strategy}")

        start_point_tame = crypto.scalar_multiply(self.start_key_tame)
        initial_tame_points = [start_point_tame] * num_walkers
        self.tame_executor.create_state_buffers(initial_tame_points)

        # Setup wild herd initial state
        target_pubkey_bytes = bytes.fromhex(self.puzzle_def['public_key'])
        start_point_wild = crypto.point_from_bytes(target_pubkey_bytes)
        initial_wild_points = [start_point_wild] * num_walkers
        self.wild_executor.create_state_buffers(initial_wild_points)

        self.total_hops_performed = 0

    def step(self) -> Optional[int]:
        """
        Executes one parallel step on the GPU for all kangaroos.

        This method dispatches the Metal kernel to perform one hop for each
        kangaroo in both herds. It then retrieves any distinguished points
        found by the GPU, checks for collisions, and adds new points to the
        appropriate traps.

        Returns:
            Optional[int]: The solved private key if a collision occurs,
                           otherwise None.
        """
        # 1. Hop all kangaroos on the GPU
        self.tame_executor.execute_step()
        self.wild_executor.execute_step()

        num_walkers = self.tame_executor.num_walkers
        self.total_hops_performed += 2 * num_walkers

        # 2. Get distinguished points from GPU
        new_distinguished_tame = self.tame_executor.get_distinguished_points()
        new_distinguished_wild = self.wild_executor.get_distinguished_points()

        # Reset GPU counters for the next step
        self.tame_executor.reset_dp_counter()
        self.wild_executor.reset_dp_counter()

        # 3. Check for collisions
        curve_n = crypto.get_curve_order_n()

        # Check if a new tame point is in the wild trap
        for pubkey, dist_tame in new_distinguished_tame:
            point_xy = pubkey.point()
            dist_wild = self.wild_trap.get_point(point_xy)
            if dist_wild is not None:
                # Collision found!
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        # Check if a new wild point is in the tame trap
        for pubkey, dist_wild in new_distinguished_wild:
            point_xy = pubkey.point()
            dist_tame = self.tame_trap.get_point(point_xy)
            if dist_tame is not None:
                # Collision found!
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        # 4. If no collision, add new points to traps
        for pubkey, dist_tame in new_distinguished_tame:
            point_xy = pubkey.point()
            self.tame_trap.add_point(point_xy, dist_tame)

        for pubkey, dist_wild in new_distinguished_wild:
            point_xy = pubkey.point()
            self.wild_trap.add_point(point_xy, dist_wild)

        return None

    def get_total_hops_performed(self) -> int:
        """
        Provides the cumulative number of hops performed by all kangaroos.

        Returns:
            int: The total number of hops.
        """
        return self.total_hops_performed
