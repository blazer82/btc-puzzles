"""
The core solver engine. Orchestrates multiple kangaroos, manages the PointTrap
instances for tame and wild herds, detects collisions, and calculates the
final private key.
"""
import random
from typing import Dict, List, Optional

import cryptography_utils as crypto
import distinguished_points as dp
import hop_strategy as hs
from kangaroo import Kangaroo


class KangarooRunnerCPU:
    """
    Orchestrates the Pollard's Kangaroo search.

    This class manages the two herds of kangaroos (tame and wild),
    the distinguished point traps, and the main search loop.
    """

    def __init__(self, puzzle_def: Dict, profile_config: Dict):
        """
        Initializes the KangarooRunnerCPU.

        Sets up the initial state for all tame and wild kangaroos based on the
        puzzle definition and solver profile. This includes pre-computing hops,
        initializing traps, and assigning unique IDs to ensure each kangaroo
        follows a unique path.

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

        # Initialize traps with memory bounds if specified
        max_trap_size = int(self.profile_config.get('max_trap_size', '0')) or None
        self.tame_trap = dp.PointTrap(max_size=max_trap_size)
        self.wild_trap = dp.PointTrap(max_size=max_trap_size)

        # Setup tame herd
        range_start = int(self.puzzle_def['range_start'], 16)
        range_end = int(self.puzzle_def['range_end'], 16)

        if start_point_strategy == 'midpoint':
            self.start_key_tame = (range_start + range_end) // 2
        elif start_point_strategy == 'random':
            self.start_key_tame = random.randint(range_start, range_end)
        else:
            raise ValueError(f"Unknown start_point_strategy: {start_point_strategy}")

        start_point_tame = crypto.scalar_multiply(self.start_key_tame)
        self.tame_kangaroos: List[Kangaroo] = []
        for i in range(num_walkers):
            k = Kangaroo(kangaroo_id=i, initial_point=start_point_tame, is_tame=True)
            self.tame_kangaroos.append(k)

        # Setup wild herd
        target_pubkey_bytes = bytes.fromhex(self.puzzle_def['public_key'])
        start_point_wild = crypto.point_from_bytes(target_pubkey_bytes)
        self.wild_kangaroos: List[Kangaroo] = []
        for i in range(num_walkers):
            k = Kangaroo(kangaroo_id=i, initial_point=start_point_wild, is_tame=False)
            self.wild_kangaroos.append(k)

        self.total_hops_performed = 0

    def step(self) -> Optional[int]:
        """
        Executes one parallel hop for all kangaroos.

        Each kangaroo in both the tame and wild herds takes a single step.
        After the step, it checks if any kangaroo has landed on a distinguished
        point. If so, it checks for a collision in the opposite herd's trap.
        If a collision is found, the private key is calculated and returned.
        Otherwise, the new distinguished points are added to their respective
        traps.

        Returns:
            Optional[int]: The solved private key if a collision occurs,
                           otherwise None.
        """
        new_distinguished_tame = []
        new_distinguished_wild = []

        # 1. Hop all kangaroos
        for k in self.tame_kangaroos + self.wild_kangaroos:
            k.hop(self.precomputed_hops)

        self.total_hops_performed += len(self.tame_kangaroos) + len(self.wild_kangaroos)

        # 2. Collect new distinguished points
        for k in self.tame_kangaroos:
            x_coord = crypto.get_x_coordinate_int(k.current_point)
            if dp.is_distinguished(x_coord, self.dp_threshold):
                new_distinguished_tame.append((k.get_xy_tuple(), k.distance))

        for k in self.wild_kangaroos:
            x_coord = crypto.get_x_coordinate_int(k.current_point)
            if dp.is_distinguished(x_coord, self.dp_threshold):
                new_distinguished_wild.append((k.get_xy_tuple(), k.distance))

        # 3. Check for collisions
        curve_n = crypto.get_curve_order_n()

        # Check if a new tame point is in the wild trap
        for point_xy, dist_tame in new_distinguished_tame:
            dist_wild = self.wild_trap.get_point(point_xy)
            if dist_wild is not None:
                # Collision found!
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        # Check if a new wild point is in the tame trap
        for point_xy, dist_wild in new_distinguished_wild:
            dist_tame = self.tame_trap.get_point(point_xy)
            if dist_tame is not None:
                # Collision found!
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        # 4. If no collision, add new points to traps
        for point_xy, dist_tame in new_distinguished_tame:
            self.tame_trap.add_point(point_xy, dist_tame)

        for point_xy, dist_wild in new_distinguished_wild:
            self.wild_trap.add_point(point_xy, dist_wild)

        return None

    def get_total_hops_performed(self) -> int:
        """
        Provides the cumulative number of hops performed by all kangaroos.

        Returns:
            int: The total number of hops.
        """
        return self.total_hops_performed
        
    def get_trap_sizes(self) -> tuple:
        """
        Returns the current sizes of the tame and wild traps.
        
        Returns:
            tuple: (tame_trap_size, wild_trap_size)
        """
        return (self.tame_trap.size(), self.wild_trap.size())
