"""
The core solver engine. Orchestrates multiple kangaroos, manages the PointTrap
instances for tame and wild herds, detects collisions, and calculates the
final private key.
"""
import random
from typing import Dict, Optional

import numpy as np

import cryptography_utils as crypto
import distinguished_points as dp
import hop_strategy as hs
import metal_utils as mu
from metal_runner import MetalRunner


class KangarooRunnerGPU:
    """
    Orchestrates the Pollard's Kangaroo search using the GPU.

    This class manages the two herds of kangaroos (tame and wild) by storing
    their state in NumPy arrays, executing hops in parallel on the GPU, and
    managing the distinguished point traps.
    """

    def __init__(self, puzzle_def: Dict, profile_config: Dict):
        """
        Initializes the KangarooRunnerGPU for GPU execution.

        Sets up the Metal runner, compiles kernels, and initializes the state
        of all kangaroos in NumPy arrays suitable for the GPU. Each kangaroo
        follows a unique path based on its thread ID.

        Args:
            puzzle_def (Dict): The parameters for the specific puzzle.
            profile_config (Dict): The parameters for the solver's behavior.
        """
        # Load configurations
        self.puzzle_def = puzzle_def
        self.profile_config = profile_config
        self.dp_threshold = int(
            self.profile_config['distinguished_point_threshold'])
        num_walkers = int(self.profile_config['num_walkers'])
        start_point_strategy = self.profile_config.get(
            'start_point_strategy', 'midpoint')

        # Initialize Metal runner and compile kernels
        self.metal_runner = MetalRunner()
        self.metal_runner.compile_library(
            "kangaroo_kernels", "src/metal/kangaroo_kernels.metal")

        # Pre-compute hops and convert to numpy format for the GPU
        precomputed_hops_list = hs.generate_precomputed_hops()
        self.precomputed_hops_np = np.array(
            [mu.point_to_ge_struct(p) for p in precomputed_hops_list], dtype=np.uint64
        )
        self.num_hops_np = np.array(
            [len(precomputed_hops_list)], dtype=np.uint32)

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
            raise ValueError(
                f"Unknown start_point_strategy: {start_point_strategy}")

        start_point_tame = crypto.scalar_multiply(self.start_key_tame)

        # Setup wild herd
        target_pubkey_bytes = bytes.fromhex(self.puzzle_def['public_key'])
        start_point_wild = crypto.point_from_bytes(target_pubkey_bytes)

        # Check for an immediate collision. If the tame start point is the
        # target, we have found the solution.
        if start_point_tame.point() == start_point_wild.point():
            self.solution = self.start_key_tame
        else:
            self.solution = None

        # Initialize tame kangaroos
        tame_kangaroos_list = [mu.point_to_gej_struct(start_point_tame) for _ in range(num_walkers)]
        self.tame_kangaroos_np = np.array(tame_kangaroos_list, dtype=np.uint64)
        self.tame_distances_np = np.zeros(num_walkers, dtype=np.uint64)

        # Initialize wild kangaroos
        wild_kangaroos_list = [mu.point_to_gej_struct(start_point_wild) for _ in range(num_walkers)]
        self.wild_kangaroos_np = np.array(wild_kangaroos_list, dtype=np.uint64)
        self.wild_distances_np = np.zeros(num_walkers, dtype=np.uint64)

        self.total_hops_performed = 0
        self.total_infinity_points = 0

    def step(self) -> Optional[int]:
        """
        Executes one parallel hop for all kangaroos on the GPU.

        All kangaroos in both herds take a single step in parallel. The GPU
        then converts the resulting Jacobian points to affine coordinates.
        These are checked for the "distinguished point" property. If a DP is
        found, it's checked for a collision in the opposite herd's trap.
        If a collision is found, the private key is calculated and returned.

        Returns:
            Optional[int]: The solved private key if a collision occurs,
                           otherwise None.
        """
        # If a solution was found during initialization, return it.
        if self.solution is not None:
            # To prevent returning it on every subsequent call, we consume it.
            solution = self.solution
            self.solution = None
            return solution

        num_walkers = int(self.profile_config['num_walkers'])

        # 1. Hop all kangaroos on the GPU
        self.metal_runner.run_kernel(
            "kangaroo_kernels", "batch_hop_kangaroos", num_walkers,
            [self.tame_kangaroos_np, self.tame_distances_np,
             self.precomputed_hops_np, self.num_hops_np]
        )
        self.metal_runner.run_kernel(
            "kangaroo_kernels", "batch_hop_kangaroos", num_walkers,
            [self.wild_kangaroos_np, self.wild_distances_np,
             self.precomputed_hops_np, self.num_hops_np]
        )
        self.total_hops_performed += 2 * num_walkers

        # 2. Convert GPU results to affine to check for distinguished points
        # Reuse the same affine arrays to avoid repeated allocations
        if not hasattr(self, '_tame_affine_np'):
            self._tame_affine_np = np.zeros((num_walkers, 11), dtype=np.uint64)
            self._wild_affine_np = np.zeros((num_walkers, 11), dtype=np.uint64)
        
        self.metal_runner.run_kernel(
            "kangaroo_kernels", "batch_ge_set_gej", num_walkers,
            [self.tame_kangaroos_np, self._tame_affine_np]
        )
        self.metal_runner.run_kernel(
            "kangaroo_kernels", "batch_ge_set_gej", num_walkers,
            [self.wild_kangaroos_np, self._wild_affine_np]
        )

        # 3. Collect new distinguished points and count infinity points
        infinity_count_tame = 0
        infinity_count_wild = 0
        
        new_distinguished_tame = []
        for i in range(num_walkers):
            # Count and skip infinity points
            if self._tame_affine_np[i][10] == 1:  # infinity flag
                infinity_count_tame += 1
                continue
                
            x_coord = mu.ge_struct_to_x_coord(self._tame_affine_np[i])
            if dp.is_distinguished(x_coord, self.dp_threshold):
                point_xy = mu.ge_struct_to_point(self._tame_affine_np[i])
                dist = int(self.tame_distances_np[i])
                new_distinguished_tame.append((point_xy, dist))

        new_distinguished_wild = []
        for i in range(num_walkers):
            # Count and skip infinity points
            if self._wild_affine_np[i][10] == 1:  # infinity flag
                infinity_count_wild += 1
                continue
                
            x_coord = mu.ge_struct_to_x_coord(self._wild_affine_np[i])
            if dp.is_distinguished(x_coord, self.dp_threshold):
                point_xy = mu.ge_struct_to_point(self._wild_affine_np[i])
                dist = int(self.wild_distances_np[i])
                new_distinguished_wild.append((point_xy, dist))
        
        # Update total infinity count
        self.total_infinity_points += infinity_count_tame + infinity_count_wild

        # 4. Check for collisions and update traps
        curve_n = crypto.get_curve_order_n()

        # First, check for collisions between the new DPs and the existing traps
        for point_xy, dist_tame in new_distinguished_tame:
            dist_wild = self.wild_trap.get_point(point_xy)
            if dist_wild is not None:
                # Collision found: new tame point is in old wild trap
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        for point_xy, dist_wild in new_distinguished_wild:
            dist_tame = self.tame_trap.get_point(point_xy)
            if dist_tame is not None:
                # Collision found: new wild point is in old tame trap
                return (self.start_key_tame + dist_tame - dist_wild) % curve_n

        # If no collisions were found, add the new distinguished points to their traps
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
    
    def get_infinity_count(self) -> int:
        """
        Returns the total number of infinity points encountered.
        
        Returns:
            int: Total infinity points across all steps
        """
        return self.total_infinity_points

    def get_memory_info(self) -> Dict:
        """
        Returns information about GPU memory usage.
        
        Returns:
            Dict with memory statistics.
        """
        buffer_info = self.metal_runner.get_buffer_cache_info()
        
        # Calculate size of our main arrays
        tame_size = self.tame_kangaroos_np.nbytes + self.tame_distances_np.nbytes
        wild_size = self.wild_kangaroos_np.nbytes + self.wild_distances_np.nbytes
        hops_size = self.precomputed_hops_np.nbytes + self.num_hops_np.nbytes
        
        affine_size = 0
        if hasattr(self, '_tame_affine_np'):
            affine_size = self._tame_affine_np.nbytes + self._wild_affine_np.nbytes
        
        total_array_size = tame_size + wild_size + hops_size + affine_size
        
        return {
            'buffer_cache': buffer_info,
            'main_arrays_bytes': total_array_size,
            'main_arrays_mb': total_array_size / (1024 * 1024),
            'trap_sizes': self.get_trap_sizes(),
            'infinity_points': self.get_infinity_count()
        }

    def clear_gpu_cache(self):
        """
        Clears the GPU buffer cache to free memory.
        Call this if memory usage becomes problematic.
        """
        self.metal_runner.clear_buffer_cache()
