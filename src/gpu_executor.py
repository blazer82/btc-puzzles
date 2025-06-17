"""
GPU execution engine for the Pollard's Kangaroo solver.

Encapsulates all interaction with the Metal API to offload the main
computational work to the Apple Silicon GPU.
"""
import os
import re
from typing import List, Tuple

import Metal as metal
import numpy as np
from coincurve.keys import PublicKey

import cryptography_utils as crypto

# --- Metal Kernel Compilation Helper ---

def _resolve_metal_includes(file_path: str, included_files: set = None) -> str:
    """
    Recursively reads a Metal source file and its #include directives,
    returning a single string with all source code. This is a workaround
    for the Metal compiler not supporting #include from source strings.
    """
    if included_files is None:
        included_files = set()

    file_path = os.path.abspath(file_path)
    if file_path in included_files:
        return ""
    included_files.add(file_path)

    base_dir = os.path.dirname(file_path)
    include_pattern = re.compile(r'#include\s+"([^"]+)"')

    final_source = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = include_pattern.match(line.strip())
                if match:
                    include_path = match.group(1)
                    abs_include_path = os.path.normpath(os.path.join(base_dir, include_path))
                    final_source.append(_resolve_metal_includes(abs_include_path, included_files))
                else:
                    final_source.append(line)
    except FileNotFoundError:
        raise FileNotFoundError(f"Metal include file not found: {file_path}")

    return "".join(final_source)


class GpuExecutor:
    """
    Manages Metal GPU resources and executes the kangaroo hop kernel.
    """

    # --- Constants ---
    # Path to the main Metal kernel file
    KERNEL_FILE = "src/metal/kangaroo_hop.metal"
    # Name of the kernel function in the Metal code
    KERNEL_NAME = "kangaroo_step"
    # 52-bit mask for limb conversion
    M52 = (1 << 52) - 1

    def __init__(self, hop_points: List[PublicKey], dp_threshold: int):
        """
        Initializes the GPU executor, compiles the kernel, and sets up read-only buffers.

        Args:
            hop_points (List[PublicKey]): The 16 pre-computed hop points.
            dp_threshold (int): The distinguished point threshold.
        """
        # 1. Initialize Metal device and command queue
        self.device = metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal-compatible GPU found.")
        self.command_queue = self.device.newCommandQueue()

        # 2. Compile the Metal kernel from source
        source = _resolve_metal_includes(self.KERNEL_FILE)
        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if error:
            raise RuntimeError(f"Metal library compilation failed: {error}")

        kernel_function = library.newFunctionWithName_(self.KERNEL_NAME)
        self.pipeline_state = self.device.newComputePipelineStateWithFunction_error_(kernel_function, None)[0]

        # 3. Define data types matching the Metal structs
        # ge_struct: fe x (5), fe y (5), bool infinity (1) -> 11 ulongs
        self.ge_dtype = np.dtype('(11,)uint64')
        # DPResult struct: ge point, ulong distance
        self.dp_result_dtype = np.dtype([('point', self.ge_dtype), ('distance', 'uint64')])

        # 4. Create read-only GPU buffers
        # Hop points buffer
        hop_points_np = np.array([self._ge_from_pubkey(p) for p in hop_points], dtype=self.ge_dtype)
        self.hop_points_buffer = self.device.newBufferWithBytes_length_options_(
            hop_points_np.tobytes(), hop_points_np.nbytes, metal.MTLResourceStorageModeShared
        )

        # DP threshold buffer
        dp_threshold_np = np.array([dp_threshold], dtype=np.uint32)
        self.dp_threshold_buffer = self.device.newBufferWithBytes_length_options_(
            dp_threshold_np.tobytes(), dp_threshold_np.nbytes, metal.MTLResourceStorageModeShared
        )

        # 5. Initialize state for kangaroo buffers (will be created later)
        self.num_walkers = 0
        self.points_buffer = None
        self.distances_buffer = None
        self.dp_counter_buffer = None
        self.dp_output_buffer = None

    # --- Data Conversion Helpers ---

    def _fe_from_int(self, value: int) -> np.ndarray:
        """Converts a Python integer to a 5x52-bit `fe` limb array."""
        n = np.zeros(5, dtype=np.uint64)
        n[0] = value & self.M52
        n[1] = (value >> 52) & self.M52
        n[2] = (value >> 104) & self.M52
        n[3] = (value >> 156) & self.M52
        n[4] = (value >> 208) & self.M52
        return n

    def _int_from_fe(self, fe_struct: np.ndarray) -> int:
        """Converts a 5x52-bit `fe` limb array back to a Python integer."""
        return (int(fe_struct[0]) |
                (int(fe_struct[1]) << 52) |
                (int(fe_struct[2]) << 104) |
                (int(fe_struct[3]) << 156) |
                (int(fe_struct[4]) << 208))

    def _ge_from_pubkey(self, pubkey: PublicKey) -> np.ndarray:
        """Converts a coincurve PublicKey to a Metal `ge` struct."""
        ge_struct = np.zeros(11, dtype=np.uint64)
        x, y = pubkey.point()
        ge_struct[0:5] = self._fe_from_int(x)
        ge_struct[5:10] = self._fe_from_int(y)
        ge_struct[10] = 0  # infinity = false
        return ge_struct

    def _pubkey_from_ge(self, ge_struct: np.ndarray) -> PublicKey:
        """Converts a Metal `ge` struct back to a coincurve PublicKey."""
        x = self._int_from_fe(ge_struct[0:5])
        y = self._int_from_fe(ge_struct[5:10])
        # Reconstruct from compressed format: 0x02/0x03 prefix + 32-byte x
        prefix = b'\x02' if y % 2 == 0 else b'\x03'
        x_bytes = x.to_bytes(32, 'big')
        return crypto.point_from_bytes(prefix + x_bytes)

    # --- Buffer Management and Kernel Execution ---

    def create_state_buffers(self, initial_points: List[PublicKey]):
        """
        Creates the main GPU buffers for storing kangaroo states.

        Args:
            initial_points (List[PublicKey]): The starting points for each kangaroo.
        """
        self.num_walkers = len(initial_points)

        # Create numpy arrays for initial state
        points_np = np.array([self._ge_from_pubkey(p) for p in initial_points], dtype=self.ge_dtype)
        distances_np = np.zeros(self.num_walkers, dtype=np.uint64)
        dp_counter_np = np.array([0], dtype=np.uint32)
        dp_output_np = np.zeros(self.num_walkers, dtype=self.dp_result_dtype) # Max 1 DP per walker per step

        # Create GPU buffers
        self.points_buffer = self.device.newBufferWithBytes_length_options_(
            points_np.tobytes(), points_np.nbytes, metal.MTLResourceStorageModeShared
        )
        self.distances_buffer = self.device.newBufferWithBytes_length_options_(
            distances_np.tobytes(), distances_np.nbytes, metal.MTLResourceStorageModeShared
        )
        self.dp_counter_buffer = self.device.newBufferWithBytes_length_options_(
            dp_counter_np.tobytes(), dp_counter_np.nbytes, metal.MTLResourceStorageModeShared
        )
        self.dp_output_buffer = self.device.newBufferWithBytes_length_options_(
            dp_output_np.tobytes(), dp_output_np.nbytes, metal.MTLResourceStorageModeShared
        )

    def execute_step(self):
        """Executes one parallel hop for all kangaroos on the GPU."""
        if not self.num_walkers:
            return

        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline_state)

        # Set all buffers for the kernel
        encoder.setBuffer_offset_atIndex_(self.points_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(self.distances_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(self.hop_points_buffer, 0, 2)
        encoder.setBuffer_offset_atIndex_(self.dp_threshold_buffer, 0, 3)
        encoder.setBuffer_offset_atIndex_(self.dp_counter_buffer, 0, 4)
        encoder.setBuffer_offset_atIndex_(self.dp_output_buffer, 0, 5)

        # Dispatch the kernel with one thread per kangaroo
        grid_size = metal.MTLSize(self.num_walkers, 1, 1)
        threadgroup_size = metal.MTLSize(min(self.num_walkers, self.pipeline_state.maxTotalThreadsPerThreadgroup()), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def get_distinguished_points(self) -> List[Tuple[PublicKey, int]]:
        """
        Reads the distinguished point results from the GPU.

        Returns:
            A list of (PublicKey, distance) tuples for each DP found.
        """
        # Read the counter to see how many DPs were found
        dp_count_buffer = self.dp_counter_buffer.contents().as_buffer(self.dp_counter_buffer.length())
        dp_count = np.frombuffer(dp_count_buffer, dtype=np.uint32, count=1)[0]
        if dp_count == 0:
            return []

        # Read the output buffer from GPU to a numpy array
        results_buffer = self.dp_output_buffer.contents().as_buffer(self.dp_output_buffer.length())
        results_np = np.frombuffer(
            results_buffer,
            dtype=self.dp_result_dtype,
            count=int(dp_count)
        )

        # Convert results to Python types
        distinguished_points = []
        for res in results_np:
            point = self._pubkey_from_ge(res['point'])
            distance = int(res['distance'])
            distinguished_points.append((point, distance))

        return distinguished_points

    def reset_dp_counter(self):
        """Resets the distinguished point counter on the GPU to zero."""
        # Create a numpy array that shares memory with the Metal buffer
        buffer = self.dp_counter_buffer.contents().as_buffer(self.dp_counter_buffer.length())
        counter_array = np.frombuffer(buffer, dtype=np.uint32, count=1)
        # Modify the numpy array, which writes directly to the GPU buffer
        counter_array[0] = 0
