"""
A helper class to manage Metal kernel compilation and execution.

This class abstracts the boilerplate code required to interact with the Metal
API, such as device initialization, library compilation from source files
(including handling of #include directives), and kernel execution.
"""
import os
import re
from typing import Dict, List, Tuple, Any

import Metal
import numpy as np


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
    with open(file_path, 'r') as f:
        for line in f:
            match = include_pattern.match(line.strip())
            if match:
                include_path = match.group(1)
                abs_include_path = os.path.normpath(os.path.join(base_dir, include_path))
                final_source.append(_resolve_metal_includes(abs_include_path, included_files))
            else:
                final_source.append(line)

    return "".join(final_source)


class MetalRunner:
    """Manages compilation and execution of Metal kernels."""

    def __init__(self):
        """Initializes the Metal device and command queue."""
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not supported on this device.")
        self.command_queue = self.device.newCommandQueue()
        self.libraries: Dict[str, Metal.MTLLibrary] = {}
        # Buffer cache: (size, dtype) -> Metal buffer
        self.buffer_cache: Dict[Tuple[int, str], Any] = {}

    def compile_library(self, library_key: str, file_path: str):
        """
        Compiles a Metal source file into a library and caches it.

        Args:
            library_key (str): A unique key to identify the compiled library.
            file_path (str): The path to the .metal source file.
        """
        if library_key in self.libraries:
            return

        source = _resolve_metal_includes(file_path)
        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if error:
            raise Exception(f"Metal library compilation failed for {file_path}: {error}")
        self.libraries[library_key] = library

    def clear_buffer_cache(self):
        """
        Clears the buffer cache to free up memory.
        Call this periodically if memory usage becomes an issue.
        """
        self.buffer_cache.clear()

    def get_buffer_cache_info(self) -> Dict:
        """
        Returns information about the current buffer cache.
        
        Returns:
            Dict with cache statistics.
        """
        total_size = sum(buffer.length() for buffer in self.buffer_cache.values())
        return {
            'num_cached_buffers': len(self.buffer_cache),
            'total_cached_bytes': total_size,
            'total_cached_mb': total_size / (1024 * 1024)
        }

    def _get_or_create_buffer(self, np_array: np.ndarray) -> Any:
        """
        Gets a cached Metal buffer or creates a new one if needed.
        
        Args:
            np_array: The numpy array to create/find a buffer for.
            
        Returns:
            A Metal buffer suitable for the array.
        """
        buffer_key = (np_array.nbytes, str(np_array.dtype))
        
        if buffer_key in self.buffer_cache:
            buffer = self.buffer_cache[buffer_key]
            # Copy data into the existing buffer
            buffer_ptr = buffer.contents()
            buffer_ptr.as_buffer(np_array.nbytes)[:] = np_array.tobytes()
            return buffer
        else:
            # Create new buffer and cache it
            buffer = self.device.newBufferWithBytes_length_options_(
                np_array.tobytes(), np_array.nbytes, Metal.MTLResourceStorageModeShared
            )
            self.buffer_cache[buffer_key] = buffer
            return buffer

    def run_kernel(
        self,
        library_key: str,
        kernel_name: str,
        num_threads: int,
        buffers_np: List[np.ndarray],
    ):
        """
        Executes a Metal kernel with the given buffers.

        The numpy arrays in `buffers_np` are used to create Metal buffers.
        After kernel execution, the data from the Metal buffers is copied
        back into the original numpy arrays, effectively updating them in place.

        Args:
            library_key (str): The key of the compiled library to use.
            kernel_name (str): The name of the kernel function to execute.
            num_threads (int): The number of threads to launch.
            buffers_np (List[np.ndarray]): A list of numpy arrays to use as buffers.
        """
        library = self.libraries[library_key]
        kernel = library.newFunctionWithName_(kernel_name)
        pipeline = self.device.newComputePipelineStateWithFunction_error_(kernel, None)[0]

        # Get or create Metal buffers from numpy arrays (with caching)
        metal_buffers = []
        for np_array in buffers_np:
            buffer = self._get_or_create_buffer(np_array)
            metal_buffers.append(buffer)

        # Execute kernel
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(metal_buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Dispatch threads
        grid_size = Metal.MTLSize(num_threads, 1, 1)
        threadgroup_size = Metal.MTLSize(min(num_threads, pipeline.maxTotalThreadsPerThreadgroup()), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)

        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Copy data back from Metal buffers to the numpy arrays
        for i, np_array in enumerate(buffers_np):
            result_bytes = metal_buffers[i].contents().as_buffer(np_array.nbytes)
            updated_array = np.frombuffer(result_bytes, dtype=np_array.dtype).reshape(np_array.shape)
            np.copyto(np_array, updated_array)
