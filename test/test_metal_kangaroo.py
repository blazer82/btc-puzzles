import pytest
import numpy as np
import Metal as metal
import os
import re

# --- Helper functions copied from test_metal_field.py ---

def _resolve_metal_includes(file_path, included_files=None):
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

class MetalTestHelper:
    """
    A helper class for running Metal compute kernels for the kangaroo tests.
    This version is adapted to handle multiple input/output buffers of
    varying types, and modifies the numpy arrays in-place.
    """
    def __init__(self, kernel_file):
        self.kernel_file = kernel_file
        self.device = metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
                                                                                                                                   
        source = _resolve_metal_includes(kernel_file)

        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if error:
            raise Exception(f"Metal library compilation failed: {error}")
        self.library = library

    def run_kernel(self, kernel_name, all_buffers_np, num_threads=1):
        """
        Executes a Metal kernel, passing a list of numpy arrays as buffers.
        The contents of the numpy arrays are updated in-place with the results.
        """
        kernel = self.library.newFunctionWithName_(kernel_name)
        pipeline = self.device.newComputePipelineStateWithFunction_error_(kernel, None)[0]
        
        metal_buffers = []
        for np_array in all_buffers_np:
            buffer = self.device.newBufferWithBytes_length_options_(
                np_array.tobytes(), np_array.nbytes, 0)
            metal_buffers.append(buffer)
        
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        
        for i, buf in enumerate(metal_buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        
        grid_size = metal.MTLSize(num_threads, 1, 1)
        threadgroup_size = metal.MTLSize(min(num_threads, pipeline.maxTotalThreadsPerThreadgroup()), 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, threadgroup_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        for i, np_array in enumerate(all_buffers_np):
            result_bytes = metal_buffers[i].contents().as_buffer(np_array.nbytes)
            np.copyto(np_array, np.frombuffer(result_bytes, dtype=np_array.dtype).reshape(np_array.shape))

# --- Test Fixtures ---

@pytest.fixture
def helper():
    """Provides a MetalTestHelper for the kangaroo kernel."""
    test_kernel_path = "test/metal/test_kangaroo.metal"
    with open(test_kernel_path, "w") as f:
        f.write('#include "../../src/metal/kangaroo_hop.metal"\n')
    
    yield MetalTestHelper(test_kernel_path)
    
    os.remove(test_kernel_path)

@pytest.fixture
def hop_points_buffer():
    """Creates a buffer of 16 dummy hop points."""
    # A `ge` struct is 11 ulongs (5 for x, 5 for y, 1 for infinity flag).
    hop_points = np.zeros((16, 11), dtype=np.uint64)
    for i in range(16):
        hop_points[i, 0] = 1000 + i  # Dummy x-coordinate
        hop_points[i, 5] = 2000 + i  # Dummy y-coordinate
        hop_points[i, 10] = 0        # Not infinity
    return hop_points

# --- Test Classes ---

@pytest.mark.gpu
class TestKangarooKernel:
    def test_kangaroo_step_updates_state(self, helper, hop_points_buffer):
        """
        Tests that a single kangaroo step correctly updates the distance
        and modifies the point.
        """
        # 1. Setup initial state for one kangaroo
        # A `ge` struct is 11 ulongs.
        initial_point = np.zeros(11, dtype=np.uint64)
        initial_point[0] = 17  # x.n[0] = 17, so hop_index will be 17 % 16 = 1
        initial_point[5] = 1   # y.n[0] = 1
        
        points_buffer = np.array([initial_point])
        distances_buffer = np.array([0], dtype=np.uint64)
        dp_threshold = np.array([16], dtype=np.uint32) # High threshold, no DPs expected
        dp_counter = np.array([0], dtype=np.uint32)
        
        dp_output_dtype = np.dtype([('point', '(11,)uint64'), ('distance', 'uint64')])
        dp_output_buffer = np.zeros(1, dtype=dp_output_dtype)

        # 2. Run the kernel
        all_buffers = [points_buffer, distances_buffer, hop_points_buffer, dp_threshold, dp_counter, dp_output_buffer]
        helper.run_kernel("kangaroo_step", all_buffers, num_threads=1)

        # 3. Verify results from the modified buffers
        # Expected distance: initial (0) + 2^hop_index (2^1) = 2
        assert distances_buffer[0] == 2

        # Point should have changed
        assert not np.array_equal(points_buffer[0], initial_point)

        # DP counter should be unchanged
        assert dp_counter[0] == 0

    def test_kangaroo_step_reports_distinguished_point(self, helper, hop_points_buffer):
        """
        Tests that a kangaroo correctly reports a distinguished point when the
        threshold is met. We force this by setting the threshold to 0.
        """
        # 1. Setup initial state
        initial_point = np.zeros(11, dtype=np.uint64)
        initial_point[0] = 3 # hop_index = 3 % 16 = 3
        points_buffer = np.array([initial_point])
        distances_buffer = np.array([100], dtype=np.uint64)
        dp_threshold = np.array([0], dtype=np.uint32) # Threshold 0 means every point is distinguished
        dp_counter = np.array([0], dtype=np.uint32)
        
        dp_output_dtype = np.dtype([('point', '(11,)uint64'), ('distance', 'uint64')])
        dp_output_buffer = np.zeros(1, dtype=dp_output_dtype)

        # 2. Run the kernel
        all_buffers = [points_buffer, distances_buffer, hop_points_buffer, dp_threshold, dp_counter, dp_output_buffer]
        helper.run_kernel("kangaroo_step", all_buffers, num_threads=1)

        # 3. Verify results
        # DP counter should have been incremented to 1
        assert dp_counter[0] == 1

        # Check the reported DP result
        reported_distance = dp_output_buffer[0]['distance']
        reported_point_x = dp_output_buffer[0]['point'][0]

        # Expected distance: initial (100) + 2^hop_index (2^3=8) = 108
        assert reported_distance == 108

        # The reported point's x coord should not be the initial one
        assert reported_point_x != initial_point[0]
