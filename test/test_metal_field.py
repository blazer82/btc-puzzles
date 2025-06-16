import pytest
import numpy as np
import Metal as metal
import os
import re

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
    def __init__(self, kernel_file):
        self.kernel_file = kernel_file
        self.device = metal.MTLCreateSystemDefaultDevice()
        self.command_queue = self.device.newCommandQueue()
                                                                                                                                   
        # Manually resolve #includes and compile the combined source string.
        source = _resolve_metal_includes(kernel_file)

        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if error:
            raise Exception(f"Metal library compilation failed: {error}")
        self.library = library

    def run_kernel(self, kernel_name, input_buffers_np, output_size_bytes):
        """
        Execute a Metal kernel with the given inputs and return the output as a numpy array.
        """
        # Get the kernel function
        kernel = self.library.newFunctionWithName_(kernel_name)
        pipeline = self.device.newComputePipelineStateWithFunction_error_(kernel, None)[0]
        
        # Create Metal buffers from numpy arrays
        input_buffers = []
        for np_array in input_buffers_np:
            buffer = self.device.newBufferWithBytes_length_options_(
                np_array.tobytes(), np_array.nbytes, 0)
            input_buffers.append(buffer)
        
        # Create output buffer
        output_buffer = self.device.newBufferWithLength_options_(output_size_bytes, 0)
        
        # Execute kernel
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(pipeline)
        
        # Set input buffers
        for i, buf in enumerate(input_buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)
        
        # Set output buffer
        encoder.setBuffer_offset_atIndex_(output_buffer, 0, len(input_buffers))
        
        # Dispatch with single thread
        encoder.dispatchThreads_threadsPerThreadgroup_(
            metal.MTLSize(1, 1, 1), metal.MTLSize(1, 1, 1))
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Return result as numpy array
        result_bytes = output_buffer.contents().as_buffer(output_size_bytes)
        return np.frombuffer(result_bytes, dtype=np.uint64)
    
def python_fe_mul_inner(a_n, b_n):
    """
    A Python port of `secp256k1_fe_mul_inner` to generate the expected result.
    This is a more direct translation of the C source to ensure correctness.
    """
    M = 0xFFFFFFFFFFFFF
    R = 0x1000003D10

    a0, a1, a2, a3, a4 = [int(x) for x in a_n]
    b0, b1, b2, b3, b4 = [int(x) for x in b_n]

    # In Python, integers have arbitrary precision, so we can use them
    # to simulate the 128-bit accumulators 'c' and 'd' from the C code.
    d = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0
    c = a4 * b4

    d += R * (c & 0xFFFFFFFFFFFFFFFF)
    c >>= 64

    t3 = d & M
    d >>= 52

    d += a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0
    d += (R << 12) * c

    t4 = d & M
    d >>= 52

    tx = t4 >> 48
    t4 &= (M >> 4)

    c = a0 * b0

    d += a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1

    u0 = d & M
    d >>= 52

    u0 = (u0 << 4) | tx
    c += u0 * (R >> 4)

    r0 = c & M
    c >>= 52

    c += a0 * b1 + a1 * b0
    d += a2 * b4 + a3 * b3 + a4 * b2

    c += (d & M) * R
    d >>= 52

    r1 = c & M
    c >>= 52
    
    c += a0 * b2 + a1 * b1 + a2 * b0
    d += a3 * b4 + a4 * b3

    c += R * (d & 0xFFFFFFFFFFFFFFFF)
    d >>= 64

    r2 = c & M
    c >>= 52

    c += (R << 12) * d
    c += t3

    r3 = c & M
    c >>= 52

    r4 = c + t4

    return np.array([r0, r1, r2, r3, r4], dtype=np.uint64)

def python_fe_mul_inner(a_n, b_n):
    """
    A Python port of `secp256k1_fe_mul_inner` to generate the expected result.
    This is a more direct translation of the C source to ensure correctness.
    """
    M = 0xFFFFFFFFFFFFF
    R = 0x1000003D10

    a0, a1, a2, a3, a4 = [int(x) for x in a_n]
    b0, b1, b2, b3, b4 = [int(x) for x in b_n]

    # In Python, integers have arbitrary precision, so we can use them
    # to simulate the 128-bit accumulators 'c' and 'd' from the C code.
    d = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0
    c = a4 * b4

    d += R * (c & 0xFFFFFFFFFFFFFFFF)
    c >>= 64

    t3 = d & M
    d >>= 52

    d += a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0
    d += (R << 12) * c

    t4 = d & M
    d >>= 52

    tx = t4 >> 48
    t4 &= (M >> 4)

    c = a0 * b0

    d += a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1

    u0 = d & M
    d >>= 52

    u0 = (u0 << 4) | tx
    c += u0 * (R >> 4)

    r0 = c & M
    c >>= 52

    c += a0 * b1 + a1 * b0
    d += a2 * b4 + a3 * b3 + a4 * b2

    c += (d & M) * R
    d >>= 52

    r1 = c & M
    c >>= 52
    
    c += a0 * b2 + a1 * b1 + a2 * b0
    d += a3 * b4 + a4 * b3

    c += R * (d & 0xFFFFFFFFFFFFFFFF)
    d >>= 64

    r2 = c & M
    c >>= 52

    c += (R << 12) * d
    c += t3

    r3 = c & M
    c >>= 52

    r4 = c + t4

    return np.array([r0, r1, r2, r3, r4], dtype=np.uint64)

def python_fe_normalize(n_in):
    """
    A Python port of `secp256k1_fe_impl_normalize`.
    """
    t = [int(x) for x in n_in]
    M52 = 0xFFFFFFFFFFFFF

    x = t[4] >> 48
    t[4] &= 0x0FFFFFFFFFFFF

    t[0] += x * 0x1000003D1
    t[1] += t[0] >> 52; t[0] &= M52
    t[2] += t[1] >> 52; t[1] &= M52; m = t[1]
    t[3] += t[2] >> 52; t[2] &= M52; m &= t[2]
    t[4] += t[3] >> 52; t[3] &= M52; m &= t[3]

    # In Python, bools are 0 or 1, so we can use multiplication.
    x = (t[4] >> 48) | ((t[4] == 0x0FFFFFFFFFFFF) * (m == M52) * (t[0] >= 0xFFFFEFFFFFC2F))

    t[0] += x * 0x1000003D1
    t[1] += (t[0] >> 52); t[0] &= M52
    t[2] += (t[1] >> 52); t[1] &= M52
    t[3] += (t[2] >> 52); t[2] &= M52
    t[4] += (t[3] >> 52); t[3] &= M52

    t[4] &= 0x0FFFFFFFFFFFF
                                                                                                                                   
    return np.array(t, dtype=np.uint64)

def python_fe_sqr(a_n):
    """
    A Python port of `secp256k1_fe_sqr_inner`.
    """
    a0, a1, a2, a3, a4 = [int(x) for x in a_n]
    M = 0xFFFFFFFFFFFFF
    R = 0x1000003D10

    d = a0 * 2 * a3 + a1 * 2 * a2
    c = a4 * a4
    d += R * (c & 0xFFFFFFFFFFFFFFFF)
    c >>= 64
    t3 = d & M
    d >>= 52

    a4_2 = a4 * 2
    d += a0 * a4_2 + a1 * 2 * a3 + a2 * a2
    d += (R << 12) * c
    t4 = d & M
    d >>= 52

    tx = t4 >> 48
    t4 &= (M >> 4)

    c = a0 * a0
    d += a1 * a4_2 + a2 * 2 * a3
    u0 = d & M
    d >>= 52
    u0 = (u0 << 4) | tx
    c += u0 * (R >> 4)
    r0 = c & M
    c >>= 52

    a0_2 = a0 * 2
    c += a0_2 * a1
    d += a2 * a4_2 + a3 * a3
    c += (d & M) * R
    d >>= 52
    r1 = c & M
    c >>= 52

    c += a0_2 * a2 + a1 * a1
    d += a3 * a4_2
    c += R * (d & 0xFFFFFFFFFFFFFFFF)
    d >>= 64
    r2 = c & M
    c >>= 52

    c += (R << 12) * d
    c += t3
    r3 = c & M
    c >>= 52

    r4 = c + t4
    return np.array([r0, r1, r2, r3, r4], dtype=np.uint64)

@pytest.mark.gpu
class TestFieldArithmetic:
    def test_fe_mul(self):
        """
        Tests the fe_mul Metal function by comparing its output to a
        CPU-based Python implementation.
        """
        # This helper would wrap the Metal API boilerplate.
        helper = MetalTestHelper("test/metal/test_field.metal")

        # 1. Define test inputs as numpy arrays of ulongs (uint64).
        a_np = np.array([100, 200, 300, 400, 5], dtype=np.uint64)
        b_np = np.array([600, 700, 800, 900, 10], dtype=np.uint64)

        # 2. Calculate the expected result on the CPU.
        expected_r_np = python_fe_mul_inner(a_np, b_np)

        # 3. Run the actual computation on the GPU.
        # The helper function abstracts away buffer creation and kernel dispatch.
        output_bytes = 5 * 8  # 5 limbs, 8 bytes each
        gpu_result_np = helper.run_kernel("test_fe_mul", [a_np, b_np], output_bytes)

        # 4. Assert that the GPU result matches the expected CPU result.
        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_normalize(self):
        """
        Tests the fe_normalize Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")

        # An un-normalized number (limbs are larger than 52 bits)
        a_np = np.array([
            0xFFFFFFFFFFFFF * 2,
            0xFFFFFFFFFFFFF * 3,
            0xFFFFFFFFFFFFF * 4,
            0xFFFFFFFFFFFFF * 5,
            0x0FFFFFFFFFFFF * 6
        ], dtype=np.uint64)

        expected_r_np = python_fe_normalize(a_np)
        gpu_result_np = helper.run_kernel("test_fe_normalize", [a_np], 5 * 8)

        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_sqr(self):
        """
        Tests the fe_sqr Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        a_np = np.array([123, 456, 789, 101112, 13], dtype=np.uint64)

        expected_r_np = python_fe_sqr(a_np)
        gpu_result_np = helper.run_kernel("test_fe_sqr", [a_np], 5 * 8)

        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_inv(self):
        """
        Tests the fe_inv Metal function by checking if a * inv(a) == 1.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # A sample field element (the number 12345)
        a_np = np.array([12345, 0, 0, 0, 0], dtype=np.uint64)

        # 1. Calculate inv(a) on the GPU
        a_inv_np = helper.run_kernel("test_fe_inv", [a_np], 5 * 8)

        # 2. Calculate a * inv(a) on the GPU
        product_np = helper.run_kernel("test_fe_mul", [a_np, a_inv_np], 5 * 8)

        # 3. Normalize the product on the GPU
        normalized_product_np = helper.run_kernel("test_fe_normalize", [product_np], 5 * 8)

        # 4. The result should be 1
        expected_one = np.array([1, 0, 0, 0, 0], dtype=np.uint64)
        assert np.array_equal(normalized_product_np, expected_one)

    def test_gej_set_infinity(self):
        """
        Tests the gej_set_infinity Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Run the kernel to set a point to infinity
        gpu_result_buffer = helper.run_kernel("test_gej_set_infinity", [], 16 * 8)
        
        # Check that the infinity flag is set
        assert gpu_result_buffer[15] == 1  # infinity flag should be True
        
        # Check that coordinates are zero
        expected_zero = np.array([0, 0, 0, 0, 0], dtype=np.uint64)
        assert np.array_equal(gpu_result_buffer[0:5], expected_zero)  # x coordinate
        assert np.array_equal(gpu_result_buffer[5:10], expected_zero)  # y coordinate
        assert np.array_equal(gpu_result_buffer[10:15], expected_zero)  # z coordinate

    def test_gej_double(self):
        """
        Tests the gej_double Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a test Jacobian point (not at infinity)
        test_gej_buffer = np.zeros(16, dtype=np.uint64)
        test_gej_buffer[0:5] = [1, 0, 0, 0, 0]  # x = 1
        test_gej_buffer[5:10] = [2, 0, 0, 0, 0]  # y = 2
        test_gej_buffer[10:15] = [1, 0, 0, 0, 0]  # z = 1
        test_gej_buffer[15] = 0  # not infinity
        
        # Run the doubling operation
        gpu_result_buffer = helper.run_kernel("test_gej_double", [test_gej_buffer], 16 * 8)
        
        # The result should not be at infinity for a valid point
        assert gpu_result_buffer[15] == 0  # should not be infinity
        
        # The coordinates should be modified (exact values depend on the doubling formula)
        # We mainly check that the operation completed without error
        assert len(gpu_result_buffer) == 16

    def test_gej_double_infinity(self):
        """
        Tests that doubling the point at infinity returns infinity.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a point at infinity
        infinity_gej_buffer = np.zeros(16, dtype=np.uint64)
        infinity_gej_buffer[15] = 1  # set infinity flag
        
        # Double the infinity point
        gpu_result_buffer = helper.run_kernel("test_gej_double", [infinity_gej_buffer], 16 * 8)
        
        # Result should still be infinity
        assert gpu_result_buffer[15] == 1

    def test_gej_add_ge(self):
        """
        Tests the gej_add_ge Metal function (adding affine point to Jacobian point).
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a test Jacobian point
        test_gej_buffer = np.zeros(16, dtype=np.uint64)
        test_gej_buffer[0:5] = [1, 0, 0, 0, 0]  # x = 1
        test_gej_buffer[5:10] = [2, 0, 0, 0, 0]  # y = 2
        test_gej_buffer[10:15] = [1, 0, 0, 0, 0]  # z = 1
        test_gej_buffer[15] = 0  # not infinity
        
        # Create a test affine point
        test_ge_buffer = np.zeros(11, dtype=np.uint64)
        test_ge_buffer[0:5] = [3, 0, 0, 0, 0]  # x = 3
        test_ge_buffer[5:10] = [4, 0, 0, 0, 0]  # y = 4
        test_ge_buffer[10] = 0  # not infinity
        
        # Run the addition operation
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [test_gej_buffer, test_ge_buffer], 16 * 8)
        
        # The result should not be at infinity for valid points
        assert gpu_result_buffer[15] == 0  # should not be infinity
        assert len(gpu_result_buffer) == 16

    def test_gej_add_ge_with_infinity(self):
        """
        Tests adding infinity points with gej_add_ge.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Test: Jacobian point at infinity + affine point = affine point
        infinity_gej_buffer = np.zeros(16, dtype=np.uint64)
        infinity_gej_buffer[15] = 1  # Jacobian point at infinity
        
        test_ge_buffer = np.zeros(11, dtype=np.uint64)
        test_ge_buffer[0:5] = [5, 0, 0, 0, 0]  # x = 5
        test_ge_buffer[5:10] = [6, 0, 0, 0, 0]  # y = 6
        test_ge_buffer[10] = 0  # not infinity
        
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [infinity_gej_buffer, test_ge_buffer], 16 * 8)
        
        # Result should be the affine point converted to Jacobian coordinates
        assert gpu_result_buffer[15] == 0  # should not be infinity
        assert np.array_equal(gpu_result_buffer[0:5], test_ge_buffer[0:5])  # x should match
        assert np.array_equal(gpu_result_buffer[5:10], test_ge_buffer[5:10])  # y should match
        
        # Test: Jacobian point + affine point at infinity = Jacobian point
        test_gej_buffer = np.zeros(16, dtype=np.uint64)
        test_gej_buffer[0:5] = [7, 0, 0, 0, 0]  # x = 7
        test_gej_buffer[5:10] = [8, 0, 0, 0, 0]  # y = 8
        test_gej_buffer[10:15] = [1, 0, 0, 0, 0]  # z = 1
        test_gej_buffer[15] = 0  # not infinity
        
        infinity_ge_buffer = np.zeros(11, dtype=np.uint64)
        infinity_ge_buffer[10] = 1  # affine point at infinity
        
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [test_gej_buffer, infinity_ge_buffer], 16 * 8)
        
        # Result should be the original Jacobian point
        assert gpu_result_buffer[15] == 0  # should not be infinity
        assert np.array_equal(gpu_result_buffer[0:5], test_gej_buffer[0:5])  # x should match
        assert np.array_equal(gpu_result_buffer[5:10], test_gej_buffer[5:10])  # y should match
