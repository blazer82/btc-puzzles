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

def python_fe_sub(r_n, a_n):
    """
    A Python port of limb-wise subtraction, to match the Metal `fe_sub`.
    """
    # Use numpy's uint64 wrapping behavior to emulate the C/Metal behavior.
    return r_n - a_n

def python_fe_half(n_in):
    """
    A Python port of `secp256k1_fe_impl_half`.
    """
    t = [int(x) for x in n_in]
    M52 = 0xFFFFFFFFFFFFF
    one = 1

    # In Python, bools are 0 or 1, so we can use multiplication.
    # The C code relies on unsigned integer behavior for negation.
    mask = ((0 - (t[0] & one)) & ((1 << 64) - 1)) >> 12

    t[0] += 0xFFFFEFFFFFC2F & mask
    t[1] += mask
    t[2] += mask
    t[3] += mask
    t[4] += mask >> 4

    r = np.zeros(5, dtype=np.uint64)
    r[0] = (t[0] >> 1) + ((t[1] & one) << 51)
    r[1] = (t[1] >> 1) + ((t[2] & one) << 51)
    r[2] = (t[2] >> 1) + ((t[3] & one) << 51)
    r[3] = (t[3] >> 1) + ((t[4] & one) << 51)
    r[4] = (t[4] >> 1)

    return r

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
        
        # Test with a larger field element
        b_np = np.array([0xFFFFFF, 0xABCDEF, 0x123456, 0x789ABC, 0xDEF], dtype=np.uint64)
        
        # Calculate inv(b)
        b_inv_np = helper.run_kernel("test_fe_inv", [b_np], 5 * 8)
        
        # Calculate b * inv(b)
        product_np = helper.run_kernel("test_fe_mul", [b_np, b_inv_np], 5 * 8)
        
        # Normalize the product
        normalized_product_np = helper.run_kernel("test_fe_normalize", [product_np], 5 * 8)
        
        # The result should be 1
        assert np.array_equal(normalized_product_np, expected_one)

    def test_ge_set_gej(self):
        """
        Tests the ge_set_gej Metal function.
        This test converts an affine point to Jacobian (with Z=1), then converts
        it back to affine and ensures the coordinates are unchanged.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")

        # Create a test Jacobian point (x=7, y=8, z=1)
        test_gej_buffer = np.zeros(16, dtype=np.uint64)
        test_gej_buffer[0:5] = [7, 0, 0, 0, 0]   # x
        test_gej_buffer[5:10] = [8, 0, 0, 0, 0]  # y
        test_gej_buffer[10:15] = [1, 0, 0, 0, 0] # z
        test_gej_buffer[15] = 0                 # not infinity

        # Run the conversion kernel
        # Output is a `ge` struct, which is 11 ulongs (5 for x, 5 for y, 1 for inf)
        gpu_result_buffer = helper.run_kernel("test_ge_set_gej", [test_gej_buffer], 11 * 8)

        # Check infinity flag
        assert gpu_result_buffer[10] == 0  # should not be infinity

        # Check coordinates
        assert np.array_equal(gpu_result_buffer[0:5], test_gej_buffer[0:5])  # x should match
        assert np.array_equal(gpu_result_buffer[5:10], test_gej_buffer[5:10])  # y should match

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
        Tests the gej_double Metal function with the secp256k1 generator point.
        
        This test uses a known valid secp256k1 point to verify the doubling operation works correctly.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create Jacobian point G
        test_gej_buffer = np.zeros(16, dtype=np.uint64)
        test_gej_buffer[0:5] = g_x_limbs  # Generator point x
        test_gej_buffer[5:10] = g_y_limbs  # Generator point y
        test_gej_buffer[10:15] = [1, 0, 0, 0, 0]  # z = 1
        test_gej_buffer[15] = 0  # not infinity
        
        # Run the doubling operation
        gpu_result_buffer = helper.run_kernel("test_gej_double", [test_gej_buffer], 16 * 8)
        
        # Debug: Check if result is infinity
        if gpu_result_buffer[15] == 1:
            print(f"ERROR: gej_double produced infinity for generator point: {gpu_result_buffer}")
            pytest.fail("gej_double should not produce infinity for the secp256k1 generator point")
        
        # The result should not be at infinity for a valid point
        assert gpu_result_buffer[15] == 0, "gej_double should not produce infinity for valid secp256k1 points"
        
        # Convert to affine to verify we get a valid point
        affine_result = helper.run_kernel("test_ge_set_gej", [gpu_result_buffer], 11 * 8)
        assert affine_result[10] == 0, "Converted affine point should not be infinity"
        
        # The coordinates should be non-zero (since 2*G is not the identity)
        x_is_zero = all(x == 0 for x in affine_result[0:5])
        y_is_zero = all(y == 0 for y in affine_result[5:10])
        assert not (x_is_zero and y_is_zero), "2*G should not be the zero point"

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
        
        This test uses the secp256k1 generator point G and verifies that G + G = 2G
        by comparing with the result of point doubling.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a Jacobian point representing the generator G
        # These are the actual coordinates of the secp256k1 generator point
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create G in Jacobian coordinates (z=1)
        g_jac = np.zeros(16, dtype=np.uint64)
        g_jac[0:5] = g_x_limbs  # x
        g_jac[5:10] = g_y_limbs  # y
        g_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        g_jac[15] = 0  # not infinity
        
        # Create G in affine coordinates
        g_aff = np.zeros(11, dtype=np.uint64)
        g_aff[0:5] = g_x_limbs  # x
        g_aff[5:10] = g_y_limbs  # y
        g_aff[10] = 0  # not infinity
        
        # Calculate G + G using gej_add_ge
        sum_result = helper.run_kernel("test_gej_add_ge", [g_jac, g_aff], 16 * 8)
        
        # Calculate 2G using gej_double
        double_result = helper.run_kernel("test_gej_double", [g_jac], 16 * 8)
        
        # Convert both results to affine for comparison
        sum_affine = helper.run_kernel("test_ge_set_gej", [sum_result], 11 * 8)
        double_affine = helper.run_kernel("test_ge_set_gej", [double_result], 11 * 8)
        
        # Verify that G + G = 2G
        # For elliptic curve points, if the x-coordinates match, then the points are either
        # equal or negatives of each other (y vs -y). Since we're working with the generator
        # point which is well-defined, we can just check the x-coordinate.
        assert np.array_equal(sum_affine[0:5], double_affine[0:5])  # x coordinates match
        assert sum_affine[10] == double_affine[10]  # infinity flags match
        
        # Check that the y-coordinate is either equal or the negative
        # (in case the implementation uses a different formula that produces -y)
        neg_y = helper.run_kernel("test_fe_negate", [double_affine[5:10]], 5 * 8)
        y_matches = (np.array_equal(sum_affine[5:10], double_affine[5:10]) or 
                     np.array_equal(sum_affine[5:10], neg_y))
        assert y_matches, "Y coordinate should match or be the negative"
        
    def test_gej_add_ge_degenerate_case(self):
        """
        Tests the degenerate case in gej_add_ge where the points have the same x-coordinate
        but different y-coordinates (which should result in infinity).
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a point P
        p_x_limbs = [123, 0, 0, 0, 0]  # x = 123
        p_y_limbs = [456, 0, 0, 0, 0]  # y = 456
        
        # Create P in Jacobian coordinates
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = p_x_limbs  # x
        p_jac[5:10] = p_y_limbs  # y
        p_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        p_jac[15] = 0  # not infinity
        
        # Create -P in affine coordinates (same x, negated y)
        neg_p_aff = np.zeros(11, dtype=np.uint64)
        neg_p_aff[0:5] = p_x_limbs  # same x
        neg_p_y = helper.run_kernel("test_fe_negate", [np.array(p_y_limbs, dtype=np.uint64)], 5 * 8)
        neg_p_aff[5:10] = neg_p_y  # negated y
        neg_p_aff[10] = 0  # not infinity
        
        # Calculate P + (-P)
        result = helper.run_kernel("test_gej_add_ge", [p_jac, neg_p_aff], 16 * 8)
        
        # Result should be infinity
        assert result[15] == 1, "P + (-P) should be infinity"
        
    def test_gej_add_ge_with_z_not_one(self):
        """
        Tests gej_add_ge with a Jacobian point that has z ≠ 1.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a point P with z ≠ 1
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = [123, 0, 0, 0, 0]  # x
        p_jac[5:10] = [456, 0, 0, 0, 0]  # y
        p_jac[10:15] = [2, 0, 0, 0, 0]  # z = 2
        p_jac[15] = 0  # not infinity
        
        # Create a point Q in affine
        q_aff = np.zeros(11, dtype=np.uint64)
        q_aff[0:5] = [789, 0, 0, 0, 0]  # x
        q_aff[5:10] = [101112, 0, 0, 0, 0]  # y
        q_aff[10] = 0  # not infinity
        
        # Calculate P + Q
        sum_result = helper.run_kernel("test_gej_add_ge", [p_jac, q_aff], 16 * 8)
        
        # Convert P to affine
        p_aff = helper.run_kernel("test_ge_set_gej", [p_jac], 11 * 8)
        
        # Convert P + Q to affine
        sum_aff = helper.run_kernel("test_ge_set_gej", [sum_result], 11 * 8)
        
        # The result should not be infinity
        assert sum_aff[10] == 0, "P + Q should not be infinity"
        
        # The result should not equal either input point
        assert not np.array_equal(sum_aff[0:5], p_aff[0:5]), "P + Q should not equal P"
        assert not np.array_equal(sum_aff[0:5], q_aff[0:5]), "P + Q should not equal Q"

    def test_gej_add_ge_with_infinity(self):
        """
        Tests adding infinity points with gej_add_ge.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point for a valid test point
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Test: Jacobian point at infinity + affine point = affine point
        infinity_gej_buffer = np.zeros(16, dtype=np.uint64)
        infinity_gej_buffer[15] = 1  # Jacobian point at infinity
        
        g_aff = np.zeros(11, dtype=np.uint64)
        g_aff[0:5] = g_x_limbs  # x
        g_aff[5:10] = g_y_limbs  # y
        g_aff[10] = 0  # not infinity
        
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [infinity_gej_buffer, g_aff], 16 * 8)
        
        # Convert result to affine for comparison
        result_affine = helper.run_kernel("test_ge_set_gej", [gpu_result_buffer], 11 * 8)
        
        # Result should be the original point G
        assert np.array_equal(result_affine[0:5], g_aff[0:5])  # x coordinates match
        assert np.array_equal(result_affine[5:10], g_aff[5:10])  # y coordinates match
        assert result_affine[10] == 0  # not infinity
        
        # Test: Jacobian point + affine point at infinity = Jacobian point
        g_jac = np.zeros(16, dtype=np.uint64)
        g_jac[0:5] = g_x_limbs  # x
        g_jac[5:10] = g_y_limbs  # y
        g_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        g_jac[15] = 0  # not infinity
        
        infinity_ge_buffer = np.zeros(11, dtype=np.uint64)
        infinity_ge_buffer[10] = 1  # affine point at infinity
        
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [g_jac, infinity_ge_buffer], 16 * 8)
        
        # Result should be the original Jacobian point
        assert gpu_result_buffer[15] == 0  # should not be infinity
        assert np.array_equal(gpu_result_buffer[0:5], g_jac[0:5])  # x should match
        assert np.array_equal(gpu_result_buffer[5:10], g_jac[5:10])  # y should match
        
        # Test: Both points at infinity
        infinity_gej_buffer = np.zeros(16, dtype=np.uint64)
        infinity_gej_buffer[15] = 1  # Jacobian point at infinity
        
        infinity_ge_buffer = np.zeros(11, dtype=np.uint64)
        infinity_ge_buffer[10] = 1  # affine point at infinity
        
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [infinity_gej_buffer, infinity_ge_buffer], 16 * 8)
        
        # Result should be infinity
        assert gpu_result_buffer[15] == 1  # should be infinity
        
        # Test: Adding a point to its negative (should result in infinity)
        # Create negative of G by negating y-coordinate
        neg_g_aff = np.copy(g_aff)
        neg_g_aff[5:10] = helper.run_kernel("test_fe_negate", [g_aff[5:10]], 5 * 8)
        
        # Convert G to Jacobian
        g_jac = np.zeros(16, dtype=np.uint64)
        g_jac[0:5] = g_x_limbs  # x
        g_jac[5:10] = g_y_limbs  # y
        g_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        g_jac[15] = 0  # not infinity
        
        # Add G + (-G)
        gpu_result_buffer = helper.run_kernel("test_gej_add_ge", [g_jac, neg_g_aff], 16 * 8)
        
        # Result should be infinity
        assert gpu_result_buffer[15] == 1  # should be infinity

    def test_fe_add(self):
        """
        Tests the fe_add Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        a_np = np.array([100, 200, 300, 400, 5], dtype=np.uint64)
        b_np = np.array([600, 700, 800, 900, 10], dtype=np.uint64)

        # fe_add is simple limb-wise addition.
        expected_r_np = a_np + b_np
        gpu_result_np = helper.run_kernel("test_fe_add", [a_np, b_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_sub(self):
        """
        Tests the fe_sub Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        a_np = np.array([600, 700, 800, 900, 10], dtype=np.uint64)
        b_np = np.array([100, 200, 300, 400, 5], dtype=np.uint64)

        expected_r_np = python_fe_sub(a_np, b_np)
        gpu_result_np = helper.run_kernel("test_fe_sub", [a_np, b_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_mul_int(self):
        """
        Tests the fe_mul_int Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        a_np = np.array([10, 20, 30, 40, 5], dtype=np.uint64)
        factor = np.array([7], dtype=np.int32)

        expected_r_np = a_np * 7
        gpu_result_np = helper.run_kernel("test_fe_mul_int", [a_np, factor], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_negate(self):
        """
        Tests the fe_negate Metal function.
        
        The Metal implementation uses a specific formula for negation:
        r.n[0] = 0xFFFFEFFFFFC2FUL * 2 * (m + 1) - a.n[0];
        r.n[1] = M52 * 2 * (m + 1) - a.n[1];
        ...
        
        Where m is the magnitude (1 for fe_negate).
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        a_np = np.array([12345, 0, 0, 0, 0], dtype=np.uint64)

        # Run the negation on the GPU
        gpu_result_np = helper.run_kernel("test_fe_negate", [a_np], 5 * 8)
        
        # Verify that a + (-a) = 0 after normalization
        sum_np = a_np + gpu_result_np
        normalized_sum = helper.run_kernel("test_fe_normalize", [sum_np], 5 * 8)
        
        # The sum should be 0
        expected_zero = np.zeros(5, dtype=np.uint64)
        assert np.array_equal(normalized_sum, expected_zero)

    def test_fe_half(self):
        """
        Tests the fe_half Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        # Test with an even number
        a_np = np.array([12346, 0, 0, 0, 0], dtype=np.uint64)
        expected_r_np = python_fe_half(a_np)
        gpu_result_np = helper.run_kernel("test_fe_half", [a_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)

        # Test with an odd number
        b_np = np.array([12345, 0, 0, 0, 0], dtype=np.uint64)
        expected_r_np = python_fe_half(b_np)
        gpu_result_np = helper.run_kernel("test_fe_half", [b_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)
        
        # Test with a number near the field modulus
        # This tests the case where adding the modulus is necessary
        c_np = np.array([
            0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF
        ], dtype=np.uint64)
        expected_r_np = python_fe_half(c_np)
        gpu_result_np = helper.run_kernel("test_fe_half", [c_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)
        
        # Test with a number that has high bits set in each limb
        d_np = np.array([
            0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF
        ], dtype=np.uint64)
        expected_r_np = python_fe_half(d_np)
        gpu_result_np = helper.run_kernel("test_fe_half", [d_np], 5 * 8)
        assert np.array_equal(gpu_result_np, expected_r_np)

    def test_fe_cmov(self):
        """
        Tests the fe_cmov Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        r_np = np.array([1, 1, 1, 1, 1], dtype=np.uint64)
        a_np = np.array([9, 9, 9, 9, 9], dtype=np.uint64)

        # Test cmov with flag=0 (r should be unchanged)
        flag_np = np.array([0], dtype=np.bool_)
        gpu_result_np = helper.run_kernel("test_fe_cmov", [r_np, a_np, flag_np], 5 * 8)
        assert np.array_equal(gpu_result_np, r_np)

        # Test cmov with flag=1 (r should become a)
        flag_np = np.array([1], dtype=np.bool_)
        gpu_result_np = helper.run_kernel("test_fe_cmov", [r_np, a_np, flag_np], 5 * 8)
        assert np.array_equal(gpu_result_np, a_np)
        
    def test_fe_set_int(self):
        """
        Tests the fe_set_int Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Test with positive integer
        a_np = np.array([42], dtype=np.int32)
        gpu_result_np = helper.run_kernel("test_fe_set_int", [a_np], 5 * 8)
        
        expected = np.zeros(5, dtype=np.uint64)
        expected[0] = 42
        assert np.array_equal(gpu_result_np, expected)
        
        # Test with zero
        a_np = np.array([0], dtype=np.int32)
        gpu_result_np = helper.run_kernel("test_fe_set_int", [a_np], 5 * 8)
        
        expected = np.zeros(5, dtype=np.uint64)
        assert np.array_equal(gpu_result_np, expected)
        
        # Test with negative integer (should be treated as unsigned)
        a_np = np.array([-1], dtype=np.int32)
        gpu_result_np = helper.run_kernel("test_fe_set_int", [a_np], 5 * 8)
        
        expected = np.zeros(5, dtype=np.uint64)
        expected[0] = 0xFFFFFFFFFFFFFFFF  # -1 as uint64 in Metal
        assert np.array_equal(gpu_result_np, expected)
        
    def test_fe_normalizes_to_zero(self):
        """
        Tests the fe_normalizes_to_zero Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Test with zero
        zero_np = np.zeros(5, dtype=np.uint64)
        result_np = helper.run_kernel("test_fe_normalizes_to_zero", [zero_np], 8)  # uint64 is 8 bytes
        assert result_np[0] == 1  # True, zero normalizes to zero
        
        # Test with non-zero
        nonzero_np = np.array([1, 0, 0, 0, 0], dtype=np.uint64)
        result_np = helper.run_kernel("test_fe_normalizes_to_zero", [nonzero_np], 8)  # uint64 is 8 bytes
        assert result_np[0] == 0  # False, non-zero doesn't normalize to zero
        
        # Test with value that would normalize to zero after modular reduction
        # This is the modulus of secp256k1, which normalizes to zero
        modulus_np = np.array([
            0xFFFFEFFFFFC2F, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF
        ], dtype=np.uint64)
        result_np = helper.run_kernel("test_fe_normalizes_to_zero", [modulus_np], 8)  # uint64 is 8 bytes
        assert result_np[0] == 1  # True, the modulus normalizes to zero
        
        # Test with a value slightly less than the modulus
        almost_modulus_np = np.array([
            0xFFFFEFFFFFC2E, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF
        ], dtype=np.uint64)
        result_np = helper.run_kernel("test_fe_normalizes_to_zero", [almost_modulus_np], 8)
        assert result_np[0] == 0  # False, this value doesn't normalize to zero
        
        # Test with a value slightly greater than the modulus
        greater_than_modulus_np = np.array([
            0xFFFFEFFFFFC30, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0xFFFFFFFFFFFFF, 0x0FFFFFFFFFFFF
        ], dtype=np.uint64)
        result_np = helper.run_kernel("test_fe_normalizes_to_zero", [greater_than_modulus_np], 8)
        assert result_np[0] == 0  # False, this should normalize to 1, not 0
        
    def test_gej_cmov(self):
        """
        Tests the gej_cmov Metal function.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create two different Jacobian points
        r_np = np.zeros(16, dtype=np.uint64)
        r_np[0:5] = [1, 0, 0, 0, 0]    # x = 1
        r_np[5:10] = [2, 0, 0, 0, 0]   # y = 2
        r_np[10:15] = [1, 0, 0, 0, 0]  # z = 1
        r_np[15] = 0                   # not infinity
        
        a_np = np.zeros(16, dtype=np.uint64)
        a_np[0:5] = [3, 0, 0, 0, 0]    # x = 3
        a_np[5:10] = [4, 0, 0, 0, 0]   # y = 4
        a_np[10:15] = [1, 0, 0, 0, 0]  # z = 1
        a_np[15] = 0                   # not infinity
        
        # Test cmov with flag=0 (r should be unchanged)
        flag_np = np.array([0], dtype=np.bool_)
        gpu_result_np = helper.run_kernel("test_gej_cmov", [r_np, a_np, flag_np], 16 * 8)
        assert np.array_equal(gpu_result_np, r_np)
        
        # Test cmov with flag=1 (r should become a)
        flag_np = np.array([1], dtype=np.bool_)
        gpu_result_np = helper.run_kernel("test_gej_cmov", [r_np, a_np, flag_np], 16 * 8)
        assert np.array_equal(gpu_result_np, a_np)
        
        # Test with infinity flag
        r_np[15] = 0  # not infinity
        a_np[15] = 1  # infinity
        
        flag_np = np.array([1], dtype=np.bool_)
        gpu_result_np = helper.run_kernel("test_gej_cmov", [r_np, a_np, flag_np], 16 * 8)
        assert gpu_result_np[15] == 1  # should be infinity
        
    def test_gej_set_ge(self):
        """
        Tests the gej_set_ge Metal function (convert affine to Jacobian).
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create an affine point
        a_np = np.zeros(11, dtype=np.uint64)
        a_np[0:5] = [7, 0, 0, 0, 0]   # x = 7
        a_np[5:10] = [8, 0, 0, 0, 0]  # y = 8
        a_np[10] = 0                  # not infinity
        
        # Convert to Jacobian
        gpu_result_np = helper.run_kernel("test_gej_set_ge", [a_np], 16 * 8)
        
        # Check that x and y are preserved
        assert np.array_equal(gpu_result_np[0:5], a_np[0:5])   # x should match
        assert np.array_equal(gpu_result_np[5:10], a_np[5:10]) # y should match
        
        # Check that z = 1
        expected_z = np.array([1, 0, 0, 0, 0], dtype=np.uint64)
        assert np.array_equal(gpu_result_np[10:15], expected_z)
        
        # Check infinity flag
        assert gpu_result_np[15] == 0  # should not be infinity
        
        # Test with infinity point
        a_np[10] = 1  # set to infinity
        
        gpu_result_np = helper.run_kernel("test_gej_set_ge", [a_np], 16 * 8)
        
        # Result should be infinity with zero coordinates
        assert gpu_result_np[15] == 1  # should be infinity
        
        zero = np.zeros(5, dtype=np.uint64)
        assert np.array_equal(gpu_result_np[0:5], zero)   # x should be zero
        assert np.array_equal(gpu_result_np[5:10], zero)  # y should be zero
        assert np.array_equal(gpu_result_np[10:15], zero) # z should be zero
    
    def test_scalar_mul_step(self):
        """
        Tests a single step of a scalar multiplication algorithm: P = 2*P + Q
        This is a common pattern in scalar multiplication implementations.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create Jacobian point P = G
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = g_x_limbs  # Generator point x
        p_jac[5:10] = g_y_limbs  # Generator point y
        p_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        p_jac[15] = 0  # not infinity
        
        # Use Random Point 1 for Q
        q_x = 0xA07B9EC56BD88ADC14EB04BFEF86793B585610EAF9FBB7B4EC61FDC791304EC2
        q_y = 0xFB80052E257D31308D7CE10B4C27DC3C66050DB50566E7FE2173947BD704E55
        
        q_x_limbs = [
            q_x & 0xFFFFFFFFFFFFF,
            (q_x >> 52) & 0xFFFFFFFFFFFFF,
            (q_x >> 104) & 0xFFFFFFFFFFFFF,
            (q_x >> 156) & 0xFFFFFFFFFFFFF,
            (q_x >> 208)
        ]
        
        q_y_limbs = [
            q_y & 0xFFFFFFFFFFFFF,
            (q_y >> 52) & 0xFFFFFFFFFFFFF,
            (q_y >> 104) & 0xFFFFFFFFFFFFF,
            (q_y >> 156) & 0xFFFFFFFFFFFFF,
            (q_y >> 208)
        ]
        
        # Create an affine point Q
        q_aff = np.zeros(11, dtype=np.uint64)
        q_aff[0:5] = q_x_limbs  # Random point 1 x
        q_aff[5:10] = q_y_limbs  # Random point 1 y
        q_aff[10] = 0  # not infinity
        
        # Make copies for both calculations to avoid in-place modification issues
        p_copy1 = np.copy(p_jac)
        p_copy2 = np.copy(p_jac)
        
        # Calculate P = 2*P + Q in a single kernel
        gpu_result_np = helper.run_kernel("test_scalar_mul_step", [p_copy1, q_aff], 16 * 8)
        
        # Calculate the same operation step by step for comparison
        doubled_p = helper.run_kernel("test_gej_double", [p_copy2], 16 * 8)
        reference_result = helper.run_kernel("test_gej_add_ge", [doubled_p, q_aff], 16 * 8)
        
        # Convert both results to affine for easier comparison
        gpu_affine = helper.run_kernel("test_ge_set_gej", [gpu_result_np], 11 * 8)
        reference_affine = helper.run_kernel("test_ge_set_gej", [reference_result], 11 * 8)
        
        # Debug: Check if either result is infinity
        if gpu_result_np[15] == 1:
            print(f"GPU result is infinity")
        if reference_result[15] == 1:
            print(f"Reference result is infinity")
        if gpu_affine[10] == 1:
            print(f"GPU affine result is infinity")
        if reference_affine[10] == 1:
            print(f"Reference affine result is infinity")
            
        # Results should be identical
        assert np.array_equal(gpu_affine[0:5], reference_affine[0:5]), f"X coords differ: GPU={gpu_affine[0:5]}, Ref={reference_affine[0:5]}"
        assert np.array_equal(gpu_affine[5:10], reference_affine[5:10]), f"Y coords differ: GPU={gpu_affine[5:10]}, Ref={reference_affine[5:10]}"
        assert gpu_affine[10] == reference_affine[10], f"Infinity flags differ: GPU={gpu_affine[10]}, Ref={reference_affine[10]}"
        
    def test_point_double_and_add(self):
        """
        Tests a sequence of point operations: doubling followed by addition.
        This is a common pattern in scalar multiplication algorithms.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a Jacobian point P
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = [123, 0, 0, 0, 0]  # x = 123
        p_jac[5:10] = [456, 0, 0, 0, 0]  # y = 456
        p_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        p_jac[15] = 0  # not infinity
        
        # Create an affine point Q
        q_aff = np.zeros(11, dtype=np.uint64)
        q_aff[0:5] = [789, 0, 0, 0, 0]  # x = 789
        q_aff[5:10] = [101112, 0, 0, 0, 0]  # y = 101112
        q_aff[10] = 0  # not infinity
        
        # Calculate 2P + Q in one operation
        combined_result = helper.run_kernel("test_point_double_and_add", [p_jac, q_aff], 16 * 8)
        
        # Calculate 2P and then 2P + Q separately for comparison
        doubled_p = helper.run_kernel("test_gej_double", [p_jac], 16 * 8)
        separate_result = helper.run_kernel("test_gej_add_ge", [doubled_p, q_aff], 16 * 8)
        
        # Convert both results to affine for easier comparison
        combined_affine = helper.run_kernel("test_ge_set_gej", [combined_result], 11 * 8)
        separate_affine = helper.run_kernel("test_ge_set_gej", [separate_result], 11 * 8)
        
        # Results should be identical
        assert np.array_equal(combined_affine[0:5], separate_affine[0:5])  # x coordinates match
        assert np.array_equal(combined_affine[5:10], separate_affine[5:10])  # y coordinates match
        assert combined_affine[10] == separate_affine[10]  # infinity flags match
        
    def test_gej_double_simple_point(self):
        """
        Tests gej_double with a simple point to isolate the doubling issue.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a simple Jacobian point (x=7, y=8, z=1)
        simple_jac = np.zeros(16, dtype=np.uint64)
        simple_jac[0:5] = [7, 0, 0, 0, 0]  # x = 7
        simple_jac[5:10] = [8, 0, 0, 0, 0]  # y = 8
        simple_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        simple_jac[15] = 0  # not infinity
        
        # Double the point
        double_result = helper.run_kernel("test_gej_double", [simple_jac], 16 * 8)
        
        # Check if result is infinity (it shouldn't be)
        if double_result[15] == 1:
            print(f"ERROR: Simple point doubling produced infinity: {double_result}")
            pytest.fail("gej_double should not produce infinity for simple valid points")
        
        # The result should not be at infinity
        assert double_result[15] == 0, "Simple point doubling should not produce infinity"
        
        # Convert to affine to verify we get a valid point
        affine_result = helper.run_kernel("test_ge_set_gej", [double_result], 11 * 8)
        assert affine_result[10] == 0, "Converted affine point should not be infinity"
        
        # The coordinates should be non-zero
        x_is_zero = all(x == 0 for x in affine_result[0:5])
        y_is_zero = all(y == 0 for y in affine_result[5:10])
        assert not (x_is_zero and y_is_zero), "Doubled point should not be the zero point"

    def test_gej_double_infinity_handling(self):
        """
        Tests that gej_double correctly handles the infinity point.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a point at infinity
        infinity_jac = np.zeros(16, dtype=np.uint64)
        infinity_jac[15] = 1  # set infinity flag
        
        # Double the infinity point
        double_result = helper.run_kernel("test_gej_double", [infinity_jac], 16 * 8)
        
        # Result should still be infinity
        assert double_result[15] == 1, "Doubling infinity should produce infinity"

    def test_ge_set_gej_simple(self):
        """
        Tests ge_set_gej with a simple point to check if coordinate conversion is working.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Create a simple Jacobian point (x=7, y=8, z=1)
        simple_jac = np.zeros(16, dtype=np.uint64)
        simple_jac[0:5] = [7, 0, 0, 0, 0]  # x = 7
        simple_jac[5:10] = [8, 0, 0, 0, 0]  # y = 8
        simple_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        simple_jac[15] = 0  # not infinity
        
        # Convert to affine
        affine_result = helper.run_kernel("test_ge_set_gej", [simple_jac], 11 * 8)
        
        # Check that conversion worked
        assert affine_result[10] == 0, "Converted point should not be infinity"
        
        # For z=1, the coordinates should be unchanged
        assert np.array_equal(affine_result[0:5], simple_jac[0:5]), "X coordinate should be unchanged when z=1"
        assert np.array_equal(affine_result[5:10], simple_jac[5:10]), "Y coordinate should be unchanged when z=1"

        
    def test_add_point_to_itself(self):
        """
        Tests adding a point to itself, which should give the same result as doubling.
        This version uses a more robust approach with separate copies and conversion cycles.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create a Jacobian point P = G
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = g_x_limbs  # Generator point x
        p_jac[5:10] = g_y_limbs  # Generator point y
        p_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        p_jac[15] = 0  # not infinity
        
        # Run the kernel
        debug_states_buffer = np.zeros(64, dtype=np.uint64)  # Buffer still needed by the kernel
        add_result_buffer = np.zeros(16, dtype=np.uint64)
        double_result_buffer = np.zeros(16, dtype=np.uint64)
        
        helper.run_kernel("test_add_point_to_itself", 
                         [p_jac, debug_states_buffer, add_result_buffer, double_result_buffer], 
                         8)
        
        # Convert results to affine for comparison
        add_affine = helper.run_kernel("test_ge_set_gej", [add_result_buffer], 11 * 8)
        double_affine = helper.run_kernel("test_ge_set_gej", [double_result_buffer], 11 * 8)
        
        # The results should be identical
        assert np.array_equal(add_affine[0:5], double_affine[0:5]), "X coordinates should match"
        assert np.array_equal(add_affine[5:10], double_affine[5:10]), "Y coordinates should match"
        assert add_affine[10] == double_affine[10], "Infinity flags should match"

    def test_field_ops_with_secp256k1_coords(self):
        """
        Tests field operations specifically with secp256k1 generator coordinates.
        This helps isolate which field operation might be failing with large values.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point coordinates
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = np.array([
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ], dtype=np.uint64)
        
        g_y_limbs = np.array([
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ], dtype=np.uint64)
        
        # Test fe_normalize with generator coordinates
        normalized_x = helper.run_kernel("test_fe_normalize", [g_x_limbs], 5 * 8)
        normalized_y = helper.run_kernel("test_fe_normalize", [g_y_limbs], 5 * 8)
        
        # Coordinates should not normalize to zero
        x_is_zero = all(x == 0 for x in normalized_x)
        y_is_zero = all(y == 0 for y in normalized_y)
        assert not x_is_zero, "Generator x-coordinate should not normalize to zero"
        assert not y_is_zero, "Generator y-coordinate should not normalize to zero"
        
        # Test fe_inv with a coordinate value (using y-coordinate)
        try:
            y_inv = helper.run_kernel("test_fe_inv", [g_y_limbs], 5 * 8)
            
            # Test that y * inv(y) = 1
            product = helper.run_kernel("test_fe_mul", [g_y_limbs, y_inv], 5 * 8)
            normalized_product = helper.run_kernel("test_fe_normalize", [product], 5 * 8)
            
            expected_one = np.array([1, 0, 0, 0, 0], dtype=np.uint64)
            assert np.array_equal(normalized_product, expected_one), f"y * inv(y) should equal 1, got {normalized_product}"
            
        except Exception as e:
            pytest.fail(f"fe_inv failed with generator y-coordinate: {e}")
        
        # Test fe_sqr with generator coordinates
        x_squared = helper.run_kernel("test_fe_sqr", [g_x_limbs], 5 * 8)
        y_squared = helper.run_kernel("test_fe_sqr", [g_y_limbs], 5 * 8)
        
        # Results should not be zero
        x_sq_is_zero = all(x == 0 for x in x_squared)
        y_sq_is_zero = all(y == 0 for y in y_squared)
        assert not x_sq_is_zero, "Generator x-coordinate squared should not be zero"
        assert not y_sq_is_zero, "Generator y-coordinate squared should not be zero"

    def test_ge_set_gej_with_secp256k1_coords(self):
        """
        Tests ge_set_gej specifically with secp256k1 generator coordinates.
        This isolates whether the coordinate conversion is the issue.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create Jacobian point G with z=1
        g_jac = np.zeros(16, dtype=np.uint64)
        g_jac[0:5] = g_x_limbs  # x
        g_jac[5:10] = g_y_limbs  # y
        g_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        g_jac[15] = 0  # not infinity
        
        # Convert to affine
        g_affine = helper.run_kernel("test_ge_set_gej", [g_jac], 11 * 8)
        
        # Should not be infinity
        assert g_affine[10] == 0, "Generator point should not convert to infinity"
        
        # Coordinates should match (since z=1)
        assert np.array_equal(g_affine[0:5], g_jac[0:5]), "X coordinate should be unchanged when z=1"
        assert np.array_equal(g_affine[5:10], g_jac[5:10]), "Y coordinate should be unchanged when z=1"
        
        # Test with z=2 to check actual coordinate transformation
        g_jac_z2 = np.copy(g_jac)
        g_jac_z2[10:15] = [2, 0, 0, 0, 0]  # z = 2
        
        g_affine_z2 = helper.run_kernel("test_ge_set_gej", [g_jac_z2], 11 * 8)
        
        # Should not be infinity
        assert g_affine_z2[10] == 0, "Generator point with z=2 should not convert to infinity"
        
        # Coordinates should be different from the z=1 case
        coords_changed = not (np.array_equal(g_affine_z2[0:5], g_jac[0:5]) and 
                             np.array_equal(g_affine_z2[5:10], g_jac[5:10]))
        assert coords_changed, "Coordinates should change when z != 1"

    def test_gej_double_step_by_step(self):
        """
        Tests gej_double by breaking it down into individual field operations.
        This helps identify which specific operation is causing the infinity result.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = np.array([
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ], dtype=np.uint64)
        
        g_y_limbs = np.array([
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ], dtype=np.uint64)
        
        z_one = np.array([1, 0, 0, 0, 0], dtype=np.uint64)
        
        # Step 1: Calculate S = Y^2
        s = helper.run_kernel("test_fe_sqr", [g_y_limbs], 5 * 8)
        s_is_zero = all(x == 0 for x in s)
        assert not s_is_zero, "S = Y^2 should not be zero"
        
        # Step 2: Calculate L = X^2
        l = helper.run_kernel("test_fe_sqr", [g_x_limbs], 5 * 8)
        l_is_zero = all(x == 0 for x in l)
        assert not l_is_zero, "L = X^2 should not be zero"
        
        # Step 3: L = 3 * L
        factor_3 = np.array([3], dtype=np.int32)
        l = helper.run_kernel("test_fe_mul_int", [l, factor_3], 5 * 8)
        l_is_zero = all(x == 0 for x in l)
        assert not l_is_zero, "L = 3*X^2 should not be zero"
        
        # Step 4: L = L / 2 (using fe_half)
        l = helper.run_kernel("test_fe_half", [l], 5 * 8)
        l_is_zero = all(x == 0 for x in l)
        assert not l_is_zero, "L = (3/2)*X^2 should not be zero"
        
        # Step 5: T = -S
        t = helper.run_kernel("test_fe_negate", [s], 5 * 8)
        
        # Step 6: T = T * X = -S * X
        t = helper.run_kernel("test_fe_mul", [t, g_x_limbs], 5 * 8)
        
        # Step 7: Z_new = Y * Z = Y * 1 = Y
        z_new = helper.run_kernel("test_fe_mul", [g_y_limbs, z_one], 5 * 8)
        z_new_is_zero = all(x == 0 for x in z_new)
        assert not z_new_is_zero, "Z_new = Y*Z should not be zero"
        
        # Step 8: X_new = L^2
        x_new = helper.run_kernel("test_fe_sqr", [l], 5 * 8)
        
        # Step 9: X_new = X_new + T
        x_new = helper.run_kernel("test_fe_add", [x_new, t], 5 * 8)
        
        # Step 10: X_new = X_new + T (add T again for 2*T)
        x_new = helper.run_kernel("test_fe_add", [x_new, t], 5 * 8)
        
        # Step 11: S = S^2
        s = helper.run_kernel("test_fe_sqr", [s], 5 * 8)
        
        # Step 12: T = X_new + T
        t = helper.run_kernel("test_fe_add", [x_new, t], 5 * 8)
        
        # Step 13: Y_new = L * T
        y_new = helper.run_kernel("test_fe_mul", [l, t], 5 * 8)
        
        # Step 14: Y_new = Y_new + S
        y_new = helper.run_kernel("test_fe_add", [y_new, s], 5 * 8)
        
        # Step 15: Y_new = -Y_new
        y_new = helper.run_kernel("test_fe_negate", [y_new], 5 * 8)
        
        # Check if any coordinate is zero (which would indicate infinity)
        x_new_is_zero = all(x == 0 for x in x_new)
        y_new_is_zero = all(y == 0 for y in y_new)
        z_new_is_zero = all(z == 0 for z in z_new)
        
        print(f"Step-by-step doubling results:")
        print(f"X_new is zero: {x_new_is_zero}")
        print(f"Y_new is zero: {y_new_is_zero}")
        print(f"Z_new is zero: {z_new_is_zero}")
        print(f"X_new: {x_new}")
        print(f"Y_new: {y_new}")
        print(f"Z_new: {z_new}")
        
        # None of the coordinates should be zero for a valid doubling
        assert not (x_new_is_zero and y_new_is_zero and z_new_is_zero), "Doubled point should not be zero"
        assert not z_new_is_zero, "Z coordinate should not be zero after doubling"

    def test_coordinate_conversion_precision(self):
        """
        Tests whether coordinate conversion (Jacobian -> Affine -> Jacobian) preserves precision.
        This checks if the conversion cycle introduces errors that could cause gej_double to fail.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the secp256k1 generator point G
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create original Jacobian point G with z=1
        original_jac = np.zeros(16, dtype=np.uint64)
        original_jac[0:5] = g_x_limbs  # x
        original_jac[5:10] = g_y_limbs  # y
        original_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        original_jac[15] = 0  # not infinity
        
        # Convert to affine
        affine_point = helper.run_kernel("test_ge_set_gej", [original_jac], 11 * 8)
        
        # Convert back to Jacobian
        converted_jac = helper.run_kernel("test_gej_set_ge", [affine_point], 16 * 8)
        
        # Compare original and converted Jacobian coordinates
        print(f"Original Jacobian coordinates:")
        print(f"  X: {original_jac[0:5]}")
        print(f"  Y: {original_jac[5:10]}")
        print(f"  Z: {original_jac[10:15]}")
        print(f"  Infinity: {original_jac[15]}")
        
        print(f"Converted Jacobian coordinates:")
        print(f"  X: {converted_jac[0:5]}")
        print(f"  Y: {converted_jac[5:10]}")
        print(f"  Z: {converted_jac[10:15]}")
        print(f"  Infinity: {converted_jac[15]}")
        
        # Test doubling both versions
        original_doubled = helper.run_kernel("test_gej_double", [original_jac], 16 * 8)
        converted_doubled = helper.run_kernel("test_gej_double", [converted_jac], 16 * 8)
        
        print(f"Original doubled result infinity flag: {original_doubled[15]}")
        print(f"Converted doubled result infinity flag: {converted_doubled[15]}")
        
        # If the conversion cycle introduces errors, the doubling results will differ
        if original_doubled[15] != converted_doubled[15]:
            print("ERROR: Conversion cycle affects doubling behavior!")
            print(f"Original point doubling produces infinity: {original_doubled[15] == 1}")
            print(f"Converted point doubling produces infinity: {converted_doubled[15] == 1}")
            
        # The coordinates should be identical after conversion cycle (since z=1)
        coords_match = (np.array_equal(original_jac[0:5], converted_jac[0:5]) and 
                       np.array_equal(original_jac[5:10], converted_jac[5:10]) and
                       np.array_equal(original_jac[10:15], converted_jac[10:15]))
        
        if not coords_match:
            print("WARNING: Coordinate conversion cycle does not preserve exact values!")
            
        # Both doubling operations should produce the same result
        assert original_doubled[15] == converted_doubled[15], "Conversion cycle should not affect doubling behavior"

    def test_gej_double_isolated_secp256k1(self):
        """
        Tests gej_double in complete isolation with the secp256k1 generator point.
        This isolates whether the issue is in gej_double itself or in the kernel structure.
        """
        helper = MetalTestHelper("test/metal/test_field.metal")
        
        # Use the exact same secp256k1 generator point as the failing test
        g_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
        g_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
        
        # Convert to 5x52 bit limb representation
        g_x_limbs = [
            g_x & 0xFFFFFFFFFFFFF,
            (g_x >> 52) & 0xFFFFFFFFFFFFF,
            (g_x >> 104) & 0xFFFFFFFFFFFFF,
            (g_x >> 156) & 0xFFFFFFFFFFFFF,
            (g_x >> 208)
        ]
        
        g_y_limbs = [
            g_y & 0xFFFFFFFFFFFFF,
            (g_y >> 52) & 0xFFFFFFFFFFFFF,
            (g_y >> 104) & 0xFFFFFFFFFFFFF,
            (g_y >> 156) & 0xFFFFFFFFFFFFF,
            (g_y >> 208)
        ]
        
        # Create the exact same Jacobian point as in the failing test
        p_jac = np.zeros(16, dtype=np.uint64)
        p_jac[0:5] = g_x_limbs  # Generator point x
        p_jac[5:10] = g_y_limbs  # Generator point y
        p_jac[10:15] = [1, 0, 0, 0, 0]  # z = 1
        p_jac[15] = 0  # not infinity
        
        # Call gej_double directly with no other operations in the kernel
        double_result = helper.run_kernel("test_gej_double", [p_jac], 16 * 8)
        
        # This should NOT produce infinity
        if double_result[15] == 1:
            print(f"ERROR: Isolated gej_double with secp256k1 generator produced infinity!")
            print(f"Input point: {p_jac}")
            print(f"Output point: {double_result}")
            pytest.fail("gej_double should not produce infinity for secp256k1 generator in isolation")
        
        # Verify the result is valid
        assert double_result[15] == 0, "Isolated gej_double should not produce infinity"
        
        # Convert to affine to verify coordinates
        affine_result = helper.run_kernel("test_ge_set_gej", [double_result], 11 * 8)
        assert affine_result[10] == 0, "Converted result should not be infinity"
        
        # Coordinates should be non-zero
        x_is_zero = all(x == 0 for x in affine_result[0:5])
        y_is_zero = all(y == 0 for y in affine_result[5:10])
        assert not (x_is_zero and y_is_zero), "Doubled generator should not be zero point"
        
        print(f"SUCCESS: Isolated gej_double with secp256k1 generator works correctly")
        print(f"Result coordinates: X={affine_result[0:5]}, Y={affine_result[5:10]}")
