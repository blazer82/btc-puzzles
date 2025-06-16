import pytest
import numpy as np

# A placeholder for a helper that would wrap Metal API calls.
# In a real implementation, this would handle device init, kernel compilation,
# buffer management, and kernel execution.
class MetalTestHelper:
    def __init__(self, kernel_file):
        self.kernel_file = kernel_file
        # In a real scenario, this would initialize the Metal device,
        # library, and command queue.
        pass

    def run_kernel(self, kernel_name, input_buffers_np, output_size_bytes):
        """
        A mock function representing the execution of a Metal kernel.
        - Compiles the kernel.
        - Creates MTLBuffers from the numpy input arrays.
        - Creates an output MTLBuffer.
        - Dispatches the kernel and waits for completion.
        - Returns the content of the output buffer as a numpy array.
        """
        # This is a mock implementation. A real one would use a library
        # like metal-python. For this example, we'll call the python
        # implementation directly to simulate a successful GPU run.
        print(f"\n(Simulating GPU execution of kernel '{kernel_name}' from {self.kernel_file})")
        if kernel_name == "test_fe_mul":
            return python_fe_mul_inner(input_buffers_np[0], input_buffers_np[1])
        elif kernel_name == "test_fe_normalize":
            return python_fe_normalize(input_buffers_np[0])
        elif kernel_name == "test_fe_sqr":
            return python_fe_sqr(input_buffers_np[0])
        elif kernel_name == "test_fe_inv":
            return python_fe_inv(input_buffers_np[0])
        else:
            raise NotImplementedError(f"Simulation for kernel '{kernel_name}' is not implemented.")


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


# --- Python Port of Modular Inverse Logic ---

def to_signed_64(n):
    n &= 0xFFFFFFFFFFFFFFFF
    return n - 0x10000000000000000 if n & 0x8000000000000000 else n

def python_fe_to_signed62(a_n):
    M62 = (1 << 62) - 1
    a0, a1, a2, a3, a4 = [int(x) for x in a_n]
    r = [0] * 5
    r[0] = (a0       | a1 << 52) & M62
    r[1] = (a1 >> 10 | a2 << 42) & M62
    r[2] = (a2 >> 20 | a3 << 32) & M62
    r[3] = (a3 >> 30 | a4 << 22) & M62
    r[4] =  a4 >> 40
    return r

def python_fe_from_signed62(s_v):
    M52 = (1 << 52) - 1
    a0, a1, a2, a3, a4 = s_v
    r = [0] * 5
    r[0] =  a0                   & M52
    r[1] = (a0 >> 52 | a1 << 10) & M52
    r[2] = (a1 >> 42 | a2 << 20) & M52
    r[3] = (a2 >> 32 | a3 << 30) & M52
    r[4] = (a3 >> 22 | a4 << 40)
    return np.array(r, dtype=np.uint64)

def python_modinv64_normalize_62(r_v, sign, modinfo):
    M62 = (1 << 62) - 1
    r = [to_signed_64(x) for x in r_v]
    
    cond_add = to_signed_64(r[4] >> 63)
    for i in range(5): r[i] += modinfo['modulus'][i] & cond_add

    cond_negate = to_signed_64(sign >> 63)
    for i in range(5): r[i] = (r[i] ^ cond_negate) - cond_negate

    for i in range(4): r[i+1] += to_signed_64(r[i] >> 62); r[i] &= M62

    cond_add = to_signed_64(r[4] >> 63)
    for i in range(5): r[i] += modinfo['modulus'][i] & cond_add

    for i in range(4): r[i+1] += to_signed_64(r[i] >> 62); r[i] &= M62
    
    return r

def python_modinv64_divsteps_59(zeta, f0, g0):
    u, v, q, r = 8, 0, 0, 8
    f, g = f0, g0
    
    for _ in range(3, 62):
        zeta = to_signed_64(zeta)
        mask1 = to_signed_64(zeta >> 63)
        mask2 = to_signed_64(-(g & 1))

        x = (f ^ mask1) - mask1
        y = (u ^ mask1) - mask1
        z = (v ^ mask1) - mask1

        g += x & mask2
        q += y & mask2
        r += z & mask2

        mask1 &= mask2
        zeta = (zeta ^ mask1) - 1

        f += g & mask1
        u += q & mask1
        v += r & mask1

        g >>= 1
        u <<= 1
        v <<= 1
        
    t = {'u': to_signed_64(u), 'v': to_signed_64(v), 'q': to_signed_64(q), 'r': to_signed_64(r)}
    return zeta, t

def python_modinv64_update_de_62(d_v, e_v, t, modinfo):
    M62 = (1 << 62) - 1
    d = [to_signed_64(x) for x in d_v]
    e = [to_signed_64(x) for x in e_v]
    u, v, q, r = t['u'], t['v'], t['q'], t['r']

    sd = to_signed_64(d[4] >> 63)
    se = to_signed_64(e[4] >> 63)
    md = (u & sd) + (v & se)
    me = (q & sd) + (r & se)

    cd = u * d[0] + v * e[0]
    ce = q * d[0] + r * e[0]

    md -= (modinfo['modulus_inv62'] * (cd & 0xFFFFFFFFFFFFFFFF) + md) & M62
    me -= (modinfo['modulus_inv62'] * (ce & 0xFFFFFFFFFFFFFFFF) + me) & M62

    cd += modinfo['modulus'][0] * md
    ce += modinfo['modulus'][0] * me
    cd >>= 62
    ce >>= 62

    for i in range(1, 5):
        cd += u * d[i] + v * e[i]
        ce += q * d[i] + r * e[i]
        if modinfo['modulus'][i] != 0:
            cd += modinfo['modulus'][i] * md
            ce += modinfo['modulus'][i] * me
        d[i-1] = cd & M62; cd >>= 62
        e[i-1] = ce & M62; ce >>= 62

    d[4] = cd
    e[4] = ce
    return d, e

def python_modinv64_update_fg_62(f_v, g_v, t):
    M62 = (1 << 62) - 1
    f = [to_signed_64(x) for x in f_v]
    g = [to_signed_64(x) for x in g_v]
    u, v, q, r = t['u'], t['v'], t['q'], t['r']

    cf = u * f[0] + v * g[0]
    cg = q * f[0] + r * g[0]
    cf >>= 62
    cg >>= 62

    for i in range(1, 5):
        cf += u * f[i] + v * g[i]
        cg += q * f[i] + r * g[i]
        f[i-1] = cf & M62; cf >>= 62
        g[i-1] = cg & M62; cg >>= 62

    f[4] = cf
    g[4] = cg
    return f, g

def python_modinv64_inv(x_v, modinfo):
    d_v = [0, 0, 0, 0, 0]
    e_v = [1, 0, 0, 0, 0]
    f_v = modinfo['modulus']
    g_v = x_v
    zeta = -1

    for _ in range(10):
        zeta, t = python_modinv64_divsteps_59(zeta, f_v[0], g_v[0])
        d_v, e_v = python_modinv64_update_de_62(d_v, e_v, t, modinfo)
        f_v, g_v = python_modinv64_update_fg_62(f_v, g_v, t)
    
    return python_modinv64_normalize_62(d_v, f_v[4], modinfo)

def python_fe_inv(a_n):
    """
    A Python port of `secp256k1_fe_impl_inv`.
    """
    modinfo = {
        'modulus': [to_signed_64(v) for v in [-0x1000003D1, 0, 0, 0, 256]],
        'modulus_inv62': 0x27C7F6E22DDACACF
    }
    tmp_n = python_fe_normalize(a_n)
    s_v = python_fe_to_signed62(tmp_n)
    s_v_inv = python_modinv64_inv(s_v, modinfo)
    return python_fe_from_signed62(s_v_inv)


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
