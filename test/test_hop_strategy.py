import pytest
from coincurve.keys import PublicKey

import cryptography_utils as crypto
import hop_strategy as hs


def test_generate_precomputed_hops():
    """Tests the generation of the 16 pre-computed hop points."""
    hops = hs.generate_precomputed_hops()

    # 1. Check type and size
    assert isinstance(hops, list)
    assert len(hops) == 16
    assert all(isinstance(p, PublicKey) for p in hops)

    # 2. Verify the first hop (2^0 * G = G)
    g = crypto.get_generator_point()
    assert hops[0].format() == g.format()

    # 3. Verify the second hop (2^1 * G = 2*G)
    g_doubled = crypto.scalar_multiply(2)
    assert hops[1].format() == g_doubled.format()

    # 4. Verify the last hop (2^15 * G)
    p_last_expected = crypto.scalar_multiply(2**15)
    assert hops[15].format() == p_last_expected.format()


def test_select_hop_index():
    """Tests the deterministic selection of a hop index."""
    num_hops = 16

    # Test with x-coordinate less than num_hops
    assert hs.select_hop_index(0, num_hops) == 0
    assert hs.select_hop_index(15, num_hops) == 15

    # Test with x-coordinate equal to num_hops
    assert hs.select_hop_index(16, num_hops) == 0

    # Test with a larger x-coordinate (100 % 16 = 4)
    assert hs.select_hop_index(100, num_hops) == 4

    # Test with a very large x-coordinate
    large_x = 0xABCDEF1234567890
    expected_index = large_x % num_hops
    assert hs.select_hop_index(large_x, num_hops) == expected_index
