import pytest
from coincurve.keys import PublicKey

import cryptography_utils as crypto


def test_get_generator_point():
    """Tests that the generator point G is returned correctly."""
    g = crypto.get_generator_point()
    assert isinstance(g, PublicKey)


def test_get_curve_order_n():
    """Tests that the curve order n is returned correctly."""
    n = crypto.get_curve_order_n()
    assert isinstance(n, int)
    # Value from secp256k1 standards
    expected_n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    assert n == expected_n


def test_scalar_multiply():
    """Tests scalar multiplication of elliptic curve points."""
    g = crypto.get_generator_point()

    # Test multiplication by 2: 2*G should equal G+G
    p1 = crypto.scalar_multiply(2)
    p2 = crypto.point_add(g, g)
    assert p1.format() == p2.format()

    # Test with a different point: 2 * (5*G) should equal 10*G
    p_start = crypto.scalar_multiply(5)
    p_doubled = crypto.scalar_multiply(2, point=p_start)
    p_ten = crypto.scalar_multiply(10)
    assert p_doubled.format() == p_ten.format()

    # Test multiplication by 1: 1*P should equal P
    p_one = crypto.scalar_multiply(1, point=p_start)
    assert p_one.format() == p_start.format()

    # Test multiplication by curve order n (should result in point at infinity, which raises ValueError)
    n = crypto.get_curve_order_n()
    with pytest.raises(ValueError):
        crypto.scalar_multiply(n)


def test_point_add():
    """Tests addition of elliptic curve points."""
    # 2*G + 3*G should equal 5*G
    p2 = crypto.scalar_multiply(2)
    p3 = crypto.scalar_multiply(3)
    p5_expected = crypto.scalar_multiply(5)
    p5_actual = crypto.point_add(p2, p3)
    assert p5_actual.format() == p5_expected.format()


def test_point_bytes_conversion():
    """Tests the conversion of points to and from byte representation."""
    g = crypto.get_generator_point()
    g_bytes = crypto.point_to_bytes(g)

    assert isinstance(g_bytes, bytes)
    # Compressed public key is 33 bytes and starts with 0x02 or 0x03
    assert len(g_bytes) == 33
    assert g_bytes[0] in (2, 3)

    # Test round-trip conversion
    g_reconstructed = crypto.point_from_bytes(g_bytes)
    assert g.format() == g_reconstructed.format()


def test_get_x_coordinate_int():
    """Tests the extraction of the x-coordinate from a point."""
    g = crypto.get_generator_point()
    x_g = crypto.get_x_coordinate_int(g)
    # Known x-coordinate of generator point G
    expected_x_g = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    assert x_g == expected_x_g

    # Test another point, 2*G
    p2 = crypto.scalar_multiply(2)
    x_p2 = crypto.get_x_coordinate_int(p2)
    expected_x_p2 = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    assert x_p2 == expected_x_p2
