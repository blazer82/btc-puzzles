"""
Encapsulates all direct interactions with the coincurve library and raw
elliptic curve mathematics specific to secp256k1.
"""
from typing import Optional

from coincurve.keys import PrivateKey, PublicKey


def get_generator_point() -> PublicKey:
    """Returns the secp256k1 generator point G."""
    # G is the public key for the private key '1'
    one_as_private_key = PrivateKey(b'\x01'.rjust(32, b'\x00'))
    return one_as_private_key.public_key


def get_curve_order_n() -> int:
    """Returns the order of the curve n."""
    # The value of n is standardized for secp256k1.
    return 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def scalar_multiply(scalar: int, point: Optional[PublicKey] = None) -> PublicKey:
    """Performs scalar multiplication of a point (defaults to G)."""
    if point is None:
        point = get_generator_point()

    scalar_bytes = scalar.to_bytes(32, 'big')
    return point.multiply(scalar_bytes)


def point_add(point1: PublicKey, point2: PublicKey) -> PublicKey:
    """Performs elliptic curve point addition."""
    return PublicKey.combine_keys([point1, point2])


def point_to_bytes(point: PublicKey) -> bytes:
    """Converts a PublicKey object to its compressed byte representation."""
    return point.format(compressed=True)


def point_from_bytes(data: bytes) -> PublicKey:
    """Converts compressed bytes back to a PublicKey object."""
    return PublicKey(data)


def get_x_coordinate_int(point: PublicKey) -> int:
    """Extracts the x-coordinate of a point as an integer."""
    return point.point()[0]
