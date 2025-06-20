"""
Defines the deterministic logic for how kangaroos select their next hop based
on their current position.
"""
import hashlib
from typing import List

from coincurve.keys import PublicKey

import cryptography_utils as crypto


def generate_precomputed_hops() -> List[PublicKey]:
    """
    Calculates and returns a list of 16 pre-computed hop points.

    The hops are defined as 2^i * G for i in [0, 15].

    Returns:
        List[PublicKey]: A list of 16 PublicKey objects representing the hops.
    """
    hops = []
    for i in range(16):
        scalar = 2**i
        hop_point = crypto.scalar_multiply(scalar)
        hops.append(hop_point)
    return hops


def select_hop_index(point_x_coordinate: int, num_precomputed_hops: int, kangaroo_id: int) -> int:
    """
    Determines the index of the next hop from the pre-computed list.

    Uses SHA-256 hash to combine the x-coordinate with the kangaroo ID,
    ensuring strong path differentiation between kangaroos even when
    starting from identical points.

    Args:
        point_x_coordinate (int): The x-coordinate of the kangaroo's current point.
        num_precomputed_hops (int): The total number of pre-computed hops available.
        kangaroo_id (int): The unique ID of the kangaroo.

    Returns:
        int: The index of the hop to use from the pre-computed list.
    """
    # Create a strong hash combining coordinate and ID
    hash_input = f"{point_x_coordinate}_{kangaroo_id}".encode('utf-8')
    hash_result = hashlib.sha256(hash_input).digest()
    # Use first 4 bytes as a 32-bit unsigned integer
    hash_int = int.from_bytes(hash_result[:4], 'big')
    return hash_int % num_precomputed_hops
