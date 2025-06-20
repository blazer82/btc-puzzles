"""
Defines the deterministic logic for how kangaroos select their next hop based
on their current position.
"""
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

    The selection is deterministic, based on the x-coordinate of the current
    point and the kangaroo's unique ID, modulo the number of available hops.
    This ensures different kangaroos take different paths.

    Args:
        point_x_coordinate (int): The x-coordinate of the kangaroo's current point.
        num_precomputed_hops (int): The total number of pre-computed hops available.
        kangaroo_id (int): The unique ID of the kangaroo.

    Returns:
        int: The index of the hop to use from the pre-computed list.
    """
    return (point_x_coordinate + kangaroo_id) % num_precomputed_hops
