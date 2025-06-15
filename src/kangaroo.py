"""
Models a single kangaroo walker, managing its state (current position,
accumulated distance) and its hopping behavior.
"""
from typing import List, Tuple

from coincurve.keys import PublicKey

import cryptography_utils as crypto
import hop_strategy as hs


class Kangaroo:
    """
    Represents a single kangaroo in the Pollard's Kangaroo algorithm.

    Manages its own state, including its current position on the elliptic curve
    and the total distance it has traveled from its starting point.
    """

    def __init__(self, initial_point: PublicKey, is_tame: bool, initial_warmup_hops: int = 0):
        """
        Initializes the kangaroo.

        Note: The actual warm-up hops specified by `initial_warmup_hops`
        are expected to be performed by the calling code (e.g., KangarooRunner)
        by repeatedly calling the `hop()` method.

        Args:
            initial_point (PublicKey): The starting point on the curve.
            is_tame (bool): True if this is a tame kangaroo, False for wild.
            initial_warmup_hops (int): A parameter indicating how many warm-up
                                       hops are needed for this kangaroo.
        """
        self.current_point: PublicKey = initial_point
        self.is_tame: bool = is_tame
        self.distance: int = 0

    def hop(self, precomputed_hops: List[PublicKey]):
        """
        Advances the kangaroo by one hop.

        The hop is chosen deterministically based on the current point's
        x-coordinate. The kangaroo's position and accumulated distance are
        updated.

        Args:
            precomputed_hops (List[PublicKey]): A list of pre-computed hop points.
        """
        # Select which hop to take based on the current position
        x_coord = crypto.get_x_coordinate_int(self.current_point)
        num_hops = len(precomputed_hops)
        hop_index = hs.select_hop_index(x_coord, num_hops)

        # Get the hop point and its corresponding distance (2^i)
        hop_point = precomputed_hops[hop_index]
        hop_distance = 2**hop_index

        # Update the kangaroo's state
        self.current_point = crypto.point_add(self.current_point, hop_point)
        self.distance += hop_distance

    def get_state(self) -> Tuple[PublicKey, int]:
        """
        Returns the current state of the kangaroo.

        Returns:
            Tuple[PublicKey, int]: A tuple containing the current point and the
                                   accumulated distance.
        """
        return self.current_point, self.distance

    def get_xy_tuple(self) -> Tuple[int, int]:
        """
        Provides the (x, y) coordinates of the current point as a tuple.

        This is useful for using the point as a dictionary key in a PointTrap.

        Returns:
            Tuple[int, int]: The (x, y) integer coordinates.
        """
        return self.current_point.point()
