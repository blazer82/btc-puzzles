"""
Implements the criteria for identifying "distinguished points" and manages
the storage (PointTrap) for these points.
"""
from typing import Dict, Optional, Tuple

# Type alias for clarity
PointXY = Tuple[int, int]


def is_distinguished(point_x_coordinate: int, threshold: int) -> bool:
    """
    Checks if a point is "distinguished" based on its x-coordinate.

    A point is distinguished if its x-coordinate has a number of trailing
    zero bits equal to or greater than the given threshold.

    Args:
        point_x_coordinate (int): The x-coordinate of the point to check.
        threshold (int): The required number of trailing zero bits.

    Returns:
        bool: True if the point is distinguished, False otherwise.
    """
    if threshold <= 0:
        return True

    # Create a mask for the lower `threshold` bits.
    # e.g., threshold=4 -> mask = 0b1111 = 15
    mask = (1 << threshold) - 1

    # If the lower bits are all zero, the AND operation will result in 0.
    return (point_x_coordinate & mask) == 0


class PointTrap:
    """
    A storage mechanism for distinguished points and their associated distances.

    Uses a dictionary for efficient O(1) average time lookups.
    """

    def __init__(self):
        """Initializes an empty PointTrap."""
        self._trap: Dict[PointXY, int] = {}

    def add_point(self, point_xy_tuple: PointXY, distance: int):
        """
        Adds or updates a distinguished point in the trap.

        If the point already exists, its distance is updated with the new value.

        Args:
            point_xy_tuple (Tuple[int, int]): The (x, y) coordinates of the point.
            distance (int): The accumulated distance to reach this point.
        """
        self._trap[point_xy_tuple] = distance

    def get_point(self, point_xy_tuple: PointXY) -> Optional[int]:
        """
        Retrieves the distance for a given point from the trap.

        Args:
            point_xy_tuple (Tuple[int, int]): The (x, y) coordinates of the point.

        Returns:
            Optional[int]: The stored distance if the point is in the trap,
                           otherwise None.
        """
        return self._trap.get(point_xy_tuple)
