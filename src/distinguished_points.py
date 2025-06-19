"""
Implements the criteria for identifying "distinguished points" and manages
the storage (PointTrap) for these points.
"""
from collections import OrderedDict
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

    Uses an OrderedDict for efficient O(1) average time lookups while maintaining
    insertion order for memory-bounded operation.
    """

    def __init__(self, max_size: int = None):
        """
        Initializes an empty PointTrap.

        Args:
            max_size (int, optional): Maximum number of points to store.
                                     If None, size is unbounded.
        """
        self._trap: OrderedDict[PointXY, int] = OrderedDict()
        self.max_size = max_size

    def add_point(self, point_xy_tuple: PointXY, distance: int):
        """
        Adds or updates a distinguished point in the trap.

        If the point already exists, its distance is updated, and it's moved
        to the end to mark it as recently used. If the trap is full and a new
        point is added, the oldest point is removed.

        Args:
            point_xy_tuple (Tuple[int, int]): The (x, y) coordinates of the point.
            distance (int): The accumulated distance to reach this point.
        """
        is_present = point_xy_tuple in self._trap

        # If the trap is full and a new point is being added, evict the oldest.
        if not is_present and self.max_size is not None and len(self._trap) >= self.max_size:
            self._trap.popitem(last=False)

        # Add or update the point's distance.
        self._trap[point_xy_tuple] = distance

        # If the point was already present, move it to the end of the OrderedDict
        # to mark it as recently used. This is crucial for the LRU logic.
        if is_present:
            self._trap.move_to_end(point_xy_tuple)

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

    def size(self) -> int:
        """
        Returns the current number of points in the trap.

        Returns:
            int: Number of stored points.
        """
        return len(self._trap)
