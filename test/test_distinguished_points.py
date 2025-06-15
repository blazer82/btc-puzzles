import pytest

import distinguished_points as dp


# Tests for is_distinguished()
def test_is_distinguished_threshold_zero():
    """Tests that with a threshold of 0, every point is distinguished."""
    assert dp.is_distinguished(12345, 0) is True
    assert dp.is_distinguished(0, 0) is True


def test_is_distinguished_threshold_one():
    """Tests that with a threshold of 1, only even numbers are distinguished."""
    assert dp.is_distinguished(10, 1) is True   # Even
    assert dp.is_distinguished(11, 1) is False  # Odd
    assert dp.is_distinguished(0, 1) is True    # Zero is even


def test_is_distinguished_higher_threshold():
    """Tests with a more significant threshold."""
    threshold = 4  # Requires 4 trailing zero bits (i.e., divisible by 16)

    # 16 = 0b10000, has 4 trailing zeros
    assert dp.is_distinguished(16, threshold) is True
    # 32 = 0b100000, has 5 trailing zeros
    assert dp.is_distinguished(32, threshold) is True
    # 15 = 0b01111, has 0 trailing zeros
    assert dp.is_distinguished(15, threshold) is False
    # 24 = 0b11000, has 3 trailing zeros
    assert dp.is_distinguished(24, threshold) is False


def test_is_distinguished_with_zero_coordinate():
    """Tests that x-coordinate 0 is always distinguished for any positive threshold."""
    assert dp.is_distinguished(0, 1) is True
    assert dp.is_distinguished(0, 8) is True
    assert dp.is_distinguished(0, 16) is True


# Tests for PointTrap class
class TestPointTrap:
    def test_initialization(self):
        """Tests that a new PointTrap is empty."""
        trap = dp.PointTrap()
        assert trap.get_point((1, 2)) is None

    def test_add_and_get_point(self):
        """Tests adding a point and retrieving it."""
        trap = dp.PointTrap()
        point = (123, 456)
        distance = 1000

        # Point should not exist initially
        assert trap.get_point(point) is None

        # Add the point
        trap.add_point(point, distance)

        # Now it should exist with the correct distance
        assert trap.get_point(point) == distance

    def test_get_nonexistent_point(self):
        """Tests that getting a non-existent point returns None."""
        trap = dp.PointTrap()
        trap.add_point((1, 1), 100)
        assert trap.get_point((2, 2)) is None

    def test_update_point(self):
        """Tests that adding an existing point updates its distance."""
        trap = dp.PointTrap()
        point = (10, 20)
        initial_distance = 50
        updated_distance = 150

        # Add the point initially
        trap.add_point(point, initial_distance)
        assert trap.get_point(point) == initial_distance

        # Add it again with a new distance
        trap.add_point(point, updated_distance)
        assert trap.get_point(point) == updated_distance
