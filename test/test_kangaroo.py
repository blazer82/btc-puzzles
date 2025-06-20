import pytest
from coincurve.keys import PublicKey

import cryptography_utils as crypto
import hop_strategy as hs
from kangaroo import Kangaroo


@pytest.fixture
def precomputed_hops():
    """Fixture to provide pre-computed hops for tests."""
    return hs.generate_precomputed_hops()


class TestKangaroo:
    def test_initialization(self):
        """Tests that a Kangaroo is initialized with the correct state."""
        g = crypto.get_generator_point()
        k = Kangaroo(kangaroo_id=0, initial_point=g, is_tame=True)

        assert k.id == 0
        assert k.is_tame is True
        assert k.distance == 0

        current_point, distance = k.get_state()
        assert current_point.format() == g.format()
        assert distance == 0

    def test_get_xy_tuple(self):
        """Tests that get_xy_tuple returns the correct coordinates."""
        g = crypto.get_generator_point()
        k = Kangaroo(kangaroo_id=0, initial_point=g, is_tame=False)

        expected_xy = g.point()
        assert k.get_xy_tuple() == expected_xy

    def test_single_hop(self, precomputed_hops):
        """Tests the state change after a single hop."""
        g = crypto.get_generator_point()
        k = Kangaroo(kangaroo_id=0, initial_point=g, is_tame=True)

        # Determine the expected hop
        x_g = crypto.get_x_coordinate_int(g)
        hop_index = hs.select_hop_index(x_g, len(precomputed_hops), k.id)
        expected_distance = 2**hop_index
        expected_next_point = crypto.point_add(g, precomputed_hops[hop_index])

        # Perform the hop
        k.hop(precomputed_hops)

        # Verify the new state
        current_point, distance = k.get_state()
        assert distance == expected_distance
        assert current_point.format() == expected_next_point.format()

    def test_multiple_hops(self, precomputed_hops):
        """Tests the state accumulation over multiple hops."""
        # Start at 5*G to have a different starting point
        start_point = crypto.scalar_multiply(5)
        k = Kangaroo(kangaroo_id=0, initial_point=start_point, is_tame=False)

        # --- First hop ---
        x1 = crypto.get_x_coordinate_int(start_point)
        idx1 = hs.select_hop_index(x1, len(precomputed_hops), k.id)
        dist1 = 2**idx1
        point1 = crypto.point_add(start_point, precomputed_hops[idx1])
        k.hop(precomputed_hops)

        # --- Second hop ---
        x2 = crypto.get_x_coordinate_int(point1)
        idx2 = hs.select_hop_index(x2, len(precomputed_hops), k.id)
        dist2 = 2**idx2
        point2 = crypto.point_add(point1, precomputed_hops[idx2])
        k.hop(precomputed_hops)

        # --- Third hop ---
        x3 = crypto.get_x_coordinate_int(point2)
        idx3 = hs.select_hop_index(x3, len(precomputed_hops), k.id)
        dist3 = 2**idx3
        point3 = crypto.point_add(point2, precomputed_hops[idx3])
        k.hop(precomputed_hops)

        # Verify final state
        expected_distance = dist1 + dist2 + dist3
        current_point, distance = k.get_state()

        assert distance == expected_distance
        assert current_point.format() == point3.format()
