"""Perception V2 day 12: the van slows for what it cannot see into.

Unknown has been kept apart from free since day 9, precisely so this question could be
asked. Until now nothing asked it.
"""
import math

import numpy as np
import pytest

from warp_av.perception.occupancy import (OccupancyGrid, FREE, OCCUPIED, UNKNOWN,
                                          DEFAULT_LANE_HALF_M, DEFAULT_BLIND_REACH_M,
                                          DEFAULT_BLIND_LOOK_M, EGO_NOSE_M)
from warp_av.behavior.behavior import (stopping_speed_for, BLIND_REACTION_S, BLIND_DECEL_MPS2)


def all_seen():
    g = OccupancyGrid()
    g.cells[:] = FREE
    g.updated = True
    return g


def unseen_at(g, x, y, side_m=1.0):
    """A patch of unseen space big enough to hide someone, centred on (x, y)."""
    n = max(1, int(round(side_m / g.cell_m)))
    for i in range(n):
        for j in range(n):
            r, c = g.to_cell(np.array([float(x) + i * g.cell_m]),
                             np.array([float(y) + j * g.cell_m]))
            g.cells[int(r[0]), int(c[0])] = UNKNOWN
    return g


def one_square_unseen_at(g, x, y):
    r, c = g.to_cell(np.array([float(x)]), np.array([float(y)]))
    g.cells[int(r[0]), int(c[0])] = UNKNOWN
    return g


# ---------------------------------------------------------------- what it cannot see


def test_a_road_it_can_see_all_of_answers_nothing():
    assert all_seen().blind_spot_ahead() is None


def test_an_unseen_pocket_beside_the_lane_is_found():
    g = unseen_at(all_seen(), 6.0, DEFAULT_LANE_HALF_M + 1.0)
    assert g.blind_spot_ahead() == pytest.approx(6.0, abs=g.cell_m)


def test_the_nearest_one_is_the_answer():
    g = all_seen()
    unseen_at(g, 11.0, 2.0)
    unseen_at(g, 6.0, 2.0)
    assert g.blind_spot_ahead() == pytest.approx(6.0, abs=g.cell_m)


def test_unseen_far_off_to_the_side_is_not_a_pocket():
    """Behind the buildings across the road is unseen and always will be."""
    g = unseen_at(all_seen(), 6.0, DEFAULT_LANE_HALF_M + DEFAULT_BLIND_REACH_M + 2.0)
    assert g.blind_spot_ahead() is None


def test_unseen_a_long_way_ahead_is_not_a_pocket_yet():
    """There is time to see it before we get there."""
    g = unseen_at(all_seen(), DEFAULT_BLIND_LOOK_M + 4.0, 2.0)
    assert g.blind_spot_ahead() is None


def test_it_does_not_look_under_the_van():
    """The van's own returns are thrown away, so the squares beneath it are always unseen.
    Asking about them would make every answer the same tiny number."""
    g = unseen_at(all_seen(), EGO_NOSE_M - 1.0, 0.0)
    assert g.blind_spot_ahead() is None


def test_one_stray_square_is_not_a_hiding_place():
    """The laser's rings leave gaps, and while driving they are everywhere. The first
    version of this rule counted them and fired on 25 readings out of 28 on a clear road."""
    g = one_square_unseen_at(all_seen(), 6.0, 2.0)
    assert g.blind_spot_ahead() is None
    for x in (5.5, 8.0, 10.0):
        one_square_unseen_at(g, x, 1.0)
    assert g.blind_spot_ahead() is None, "scattered single squares are still not a pocket"


def test_a_pocket_must_be_deep_as_well_as_wide():
    """A thin line of unseen squares across the lane is a ring gap, not a doorway."""
    g = all_seen()
    for j in range(8):
        one_square_unseen_at(g, 7.0, 0.5 + j * g.cell_m)
    assert g.blind_spot_ahead() is None


def test_blocked_is_not_unseen():
    """A wall we can SEE is an obstacle, not a pocket -- the object list already has it."""
    g = all_seen()
    r, c = g.to_cell(np.array([6.0]), np.array([2.0]))
    g.cells[int(r[0]), int(c[0])] = OCCUPIED
    assert g.blind_spot_ahead() is None


# ---------------------------------------------------------------- what speed that is worth


def test_the_speed_is_one_you_could_actually_stop_from():
    """The rule, checked by doing the sum the other way round."""
    for d in (3.0, 5.5, 8.0, 12.0):
        v = stopping_speed_for(d)
        travelled = v * BLIND_REACTION_S + v * v / (2.0 * BLIND_DECEL_MPS2)
        assert travelled == pytest.approx(d, rel=1e-6)


def test_seeing_further_allows_more_speed():
    speeds = [stopping_speed_for(d) for d in (3.0, 5.5, 8.0, 12.0, 15.0)]
    assert speeds == sorted(speeds)


def test_it_stops_mattering_by_itself_at_a_sensible_range():
    """No threshold needed: past about 14 m the answer is already above cruising speed."""
    assert stopping_speed_for(5.5) < 5.0
    assert stopping_speed_for(DEFAULT_BLIND_LOOK_M) > 8.0


def test_nothing_is_ever_a_negative_speed():
    assert stopping_speed_for(0.0) == 0.0
    assert stopping_speed_for(-5.0) == 0.0


# ---------------------------------------------------------------- and it reaches the driving

from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior


def decide(speed=8.0, stop=False, blind=None, sensor_cap=None):
    b = BehaviorSystem()
    b._speed_cap_mps = sensor_cap
    b._blind_spot_m = blind
    return b._decide(DrivingBehavior.FOLLOWING_ROUTE, "cruising", speed, stop)


def test_with_nothing_hidden_the_van_is_not_slowed():
    out = decide(speed=8.0, blind=None)
    assert out.desired_speed_mps == 8.0
    assert "cannot see" not in out.reason


def test_a_pocket_beside_the_lane_slows_it_and_says_why():
    out = decide(speed=8.0, blind=5.5)
    assert out.desired_speed_mps == pytest.approx(stopping_speed_for(5.5))
    assert out.desired_speed_mps < 5.0
    assert "cannot see past 5.5 m" in out.reason


def test_a_pocket_it_could_already_stop_for_changes_nothing():
    """Going slowly enough already: no cap, and no confusing message."""
    out = decide(speed=3.0, blind=8.0)
    assert out.desired_speed_mps == 3.0
    assert "cannot see" not in out.reason


def test_the_tightest_cap_wins_and_names_itself():
    missing_sensor = decide(speed=8.0, blind=12.0, sensor_cap=2.0)
    assert missing_sensor.desired_speed_mps == 2.0
    assert "a sensor is missing" in missing_sensor.reason

    blind_pocket = decide(speed=8.0, blind=4.0, sensor_cap=7.0)
    assert blind_pocket.desired_speed_mps == pytest.approx(stopping_speed_for(4.0))
    assert "cannot see past 4.0 m" in blind_pocket.reason


def test_no_cap_can_ever_speed_the_van_up():
    assert decide(speed=1.0, blind=12.0).desired_speed_mps == 1.0


def test_no_cap_can_turn_a_stop_into_driving():
    out = decide(speed=0.0, stop=True, blind=12.0)
    assert out.should_stop is True and out.desired_speed_mps == 0.0
