"""Perception V2 day 12 part two: the map remembers a moment ago.

Measured live in Town10HD on 2026-09-09, standing still on a clear road: 'free ahead'
swung between 10.5 and 17.5 m, jumping 2 m or more in 17 readings out of 40. Walking the
same strip by hand showed what it kept stopping at -- UNSEEN squares, 24 times out of 24,
and never once anything solid. Those are the gaps between the laser's rings.
"""
import math
import time

import numpy as np
import pytest

from warp_av.perception.occupancy import (OccupancyGrid, FREE, OCCUPIED, UNKNOWN,
                                          FREE_MEMORY_TICKS, BLOCKED_MEMORY_TICKS,
                                          MAX_CARRY_TURN_DEG)


def fan(hit_m=25.0, skip_deg=None):
    """A sweep: returns off something solid all round, optionally with a wedge missing."""
    pts = []
    for deg in np.arange(-60.0, 60.0, 0.5):
        if skip_deg and skip_deg[0] <= deg <= skip_deg[1]:
            continue
        a = math.radians(deg)
        pts.append([hit_m * math.cos(a), hit_m * math.sin(a)])
    arr = np.array(pts, dtype=np.float32)
    return arr, np.ones(arr.shape[0], dtype=bool)


def nothing():
    return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=bool)


STILL = (0.0, 0.0, 0.0)


def test_a_gap_in_the_rings_reads_as_a_wall_without_memory():
    """The fault being fixed, stated as a test so it cannot come back unnoticed."""
    g = OccupancyGrid()
    g.update(*fan(skip_deg=(-8.0, 8.0)))
    assert g.free_distance() == pytest.approx(3.0, abs=0.3)


def test_and_does_not_with_memory():
    g = OccupancyGrid()
    g.update(*fan())
    clear = g.free_distance()
    assert clear > 20.0
    g.update(*fan(skip_deg=(-8.0, 8.0)), moved=STILL)
    assert g.free_distance() == pytest.approx(clear, abs=0.3)


def test_empty_space_is_not_believed_for_long():
    """Half a second ago it was empty. Two seconds ago somebody may be standing in it."""
    g = OccupancyGrid()
    g.update(*fan())
    for _ in range(FREE_MEMORY_TICKS):
        g.update(*fan(skip_deg=(-8.0, 8.0)), moved=STILL)
        assert g.free_distance() > 20.0                 # still believed
    g.update(*fan(skip_deg=(-8.0, 8.0)), moved=STILL)
    assert g.free_distance() == pytest.approx(3.0, abs=0.3)


def test_blocked_is_remembered_longer_than_free():
    assert BLOCKED_MEMORY_TICKS > FREE_MEMORY_TICKS
    g = OccupancyGrid()
    g.update(np.array([[8.0, 0.0]], dtype=np.float32), np.ones(1, dtype=bool))
    assert g.at(8.0, 0.0) == OCCUPIED
    for _ in range(FREE_MEMORY_TICKS + 1):
        g.update(*nothing(), moved=STILL)
    assert g.at(8.0, 0.0) == OCCUPIED, "an obstacle must not evaporate as fast as empty space"
    for _ in range(BLOCKED_MEMORY_TICKS):
        g.update(*nothing(), moved=STILL)
    assert g.at(8.0, 0.0) == UNKNOWN


def test_a_sharp_turn_rebuilds_rather_than_guesses():
    """Sliding whole squares is a slide, not a rotation. Turning, the map starts again --
    which is exactly what it always used to do."""
    g = OccupancyGrid()
    g.update(np.array([[10.0, 0.0]], dtype=np.float32), np.ones(1, dtype=bool))
    g.update(*nothing(), moved=(0.0, 0.0, MAX_CARRY_TURN_DEG + 0.5))
    assert g.at(10.0, 0.0) == UNKNOWN


def test_what_the_laser_says_now_always_wins():
    """Memory is painted over by the new sweep, never the other way round."""
    g = OccupancyGrid()
    g.update(*fan())
    assert g.at(10.0, 0.0) == FREE
    g.update(np.array([[10.0, 0.0]], dtype=np.float32), np.ones(1, dtype=bool),
             moved=STILL)
    assert g.at(10.0, 0.0) == OCCUPIED


def test_the_map_moves_with_the_van():
    """Something 10 m ahead is 8 m ahead after the van has gone 2 m."""
    g = OccupancyGrid()
    g.update(np.array([[10.0, 0.0]], dtype=np.float32), np.ones(1, dtype=bool))
    assert g.at(10.0, 0.0) == OCCUPIED
    g.update(*nothing(), moved=(2.0, 0.0, 0.0))
    assert g.at(8.0, 0.0) == OCCUPIED
    assert g.at(10.0, 0.0) != OCCUPIED


def occupied_spots(g):
    return [g.to_metres(r, c) for r, c in zip(*np.where(g.cells == OCCUPIED))]


def test_the_map_slides_sideways_too():
    g = OccupancyGrid()
    g.update(np.array([[10.0, 0.0]], dtype=np.float32), np.ones(1, dtype=bool))
    g.update(*nothing(), moved=(0.0, 1.5, 0.0))       # the van moved 1.5 m to its right
    spots = occupied_spots(g)
    assert len(spots) == 1
    x, y = spots[0]
    assert x == pytest.approx(10.0, abs=0.3)
    assert y == pytest.approx(-1.5, abs=0.3), "what was straight ahead is now to our left"


def test_without_a_motion_it_behaves_exactly_as_it_used_to():
    """No `moved` = rebuilt from this sweep alone. The offline scorer relies on it."""
    g = OccupancyGrid()
    g.update(*fan())
    g.update(*fan(skip_deg=(-8.0, 8.0)))
    assert g.free_distance() == pytest.approx(3.0, abs=0.3)


def test_carrying_the_map_is_cheap_enough_to_do_every_tick():
    g = OccupancyGrid()
    p, o = fan()
    g.update(p, o)
    t0 = time.perf_counter()
    for i in range(50):
        g.update(p, o, moved=(0.08, 0.0, 0.1), now=200.0 + i * 0.1)
    per_ms = (time.perf_counter() - t0) / 50 * 1000.0
    assert per_ms < 6.0, f"{per_ms:.1f} ms per sweep would cost the van decisions"
