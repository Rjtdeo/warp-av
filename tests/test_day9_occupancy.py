"""
Perception V2 day 9: a map of free space.

A list of objects answers "stop for that". It cannot answer "where can I go?",
and it turns one long wall into a crowd of invented objects. The grid answers
both: every square around the van is free, blocked, or not seen.

Unknown is kept apart from free on purpose. Behind a parked van is unknown,
not empty, and a planner that treats the two the same drives into whatever is
hiding there.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.occupancy import (FREE, OCCUPIED, UNKNOWN, OccupancyGrid)  # noqa: E402


def ray(bearing_deg, ranges):
    """Laser returns along one bearing, at the given ranges."""
    a = math.radians(bearing_deg)
    return [(r * math.cos(a), r * math.sin(a)) for r in ranges]


def scan(hits, road=()):
    """hits are solid returns, road are ground returns."""
    pts = list(hits) + list(road)
    occ = [True] * len(hits) + [False] * len(road)
    return np.array(pts, dtype=np.float32).reshape(-1, 2), np.array(occ, dtype=bool)


def test_an_empty_grid_claims_nothing():
    g = OccupancyGrid()
    assert np.all(g.cells == UNKNOWN)
    assert g.at(5.0, 0.0) == UNKNOWN
    from warp_av.perception.occupancy import EGO_NOSE_M
    assert g.summary().free_ahead_m == pytest.approx(EGO_NOSE_M, abs=0.3), \
        "with nothing seen, the van may not claim any road beyond its own nose"


def test_a_beam_makes_the_space_it_crossed_free_and_where_it_stopped_blocked():
    g = OccupancyGrid(range_m=20.0)
    g.update(*scan(ray(0.0, [10.0])))
    assert g.at(10.0, 0.0) == OCCUPIED
    for d in (1.0, 4.0, 8.0, 9.5):
        assert g.at(d, 0.0) == FREE, f"the beam went through {d} m to get there"


def test_what_is_behind_a_thing_is_unknown_not_empty():
    g = OccupancyGrid(range_m=20.0)
    g.update(*scan(ray(0.0, [10.0])))
    for d in (11.0, 14.0, 18.0):
        assert g.at(d, 0.0) == UNKNOWN, "the van has not seen behind the thing at 10 m"


def test_the_nearest_thing_along_a_bearing_is_what_stops_the_beam():
    g = OccupancyGrid(range_m=20.0)
    g.update(*scan(ray(0.0, [6.0, 12.0])))
    assert g.at(6.0, 0.0) == OCCUPIED
    assert g.at(3.0, 0.0) == FREE
    assert g.at(9.0, 0.0) == UNKNOWN, "hidden behind the nearer one"


def test_road_returns_open_up_free_space_with_nothing_blocking():
    g = OccupancyGrid(range_m=20.0)
    g.update(*scan([], road=ray(0.0, [15.0])))
    assert g.at(15.0, 0.0) in (FREE, UNKNOWN)
    assert g.at(8.0, 0.0) == FREE
    assert g.at(18.0, 0.0) == UNKNOWN


def test_how_far_it_can_go_stops_at_the_first_thing_in_the_van_s_width():
    g = OccupancyGrid(range_m=20.0)
    hits = []
    for b in np.arange(-40.0, 40.1, 0.25):
        hits += ray(float(b), [18.0])
    hits += ray(2.0, [8.0])                       # one thing, just off dead ahead
    g.update(*scan(hits))
    assert g.free_distance(0.0, half_width_m=1.5) < 8.5
    assert g.free_distance(0.0, half_width_m=0.1) > 8.5, "a narrow line squeezes past it"


def test_unknown_space_is_not_a_road_the_van_may_use():
    """The strip test must refuse unseen squares, or the van drives into the dark."""
    g = OccupancyGrid(range_m=20.0)
    g.update(*scan(ray(0.0, [12.0])))             # only one bearing was ever measured
    from warp_av.perception.occupancy import EGO_NOSE_M
    assert g.free_distance(0.0, half_width_m=1.5) <= EGO_NOSE_M + 0.3


def test_a_wall_is_one_surface_and_not_a_crowd_of_objects():
    """The point of the grid: a long flat thing reads as a line of blocked squares."""
    g = OccupancyGrid(range_m=25.0)
    hits = [(x, 6.0) for x in np.arange(-8.0, 20.0, 0.2)]
    g.update(*scan(hits))
    blocked = int(np.count_nonzero(g.cells == OCCUPIED))
    assert blocked > 60, "the whole wall should be marked, not a few blobs"
    assert g.at(10.0, 6.0) == OCCUPIED
    assert g.at(10.0, 3.0) == FREE, "the road this side of it is open"
    assert g.at(10.0, 9.0) == UNKNOWN, "and the far side is hidden by it"


def test_a_thing_between_two_slices_still_casts_a_shadow():
    """A post or a cone can fall between two bearings. Its neighbours are held back too,
    which gives away a little free space rather than sweeping straight past it."""
    wide = OccupancyGrid(range_m=20.0, spread_slices=4)
    narrow = OccupancyGrid(range_m=20.0, spread_slices=0)
    pts, occ = scan(ray(0.0, [10.0]) + ray(0.35, [19.0]))
    wide.update(pts, occ)
    narrow.update(pts, occ)
    a = math.radians(0.35)
    assert narrow.at(14.0 * math.cos(a), 14.0 * math.sin(a)) == FREE
    assert wide.at(14.0 * math.cos(a), 14.0 * math.sin(a)) == UNKNOWN


def test_the_summary_is_what_a_planner_reads():
    g = OccupancyGrid(range_m=20.0)
    hits = []
    for b in np.arange(-60.0, 60.1, 0.25):
        hits += ray(float(b), [15.0])
    g.update(*scan(hits))
    s = g.summary()
    assert s.free_ahead_m == pytest.approx(15.0, abs=1.0)
    assert s.free_cells > 0 and s.occupied_cells > 0
    d = s.as_dict()
    assert 0.0 <= d["seen_share"] <= 1.0
    import json
    json.dumps(d)


def test_it_is_fast_enough_to_run_every_tick():
    import time
    g = OccupancyGrid()
    rng = np.random.default_rng(3)
    n = 7000
    b = rng.uniform(0, 360, n)
    r = rng.uniform(2, 28, n)
    pts = np.stack([r * np.cos(np.radians(b)), r * np.sin(np.radians(b))], axis=1).astype(np.float32)
    occ = rng.random(n) < 0.15
    t0 = time.perf_counter()
    for _ in range(5):
        g.update(pts, occ)
    ms = (time.perf_counter() - t0) * 1000.0 / 5
    assert ms < 40.0, f"{ms:.1f} ms per turn is too slow for a 10 Hz loop"


def test_coordinates_round_trip():
    g = OccupancyGrid(range_m=10.0, cell_m=0.5)
    for x, y in ((0.0, 0.0), (3.2, -1.7), (-4.0, 8.0)):
        r, c = g.to_cell(x, y)
        bx, by = g.to_metres(int(r), int(c))
        assert abs(bx - x) <= g.cell_m and abs(by - y) <= g.cell_m


def test_it_never_claims_a_square_it_did_not_see():
    """Every square the grid calls free must be nearer than a return on its bearing."""
    g = OccupancyGrid(range_m=15.0)
    hits = []
    for b in np.arange(0.0, 360.0, 1.0):
        hits += ray(float(b), [9.0])
    g.update(*scan(hits))
    rows, cols = np.nonzero(g.cells == FREE)
    xs, ys = g.to_metres(rows, cols)
    assert np.max(np.hypot(xs, ys)) <= 9.2, "free space must stop at the ring of returns"
