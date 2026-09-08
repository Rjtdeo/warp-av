"""
Perception V2 day 10: where the road stops.

Day 9's map says where there is space. It does not say where the van may put a
wheel: a laser beam travels along a pavement perfectly well, so a pavement came
back as free.

Two signals, and the measurement that decided how much weight each gets.

The kerb itself is a step of ten or fifteen centimetres running alongside the
road. That is the textbook answer, and in this town it is thin evidence: the
pavement sits between 0 and 15 cm above the road, sometimes level with it, and
the laser returns as few as six points from a kerb in twenty metres. So the
kerb line is reported only when there is real support for it, and honestly
declined otherwise.

Flat ground is the well-supported signal: thousands of returns land on it. It
does not separate carriageway from pavement, but it does separate ground the
van could roll on from free space it merely looked through.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.occupancy import FREE, NOT_ROAD, OCCUPIED, ROAD, OccupancyGrid  # noqa: E402
from warp_av.perception.road_edges import (KERB_MAX_M, KERB_MIN_M, MIN_POINTS, RoadEdge,  # noqa: E402
                                           RoadEdges, find_road_edges)


def kerb_points(y, x_from=2.0, x_to=20.0, step=0.4, height=0.14, drift=0.0):
    """A kerb running alongside the van at lateral offset y."""
    xs = np.arange(x_from, x_to, step)
    pts = np.stack([xs, np.full_like(xs, y) + drift * xs], axis=1)
    return pts, np.full(len(xs), height, dtype=np.float32)


def road_points(n=400, width=8.0, ahead=25.0):
    rng = np.random.default_rng(5)
    xs = rng.uniform(1.0, ahead, n)
    ys = rng.uniform(-width / 2, width / 2, n)
    return np.stack([xs, ys], axis=1), np.zeros(n, dtype=np.float32)


def test_a_straight_kerb_is_found_where_it_is():
    pts, h = kerb_points(-3.2)
    edges = find_road_edges(pts, h)
    assert edges.left is not None and edges.left.confident
    assert edges.left.offset_m == pytest.approx(-3.2, abs=0.2)
    assert abs(edges.left.heading_deg) < 3.0
    assert edges.right is None


def test_both_sides_give_the_width_of_the_road():
    left, hl = kerb_points(-3.5)
    right, hr = kerb_points(3.5)
    edges = find_road_edges(np.vstack([left, right]), np.concatenate([hl, hr]))
    assert edges.width_m == pytest.approx(7.0, abs=0.4)


def test_a_kerb_that_turns_away_is_reported_turning():
    pts, h = kerb_points(-3.0, drift=-0.1)         # widens by 10 cm per metre ahead
    edges = find_road_edges(pts, h)
    assert edges.left is not None
    assert edges.left.heading_deg < -3.0, "it should read as running away from the van"
    assert edges.left.lateral_at(10.0) < edges.left.lateral_at(0.0)
    closing, hc = kerb_points(-3.0, drift=0.1)     # and the other way round
    assert find_road_edges(closing, hc).left.heading_deg > 3.0


def test_a_few_stray_points_do_not_drag_the_line():
    pts, h = kerb_points(-3.0)
    strays = np.array([[6.0, -6.0], [7.0, -6.4], [8.0, -5.9]], dtype=np.float32)
    edges = find_road_edges(np.vstack([pts, strays]),
                            np.concatenate([h, np.full(3, 0.2, dtype=np.float32)]))
    assert edges.left.offset_m == pytest.approx(-3.0, abs=0.3), "the wheels of a parked car"


def test_nothing_is_claimed_when_there_is_nothing_to_see():
    pts, h = road_points()
    edges = find_road_edges(pts, h)
    assert edges.left is None or not edges.left.confident
    assert edges.right is None or not edges.right.confident
    assert edges.width_m is None
    assert edges.drivable(8.0, 0.0) is None, "unknown must not be reported as fine"


def test_too_few_points_is_declined_not_guessed():
    pts, h = kerb_points(-3.0, x_from=2.0, x_to=4.0, step=0.5)
    assert len(pts) < MIN_POINTS
    assert find_road_edges(pts, h).left is None


def test_only_the_kerb_band_is_used():
    """A wall is not a kerb, and neither is a flat road."""
    pts, _ = kerb_points(-3.0)
    tall = np.full(len(pts), 1.8, dtype=np.float32)
    flat = np.full(len(pts), 0.01, dtype=np.float32)
    assert find_road_edges(pts, tall).left is None
    assert find_road_edges(pts, flat).left is None
    assert KERB_MIN_M < 0.14 < KERB_MAX_M


def test_inside_the_kerbs_is_drivable_and_outside_is_not():
    left, hl = kerb_points(-3.5)
    right, hr = kerb_points(3.5)
    edges = find_road_edges(np.vstack([left, right]), np.concatenate([hl, hr]))
    assert edges.drivable(8.0, 0.0) is True
    assert edges.drivable(8.0, 4.5) is False, "that is the pavement"
    assert edges.drivable(8.0, -4.5) is False


def test_free_space_is_not_the_same_as_ground_to_drive_on():
    """The day-10 point: a beam that passes over a drop leaves free space behind it,
    and the van must not treat that as road."""
    g = OccupancyGrid(range_m=20.0)
    hits, occ = [], []
    for b in np.arange(-30.0, 30.1, 0.5):
        a = math.radians(b)
        hits.append((14.0 * math.cos(a), 14.0 * math.sin(a)))
        occ.append(False)                       # ground returns at 14 m
    g.update(np.array(hits, dtype=np.float32), np.array(occ, dtype=bool))
    assert g.at(8.0, 0.0) == FREE
    assert g.drivable_at(8.0, 0.0) is True, "ground was seen along that bearing"
    assert g.drivable_at(8.0, 12.0) is False, "no beam ever landed out there"


def test_the_ground_layer_is_filled_not_drawn_in_rings():
    """Marking only the squares the returns land in draws thin arcs, because the laser
    touches the ground in rings. The gap between rings is still ground."""
    g = OccupancyGrid(range_m=20.0)
    hits, occ = [], []
    for b in np.arange(-40.0, 40.1, 0.5):
        for r in (6.0, 12.0):                   # two rings, nothing between them
            a = math.radians(b)
            hits.append((r * math.cos(a), r * math.sin(a)))
            occ.append(False)
    g.update(np.array(hits, dtype=np.float32), np.array(occ, dtype=bool))
    assert g.drivable_at(9.0, 0.0) is True, "between the rings is still ground"
    assert g.road_edge(6.0, "right") is not None


def test_the_ground_layer_stops_at_a_solid_thing():
    g = OccupancyGrid(range_m=20.0)
    hits, occ = [], []
    for b in np.arange(-20.0, 20.1, 0.5):
        a = math.radians(b)
        hits.append((16.0 * math.cos(a), 16.0 * math.sin(a)))
        occ.append(False)
    for b in np.arange(-8.0, 8.1, 0.25):        # a wall across the road at 7 m
        a = math.radians(b)
        hits.append((7.0 * math.cos(a), 7.0 * math.sin(a)))
        occ.append(True)
    g.update(np.array(hits, dtype=np.float32), np.array(occ, dtype=bool))
    assert g.drivable_at(5.0, 0.0) is True
    assert g.drivable_at(10.0, 0.0) is False, "the ground behind it was never seen"


def test_the_summary_carries_where_the_ground_reaches():
    g = OccupancyGrid(range_m=20.0)
    hits, occ = [], []
    for b in np.arange(-60.0, 60.1, 0.5):
        for r in np.arange(2.0, 15.0, 1.0):
            a = math.radians(b)
            hits.append((r * math.cos(a), r * math.sin(a)))
            occ.append(False)
    g.update(np.array(hits, dtype=np.float32), np.array(occ, dtype=bool))
    s = g.summary()
    assert s.road_left_m is not None and s.road_right_m is not None
    assert s.road_width_m > 4.0
    d = s.as_dict()
    assert set(d["road"]) == {"left_m", "right_m", "width_m"}


def test_an_edge_reports_where_it_will_be_further_ahead():
    e = RoadEdge(side="right", offset_m=3.0, heading_deg=5.0, length_m=12.0,
                 points=40, inlier_share=0.9)
    assert e.confident
    assert e.lateral_at(0.0) == pytest.approx(3.0)
    assert e.lateral_at(10.0) == pytest.approx(3.0 + 10.0 * math.tan(math.radians(5.0)), abs=0.01)


def test_an_edge_across_the_road_is_not_believed():
    e = RoadEdge(side="right", offset_m=3.0, heading_deg=70.0, length_m=12.0,
                 points=40, inlier_share=0.9)
    assert not e.confident, "a kerb runs alongside the road, not across it"
