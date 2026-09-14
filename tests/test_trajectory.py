"""
Planning V2 task 3: the next few seconds, written down.

The van used to have "aim there, go this fast" and nothing that said where it would be in two
seconds. planning/trajectory.py writes that down every tick -- the path the controller would
drive (pure pursuit at the route), the speed the behaviour asked for held to the bend cap and
comfortable acceleration and braking, and the time each point is reached -- and the off-route
body sweep (task 1) now slides along that path instead of a straight line from the nose.

Frame: CARLA's. yaw turns +x towards +y, so +y is the van's RIGHT and positive curvature
turns it right. Route along +x at y = 0 unless said otherwise.
"""
import math

import pytest

from warp_av.planning.trajectory import (plan_trajectory, Trajectory, TrajectoryPoint, HORIZON_M,
                                         ACCEL_MPS2, MAX_CURVATURE)
from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.footprint import VehicleFootprint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
DECEL, A_LAT, V_MIN = RoutePlanner.A_DECEL, RoutePlanner.A_LAT_MAX, RoutePlanner.V_TURN_MIN


def straight(n=60):
    return [Waypoint(x=i * 2.0, y=0.0) for i in range(n)]


def bend(radius=8.0, straight_in=14.0):
    """Straight along +x, then a right-hand quarter circle of `radius`, then straight."""
    pts = [(x, 0.0) for x in range(0, int(straight_in) + 1, 2)]
    cx, cy = straight_in, radius
    for k in range(1, 10):
        a = -math.pi / 2 + (math.pi / 2) * k / 9
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    ex, ey = pts[-1]
    pts += [(ex, ey + d) for d in range(2, 30, 2)]
    return [Waypoint(x=x, y=y) for x, y in pts]


def off_the_route(p, route):
    """How far a point sits from the route LINE (not from its nearest waypoint)."""
    from warp_av.planning.footprint import _project, _polyline
    return abs(_project(p.x, p.y, _polyline(route))[1])


# ---- geometry -------------------------------------------------------------------------------

def test_on_the_route_it_follows_the_route():
    t = plan_trajectory((10.0, 0.0, 0.0), 4.0, straight(), (18.0, 0.0), 4.0)
    assert t.points[0].x == 10.0 and t.points[0].y == 0.0 and t.points[0].s == 0.0
    assert t.length_m == pytest.approx(HORIZON_M)
    assert all(abs(p.y) < 0.05 and abs(p.curvature) < 1e-3 for p in t.points)
    assert t.points[-1].x == pytest.approx(35.0, abs=0.1)
    assert all(p.speed == pytest.approx(4.0) for p in t.points)
    assert t.points[-1].t == pytest.approx(HORIZON_M / 4.0, abs=0.05)
    ss = [p.s for p in t.points]
    assert ss == sorted(ss) and all(b - a > 0 for a, b in zip(ss, ss[1:]))


def test_off_the_route_it_rejoins_it_smoothly():
    # 3.5 m to the LEFT of its line (y = -3.5), pointing along it, aim 6 m ahead on the line
    t = plan_trajectory((10.0, -3.5, 0.0), 2.0, straight(), (16.0, 0.0), 4.0)
    first = [p for p in t.points if 0.0 < p.s <= 5.0]
    assert all(p.curvature > 0.0 for p in first), "rejoining a line on its right starts by turning right"
    assert max(abs(p.curvature) for p in t.points) < 0.2, "a gentle arc, not a swerve"
    assert max(p.y for p in t.points) < 0.3, "it does not swing past the line"
    later = [p for p in t.points if p.s > 8.0]
    assert all(abs(p.curvature) < 0.05 for p in later), "then eases straight, no second swerve"
    late = [p for p in t.points if p.s > 17.0]
    assert late and all(abs(p.y) < 0.15 for p in late), "back on the line within 17 m"
    assert all(abs(p.yaw) < math.radians(5.0) for p in late), "...and pointing along it"
    assert max(abs(p.yaw) for p in t.points) < math.radians(25)
    # no kink: heading never jumps more than a step of the tightest arc allows
    jumps = [abs(b.yaw - a.yaw) for a, b in zip(t.points, t.points[1:])]
    assert max(jumps) < 0.2 * 0.5 + 1e-6


def test_positive_curvature_is_a_right_turn_in_carlas_frame():
    right = plan_trajectory((0.0, 0.0, 0.0), 3.0, [Waypoint(x=x, y=3.0) for x in range(0, 60, 2)], (6.0, 3.0), 3.0)
    left = plan_trajectory((0.0, 0.0, 0.0), 3.0, [Waypoint(x=x, y=-3.0) for x in range(0, 60, 2)], (6.0, -3.0), 3.0)
    assert right.points[1].curvature > 0 and right.points[8].y > 0.5
    assert left.points[1].curvature < 0 and left.points[8].y < -0.5


def test_through_a_bend_it_stays_near_the_route():
    route = bend()
    t = plan_trajectory((2.0, 0.0, 0.0), 3.0, route, (8.0, 0.0), 4.0)
    assert all(off_the_route(p, route) < 0.6 for p in t.points), "cuts the corner by less than a wheel"
    turning = [p for p in t.points if 16.0 < p.s < 24.0]
    assert turning and all(p.curvature > 0.05 for p in turning), "a right-hand bend reads as right turns"


def test_no_route_means_straight_on_and_no_aim_means_the_route_from_here():
    t = plan_trajectory((5.0, 2.0, math.radians(30)), 3.0, [], None, 3.0)
    assert t.length_m == pytest.approx(HORIZON_M)
    end = t.points[-1]
    assert end.x == pytest.approx(5.0 + HORIZON_M * math.cos(math.radians(30)), abs=0.01)
    assert end.y == pytest.approx(2.0 + HORIZON_M * math.sin(math.radians(30)), abs=0.01)
    assert all(p.curvature == 0.0 for p in t.points)
    u = plan_trajectory((10.0, -2.0, 0.0), 3.0, straight(), None, 3.0)
    assert abs(u.points[-1].y) < 0.15, "without an aim point it still rejoins the route"


def test_the_route_running_out_means_straight_on_from_its_end():
    t = plan_trajectory((10.0, 0.0, 0.0), 3.0, straight(9), (16.0, 0.0), 3.0)     # route ends at x = 16
    assert t.length_m == pytest.approx(HORIZON_M)
    assert all(abs(p.y) < 0.05 for p in t.points) and t.points[-1].x == pytest.approx(35.0, abs=0.1)


def test_an_aim_point_behind_cannot_ask_for_a_spiral():
    t = plan_trajectory((10.0, 0.0, math.pi), 1.0, straight(), (14.0, 0.0), 2.0)
    assert all(abs(p.curvature) <= MAX_CURVATURE + 1e-9 for p in t.points)
    assert t.length_m == pytest.approx(HORIZON_M)


# ---- speed and time --------------------------------------------------------------------------

def test_pulling_away_is_no_harder_than_the_controller_allows():
    t = plan_trajectory((0.0, 0.0, 0.0), 0.0, straight(), (5.0, 0.0), 4.0)
    assert t.points[0].speed == 0.0
    for p in t.points[1:]:
        assert p.speed <= math.sqrt(2.0 * ACCEL_MPS2 * p.s) + 1e-6
        assert p.speed <= 4.0 + 1e-9
    assert t.points[-1].speed == pytest.approx(4.0)
    ts = [p.t for p in t.points]
    assert ts == sorted(ts) and ts[1] > 0.0


def test_asked_to_stop_the_speed_falls_to_zero_and_the_geometry_still_goes_on():
    t = plan_trajectory((10.0, 0.0, 0.0), 4.0, straight(), (18.0, 0.0), 0.0)
    assert t.stops_at_m == pytest.approx(16.0 / (2.0 * DECEL))
    assert t.points[0].speed == 4.0
    beyond = [p for p in t.points if p.s > t.stops_at_m + 0.5]
    assert beyond and all(p.speed == 0.0 for p in beyond)
    assert t.length_m == pytest.approx(HORIZON_M), "the path it would take stays written down"
    standing = plan_trajectory((10.0, 0.0, 0.0), 0.0, straight(), (18.0, 0.0), 0.0)
    assert standing.stops_at_m == 0.0 and all(p.speed == 0.0 for p in standing.points)


def test_a_bend_ahead_is_taken_at_the_bend_cap_and_braked_for_in_time():
    route = bend()
    t = plan_trajectory((2.0, 0.0, 0.0), 6.0, route, (8.0, 0.0), 8.0)
    turning = [p for p in t.points if 16.0 < p.s < 24.0]
    for p in turning:
        assert p.speed <= max(V_MIN, math.sqrt(A_LAT / p.curvature)) + 1e-6
    for a, b in zip(t.points, t.points[1:]):           # never slows harder than decel
        assert a.speed <= math.sqrt(b.speed ** 2 + 2.0 * DECEL * (b.s - a.s)) + 1e-6


def test_time_moves_even_for_a_crawling_van():
    t = plan_trajectory((0.0, 0.0, 0.0), 0.0, straight(), (5.0, 0.0), 0.3)
    assert all(b.t > a.t for a, b in zip(t.points, t.points[1:]))
    assert t.duration_s < 200.0


def test_at_time_and_the_api_dict():
    t = plan_trajectory((0.0, 0.0, 0.0), 4.0, straight(), (8.0, 0.0), 4.0)
    p = t.at_time(1.0)
    assert p is not None and p.s == pytest.approx(4.0, abs=0.5)
    assert t.at_time(99.0) is t.points[-1]
    d = t.as_dict()
    assert d["length_m"] == 25.0 and d["stops_at_m"] is None and d["aim"] == [8.0, 0.0]
    assert 12 <= len(d["points"]) <= 15, "one point every 2 m plus the last"
    assert set(d["points"][0]) == {"s", "x", "y", "yaw_deg", "curvature", "speed", "t"}
    assert d["points"][-1]["s"] == 25.0


# ---- the first thing that reads it: the off-route body sweep -----------------------------------

def run_filter(ego, things, intended_path=None):
    per = PerceptionOutput(objects=things)
    return RoutePlanner.__new__(RoutePlanner).filter_to_route_corridor(
        per, Route(waypoints=straight()), *ego, footprint=FOOT, intended_path=intended_path)


def obj_at_world(wx, wy, ego, kind=ObjectType.OBSTACLE):
    ex, ey, eyaw = ego
    dx, dy = wx - ex, wy - ey
    c, s = math.cos(-eyaw), math.sin(-eyaw)
    return DetectedObject(object_type=kind, x=dx * c - dy * s, y=dx * s + dy * c,
                          distance=math.hypot(dx, dy), speed=0.0)


def test_steering_home_the_sweep_follows_the_path_not_the_nose():
    # 3.5 m left of its line, steering back to it (aim 6 m ahead on the line). A mailbox 10 m
    # dead ahead of the nose is in the NEXT lane the van is leaving, not on its path.
    ego = (10.0, -3.5, 0.0)
    path = plan_trajectory(ego, 2.0, straight(), (16.0, 0.0), 4.0).xy()
    mailbox = obj_at_world(20.0, -3.5, ego)
    assert run_filter(ego, [mailbox]).path_blocked is True, "straight-ahead sweep (task 1): blocked"
    assert run_filter(ego, [mailbox], intended_path=path).path_blocked is False, "on the real path: not in the way"
    # ...but a thing standing ON that path, though not straight ahead of the nose, is
    on_path = obj_at_world(17.0, -0.3, ego)
    out = run_filter(ego, [on_path], intended_path=path)
    assert out.path_blocked is True and out.closest_obstacle_distance == pytest.approx(7.0)


def test_dead_ahead_on_the_path_still_blocks_with_a_path():
    ego = (10.0, 3.0, 0.0)
    path = plan_trajectory(ego, 0.0, straight(), (16.0, 0.0), 4.0).xy()
    out = run_filter(ego, [obj_at_world(14.0, 2.6, ego)], intended_path=path)   # 4 m ahead, on the arc
    assert out.path_blocked is True


def test_on_the_route_the_path_changes_nothing():
    ego = (10.0, 0.0, 0.0)
    path = plan_trajectory(ego, 4.0, straight(), (18.0, 0.0), 4.0).xy()
    for things in ([obj_at_world(15.0, 0.2, ego)], [obj_at_world(15.0, 3.0, ego)]):
        a, b = run_filter(ego, things), run_filter(ego, things, intended_path=path)
        assert (a.path_blocked, a.closest_obstacle_distance) == (b.path_blocked, b.closest_obstacle_distance)


def test_a_too_short_path_falls_back_to_the_straight_line():
    ego = (10.0, 3.0, 0.0)
    out = run_filter(ego, [obj_at_world(15.0, 3.0, ego)], intended_path=[(10.0, 3.0)])
    assert out.path_blocked is True


def test_main_builds_it_every_tick_and_hands_it_to_the_sweep():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert "self._trajectory = plan_trajectory(" in src
    assert "intended_path=(self._trajectory.xy()" in src
    assert '"trajectory": (self._trajectory.as_dict()' in src
    i = src.index("self._trajectory = plan_trajectory(")
    assert "behavior_output.should_stop" in src[i:i + 600], "a stop is written down as a stop"
