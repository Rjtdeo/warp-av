"""Troy fix #2/#3: slow down for left AND right turns, gradually, before the bend."""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint


def planner():
    p = RoutePlanner.__new__(RoutePlanner)   # skip CARLA constructor
    return p


def straight(n=60, spacing=2.0):
    return Route(waypoints=[Waypoint(x=i * spacing, y=0.0) for i in range(n)])


def route_with_turn(turn_at_m=40.0, radius=8.0, left=True, spacing=2.0):
    """Straight, then a 90-degree arc of the given radius, then straight again."""
    wps = []
    x = 0.0
    while x < turn_at_m:
        wps.append(Waypoint(x=x, y=0.0))
        x += spacing
    steps = max(4, int((math.pi / 2 * radius) / spacing))
    sgn = 1.0 if left else -1.0
    for i in range(1, steps + 1):
        a = (math.pi / 2) * i / steps
        wps.append(Waypoint(x=turn_at_m + radius * math.sin(a),
                            y=sgn * (radius - radius * math.cos(a))))
    lx, ly = wps[-1].x, wps[-1].y
    for i in range(1, 20):
        wps.append(Waypoint(x=lx, y=ly + sgn * i * spacing))
    return Route(waypoints=wps)


def test_straight_road_no_cap():
    assert planner().curve_speed_cap(straight(), 0.0, 0.0, cruise=8.0) == 8.0


def test_cap_tightens_as_turn_approaches():
    p = planner()
    r = route_with_turn(turn_at_m=40.0, radius=8.0)
    far = p.curve_speed_cap(r, 0.0, 0.0, cruise=8.0)        # turn 40 m away: outside horizon
    mid = p.curve_speed_cap(r, 20.0, 0.0, cruise=8.0)       # 20 m away
    near = p.curve_speed_cap(r, 34.0, 0.0, cruise=8.0)      # 6 m away
    inside = p.curve_speed_cap(r, 44.0, 2.0, cruise=8.0)    # in the bend
    assert far == 8.0
    assert 8.0 >= mid >= near >= inside
    assert near < 7.0, f"not slowing before the corner: {near:.1f}"
    # 8 m radius at 2 m/s^2 comfort -> ~4 m/s in the turn
    assert 2.5 <= inside <= 5.0, f"in-turn speed {inside:.1f} not in the sane range"


def test_left_and_right_turns_treated_equally():
    p = planner()
    l = p.curve_speed_cap(route_with_turn(left=True), 34.0, 0.0, cruise=8.0)
    r = p.curve_speed_cap(route_with_turn(left=False), 34.0, 0.0, cruise=8.0)
    assert abs(l - r) < 0.01, f"left {l:.2f} vs right {r:.2f} must match"


def test_gentle_curve_barely_caps():
    p = planner()
    r = route_with_turn(turn_at_m=20.0, radius=40.0)   # sweeping highway bend
    cap = p.curve_speed_cap(r, 18.0, 0.0, cruise=8.0)
    assert cap > 5.5, f"gentle bend over-slowing: {cap:.1f}"


def test_never_below_minimum():
    p = planner()
    r = route_with_turn(turn_at_m=10.0, radius=2.5)    # hairpin
    cap = p.curve_speed_cap(r, 9.0, 0.0, cruise=8.0)
    assert cap >= p.V_TURN_MIN


# ---------------------------------------------------------------------------
# Corner-cutting regression (kerb/divider clipping): drive the full pipeline
# through a 90-degree corner and measure worst distance from the lane centre.
# ---------------------------------------------------------------------------
from warp_av.control.controller import VehicleController
from test_controller_stability import SimVan, DT


def _dist_to_polyline(px, py, pts):
    best = float("inf")
    for a, b in zip(pts, pts[1:]):
        ax, ay, bx, by = a.x, a.y, b.x, b.y
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def _drive_corner(left):
    """Full pipeline: curve cap + bend lookahead + centreline correction."""
    p = planner()
    route = route_with_turn(turn_at_m=40.0, radius=8.0, left=left)
    van = SimVan(speed=8.0)
    ctrl = VehicleController()
    worst = 0.0
    for _ in range(int(45 / DT)):
        cap = p.curve_speed_cap(route, van.x, van.y, cruise=8.0)
        desired = min(8.0, cap)
        la = max(5.0, min(13.0, 1.6 * van.speed))
        if cap < 7.5:
            la = min(la, 5.5)
        wp = p.get_next_waypoint(route, van.x, van.y, lookahead=la)
        ct = p.signed_cross_track(route, van.x, van.y)
        cmd = ctrl.compute_command(van.x, van.y, van.yaw, van.speed,
                                   wp.x, wp.y, desired, should_stop=False,
                                   cross_track_m=ct)
        van.step(cmd)
        worst = max(worst, _dist_to_polyline(van.x, van.y, route.waypoints))
        if math.hypot(van.x - route.waypoints[-1].x, van.y - route.waypoints[-1].y) < 4.0:
            break
    return worst


def test_no_corner_cutting_left_or_right():
    # CARLA lane half-width ~1.75 m, van half-width ~1.0 m: centre deviation
    # under 0.9 m keeps the body inside the lane (no divider/footpath contact).
    worst_left = _drive_corner(left=True)
    worst_right = _drive_corner(left=False)
    assert worst_left < 0.9, f"LEFT corner: strayed {worst_left:.2f} m from lane centre"
    assert worst_right < 0.9, f"RIGHT corner: strayed {worst_right:.2f} m from lane centre"


def test_cross_track_sign():
    p = planner()
    r = straight()
    assert p.signed_cross_track(r, 10.0, 2.0) > 1.9      # left of path
    assert p.signed_cross_track(r, 10.0, -2.0) < -1.9    # right of path
    assert abs(p.signed_cross_track(r, 10.0, 0.0)) < 0.01


def test_aim_point_lies_on_the_road():
    # The interpolated aim point must sit ON the route polyline even mid-bend —
    # this is the property that prevents aiming across the inside of a corner.
    p = planner()
    route = route_with_turn(turn_at_m=40.0, radius=8.0, left=True)
    for x, y in [(30.0, 0.0), (38.0, 0.0), (41.0, 0.5), (44.0, 2.0)]:
        wp = p.get_next_waypoint(route, x, y, lookahead=6.0)
        assert _dist_to_polyline(wp.x, wp.y, route.waypoints) < 0.05
        # anchor node can sit up to one waypoint spacing ahead of the vehicle
        assert math.hypot(wp.x - x, wp.y - y) <= 6.0 + 2.1, "aim point far beyond the requested lookahead"
