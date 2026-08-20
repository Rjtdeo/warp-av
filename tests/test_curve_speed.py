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
    assert cap > 6.5, f"gentle bend over-slowing: {cap:.1f}"


def test_never_below_minimum():
    p = planner()
    r = route_with_turn(turn_at_m=10.0, radius=2.5)    # hairpin
    cap = p.curve_speed_cap(r, 9.0, 0.0, cruise=8.0)
    assert cap >= p.V_TURN_MIN
