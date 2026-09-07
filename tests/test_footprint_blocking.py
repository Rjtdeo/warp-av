"""
Planning V2, phase 1B: the swept van body decides hard-blocks for STATIONARY
objects when a footprint is passed; without one, the old centre-line bands
apply unchanged.
"""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint, FOOTPRINT_STATIONARY_REACH_M
from warp_av.planning.footprint import VehicleFootprint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)   # swept half-width 1.29 m


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def straight_route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def route_with_junction(x_from=14.0, x_to=20.0):
    wps = [Waypoint(x=i * 2.0, y=0.0, is_junction=(x_from <= i * 2.0 <= x_to)) for i in range(60)]
    return Route(waypoints=wps)


def left_bend_route(radius=6.0, straight_in=10.0):
    """Straight along +x, then a left-hand arc. No junction tags."""
    pts = [(x, 0.0) for x in range(0, int(straight_in) + 1, 1)]
    cx, cy = straight_in, radius
    n = 30
    for k in range(1, n + 1):
        a = math.radians(90.0) * k / n
        pts.append((cx + radius * math.sin(a), cy - radius * math.cos(a)))
    return Route(waypoints=[Waypoint(x=x, y=y) for x, y in pts])


def ego_frame(obj_wx, obj_wy, ego_x, ego_y, ego_yaw):
    dx, dy = obj_wx - ego_x, obj_wy - ego_y
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    return dx * c - dy * s, dx * s + dy * c


def obj_at_world(wx, wy, ego, kind=ObjectType.VEHICLE, speed=0.0):
    ex, ey, eyaw = ego
    x, y = ego_frame(wx, wy, ex, ey, eyaw)
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(wx - ex, wy - ey), speed=speed)


def run(route, ego, objects, footprint=None):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = True
    p.closest_obstacle_distance = min(o.distance for o in objects)
    return planner().filter_to_route_corridor(p, route, *ego, footprint=footprint)


def same_verdict(a, b):
    return (a.path_blocked, a.closest_obstacle_distance, a.closest_obstacle_type,
            a.closest_obstacle_speed, a.closest_obstacle_lateral_m) == \
           (b.path_blocked, b.closest_obstacle_distance, b.closest_obstacle_type,
            b.closest_obstacle_speed, b.closest_obstacle_lateral_m)


def test_flag_off_is_the_old_behaviour_including_the_band_block():
    ego = (10.0, 0.0, 0.0)
    # a stationary planter 1.9 m off the line, 6 m ahead: the old wide-body band
    # (1.40-2.20 m, stationary, within 12 m) hard-blocks it
    out = run(straight_route(), ego, [obj_at_world(16.0, 1.9, ego, kind=ObjectType.OBSTACLE)])
    assert out.path_blocked is True
    # and the default of the new argument is None, i.e. OFF
    import inspect
    assert inspect.signature(RoutePlanner.filter_to_route_corridor).parameters["footprint"].default is None


def test_flag_on_stationary_object_outside_the_swept_path_does_not_block():
    ego = (10.0, 0.0, 0.0)
    planter = obj_at_world(16.0, 1.9, ego, kind=ObjectType.OBSTACLE)   # radius 0.5 -> nearest edge 1.4 m > 1.29 m
    out = run(straight_route(), ego, [planter], footprint=FOOT)
    assert out.path_blocked is False
    # it is still SEEN (slow zone bookkeeping unchanged): closest distance is set
    assert out.closest_obstacle_distance == 6.0


def test_flag_on_stationary_object_inside_the_swept_path_blocks():
    ego = (10.0, 0.0, 0.0)
    out = run(straight_route(), ego, [obj_at_world(16.0, 1.0, ego, kind=ObjectType.OBSTACLE)], footprint=FOOT)
    assert out.path_blocked is True
    # a parked CAR 1.6 m off the line (run 62): half a car reaches into our margin -> still blocks
    out = run(straight_route(), ego, [obj_at_world(16.0, 1.6, ego)], footprint=FOOT)
    assert out.path_blocked is True
    # a true lead car at 0.6 m blocks in both modes
    assert run(straight_route(), ego, [obj_at_world(16.0, 0.6, ego)], footprint=FOOT).path_blocked is True


def test_bend_case_the_old_centre_line_rule_misses_is_caught():
    route = left_bend_route(radius=6.0, straight_in=10.0)
    ego = (6.0, 0.0, 0.0)          # bin is ~8.7 m ahead along the route, inside the 12 m stationary reach
    # a stationary bin 2.35 m OUTSIDE the arc, half-way round the bend:
    cx, cy, r = 10.0, 6.0, 6.0
    a = math.radians(45.0)
    px, py = cx + r * math.sin(a), cy - r * math.cos(a)
    nx, ny = (px - cx) / r, (py - cy) / r
    bin_ = obj_at_world(px + 2.35 * nx, py + 2.35 * ny, ego, kind=ObjectType.OBSTACLE)
    # old rules: 2.35 m > 2.20 m from the line -> the object is not even considered
    old = run(route, ego, [bin_])
    assert old.path_blocked is False
    # swept body: the outer front corner swings ~1.6 m wide in a 6 m bend, plus
    # 0.30 m margin plus the bin's 0.5 m radius reaches past 2.35 m -> blocked
    new = run(route, ego, [bin_], footprint=FOOT)
    assert new.path_blocked is True


def test_moving_vehicle_verdicts_are_identical_with_the_flag_on():
    ego = (10.0, 0.0, 0.0)
    for lat in (0.6, 1.6, 2.0):
        objs = [obj_at_world(16.0, lat, ego, speed=6.0)]
        assert same_verdict(run(straight_route(), ego, objs), run(straight_route(), ego, objs, footprint=FOOT))
    # a moving car drifting through the band never wide-blocks in either mode
    assert run(straight_route(), ego, [obj_at_world(16.0, 1.6, ego, speed=6.0)], footprint=FOOT).path_blocked is False


def test_junction_verdicts_are_identical_with_the_flag_on():
    ego = (10.0, 0.0, 0.0)
    waiter = [obj_at_world(16.0, 1.6, ego)]                   # stationary, junction-tagged stretch
    off, on = run(route_with_junction(), ego, waiter), run(route_with_junction(), ego, waiter, footprint=FOOT)
    assert same_verdict(off, on) and on.path_blocked is False
    # a stationary car right in our lane at a junction still blocks in both modes
    lead = [obj_at_world(16.0, 0.5, ego)]
    off, on = run(route_with_junction(), ego, lead), run(route_with_junction(), ego, lead, footprint=FOOT)
    assert same_verdict(off, on) and on.path_blocked is True


def test_pedestrian_verdicts_are_identical_with_the_flag_on():
    ego = (10.0, 0.0, 0.0)
    walker = [obj_at_world(16.0, 0.8, ego, kind=ObjectType.PEDESTRIAN, speed=1.4)]
    assert same_verdict(run(straight_route(), ego, walker), run(straight_route(), ego, walker, footprint=FOOT))


def test_stationary_object_beyond_the_sweep_reach_is_not_blocked_by_the_sweep():
    ego = (0.0, 0.0, 0.0)
    far = [obj_at_world(FOOTPRINT_STATIONARY_REACH_M + 6.0, 0.3, ego, kind=ObjectType.OBSTACLE)]
    out = run(straight_route(), ego, far, footprint=FOOT)
    assert out.path_blocked is False and out.closest_obstacle_distance == FOOTPRINT_STATIONARY_REACH_M + 6.0


def test_planner_flag_defaults_off():
    p = planner()
    RoutePlanner.__init__  # constructing needs CARLA; check the attribute the constructor sets
    import inspect
    src = inspect.getsource(RoutePlanner.__init__)
    assert "self.use_footprint_blocking = False" in src
