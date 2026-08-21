"""
Route-corridor in-path check: the dashboard bug where a tilted (mid-turn) van
flagged a vehicle in the NEIGHBOURING lane as "ahead" and stopped.
"""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def straight_route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def route_with_junction(x_from=14.0, x_to=20.0):
    """Straight route whose waypoints in [x_from, x_to] are junction-tagged."""
    wps = [Waypoint(x=i * 2.0, y=0.0, is_junction=(x_from <= i * 2.0 <= x_to))
           for i in range(60)]
    return Route(waypoints=wps)


def ego_frame(obj_wx, obj_wy, ego_x, ego_y, ego_yaw):
    """World -> ego frame, same math as perception._actor_to_object."""
    dx, dy = obj_wx - ego_x, obj_wy - ego_y
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    return dx * c - dy * s, dx * s + dy * c


def obj_at_world(wx, wy, ego, kind=ObjectType.VEHICLE, speed=0.0):
    ex, ey, eyaw = ego
    x, y = ego_frame(wx, wy, ex, ey, eyaw)
    return DetectedObject(object_type=kind, x=x, y=y,
                          distance=math.hypot(wx - ex, wy - ey), speed=speed)


def run_filter(route, ego, objects):
    p = PerceptionOutput(objects=objects)
    # ego-box verdict first (what perception would say): worst case all blocked
    p.path_blocked = True
    p.closest_obstacle_distance = min(o.distance for o in objects)
    return planner().filter_to_route_corridor(p, route, *ego)


def test_tilted_van_ignores_vehicle_off_route():
    # Van mid-turn at (10, 0), nose 50 degrees off the route. A car sits dead
    # ahead of the NOSE at 5 m -> world (13.2, 3.8): 3.8 m off the corridor.
    ego = (10.0, 0.0, math.radians(50))
    car_w = (10 + 5 * math.cos(ego[2]), 0 + 5 * math.sin(ego[2]))
    out = run_filter(straight_route(), ego, [obj_at_world(*car_w, ego)])
    assert out.path_blocked is False, "off-route vehicle must not block the path"
    assert out.closest_obstacle_distance == 999.0


def test_vehicle_on_route_still_blocks_even_when_tilted():
    ego = (10.0, 0.0, math.radians(50))
    out = run_filter(straight_route(), ego, [obj_at_world(16.0, 0.3, ego)])
    assert out.path_blocked is True
    assert 5.0 < out.closest_obstacle_distance < 7.0    # ~6 m along the route


def test_obstacle_around_the_corner_now_seen():
    # Route turns right at x=20; obstacle sits on the route AFTER the bend.
    wps = [Waypoint(x=i * 2.0, y=0.0) for i in range(11)]
    wps += [Waypoint(x=20.0, y=-(i * 2.0)) for i in range(1, 15)]
    route = Route(waypoints=wps)
    ego = (14.0, 0.0, 0.0)                      # still on the straight, nose forward
    obstacle_w = (20.0, -6.0)                   # around the corner, on the route
    out = run_filter(route, ego, [obj_at_world(*obstacle_w, ego, kind=ObjectType.OBSTACLE)])
    assert out.path_blocked is False            # ~12 m along the route: not yet danger
    assert 10.0 < out.closest_obstacle_distance < 14.0
    ego_close = (19.0, 0.0, 0.0)
    out = run_filter(route, ego_close, [obj_at_world(*obstacle_w, ego_close, kind=ObjectType.OBSTACLE)])
    assert out.path_blocked is True             # ~8 m along the route: stop


def test_object_behind_on_route_ignored():
    ego = (30.0, 0.0, 0.0)
    out = run_filter(straight_route(), ego, [obj_at_world(20.0, 0.0, ego)])
    assert out.path_blocked is False and out.closest_obstacle_distance == 999.0


def test_no_route_keeps_ego_box_verdict():
    p = PerceptionOutput(objects=[DetectedObject(object_type=ObjectType.VEHICLE, x=5, y=0, distance=5)],
                         path_blocked=True, closest_obstacle_distance=5.0)
    out = planner().filter_to_route_corridor(p, None, 0, 0, 0)
    assert out.path_blocked is True and out.closest_obstacle_distance == 5.0


def test_cross_street_waiter_slows_but_does_not_block():
    """The junction-deadlock bug: a car waiting at the cross-street stop line,
    1.6 m off our path AT A JUNCTION, must slow us — never freeze the
    mission. (Junction-tagged: cross-street geometry is give-way's job.)"""
    ego = (10.0, 0.0, 0.0)
    out = run_filter(route_with_junction(), ego, [obj_at_world(16.0, 1.6, ego)])
    assert out.path_blocked is False, "off-centre waiter must not hard-block"
    assert out.closest_obstacle_distance < 8.0      # still seen -> slow zone
    assert out.closest_obstacle_lateral_m == 1.6    # and the evidence is visible


def test_parked_car_in_narrow_bay_blocks_instead_of_scraping():
    """Sweep run 62: a STATIONARY car 1.6 m off-centre on a plain straight
    is a physical-width conflict (van 2.0 m + car 1.8 m > 2×1.6 m) — the van
    must stop, not squeeze past and scrape."""
    ego = (10.0, 0.0, 0.0)
    out = run_filter(straight_route(), ego, [obj_at_world(16.0, 1.6, ego)])
    assert out.path_blocked is True, "narrow-gap parked car must hard-block"


def test_moving_car_in_band_does_not_wide_block():
    """A MOVING car drifting through the 1.4-2.05 m band (overtake, merge)
    must not phantom-brake the van via the wide-body rule."""
    ego = (10.0, 0.0, 0.0)
    out = run_filter(straight_route(), ego,
                     [obj_at_world(16.0, 1.6, ego, speed=6.0)])
    assert out.path_blocked is False


def test_true_lead_car_still_blocks():
    ego = (10.0, 0.0, 0.0)
    out = run_filter(straight_route(), ego, [obj_at_world(16.0, 0.6, ego)])
    assert out.path_blocked is True
    assert out.closest_obstacle_lateral_m == 0.6
