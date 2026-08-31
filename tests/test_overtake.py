"""Go-around a dead vehicle: swing one lane left, pass, rejoin.
V1 contract: straights only, never near junctions, needs a real lane,
and the rewritten route must free the corridor of the dead car."""
import math

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def straight_route(n=60, step=2.0):
    return Route(waypoints=[Waypoint(x=i * step, y=0.0) for i in range(n)])


def test_overtake_path_shifts_left_and_rejoins():
    r = straight_route()
    rejoin = planner().plan_overtake(r, 0.0, 0.0, obstacle_along_m=15.0,
                                     lane_ok=lambda x, y: True)
    assert rejoin is not None
    # CARLA frame: LEFT of +x heading is negative y.
    # Clearance builds along the whole approach: >2.2 m by the car's rear
    # corner (~x 12.5), full lane beside and past it.
    corner = [wp.y for wp in r.waypoints if 12.0 <= wp.x <= 14.0]
    assert corner and all(y < -2.2 for y in corner), \
        f"need >2.2 m clearance at the rear corner, got {corner}"
    beside = [wp.y for wp in r.waypoints if 16.0 <= wp.x <= 23.0]
    assert beside and all(y < -3.3 for y in beside), \
        f"full lane offset beside/past the obstacle, got {beside}"
    tail = [wp.y for wp in r.waypoints if wp.x >= 34.0]
    assert tail and all(abs(y) < 0.25 for y in tail), "must rejoin the lane"
    assert math.hypot(rejoin.x - 31.0, rejoin.y) < 4.0


def test_overtake_refuses_without_a_lane():
    r = straight_route()
    assert planner().plan_overtake(r, 0.0, 0.0, 15.0,
                                   lane_ok=lambda x, y: False) is None
    assert all(abs(wp.y) < 1e-9 for wp in r.waypoints), "route untouched on refusal"


def test_overtake_refuses_near_junctions_and_bends():
    r = straight_route()
    for wp in r.waypoints:
        if 18.0 <= wp.x <= 24.0:
            wp.is_junction = True
    assert planner().plan_overtake(r, 0.0, 0.0, 15.0, lane_ok=lambda x, y: True) is None

    bend = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=math.radians(i * 2.5))
                            for i in range(60)])
    assert planner().plan_overtake(bend, 0.0, 0.0, 15.0, lane_ok=lambda x, y: True) is None


def test_overtake_refuses_when_destination_is_too_close():
    r = straight_route(n=12)      # ~22 m of route, rejoin needs ~31 m
    assert planner().plan_overtake(r, 0.0, 0.0, 15.0, lane_ok=lambda x, y: True) is None


def test_dead_car_no_longer_blocks_the_rewritten_route():
    r = straight_route()
    p = planner()
    assert p.plan_overtake(r, 0.0, 0.0, 15.0, lane_ok=lambda x, y: True) is not None
    # the dead car sits at (15, 0) — world == ego frame for pose (0,0,yaw 0)
    car = DetectedObject(object_type=ObjectType.VEHICLE, x=15.0, y=0.0,
                         distance=15.0, speed=0.0)
    out = p.filter_to_route_corridor(PerceptionOutput(objects=[car]), r, 0.0, 0.0, 0.0)
    assert out.path_blocked is False, \
        "after the rewrite the dead car must sit a lane away from the corridor"


def test_departure_blend_eases_out_of_a_bay():
    """Van parked 6 m right of the lane: the route head must start AT the
    van's side and decay to the lane centre — no full-lock swing."""
    r = straight_route()
    ok = planner().blend_departure(r, 0.0, 6.0)     # ego in the right bay
    assert ok
    head = r.waypoints[0]
    assert abs(head.y - 6.0) < 0.8, f"route must start beside the van, got y={head.y}"
    mid = [wp.y for wp in r.waypoints if 4.0 <= wp.x <= 8.0]
    assert mid and all(0.5 < y < 5.5 for y in mid), "offset must decay smoothly"
    tail = [wp.y for wp in r.waypoints if wp.x >= 16.0]
    assert tail and all(abs(y) < 0.3 for y in tail), "must converge to the lane"


def test_departure_blend_noop_when_already_in_lane():
    r = straight_route()
    ok = planner().blend_departure(r, 0.0, 0.4)
    assert not ok
    assert all(abs(wp.y) < 1e-9 for wp in r.waypoints)
