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
    # CARLA frame: LEFT of +x heading is negative y
    ys = {round(wp.x): wp.y for wp in r.waypoints}
    beside = [wp.y for wp in r.waypoints if 12.0 <= wp.x <= 22.0]
    assert beside and all(y < -3.0 for y in beside), \
        f"path must run a full lane left beside the obstacle, got {beside}"
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
