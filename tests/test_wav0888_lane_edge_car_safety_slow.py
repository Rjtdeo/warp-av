"""This test exists because WAV-0888 came to rest 0.26 m from a car parked at the lane edge of
the north-east bend (V1.6/V1.7, 2026-09-17). The corridor judges a parked car by its centre's
distance from the ROUTE LINE (2.9 m here: clear), while the van, 0.8 m outside the line and
heading outward, had the car 13 m ahead and a metre to the side of its nose. V1.6 added a
heading-line check that was only a comfort slow and vanished with each steering tick. V1.7
holds it and makes it required.

Frame: route along +x at y = 0; van pose (x, y, yaw); objects handed over in the van's frame
exactly as the planner converts them.
"""
import math

import pytest

from warp_av.behavior.behavior import BehaviorSystem
from warp_av.behavior import transitions as T
from warp_av.localization.localization import Pose
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PATH_SLOW, PATH_CLEAR
from warp_av.planning.planner import RoutePlanner, Route, Waypoint

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
CAR = (21.0, 2.9)            # world: 11 m ahead along the route, centre 2.9 m right of the line
CAR_LEN, CAR_WID = 4.7, 1.9


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def car_seen_from(ego, wx=CAR[0], wy=CAR[1]):
    ex, ey, yaw = ego
    dx, dy = wx - ex, wy - ey
    x = math.cos(yaw) * dx + math.sin(yaw) * dy
    y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y), speed=0.0,
                          length_m=CAR_LEN, width_m=CAR_WID, stationary=True)


def decide(pl, ego, objects):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = False
    p.closest_obstacle_distance = 999.0
    pl.filter_to_route_corridor(p, route(), *ego, footprint=FOOT)
    return pl.last_decision


def test_case_a_wav0888_lane_edge_car_is_a_required_slow_and_survives_a_steering_tick():
    pl = planner()
    outward = (10.0, 0.8, math.radians(12.0))       # inside the corridor, nose 12 deg outward
    out = decide(pl, outward, [car_seen_from(outward)])
    assert out.level == PATH_SLOW and out.blocked is False and out.edge_hold is True
    assert 10.0 <= out.closest_distance_m <= 12.0
    # the behaviour makes it required (so the V1.5 ease-off leaves it alone)
    b = BehaviorSystem(); b.set_mission(); b.slow_speed = 2.0
    o = b.update(perception=PerceptionOutput(), pose=Pose(healthy=True, speed=6.0), destination_distance=115.0,
                 safety_ok=True, path=out)
    assert o.why == T.OBJECT_AHEAD_SLOW and o.desired_speed_mps == 2.0 and o.safety_required is True
    # one steering tick swings the nose back toward the line: the heading no longer points at
    # the car, but the car is still beside the body -- the slow must stay
    back = (10.6, 0.85, math.radians(-4.0))
    out2 = decide(pl, back, [car_seen_from(back)])
    assert out2.level == PATH_SLOW and out2.edge_hold is True and out2.blocked is False


def test_case_b_a_safely_separated_shoulder_car_stays_clear():
    pl = planner()
    on_line = (10.0, 0.0, 0.0)                      # on the line, pointing along it
    out = decide(pl, on_line, [car_seen_from(on_line)])
    assert out.level == PATH_CLEAR and out.edge_hold is False and out.closest_distance_m == 999.0
    # a moving vehicle and a far one are left to the existing rules
    moving = car_seen_from((10.0, 0.8, math.radians(12.0)))
    moving.speed = 5.0
    moving.stationary = False
    assert decide(planner(), (10.0, 0.8, math.radians(12.0)), [moving]).level == PATH_CLEAR


def test_case_c_the_hold_releases_once_the_car_is_passed_or_safely_clear():
    pl = planner()
    outward = (10.0, 0.8, math.radians(12.0))
    assert decide(pl, outward, [car_seen_from(outward)]).edge_hold is True
    # the van has steered well back onto its line and is passing wide: safely clear -> released
    wide = (14.0, -0.6, 0.0)
    out = decide(pl, wide, [car_seen_from(wide)])
    assert out.edge_hold is False and out.level == PATH_CLEAR
    # and a car already behind the nose is never held
    pl2 = planner()
    assert decide(pl2, outward, [car_seen_from(outward)]).edge_hold is True
    passed = (26.0, 0.8, 0.0)                       # 5 m past the car
    out = decide(pl2, passed, [car_seen_from(passed)])
    assert out.edge_hold is False and out.level == PATH_CLEAR
