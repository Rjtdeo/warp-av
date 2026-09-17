"""This test exists because WAV-0888 came to rest 0.26 m from a car parked at the lane edge of
the north-east bend (V1.6, 2026-09-17), after V1.5 had removed the eased-away slowdown.

The corridor judges a parked car by its centre's distance from the ROUTE LINE. That car's
centre was ~2.9 m from the line -- clear by every route-frame rule -- but the van was 0.8 m
outside the line and heading outward, so in the van's own frame the car was 13 m ahead and
about a metre to the side. Nothing looked along the van's heading until it was a full
corridor width off the route, and by then the true gap was 0.36 m.

Frame: route along +x at y = 0; van pose (x, y, yaw); objects handed over in the van's frame
exactly as the planner converts them (so world positions below are what matter).
"""
import math

import pytest

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PATH_SLOW, PATH_CLEAR
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)   # the live body
CAR_LEN, CAR_WID = 4.7, 1.9                                                        # the parked Mustang


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def straight_route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def car_at_world(ego, wx, wy, yaw_deg=0.0):
    """A stationary vehicle at a world position, expressed in the van's frame the way the planner undoes it."""
    ex, ey, yaw = ego
    dx, dy = wx - ex, wy - ey
    x = math.cos(yaw) * dx + math.sin(yaw) * dy
    y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y), speed=0.0,
                          length_m=CAR_LEN, width_m=CAR_WID, yaw_deg=yaw_deg, stationary=True)


def decide(ego, objects):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = False
    p.closest_obstacle_distance = 999.0
    pl = planner()
    pl.filter_to_route_corridor(p, straight_route(), *ego, footprint=FOOT)
    return pl.last_decision


def test_wav0888_a_lane_edge_car_on_the_vans_heading_slows_it_early():
    # the van 0.8 m right of the line (inside the corridor, so not "off route"), heading 12 deg
    # further out; the car's centre 2.9 m right of the route line, 11 m ahead along it
    ego = (10.0, 0.8, math.radians(12.0))
    out = decide(ego, [car_at_world(ego, 21.0, 2.9)])
    assert out.level == PATH_SLOW and out.blocked is False      # SLOW, not a stop
    assert out.closest_kind == ObjectType.VEHICLE
    assert 10.0 <= out.closest_distance_m <= 12.0              # at its real distance, early
    assert out.used_footprint is True


def test_the_same_car_beside_a_van_that_is_on_its_line_is_clear():
    # on the line, pointing along it: the car's near side is 1.95 m off the line, the van's
    # swept body reaches 1.29 m -- a safely separated shoulder car must not slow the van
    ego = (10.0, 0.0, 0.0)
    out = decide(ego, [car_at_world(ego, 21.0, 2.9)])
    assert out.level == PATH_CLEAR and out.blocked is False and out.closest_distance_m == 999.0


def test_a_moving_vehicle_and_a_far_one_are_left_to_the_existing_rules():
    ego = (10.0, 0.8, math.radians(12.0))
    moving = car_at_world(ego, 21.0, 2.9)
    moving.speed = 5.0
    moving.stationary = False
    assert decide(ego, [moving]).level == PATH_CLEAR
    assert decide(ego, [car_at_world(ego, 40.0, 2.9)]).level == PATH_CLEAR      # 30 m ahead: beyond the reach
