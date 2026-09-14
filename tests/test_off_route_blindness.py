"""
Planning V2 task 1: the van is blind to what is straight ahead of it once it has drifted off
its route.

filter_to_route_corridor judges every object by its distance from the ROUTE LINE and never
looks further than 2.2-2.6 m from it. A van 3 m off its route with a car five metres in
front of its nose therefore reported "clear, 999 m" -- and live on 2026-09-13 it drove into
a mailbox it had seen on every reading. The fix: while the van's centre is outside the
corridor it is checking, its own body is also slid straight ahead from where it really is.

Frame: route along +x at y = 0; van pose (x, y, yaw); objects in the van's frame.
"""
import math

import pytest

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import BLOCKED_SWEPT_PATH
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)  # the live body
OFF_ROUTE_Y = 3.0     # 3 m off the line: past every lateral limit the corridor rules have


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def straight_route():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def route_with_junction(x_from=14.0, x_to=20.0):
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, is_junction=(x_from <= i * 2.0 <= x_to))
                            for i in range(60)])


def obj_ahead(x, y=0.0, kind=ObjectType.VEHICLE, speed=0.0):
    """An object x m ahead of the steering point and y m to the right, in the van's frame."""
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed)


def run(route, ego, objects, footprint=FOOT):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = False
    p.closest_obstacle_distance = 999.0
    return planner().filter_to_route_corridor(p, route, *ego, footprint=footprint)


# TEST 1 -- the bug: off route, stationary thing dead ahead, must block at its real distance
@pytest.mark.parametrize("kind", [ObjectType.VEHICLE, ObjectType.OBSTACLE])
def test_off_route_van_sees_what_is_straight_ahead(kind):
    ego = (10.0, OFF_ROUTE_Y, 0.0)                  # 3 m right of the route, pointing along it
    out = run(straight_route(), ego, [obj_ahead(5.0, kind=kind)])
    assert out.path_blocked is True, "5 m dead ahead of an off-route van must block"
    assert out.closest_obstacle_distance == pytest.approx(5.0)


def test_off_route_block_is_reported_as_the_swept_body():
    ego = (10.0, OFF_ROUTE_Y, 0.0)
    pl = planner()
    p = PerceptionOutput(objects=[obj_ahead(5.0, kind=ObjectType.OBSTACLE)])
    pl.filter_to_route_corridor(p, straight_route(), *ego, footprint=FOOT)
    assert pl.last_decision.reason == BLOCKED_SWEPT_PATH
    assert pl.last_decision.used_footprint is True
    assert pl.last_decision.blocker_distance_m == pytest.approx(5.0)


def test_off_route_van_pointing_back_at_the_route_still_sees_ahead():
    # wandered 6 m right (the mailbox run: off the road), now steering back towards the
    # line at 30 degrees; a parked car sits 5 m in front of its nose -- world (14.3, 3.5),
    # 3.5 m from the route line where no corridor rule looks, but squarely in its way
    ego = (10.0, 6.0, math.radians(-30))
    out = run(straight_route(), ego, [obj_ahead(5.0)])
    assert out.path_blocked is True
    assert out.closest_obstacle_distance == pytest.approx(5.0)


def test_corridor_only_mode_is_left_alone():
    # No body to sweep (footprint blocking off): the old corridor verdict, unchanged. This
    # is what the bug looked like; the fix needs the body and is only active with it.
    ego = (10.0, OFF_ROUTE_Y, 0.0)
    out = run(straight_route(), ego, [obj_ahead(5.0)], footprint=None)
    assert out.path_blocked is False
    assert out.closest_obstacle_distance == 999.0


# TEST 2 -- on route (and merely within the corridor) nothing new runs
@pytest.mark.parametrize("y", [0.0, 1.0, 1.75])
def test_on_route_verdicts_are_the_corridor_rules_as_before(y):
    ego = (10.0, y, 0.0)
    # a car 5 m ahead in our lane: the band rule blocks it, exactly as before
    out = run(straight_route(), ego, [obj_ahead(5.0)])
    assert out.path_blocked is True
    assert out.closest_obstacle_distance == pytest.approx(5.0)
    # a planter 5 m ahead but 3 m out to the side: not in the corridor, not blocked
    out = run(straight_route(), ego, [obj_ahead(5.0, 3.0, kind=ObjectType.OBSTACLE)])
    assert out.path_blocked is False


# TEST 3 -- the false positive the old test guards: mid-turn, nose across the next lane
def test_tilted_van_on_route_still_ignores_the_car_its_nose_points_at():
    # the original test_tilted_van_ignores_vehicle_off_route, with the live body switched on:
    # van ON its route at (10, 0), nose 50 degrees off it, a car 5 m dead ahead of the nose
    ego = (10.0, 0.0, math.radians(50))
    out = run(straight_route(), ego, [obj_ahead(5.0)])
    assert out.path_blocked is False
    assert out.closest_obstacle_distance == 999.0


def test_off_route_van_in_a_junction_does_not_block_on_cross_traffic():
    # off route AND turning through a junction: a straight-ahead sweep would say the car
    # waiting in the cross street is in our way. It is not -- we are turning. Off junctions
    # the sweep stays out, exactly as the route sweep does.
    ego = (12.0, OFF_ROUTE_Y, math.radians(50))
    out = run(route_with_junction(), ego, [obj_ahead(5.0)])
    assert out.path_blocked is False


def test_off_route_moving_car_is_left_to_the_band_rules():
    # a car crossing 5 m ahead at 6 m/s: moving, so the nose sweep never looks at it, and
    # 3 m off the route line the band rules ignore it -- as before
    ego = (10.0, OFF_ROUTE_Y, 0.0)
    out = run(straight_route(), ego, [obj_ahead(5.0, speed=6.0)])
    assert out.path_blocked is False
    assert out.closest_obstacle_distance == 999.0


# TEST 4 -- off route with a clear road ahead stays clear
def test_off_route_van_with_nothing_in_its_way_is_clear():
    ego = (10.0, OFF_ROUTE_Y, 0.0)
    assert run(straight_route(), ego, []).path_blocked is False
    # a parked car 5 m ahead but 3.5 m out to the side: the body slides past it
    out = run(straight_route(), ego, [obj_ahead(5.0, 3.5)])
    assert out.path_blocked is False
    assert out.closest_obstacle_distance == 999.0
    # a kerb-height strip level with our side, not ahead: never a nose hit
    out = run(straight_route(), ego, [obj_ahead(-0.5, 2.5, kind=ObjectType.OBSTACLE)])
    assert out.path_blocked is False


def test_off_route_sweep_reaches_the_danger_distance_only():
    # 8 m (danger_m) plus the body's half-length: a car 20 m ahead is for the corridor rules,
    # not the nose sweep (the van drives at recovery speed off route -- see main.py)
    ego = (10.0, OFF_ROUTE_Y, 0.0)
    out = run(straight_route(), ego, [obj_ahead(20.0)])
    assert out.path_blocked is False
    out = run(straight_route(), ego, [obj_ahead(10.0)])
    assert out.path_blocked is True                 # 10 m ahead: nose reaches it
