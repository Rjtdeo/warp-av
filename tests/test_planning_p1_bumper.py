"""The van stops for what is in front of it, measured bumper to bumper.

The fault, found by phase 0's own instrumentation on its first live run: eleven ticks in a
row reported "OBSTACLE in path at 0.0m", blocked by object 3845, with nothing in front of
the van. Before phase 0 that was a bare `blocked = true` and there was nothing to chase.

Why it happened. Distance along the route was measured from the point the van STEERS ABOUT,
which sits 2.96 m behind its nose, and the corridor kept everything from a metre behind that
point. So an object level with the driver's door counted as an obstacle "ahead", one beside
the rear wheels was reported at 0.0 m -- max(0.0, along) clamped it -- and both hard-stopped
the van. Driving forward cannot reach either of them.

The object's own body counts as well, because "beside the van" depends on how long the thing
is: a lorry whose middle is level with our door has its nose well past our bumper.
"""
import math

import pytest

from warp_av.planning.planner import (RoutePlanner, Route, Waypoint,
                                      VAN_HALF_LENGTH_M, UNMEASURED_REACH_M, reach_toward_us_m)
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def road():
    """Route running 20 m BEHIND the van as well as ahead.

    A route that starts exactly at the van makes every object behind it project to arc zero,
    which hides the bug being tested here. Worth stating: the first attempt at this test did
    exactly that and reported the van blocking on things it was in fact ignoring.
    """
    return Route(waypoints=[Waypoint(x=-20.0 + i * 2.0, y=0.0) for i in range(61)])


def thing(x, y=0.3, kind=ObjectType.OBSTACLE, size=None, speed=0.0, ident=1):
    o = DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y))
    o.speed = speed
    o.id = ident
    if size:
        o.length_m, o.width_m, o.height_m = size
    return o


def blocks(objs, ego=(0.0, 0.0, 0.0), footprint=None):
    p = planner()
    per = PerceptionOutput(objects=list(objs))
    p.filter_to_route_corridor(per, road(), ego[0], ego[1], ego[2], footprint=footprint)
    return per.path_blocked, p.last_decision


CONE = (0.4, 0.4, 0.7)
CAR = (4.5, 1.8, 1.5)
LORRY = (9.0, 2.4, 3.0)


# ---------------------------------------------------------------- the fault itself


@pytest.mark.parametrize("x", [-1.2, -0.9, -0.5, 0.0, 1.0, 2.0])
def test_nothing_beside_the_van_stops_it(x):
    """A cone level with the wheels or the door cannot be hit by driving forward."""
    blocked, _ = blocks([thing(x, size=CONE)])
    assert blocked is False, f"stopped for a cone {x:+.1f} m from its own centre"


def test_no_blocker_is_ever_reported_at_zero_metres():
    """0.0 m means touching the bumper. It was max(0.0, along) clamping a negative."""
    for x in (-0.9, -0.5, 0.0, 0.5):
        _, decision = blocks([thing(x, size=CONE)])
        assert decision.blocker_distance_m is None or decision.blocker_distance_m > 0.05, \
            f"reported a blocker at {decision.blocker_distance_m} m"


# ---------------------------------------------------------------- what must still stop it


@pytest.mark.parametrize("x", [3.2, 4.0, 6.0, 7.5])
def test_it_still_stops_for_what_is_in_front(x):
    blocked, d = blocks([thing(x, size=CONE)])
    assert blocked is True, f"drove at a cone {x} m ahead"
    assert d.blocker_distance_m == pytest.approx(x, abs=0.2)


def test_a_lorry_beside_our_door_still_stops_us():
    """Its MIDDLE is level with the driver's door; its nose is well past our bumper. This is
    the case that makes the object's own size part of the question rather than a detail."""
    blocked, _ = blocks([thing(1.0, size=LORRY)])
    assert blocked is True, "drove into a nine-metre lorry because its centre was beside us"


def test_a_car_beside_our_door_still_stops_us():
    blocked, _ = blocks([thing(1.0, size=CAR)])
    assert blocked is True


def test_something_never_measured_still_stops_us():
    """Not knowing how big a thing is has never been a reason to drive at it."""
    blocked, _ = blocks([thing(1.0, size=None)])
    assert blocked is True
    assert reach_toward_us_m(thing(1.0, size=None)) == UNMEASURED_REACH_M


def test_but_an_unmeasured_thing_well_behind_us_does_not():
    """The conservative default must not bring the original fault back by another door."""
    blocked, _ = blocks([thing(-8.0, size=None)])
    assert blocked is False


# ---------------------------------------------------------------- the geometry helper


def test_reach_uses_the_measured_body_not_the_politeness_radius():
    """obstacle_radius_m answers 'how much room does this deserve' and floors an unnamed lump
    at 0.4 m. Using it here read a nine-metre lorry as half a metre long."""
    assert reach_toward_us_m(thing(0.0, size=LORRY)) > 4.0
    assert reach_toward_us_m(thing(0.0, size=CONE)) < 0.5


def test_the_bumper_is_where_the_body_says_it_is():
    assert VAN_HALF_LENGTH_M == pytest.approx(2.96, abs=0.01)


def test_the_swept_body_check_is_deliberately_not_gated():
    """The centre-line rules ask 'is something in my path'. The swept body asks 'would my
    body touch it as I move', which is a different question: a thing overlapping our side
    IS scraped as we drive past it. So the bumper gate applies to the first and not the
    second, and sweep_conflict keeps its own rule of ignoring what is behind the centre."""
    from warp_av.planning.footprint import sweep_conflict, VehicleFootprint
    foot = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)
    pts = [(-20.0 + i * 2.0, 0.0) for i in range(61)]
    behind = sweep_conflict(pts, (0.0, 0.0), foot, (-6.0, 0.5), obstacle_radius=0.3)
    assert behind is None, "the sweep looked backwards"
    alongside = sweep_conflict(pts, (0.0, 0.0), foot, (1.0, 1.2), obstacle_radius=0.3)
    assert alongside is not None, "the sweep ignored something it would scrape along"
