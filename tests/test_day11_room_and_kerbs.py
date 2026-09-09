"""Perception V2 day 11: the van uses two things it already measured.

  * how much room each object needs (perception measured it; the planner ignored it)
  * where the kerb is (the van fits a line, then reported chips of that line as obstacles)
"""
import math
import types

import pytest

from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput
from warp_av.planning.planner import (RoutePlanner, Route, Waypoint, block_band_m,
                                      obstacle_radius_m, VAN_HALF_WIDTH_M,
                                      DEFAULT_OBSTACLE_RADIUS_M, MAX_BLOCK_HALFWIDTH_M)
from warp_av.perception.camera_lidar_perception import (KERB_CRUMB_MAX_HEIGHT_M,
                                                        KERB_CRUMB_TOLERANCE_M)
from warp_av.perception.ground_filter import ROAD_EDGE_MIN_CENTRE_LATERAL_M
from warp_av.perception.road_edges import RoadEdge, RoadEdges


# ---------------------------------------------------------------- how much room


def obj(kind, x, y, clearance=None, speed=0.0):
    o = DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed)
    if clearance is not None:
        o.clearance_radius_m = clearance
    return o


def test_the_measured_room_is_the_one_used():
    """It used to look for `.radius`, which nothing sets, and always fell back."""
    assert obstacle_radius_m(obj(ObjectType.PEDESTRIAN, 5.0, 0.0, clearance=0.6)) == 0.6
    assert obstacle_radius_m(obj(ObjectType.CYCLIST, 5.0, 0.0, clearance=0.8)) == 0.8
    assert obstacle_radius_m(obj(ObjectType.VEHICLE, 5.0, 0.0, clearance=1.2)) == 1.2


def test_the_room_never_drops_below_what_the_kind_needs():
    """A bad measurement may widen the margin, never narrow it."""
    tiny = obj(ObjectType.PEDESTRIAN, 5.0, 0.0, clearance=0.05)
    assert obstacle_radius_m(tiny) == DEFAULT_OBSTACLE_RADIUS_M["pedestrian"]
    for bad in (None, float("nan"), 0.0, -1.0, "wide"):
        o = obj(ObjectType.VEHICLE, 5.0, 0.0)
        o.clearance_radius_m = bad
        assert obstacle_radius_m(o) == DEFAULT_OBSTACLE_RADIUS_M["vehicle"]


def test_a_person_gets_a_wider_band_than_a_bin():
    person = obj(ObjectType.PEDESTRIAN, 6.0, 1.5, clearance=0.6)
    bin_ = obj(ObjectType.OBSTACLE, 6.0, 1.5, clearance=0.4)
    assert block_band_m(person, 1.40) == pytest.approx(VAN_HALF_WIDTH_M + 0.6)
    assert block_band_m(person, 1.40) > 1.40
    assert block_band_m(bin_, 1.40) == 1.40          # unchanged


def test_a_vehicle_keeps_the_tuned_band():
    """Widening it freezes the mission on every cross-street waiter."""
    car = obj(ObjectType.VEHICLE, 6.0, 1.6, clearance=1.2)
    assert block_band_m(car, 1.40) == 1.40


def test_no_band_reaches_past_what_the_corridor_looks_at():
    huge = obj(ObjectType.CYCLIST, 6.0, 1.0, clearance=9.0)
    assert block_band_m(huge, 1.40) <= MAX_BLOCK_HALFWIDTH_M


def straight_route():
    return Route(waypoints=[Waypoint(x=float(i), y=0.0, z=0.0, yaw=0.0, speed=8.0)
                            for i in range(0, 120, 2)])


def run(objects, ego=(10.0, 0.0, 0.0)):
    p = PerceptionOutput(objects=objects, healthy=True)
    return RoutePlanner.filter_to_route_corridor(
        types.SimpleNamespace(), p, straight_route(), ego[0], ego[1], ego[2])


def test_a_walking_person_just_off_the_line_now_stops_the_van():
    """1.5 m off the line: inside a person's own room, outside the old fixed 1.40 m.

    They are MOVING on purpose. A standing object out there is already stopped for by the
    wide-body rule, so only a moving one isolates what changed today.
    """
    walker = obj(ObjectType.PEDESTRIAN, 6.0, 1.5, clearance=0.6, speed=1.4)
    walker.x, walker.y = 6.0, -1.5          # ego frame: y is to the LEFT here
    assert run([walker]).path_blocked is True


def test_a_moving_car_at_the_same_spot_still_does_not():
    """The cross-street waiter. Its band was tuned and today does not touch it."""
    car = obj(ObjectType.VEHICLE, 6.0, 1.5, clearance=1.2, speed=6.0)
    car.x, car.y = 6.0, -1.5
    assert run([car]).path_blocked is False


# ---------------------------------------------------------------- the kerb it already found


class FakePerception:
    """Just the two attributes _sits_on_the_kerb touches."""

    from warp_av.perception.camera_lidar_perception import CameraLidarPerception
    _sits_on_the_kerb = CameraLidarPerception._sits_on_the_kerb

    def __init__(self, edges):
        self.road_edges = edges


def kerb_line(offset=4.46, heading=0.8, points=189, length=27.7, share=0.91, side="right"):
    return RoadEdges(right=RoadEdge(side=side, offset_m=offset, heading_deg=heading,
                                    length_m=length, points=points, inlier_share=share))


def crumb(x=12.0, y=4.4, height=0.14):
    return {"x": x, "y": y, "height": height}


def test_a_chip_of_the_kerb_is_kerb():
    """The live case: 0.38 x 0.00 x 0.14 m at 4.4 m right, on a line fitted at 4.46 m."""
    p = FakePerception(kerb_line())
    assert p._sits_on_the_kerb(crumb()) is True


def test_the_line_is_followed_along_its_length():
    """A kerb that runs away from the van is still the kerb further ahead."""
    p = FakePerception(kerb_line(offset=4.46, heading=5.0))
    far = kerb_line(offset=4.46, heading=5.0).right.lateral_at(20.0)
    assert p._sits_on_the_kerb(crumb(x=20.0, y=far)) is True
    assert p._sits_on_the_kerb(crumb(x=20.0, y=4.46)) is False   # the OLD offset, not the line


def test_nothing_that_stands_up_is_ever_kerb():
    p = FakePerception(kerb_line())
    assert p._sits_on_the_kerb(crumb(height=KERB_CRUMB_MAX_HEIGHT_M)) is False
    assert p._sits_on_the_kerb(crumb(height=0.9)) is False
    assert p._sits_on_the_kerb({"x": 12.0, "y": 4.4, "height": None}) is False


def test_nothing_the_van_could_hit_is_ever_kerb():
    """Even with a kerb line right there, the corridor is untouchable."""
    p = FakePerception(kerb_line(offset=1.0))
    assert p._sits_on_the_kerb(crumb(y=1.0)) is False
    assert p._sits_on_the_kerb(crumb(y=ROAD_EDGE_MIN_CENTRE_LATERAL_M - 0.01)) is False


def test_a_thing_off_the_line_survives():
    """A real low obstacle beside the road, not ON the kerb, stays an object.

    Measured from where the line actually is at that distance, not from its offset at the
    van -- the first version of this test forgot the line runs at an angle and caught me.
    """
    edges = kerb_line()
    p = FakePerception(edges)
    on_line = edges.right.lateral_at(12.0)
    assert p._sits_on_the_kerb(crumb(x=12.0, y=on_line)) is True
    assert p._sits_on_the_kerb(crumb(x=12.0, y=on_line + KERB_CRUMB_TOLERANCE_M + 0.1)) is False
    assert p._sits_on_the_kerb(crumb(x=12.0, y=on_line - KERB_CRUMB_TOLERANCE_M - 0.1)) is False


def test_without_a_confident_line_nothing_is_dropped():
    assert FakePerception(RoadEdges())._sits_on_the_kerb(crumb()) is False
    assert FakePerception(None)._sits_on_the_kerb(crumb()) is False
    shaky = kerb_line(points=3, length=2.0, share=0.2)
    assert shaky.right.confident is False
    assert FakePerception(shaky)._sits_on_the_kerb(crumb()) is False


def test_the_side_matters():
    """A right-hand kerb line says nothing about something on the left."""
    p = FakePerception(kerb_line())
    assert p._sits_on_the_kerb(crumb(y=-4.4)) is False
