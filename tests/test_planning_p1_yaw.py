"""Obstacles as the rectangles perception measured, not circles round their centres.

The fault, measured live on 2026-09-10: the van stopped for kerb 2.5 m out to its side on 19
of 23 blocked ticks in one drive. Every object reached the swept-path check as a CIRCLE, and
a circle has to be big enough to cover the object's far corner in every direction -- so a
kerb strip measured 3.5 x 0.1 m became a disc of radius 1.75 m, reaching 1.75 m across the
road when the strip itself reaches 0.05 m.

Perception measures each object's heading. Planning threw it away. These tests pin what
using it changes -- and, just as important, the cases where orientation must make the van
MORE careful, not less. A car turned across the lane is further from the line at its
centre than one parked along it, and must still stop the van.
"""
import math

import pytest

from warp_av.planning.footprint import (VehicleFootprint, ObstacleBox, sweep_conflict,
                                        _boxes_touch)
from warp_av.planning.planner import (RoutePlanner, Route, Waypoint, obstacle_box_for,
                                      YAW_TOLERANCE_DEG, OBSTACLE_BOX_PAD_M)
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject

FOOT = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)


# ---------------------------------------------------------------- the rectangle test itself


def test_two_overlapping_rectangles_touch():
    assert _boxes_touch(0, 0, 0.0, 2.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0)


def test_rectangles_side_by_side_with_a_gap_do_not():
    assert not _boxes_touch(0, 0, 0.0, 2.0, 1.0, 0.0, 2.5, 0.0, 2.0, 1.0)


def test_a_turned_rectangle_that_only_reaches_across_is_caught():
    """A long thin thing turned across the gap reaches where its centre does not."""
    assert _boxes_touch(0, 0, 0.0, 2.0, 1.0, 0.0, 2.6, math.pi / 2, 2.0, 0.1)
    assert not _boxes_touch(0, 0, 0.0, 2.0, 1.0, 0.0, 2.6, 0.0, 2.0, 0.1)


def test_corner_to_corner_near_miss_is_a_miss():
    """The case a circle gets most wrong: two rectangles meeting at corners with a gap."""
    assert not _boxes_touch(0, 0, 0.0, 1.0, 1.0, 2.3, 2.3, 0.0, 1.0, 1.0)


def test_a_rectangle_is_the_same_turned_half_a_circle():
    for h in (0.0, 0.7, 1.9):
        assert _boxes_touch(0, 0, 0.0, 2, 1, 1, 1.5, h, 1, 0.2) == \
               _boxes_touch(0, 0, 0.0, 2, 1, 1, 1.5, h + math.pi, 1, 0.2)


# ---------------------------------------------------------------- the planner, on the road


def road():
    return Route(waypoints=[Waypoint(x=-20.0 + i * 2.0, y=0.0) for i in range(61)])


def thing(x, y, size, yaw_deg=0.0, kind=ObjectType.OBSTACLE):
    o = DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y))
    o.speed = 0.0
    o.id = 1
    o.length_m, o.width_m, o.height_m = size
    o.yaw_deg = yaw_deg
    return o


def blocks(o, footprint=FOOT):
    p = RoutePlanner.__new__(RoutePlanner)
    per = PerceptionOutput(objects=[o])
    p.filter_to_route_corridor(per, road(), 0.0, 0.0, 0.0, footprint=footprint)
    return per.path_blocked


def test_a_kerb_strip_along_the_road_no_longer_stops_the_van():
    """The live fault. 3.5 x 0.1 m, lying along the road, 2.5 m to the side: 1.2 m clear of
    the van's body. As a circle it reached 1.75 m across and stopped the van."""
    assert blocks(thing(6.0, 2.5, (3.5, 0.1, 0.8), yaw_deg=0.0)) is False


def test_the_same_strip_turned_across_the_lane_does():
    """Orientation cuts both ways. A barrier across the lane reaches into our path from the
    same centre position, and must stop the van."""
    assert blocks(thing(6.0, 2.2, (3.5, 0.1, 0.8), yaw_deg=90.0)) is True


def test_a_parked_car_half_in_our_margin_still_stops_us():
    """Run 62, the reason the swept body exists: a parked car 1.6 m off the line."""
    assert blocks(thing(6.0, 1.6, (4.5, 1.8, 1.5), yaw_deg=0.0, kind=ObjectType.VEHICLE)) is True


def test_a_car_seen_end_on_is_not_mistaken_for_a_thin_one():
    """The laser sees one face: a car end-on can measure 1.8 x 0.5 m. The vehicle width
    floor keeps it a car. Without it, a car beside the lane would read as a post."""
    box = obstacle_box_for(thing(0, 0, (1.8, 0.5, 1.5), kind=ObjectType.VEHICLE), 0.0)
    assert box.half_width >= 0.9


def test_a_lead_car_in_the_lane_still_stops_us():
    assert blocks(thing(6.0, 0.3, (4.5, 1.8, 1.5), yaw_deg=0.0, kind=ObjectType.VEHICLE)) is True


def test_a_car_turned_across_the_road_stops_us():
    """Its centre is further from the line than a parked one's would be; its nose is in our
    lane. This is what the circle happened to get right and the box must not get wrong."""
    assert blocks(thing(6.0, 2.4, (4.5, 1.8, 1.5), yaw_deg=90.0, kind=ObjectType.VEHICLE)) is True


def test_an_unmeasured_thing_falls_back_to_the_cautious_circle():
    """Not knowing a thing's shape is not a reason to assume it is thin."""
    o = DetectedObject(object_type=ObjectType.OBSTACLE, x=6.0, y=1.0, distance=6.1)
    o.speed = 0.0
    o.id = 1
    assert obstacle_box_for(o, 0.0) is None
    assert blocks(o) is True


def test_the_box_is_never_smaller_than_what_was_measured():
    """The laser sees only the near face, so a measurement is a lower bound on the thing."""
    box = obstacle_box_for(thing(0, 0, (3.5, 0.1, 0.8)), 0.0)
    assert box.half_length >= 1.75
    assert box.half_width >= 0.05 + OBSTACLE_BOX_PAD_M


def test_heading_error_is_allowed_for():
    """The yaw comes from the sighting whose length was nearest the median, in the van's frame
    AT THAT SIGHTING. The box is widened by what a YAW_TOLERANCE_DEG error would swing its
    ends through, so a slightly stale heading cannot make a long thing look thinner."""
    short = obstacle_box_for(thing(0, 0, (0.4, 0.1, 0.8)), 0.0)
    long_ = obstacle_box_for(thing(0, 0, (3.5, 0.1, 0.8)), 0.0)
    assert long_.half_width > short.half_width, "a long thing's heading error must cost it width"
    assert long_.half_width < 1.0, "and still nothing like the 1.75 m the circle claimed"


def test_heading_is_turned_into_the_world_frame():
    """Measured in the van's frame; the sweep runs in the world's."""
    box = obstacle_box_for(thing(0, 0, (3.5, 0.1, 0.8), yaw_deg=30.0), math.radians(45.0))
    assert box.heading == pytest.approx(math.radians(75.0))
