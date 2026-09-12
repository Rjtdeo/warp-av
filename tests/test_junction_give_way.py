"""Give way at a junction to what is actually coming.

Until 2026-09-11 the rule was "moving, nearby, and not in our own lane" -- a radius, not a
right of way. A car LEAVING the junction down the crossing road held the van at the line
exactly as long as one arriving, and after 12 s of that the van crept out anyway.

Now each vehicle's own velocity decides: does its motion bring it onto our line of travel,
soon, and near us when it gets there?
"""
import math

import pytest

from warp_av.world_model import WorldModel as WM, WorldObject, CROSSING_WITHIN_S, CROSSING_NEAR_M


def car(x, y, vx=0.0, vy=0.0, speed=None, oid=1):
    return WorldObject(id=oid, kind="vehicle", x=x, y=y, distance_m=math.hypot(x, y),
                       world_x=x, world_y=y, vx_world=vx, vy_world=vy,
                       speed_mps=math.hypot(vx, vy) if speed is None else speed,
                       stationary=False)


def world(*objects):
    return WM(objects=list(objects))


#: the van sits at the origin facing +x (east), so "across" is +y (its right)
EAST = 0.0


def test_a_car_coming_across_from_the_right_is_given_way_to():
    coming = car(x=8.0, y=14.0, vx=0.0, vy=-6.0)          # driving toward our line
    got = world(coming).crossing_vehicles(25.0, ego_yaw_rad=EAST)
    assert [o.id for o in got] == [1]


def test_a_car_driving_AWAY_down_the_crossing_road_is_not():
    """The live fault: it counted, and the van waited for a car that was leaving."""
    leaving = car(x=8.0, y=14.0, vx=0.0, vy=6.0)          # driving away from our line
    assert world(leaving).crossing_vehicles(25.0, ego_yaw_rad=EAST) == []


def test_a_car_that_will_cross_long_after_we_are_gone_is_not_a_conflict():
    far_off = car(x=8.0, y=CROSSING_WITHIN_S * 4.0 + 20.0, vx=0.0, vy=-4.0)
    assert world(far_off).crossing_vehicles(60.0, ego_yaw_rad=EAST) == []


def test_nor_one_that_crosses_our_line_far_behind_us():
    behind = car(x=-2.5, y=14.0, vx=-9.0, vy=-6.0)        # reaches our line well behind
    assert world(behind).crossing_vehicles(30.0, ego_yaw_rad=EAST) == []


def test_a_car_edging_onto_our_line_counts():
    edging = car(x=10.0, y=2.6, vx=-1.0, vy=-2.0)         # coming at us and drifting across
    got = world(edging).crossing_vehicles(25.0, ego_yaw_rad=EAST)
    assert [o.id for o in got] == [1]


def test_a_car_in_our_own_lane_is_the_corridor_check_business_not_give_way():
    head_on = car(x=10.0, y=0.5, vx=-4.0, vy=0.0)
    assert world(head_on).crossing_vehicles(25.0, ego_yaw_rad=EAST) == []


def test_parked_cars_and_far_cars_are_still_ignored():
    parked = car(x=8.0, y=14.0, vx=0.0, vy=0.0, speed=0.0)
    far = car(x=60.0, y=40.0, vx=0.0, vy=-6.0, oid=2)
    assert world(parked, far).crossing_vehicles(25.0, ego_yaw_rad=EAST) == []


def test_without_a_heading_the_old_radius_answer_stands():
    """Callers with no pose (the bare-perception path) get what they always got."""
    leaving = car(x=8.0, y=14.0, vx=0.0, vy=6.0)
    assert len(world(leaving).crossing_vehicles(25.0)) == 1


def test_the_behaviour_hands_the_heading_over():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "behavior" / "behavior.py").read_text()
    assert "ego_yaw_rad=getattr(now.pose" in src


# ---- never enter a junction the van cannot clear -----------------------------------------

from warp_av.behavior.behavior import (BehaviorSystem, DrivingBehavior,  # noqa: E402
                                       JUNCTION_ENTER_WITHIN_M)
from warp_av.behavior import transitions as T  # noqa: E402
from warp_av.localization.localization import Pose  # noqa: E402
from warp_av.perception.perception import PerceptionOutput, ObjectType  # noqa: E402
from warp_av.planning.planner import RoutePlanner, Route, Waypoint  # noqa: E402


def van_at_a_junction(**kw):
    b = BehaviorSystem()
    b.set_mission()
    seen = PerceptionOutput(closest_obstacle_distance=kw.pop("blocker_m", 999.0),
                            closest_obstacle_type=ObjectType.VEHICLE)
    return b.update(perception=seen, pose=Pose(healthy=True), destination_distance=200.0,
                    safety_ok=True, **kw)


def test_it_waits_on_this_side_when_the_way_out_is_blocked():
    out = van_at_a_junction(junction_span=(4.0, 18.0), blocker_m=22.0)
    assert out.why == T.JUNCTION_KEEP_CLEAR and out.should_stop
    assert "way out is blocked" in out.reason


def test_it_goes_when_there_is_room_beyond():
    out = van_at_a_junction(junction_span=(4.0, 18.0), blocker_m=40.0)
    assert out.why != T.JUNCTION_KEEP_CLEAR


def test_something_on_THIS_side_of_the_junction_is_not_a_box_problem():
    out = van_at_a_junction(junction_span=(10.0, 24.0), blocker_m=6.0)
    assert out.why != T.JUNCTION_KEEP_CLEAR


def test_it_only_asks_when_it_is_about_to_enter():
    out = van_at_a_junction(junction_span=(JUNCTION_ENTER_WITHIN_M + 5.0, 30.0), blocker_m=33.0)
    assert out.why != T.JUNCTION_KEEP_CLEAR


def test_where_the_junction_begins_and_ends_comes_from_the_route():
    p = RoutePlanner.__new__(RoutePlanner)
    wps = []
    for i in range(40):
        x = i * 2.0
        wps.append(Waypoint(x=x, y=0.0, yaw=0.0, is_junction=20.0 <= x <= 32.0))
    span = p.junction_span(Route(waypoints=wps), 0.0, 0.0)
    assert span is not None
    entry, exit_m = span
    assert abs(entry - 20.0) < 2.1 and abs(exit_m - 34.0) < 2.1
    assert p.junction_span(Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(40)]),
                           0.0, 0.0) is None
