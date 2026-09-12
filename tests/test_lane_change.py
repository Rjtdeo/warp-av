"""Changing lane properly.

CARLA's route planner changes lane by putting the next waypoint in the next lane: a 3.5 m
sideways step between two points 2 m apart. The van chased that with the steering, which is a
swerve, and nothing ever looked at the lane it was swerving into. The geometry is fixed here;
looking is `lane_change_blocker`, which is the same question the go-around asks.
"""
import math

from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.planner import RoutePlanner, Route, Waypoint


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def stepped_route(change_at=40.0, over=3.5, n=60):
    """Straight along +x, stepping `over` metres to the right at `change_at`."""
    return Route(waypoints=[Waypoint(x=i * 2.0, y=(over if i * 2.0 >= change_at else 0.0), yaw=0.0)
                            for i in range(n)])


def lateral_steps(route):
    out = []
    for a, b in zip(route.waypoints, route.waypoints[1:]):
        c, s = math.cos(a.yaw), math.sin(a.yaw)
        out.append(abs(-(b.x - a.x) * s + (b.y - a.y) * c))
    return out


def test_the_route_from_the_map_really_does_step_sideways():
    assert max(lateral_steps(stepped_route())) > 3.0


def test_after_smoothing_there_is_no_step_left():
    """A smoothstep over 18 m moves sideways fastest in the middle, at about 1.5 x 3.5 / 18
    metres per metre -- under 0.8 m between two waypoints 2 m apart, against 3.5 m before."""
    p, r = planner(), stepped_route()
    assert p.smooth_lane_changes(r) == 1
    assert max(lateral_steps(r)) < 0.8, "a lane change is a move, not a jump"


def test_it_arrives_in_the_new_lane_where_the_map_says_and_not_before():
    p, r = planner(), stepped_route(change_at=40.0)
    p.smooth_lane_changes(r)
    at = {round(w.x): w.y for w in r.waypoints}
    assert abs(at[40] - 3.5) < 0.05, "in the new lane at the change point"
    assert abs(at[20] - 3.5) > 1.0, "still moving over 20 m before it"
    assert abs(at[18]) < 1.2, "and not yet started a whole change-length before it"
    assert abs(at[50] - 3.5) < 0.05, "and it stays there"


def test_it_spreads_the_move_over_the_length_we_asked_for():
    p, r = planner(), stepped_route(change_at=40.0)
    p.smooth_lane_changes(r, over_m=10.0)
    at = {round(w.x): w.y for w in r.waypoints}
    assert abs(at[28]) < 0.3, "nothing happens more than 10 m before the change"
    assert at[34] > 0.5


def test_two_changes_in_one_route_are_both_spread():
    wps = []
    for i in range(80):
        x = i * 2.0
        y = 0.0 if x < 40 else (3.5 if x < 100 else 7.0)
        wps.append(Waypoint(x=x, y=y, yaw=0.0))
    r = Route(waypoints=wps)
    assert planner().smooth_lane_changes(r) == 2
    assert max(lateral_steps(r)) < 0.8


def test_a_straight_route_is_left_alone():
    r = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(40)])
    before = [(w.x, w.y) for w in r.waypoints]
    assert planner().smooth_lane_changes(r) == 0
    assert [(w.x, w.y) for w in r.waypoints] == before


def test_every_route_the_van_plans_gets_this():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "planning" / "planner.py").read_text()
    i = src.index("def plan_route(")
    assert "self.smooth_lane_changes(planned)" in src[i:i + 3000]


# ---- looking into the lane before moving into it -----------------------------------------

from warp_av.planning.planner import (lane_change_blocker, LANE_CHANGE_AHEAD_M,  # noqa: E402
                                      LANE_CHANGE_BEHIND_M, LANE_CHANGE_STANDING_M)
from warp_av.behavior import transitions as T  # noqa: E402

LEFT, RIGHT = -1, 1


def thing(x, y, speed=0.0, vx=0.0, kind=ObjectType.VEHICLE, h=1.5, w=1.8, l=4.5):
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed,
                          stationary=speed <= 0.3, vx_world=vx, vy_world=0.0,
                          height_m=h, width_m=w, length_m=l,
                          box_length_m=l, box_width_m=w, box_yaw_deg=0.0)


def test_an_empty_lane_is_a_lane_it_may_move_into():
    assert lane_change_blocker([], LEFT, 0.0) is None
    assert lane_change_blocker([thing(10.0, 0.0)], LEFT, 0.0) is None       # our own lane


def test_a_car_standing_CLOSE_in_that_lane_stops_the_move():
    assert "standing" in lane_change_blocker([thing(4.0, -3.5)], LEFT, 0.0)
    assert "standing" in lane_change_blocker([thing(4.0, 3.5)], RIGHT, 0.0)


def test_but_a_parked_car_further_up_that_lane_does_not():
    """Live 2026-09-11: waiting for parked cars to move ran a mission out of time 166 m short.
    Changing lane behind one is fine -- what is in front is the corridor check's business."""
    assert lane_change_blocker([thing(LANE_CHANGE_STANDING_M + 1.0, -3.5)], LEFT, 0.0) is None


def test_a_car_moving_in_that_lane_ahead_stops_it_too():
    moving = thing(20.0, -3.5, speed=6.0, vx=6.0)
    assert "moving" in lane_change_blocker([moving], LEFT, 0.0)


def test_and_one_coming_up_it_from_behind():
    behind = thing(-18.0, -3.5, speed=7.0, vx=7.0)
    assert "coming up" in lane_change_blocker([behind], LEFT, 0.0)


def test_but_not_one_dropping_back_behind_us():
    dropping = thing(-18.0, -3.5, speed=7.0, vx=-7.0)
    assert lane_change_blocker([dropping], LEFT, 0.0) is None


def test_the_far_lane_is_not_the_next_lane():
    two_over = thing(12.0, -7.5)
    assert lane_change_blocker([two_over], LEFT, 0.0) is None


def test_the_side_is_the_side_the_route_goes():
    car_on_the_right = thing(4.0, 3.5)
    assert lane_change_blocker([car_on_the_right], LEFT, 0.0) is None
    assert lane_change_blocker([car_on_the_right], RIGHT, 0.0) is not None


def test_where_the_next_lane_change_is_and_which_way():
    p, r = planner(), stepped_route(change_at=40.0)
    p.smooth_lane_changes(r)
    got = p.next_lane_change(r, 0.0, 0.0, 0.0)
    assert got is not None
    start_m, side = got
    assert side == RIGHT and 15.0 < start_m < 26.0, "it starts moving over before the change"
    straight = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(40)])
    assert p.next_lane_change(straight, 0.0, 0.0, 0.0) is None


def test_the_van_holds_its_lane_and_says_so():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("def _wait_for_a_gap")
    body = src[i:i + 3400]
    assert "lane_change_blocker(perception.objects, side, pose.yaw)" in body
    assert "LANE_CHANGE_GIVE_UP_S" in body, "it never waits for ever for something standing"
    assert "_gap_given_up_until" in body, "and having given up, it stays given up"
    assert "behavior_output.why = LANE_CHANGE_WAIT" in body
    assert "LANE_CHANGE_WAITING" in body and "LANE_CHANGE_GO" in body
    assert T.LANE_CHANGE_WAIT in T.ALL_WHY
    assert T.LANE_CHANGE_WAITING in T.ALL_MOVES and T.LANE_CHANGE_GO in T.ALL_MOVES
