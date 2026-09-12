"""When may the van swing out past a dead vehicle?

Moving traffic is judged by where it is and where it is going (planner.overtake_blocker).
Things standing still are judged by the path itself: the van's body slid along the planned
way round (RoutePlanner.plan_overtake, then pull_in_blocker over it).

Live on 2026-09-11 three cars were parked in the van's lane, 35 m apart. The van stopped
behind the first and never went round it: every look, a "moving vehicle" 41-49 m ahead
vetoed the pass. It was the SECOND parked car, half hidden behind the first, whose visible
pieces slid about -- in the van's own lane, 35 m beyond the first car, far past where the van
pulls back in. And no look at all went behind the van, into the lane it would pull out into.
Then, the same day, an SUV parked in the oncoming lane beside the van held it for three
minutes: the average of the SUV's laser points sat inside the old "standing in the passing
lane" band, 1.4 m clear of the path the van would have taken.
"""
import math

from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.planner import RoutePlanner, Route, Waypoint, overtake_blocker, body_centre

LEAD = 12.0                      # the dead car's back, metres ahead of the van's centre
REJOIN = LEAD + 16.0 + 8.0       # main.py: back in lane, plus room


def thing(x, y, kind=ObjectType.VEHICLE, speed=0.0, stationary=True, vx=0.0, h=1.5, w=1.8, l=4.5,
          box_dx=0.0, box_dy=0.0):
    """x ahead, y to the RIGHT of the van (as perception reports); a fitted box of l x w."""
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed,
                          stationary=stationary, vx_world=vx, vy_world=0.0,
                          height_m=h, width_m=w, length_m=l,
                          box_length_m=l, box_width_m=w, box_yaw_deg=0.0, box_dx=box_dx, box_dy=box_dy)


def lead():
    return thing(LEAD + 2.25, 0.0)


# ---- moving traffic -------------------------------------------------------------------

def test_an_empty_road_lets_it_pass():
    assert overtake_blocker([lead()], LEAD, REJOIN, 0.0) is None


def test_a_parked_car_far_beyond_the_first_does_not_stop_the_pass():
    """The live case: the second car 35 m past the first, its pieces 'moving'."""
    second = thing(LEAD + 35.0, 0.2, speed=1.4, stationary=False)
    assert overtake_blocker([lead(), second], LEAD, REJOIN, 0.0) is None


def test_someone_moving_where_the_van_pulls_back_in_stops_it():
    walker = thing(LEAD + 12.0, 0.5, kind=ObjectType.PEDESTRIAN, speed=1.2, stationary=False,
                   h=1.7, w=0.5, l=0.5)
    assert "pull back in" in overtake_blocker([lead(), walker], LEAD, REJOIN, 0.0)


def test_traffic_in_the_passing_lane_ahead_stops_it():
    """Coming AT us, what matters is whether the pass can be finished before it arrives."""
    oncoming = thing(30.0, -3.5, speed=6.0, stationary=False, vx=-6.0)
    why = overtake_blocker([lead(), oncoming], LEAD, REJOIN, 0.0, pass_takes_s=9.0)
    assert "oncoming" in why and "meets us in 5 s" in why


def test_a_gap_in_the_oncoming_lane_big_enough_for_the_pass_is_taken():
    far_off = thing(150.0, -3.5, speed=6.0, stationary=False, vx=-6.0)     # 25 s away
    assert overtake_blocker([lead(), far_off], LEAD, REJOIN, 0.0, pass_takes_s=9.0) is None


def test_a_car_going_the_same_way_in_that_lane_is_judged_by_distance_not_time():
    same_way = thing(30.0, -3.5, speed=6.0, stationary=False, vx=6.0)
    assert "passing lane" in overtake_blocker([lead(), same_way], LEAD, REJOIN, 0.0,
                                              pass_takes_s=9.0)


def test_a_car_coming_up_behind_in_the_passing_lane_stops_it():
    behind = thing(-20.0, -3.5, speed=5.0, stationary=False, vx=5.0)
    assert "behind" in overtake_blocker([lead(), behind], LEAD, REJOIN, 0.0)


def test_one_behind_going_the_other_way_does_not():
    leaving = thing(-20.0, -3.5, speed=5.0, stationary=False, vx=-5.0)
    assert overtake_blocker([lead(), leaving], LEAD, REJOIN, 0.0) is None


def test_a_kerb_fragment_sliding_along_the_far_kerb_does_not():
    """Beyond the passing lane, at the kerb: its pieces 'move', but it is kerb."""
    sliver = thing(25.0, -7.2, kind=ObjectType.OBSTACLE, speed=3.0, stationary=False,
                   h=0.4, w=0.2, l=0.3)
    assert overtake_blocker([lead(), sliver], LEAD, REJOIN, 0.0) is None


def test_moving_things_on_the_right_never_matter():
    kerbside = thing(20.0, 3.5, speed=3.0, stationary=False)
    assert overtake_blocker([lead(), kerbside], LEAD, REJOIN, 0.0) is None


def test_a_body_is_where_its_box_is_not_where_its_near_side_is():
    """The live SUV: points averaged 6.2 m left, box centred 0.76 m further out."""
    suv = thing(-0.1, -6.2, w=1.87, l=4.27, box_dx=-0.22, box_dy=-0.76)
    x, y = body_centre(suv)
    assert abs(y + 6.96) < 1e-6 and abs(x + 0.32) < 1e-6
    moving_suv = thing(20.0, -6.2, speed=5.0, stationary=False, vx=-5.0, w=1.87, l=4.27, box_dy=-0.76)
    assert overtake_blocker([lead(), moving_suv], LEAD, REJOIN, 0.0) is None     # its own lane


# ---- things standing still: the path round ---------------------------------------------

class _Seen:
    def __init__(self, objects):
        self.objects = objects


def way_round():
    """A straight road, the van at x = 0 facing +x (CARLA: + y is right), the path round a
    dead car planned on it."""
    p = RoutePlanner.__new__(RoutePlanner)          # no CARLA: geometry only
    road = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
    trial = Route(waypoints=list(road.waypoints))
    assert p.plan_overtake(trial, 0.0, 0.0, LEAD) is not None
    return p, trial


def standing_in_the_way(objects):
    p, trial = way_round()
    van = VehicleFootprint(half_length=2.95, half_width=0.99, safety_margin=0.30)
    return p.pull_in_blocker(_Seen(objects), trial, 0.0, 0.0, 0.0, van, horizon_m=REJOIN)


def test_the_way_round_swings_one_lane_left_and_back():
    _, trial = way_round()
    ys = {round(w.x): w.y for w in trial.waypoints}
    assert abs(ys[20] + 3.6) < 0.05 and abs(ys[40]) < 0.05


def test_the_way_round_clears_the_dead_car_itself():
    assert standing_in_the_way([lead()]) is None


def test_the_suv_parked_in_the_oncoming_lane_beside_the_van_does_not_stop_it():
    """The live case, as perception measured it."""
    suv = thing(-0.1, -6.2, w=1.87, l=4.27, h=2.1, box_dx=-0.22, box_dy=-0.76)
    assert standing_in_the_way([lead(), suv]) is None
    ahead = thing(20.0, -6.2, w=1.87, l=4.27, h=2.1, box_dy=-0.76)       # nor one further on
    assert standing_in_the_way([lead(), ahead]) is None


def test_a_car_standing_in_the_passing_lane_beside_the_lead_stops_it():
    parked_left = thing(LEAD + 4.0, -3.5)
    got = standing_in_the_way([lead(), parked_left])
    assert got is not None and got[0] is parked_left


def test_so_does_a_box_in_the_passing_lane():
    box = thing(20.0, -3.2, kind=ObjectType.OBSTACLE, h=0.6, w=0.6, l=0.6)
    got = standing_in_the_way([lead(), box])
    assert got is not None and got[0] is box


def test_so_does_a_car_where_the_van_pulls_back_in():
    close = thing(LEAD + 17.0, 0.0)
    got = standing_in_the_way([lead(), close])
    assert got is not None and got[0] is close


def test_but_not_one_well_past_where_it_is_back_in_lane():
    far = thing(LEAD + 35.0, 0.0)
    assert standing_in_the_way([lead(), far]) is None


def test_things_beside_the_lane_on_the_right_do_not():
    """The course's bin on the shoulder, a cone past the lane edge, a box poking 0.15 m in."""
    bin_ = thing(16.0, 2.8, kind=ObjectType.OBSTACLE, h=1.1, w=0.6, l=0.6)
    cone = thing(24.0, 2.3, kind=ObjectType.OBSTACLE, h=0.7, w=0.4, l=0.4)
    box = thing(20.0, 1.9, kind=ObjectType.OBSTACLE, h=0.5, w=0.5, l=0.5)
    assert standing_in_the_way([lead(), bin_, cone, box]) is None


# ---- what may be gone round at all (2026-09-11: not only cars) ---------------------------

from warp_av.planning.planner import pass_refused   # noqa: E402


def test_a_barrel_in_the_lane_may_be_passed():
    """Live that day: a 0.45 m barrel in the lane stopped the van 8.7 m short and it was still
    standing there when the test ended."""
    assert pass_refused("obstacle", 0.0, camera_degraded=False) is None
    assert pass_refused("unknown", 0.0, camera_degraded=False) is None
    assert pass_refused("vehicle", 0.0, camera_degraded=False) is None


def test_a_person_is_never_driven_round():
    assert "waited for" in pass_refused("pedestrian", 0.0, camera_degraded=False)
    assert "waited for" in pass_refused("cyclist", 0.0, camera_degraded=False)
    assert "waited for" in pass_refused("pedestrian", 0.0, camera_degraded=True)


def test_something_moving_is_followed_not_passed():
    assert "moving" in pass_refused("vehicle", 1.5, camera_degraded=False)
    assert "moving" in pass_refused("obstacle", 0.9, camera_degraded=False)
    assert pass_refused("vehicle", 0.2, camera_degraded=False) is None      # noise, not motion


def test_an_unnamed_thing_is_only_passed_while_the_camera_works():
    """Dead ahead in the camera's view, a thing it has not called a person is a thing it
    looked at and did not call a person. With the camera stale, that is ignorance."""
    assert "not naming things" in pass_refused("obstacle", 0.0, camera_degraded=True)
    assert "not naming things" in pass_refused("unknown", 0.0, camera_degraded=True)
    assert pass_refused("vehicle", 0.0, camera_degraded=True) is None       # a car is a car


def test_the_van_asks_this_before_it_goes_round_anything():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert "why_not = pass_refused(" in src
    i = src.index("OVERTAKE_STATES = ")
    assert "STOPPED_OBSTACLE" in src[i:i + 200] and "STOPPED_BLOCKED" in src[i:i + 200]


# ---- the smallest way round: a nudge inside our own lane (2026-09-11) --------------------

from warp_av.planning.planner import (pass_options, LANE_EDGE_KEEP_M,  # noqa: E402
                                      SHOULDER_SHIFT_M)


def test_the_nudge_is_tried_before_the_whole_lane_and_the_shoulder_last():
    ways = pass_options(3.5, 0.99, 3.6)
    assert [w[0] for w in ways] == [0.71, -0.71, 3.6, -SHOULDER_SHIFT_M]
    assert [w[1] for w in ways] == [True, True, False, False]
    assert [w[2] for w in ways] == [False, False, False, True], "only the last may use it"


def test_a_nudge_never_puts_the_body_over_the_line():
    for lane in (2.6, 3.0, 3.5, 4.2):
        over = pass_options(lane, 0.99, 3.6)[0][0]
        assert over + 0.99 <= lane / 2.0 - LANE_EDGE_KEEP_M + 1e-9


def test_a_lane_too_narrow_to_move_in_offers_the_lane_and_the_shoulder():
    assert pass_options(2.0, 0.99, 3.6) == [(3.6, False, False), (-SHOULDER_SHIFT_M, False, True)]


def test_the_path_of_a_nudge_stays_where_it_should():
    p, _ = way_round()
    road = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
    trial = Route(waypoints=list(road.waypoints))
    assert p.plan_overtake(trial, 0.0, 0.0, LEAD, shift_m=0.7) is not None
    ys = {round(w.x): w.y for w in trial.waypoints}
    assert abs(ys[20] + 0.7) < 0.02 and abs(ys[40]) < 0.02          # over, then back


def test_it_can_move_the_other_way_too():
    p, _ = way_round()
    road = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
    trial = Route(waypoints=list(road.waypoints))
    assert p.plan_overtake(trial, 0.0, 0.0, LEAD, shift_m=-0.7) is not None
    assert abs({round(w.x): w.y for w in trial.waypoints}[20] - 0.7) < 0.02


def test_a_nudge_gets_past_a_box_poking_into_the_lane_and_a_lane_change_is_not_needed():
    """Live 2026-09-11: the van stood for 116 s in front of a box reaching 0.15 m into its
    lane, with 0.6 m of road beside it."""
    p, _ = way_round()
    box = thing(LEAD + 2.0, 1.5, kind=ObjectType.OBSTACLE, h=0.7, w=0.66, l=0.65)
    van = VehicleFootprint(half_length=2.95, half_width=0.99, safety_margin=0.30)
    straight = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
    assert p.pull_in_blocker(_Seen([box]), straight, 0.0, 0.0, 0.0, van,
                             horizon_m=REJOIN) is not None, "it is in the way to begin with"
    for over, in_lane, _shoulder in pass_options(3.5, 0.99, 3.6):
        trial = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
        assert p.plan_overtake(trial, 0.0, 0.0, LEAD, shift_m=over) is not None
        if p.pull_in_blocker(_Seen([box]), trial, 0.0, 0.0, 0.0, van,
                             horizon_m=REJOIN) is None:
            assert in_lane and over > 0, "it should slide left inside its own lane"
            return
    raise AssertionError("no way round was found at all")


def test_but_something_in_the_middle_of_the_lane_still_needs_a_lane_to_borrow():
    p, _ = way_round()
    barrel = thing(LEAD + 2.0, 0.0, kind=ObjectType.OBSTACLE, h=0.8, w=0.48, l=0.45)
    van = VehicleFootprint(half_length=2.95, half_width=0.99, safety_margin=0.30)
    taken = None
    for over, in_lane, _shoulder in pass_options(3.5, 0.99, 3.6):
        trial = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(60)])
        if p.plan_overtake(trial, 0.0, 0.0, LEAD, shift_m=over) is None:
            continue
        if p.pull_in_blocker(_Seen([barrel]), trial, 0.0, 0.0, 0.0, van, horizon_m=REJOIN) is None:
            taken = (over, in_lane)
            break
    assert taken == (3.6, False)


def test_the_van_tries_them_in_that_order_and_creeps_while_it_squeezes():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert "for over_m, in_lane, on_shoulder in pass_options(" in src
    assert "SQUEEZE_ABORT_M if (in_lane or on_shoulder) else PASS_ABORT_M" in src
    assert "SQUEEZE_SPEED_MPS if (in_lane or on_shoulder)" in src
    assert "self._shoulder_ok if on_shoulder else self._lane_ok" in src, \
        "only the shoulder option may use the shoulder"
    assert "pass_takes_s=pass_takes_s, ego_speed_mps=pose.speed" in src
