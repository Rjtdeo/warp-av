"""Nothing is driven into on the way to a parking spot -- including a stop in the lane.

Live in E3 on 2026-09-15 the van passed a car on its left, and 1.4 s later hit it. Two things
went wrong together and either one alone would have prevented it.

The spot was re-chosen while the pass was still being driven. Choosing again cuts the route
back towards the pin, and the pin was BEHIND the car the van was in the middle of passing, so
the new route turned it straight across that car.

And the spot it cut to was a stop in the lane. Every other kind is asked what the van's body
would touch on the way in (_confirm_parking_spot, planner.pull_in_blocker). A lane stop is the
last resort, so it was written down already confirmed and nothing ever looked. At the moment
of contact the planner reported nothing in the way at all, because the car was 0.63 m BEHIND
the van's centre and the corridor check only looks forward.
"""
import math
from types import SimpleNamespace

from warp_av.main import WarpAV
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.planner import Route, RoutePlanner, Waypoint

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)


def road(n=40, step=2.0):
    return Route(waypoints=[Waypoint(x=i * step, y=0.0, yaw=0.0) for i in range(n)],
                 total_distance=(n - 1) * step)


def parked(x, y, length=4.79, w=2.16):
    """Something standing still, in the van's frame: x ahead, y to the right."""
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y),
                          speed=0.0, stationary=True, height_m=1.49, length_m=length, width_m=w,
                          box_length_m=length, box_width_m=w, box_yaw_deg=0.0)


def van(overtaking=False, in_the_way=None, spot=None, choose=None, pin_index=12):
    s = SimpleNamespace()
    s.said, s.logged = [], []
    s._overtake_point = object() if overtaking else None
    s._route = road()
    s._route_base = list(road(n=40).waypoints)
    s._pin_index = pin_index
    s._parking_spot = spot if spot is not None else {"kind": "bay", "x": 30.0, "y": 2.0}
    s._parking_rejected = []
    s._parking_wait_since = 123.0
    s._signal_lookahead = None
    s.behavior = SimpleNamespace(_park_best_d=7.0)
    s.footprint_blocking = SimpleNamespace(footprint=FOOT)
    s._last_perception = SimpleNamespace(objects=list(in_the_way or []))
    s.planner = RoutePlanner.__new__(RoutePlanner)          # the real pull_in_blocker
    s.logger = SimpleNamespace(log_event=lambda *a, **k: s.logged.append((a, k)))
    s._note_move = lambda what, text: s.said.append((what, text))
    s._choose_spot = lambda ahead_of=None: choose
    s._way_in_blocker = lambda wps, pose: WarpAV._way_in_blocker(s, wps, pose)
    s._check_the_kerb_is_empty = lambda sp, pose: s.said.append(("kerb_checked", ""))
    s.pose = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, speed=0.5)
    s.rechoose = lambda why="the spot will not do": WarpAV._rechoose_parking(s, s.pose, why)
    s.confirm = lambda d=7.7: WarpAV._confirm_parking_spot(
        s, s.pose, SimpleNamespace(behavior=None), d)
    return s


def events(s, name):
    return [(a, k) for a, k in s.logged if a[0] == name]


# ---- the pass finishes before the spot moves -------------------------------------------

def test_the_spot_is_not_re_chosen_while_a_way_round_is_being_driven():
    """E3's first fault. Cutting the route back mid-pass points the van at a spot behind the
    thing it is passing."""
    s = van(overtaking=True)
    before = list(s._route.waypoints)
    s.rechoose()
    assert s._route.waypoints == before, "the route was left alone"
    assert s._parking_spot["kind"] == "bay", "and so was the spot"
    assert s._parking_rejected == [], "nothing was turned down either"
    assert s.said == [] and s.logged == []


def test_and_it_is_re_chosen_as_soon_as_the_pass_is_over():
    other = (list(road(n=20).waypoints), {"kind": "bay", "from_pin_m": 4, "x": 20.0, "y": 2.0})
    s = van(overtaking=False, choose=other)
    s.rechoose()
    assert s._parking_spot["kind"] == "bay" and s._parking_spot["x"] == 20.0
    assert len(s._route.waypoints) == 20, "the new route was taken"
    assert any(w == "spot_rechosen" for w, _ in s.said)


def test_the_gate_is_the_pass_and_nothing_else():
    """Same call, same scene, one difference."""
    other = (list(road(n=20).waypoints), {"kind": "bay", "from_pin_m": 4, "x": 20.0, "y": 2.0})
    held = van(overtaking=True, choose=other)
    free = van(overtaking=False, choose=other)
    held.rechoose()
    free.rechoose()
    assert len(held._route.waypoints) == 40 and len(free._route.waypoints) == 20


# ---- a stop in the lane is asked the same question as every other spot ---------------------

def test_a_lane_stop_the_van_would_sweep_a_car_to_reach_is_not_cut_to():
    """E3's second fault. No other spot will do, so the route is cut back to the pin -- but
    only if the van can actually get there."""
    car = parked(6.0, 0.6)                       # on the cut route, 6 m ahead
    car.id = 849
    s = van(overtaking=False, choose=None, in_the_way=[car])
    before = list(s._route.waypoints)
    s.rechoose()
    assert s._route.waypoints == before, "the route was NOT cut to a stop it cannot reach"
    assert s._parking_spot["kind"] == "bay", "and the lane stop was not adopted"
    got = events(s, "parking_lane_stop_blocked")
    assert len(got) == 1, "and it said so"
    data = got[0][1]["data"]
    assert data["blocker_id"] == 849 and data["blocker_kind"] == "vehicle"
    assert data["touch_along_m"] is not None and data["blocker_distance_m"] > 0


def test_a_clear_lane_stop_is_still_taken_exactly_as_before():
    """The last resort must still work when the way in really is clear."""
    s = van(overtaking=False, choose=None, in_the_way=[])
    s.rechoose()
    assert s._parking_spot["kind"] == "lane"
    assert s._parking_spot["confirmed"] is True
    assert len(s._route.waypoints) == s._pin_index + 1, "cut back to the pin, as before"
    assert any("stopping straight in the lane" in t for _, t in s.said)


def test_something_off_to_one_side_does_not_stop_the_lane_fallback():
    """The test is what the van's body would touch, not what is nearby."""
    far = parked(6.0, 6.0)
    far.id = 77
    s = van(overtaking=False, choose=None, in_the_way=[far])
    s.rechoose()
    assert s._parking_spot["kind"] == "lane", "6 m to the side is not in the way"


# ---- and an existing lane stop is looked at, where nothing looked before ---------------------

def test_a_lane_stop_already_chosen_is_checked_and_reported():
    car = parked(6.0, 0.6)
    car.id = 849
    s = van(spot={"kind": "lane", "x": 24.0, "y": 0.0, "confirmed": True}, in_the_way=[car])
    s.confirm()
    assert s._parking_spot["way_in"] == "blocked"
    assert len(events(s, "parking_lane_stop_way_in")) == 1
    data = events(s, "parking_lane_stop_way_in")[0][1]["data"]
    assert data["blocker_id"] == 849


def test_a_clear_lane_stop_is_reported_clear():
    s = van(spot={"kind": "lane", "x": 24.0, "y": 0.0, "confirmed": True}, in_the_way=[])
    s.confirm()
    assert s._parking_spot["way_in"] == "clear"
    assert events(s, "parking_lane_stop_way_in") == []


def test_looking_at_a_lane_stop_never_changes_the_route():
    """This one only reports: the lane stop IS the route, so there is nothing to change."""
    car = parked(6.0, 0.6)
    car.id = 849
    s = van(spot={"kind": "lane", "x": 24.0, "y": 0.0, "confirmed": True}, in_the_way=[car])
    before = list(s._route.waypoints)
    s.confirm()
    assert s._route.waypoints == before


def test_a_bay_still_goes_down_the_path_it_always_did():
    """No regression for the kinds that were already checked."""
    s = van(spot={"kind": "kerb", "x": 24.0, "y": 2.0})
    s.confirm()
    assert ("kerb_checked", "") in s.said, "a kerb spot still gets the kerb check"


def test_nothing_is_looked_at_from_too_far_out():
    s = van(spot={"kind": "lane", "x": 24.0, "y": 0.0, "confirmed": True}, in_the_way=[parked(6.0, 0.6)])
    s.confirm(d=999.0)
    assert "way_in" not in s._parking_spot, "out of range, as before"


def test_the_way_in_check_never_throws_on_a_broken_scene():
    s = van(overtaking=False, choose=None, in_the_way=[])
    s._last_perception = None
    s.rechoose()
    assert s._parking_spot["kind"] == "lane", "no perception is not a reason to refuse to park"
