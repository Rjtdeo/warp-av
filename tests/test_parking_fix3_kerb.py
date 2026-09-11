"""Fix 3: the van stalled at the kerb when pulling into a parking spot (2026-09-10).

Two causes, found in 4 stalled scenarios:
  * 3 of 4: bays one slot long. The van cannot reverse, so it turns in from the lane, and
    its body leaves the lane 13.1 m before the slot's centre -- over bare kerb, before a
    short bay begins. Now a slot needs APPROACH_BAY_BEHIND_M of free bay behind it.
  * 1 of 4: a kerb beside a long bay, 0.5 m from where the van would stand, against 0.55 m
    of padding. Now a kerb-height thing needs tyre clearance only.
And whatever is left: blocked near the spot by something that will not move, the van
stops waiting and finishes there.
"""
import math

import pytest

from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.planner import (KERB_CLEARANCE_M, RoutePlanner, Route, Waypoint,
                                      WaitingIsPointless, kerb_like, obstacle_box_for)
from test_parking import _mock_bays, straight_route

SPRINTER = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def thing(x, y, size, kind=ObjectType.OBSTACLE, stationary=True, ident=1, **kw):
    o = DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), id=ident,
                       stationary=stationary, **kw)
    o.length_m, o.width_m, o.height_m = size
    return o


def blocked(obj):
    per = PerceptionOutput(objects=[obj])
    road = Route(waypoints=[Waypoint(x=-20.0 + i * 2.0, y=0.0) for i in range(61)])
    p = planner()
    p.filter_to_route_corridor(per, road, 0.0, 0.0, 0.0, footprint=SPRINTER)
    return per.path_blocked


# ---- what counts as a kerb ---------------------------------------------------------------

def test_a_kerb_piece_is_kerb_like():
    assert kerb_like(thing(8.0, 1.6, (1.27, 0.17, 0.15)))


def test_nothing_taller_or_moving_or_named_is_kerb_like():
    assert not kerb_like(thing(8.0, 1.6, (1.27, 0.17, 0.35))), "taller than a kerb"
    assert not kerb_like(thing(8.0, 1.6, (1.27, 0.17, 0.15), stationary=False)), "moving"
    assert not kerb_like(thing(8.0, 1.6, (0.4, 0.3, 0.15), kind=ObjectType.PEDESTRIAN)), "a person"
    assert not kerb_like(thing(8.0, 1.6, (0.0, 0.0, 0.0))), "never measured"


def test_a_kerb_is_taken_at_its_measured_width():
    box = obstacle_box_for(thing(8.0, 1.6, (1.27, 0.17, 0.15)), 0.0)
    wall = obstacle_box_for(thing(8.0, 1.6, (1.27, 0.17, 0.60)), 0.0)
    assert box.half_width < wall.half_width, "the 0.15 m floor is for things that can hide a car"


# ---- case A: the kerb beside the slot ------------------------------------------------------

def test_the_kerb_beside_a_parked_van_no_longer_stops_it():
    """WAV-0223: kerb 1.27 x 0.17 x 0.15 m, 1.59 m off the slot's centre line."""
    assert not blocked(thing(8.0, 1.59, (1.27, 0.17, 0.15)))


def test_the_same_thing_half_a_metre_tall_still_does():
    assert blocked(thing(8.0, 1.59, (1.27, 0.17, 0.50)))


def test_a_kerb_height_thing_in_the_path_still_stops_the_van():
    assert blocked(thing(8.0, 0.3, (1.0, 0.3, 0.15)))
    # and one the tyres would clip: body side 0.994 m, tyre clearance 0.10 m
    assert blocked(thing(8.0, 0.994 + KERB_CLEARANCE_M, (0.6, 0.1, 0.15)))


# ---- case B: slots the van cannot drive into ---------------------------------------------

def run(length_m):
    return [(float(t), 2.8, 0.0, 2.0) for t in range(0, int(length_m) + 1, 2)]


def test_slots_know_their_bay_and_how_much_bay_is_behind_them():
    slots = planner()._slice_run_into_slots(run(30), bay_id=4)
    assert [s["k"] for s in slots] == [0, 1, 2, 3]
    assert all(s["bay"] == 4 for s in slots)
    assert [s["bay_behind_m"] for s in slots] == [3.5, 10.5, 17.5, 24.5]


def test_a_bay_one_slot_long_is_never_chosen():
    slots = planner()._slice_run_into_slots(run(8))
    for s in slots:
        s["occupied"] = False
    assert len(slots) == 1
    assert RoutePlanner.choose_free_slot(slots) is None


def test_the_first_slot_the_van_can_drive_into_is_the_third():
    slots = planner()._slice_run_into_slots(run(22))
    for s in slots:
        s["occupied"] = False
    assert RoutePlanner.choose_free_slot(slots) == 2


def test_the_turn_in_needs_the_slots_behind_free():
    slots = planner()._slice_run_into_slots(run(30))
    for s in slots:
        s["occupied"] = False
    slots[2]["occupied"] = True         # the van would sweep through it to reach slot 3
    assert not RoutePlanner.slot_reachable(slots, 3)
    assert RoutePlanner.choose_free_slot(slots) is None


def test_a_free_slot_in_ANOTHER_bay_is_no_approach():
    """The old rule took 'the slot before it in the list' -- across the kerb between bays."""
    p = planner()
    slots = p._slice_run_into_slots(run(22), bay_id=0)
    far = [(40.0 + t, 2.8, 0.0, 2.0) for t in range(0, 9, 2)]
    slots += p._slice_run_into_slots(far, bay_id=1)       # one slot nearest the destination
    for s in slots:
        s["occupied"] = False
    assert RoutePlanner.choose_free_slot(slots) == 2, "not the lone slot in the next bay"


def test_find_parking_slots_numbers_the_bays():
    p = planner()
    _mock_bays(p, span=(100.0, 150.0))
    slots = p.find_parking_slots(straight_route(80))
    assert slots and all("bay" in s and "k" in s for s in slots)


# ---- the kerbside pull-over has the same need ---------------------------------------------

def test_the_pull_over_skips_a_bay_too_short_to_turn_into():
    p = planner()
    _mock_bays(p, span=(108.0, 116.0))              # 8 m of bay, just before the pin
    r = straight_route(60)                           # ends at x = 118
    spot = p.apply_pullover(r)
    assert spot is None or spot["kind"] != "bay"


def test_the_pull_over_still_uses_a_long_bay():
    p = planner()
    _mock_bays(p, span=(60.0, 118.0))
    spot = p.apply_pullover(straight_route(60))
    assert spot is not None and spot["kind"] == "bay"


# ---- whatever is left: stop waiting for what will not move ---------------------------------

def test_blocked_by_a_kerb_near_the_spot_it_finishes_after_six_seconds():
    w = WaitingIsPointless()
    kerb = thing(9.0, 1.5, (1.2, 0.2, 0.15))
    assert w.update(True, 12.0, kerb, now=0.0) is None
    assert w.update(True, 12.0, kerb, now=5.0) is None
    assert w.update(True, 12.0, kerb, now=6.5) == "kerb"


def test_a_static_post_is_named_as_one():
    w = WaitingIsPointless()
    post = thing(9.0, 1.0, (0.2, 0.2, 3.5), motion_class="static", static_rule="pole")
    w.update(True, 10.0, post, now=0.0)
    assert w.update(True, 10.0, post, now=7.0) == "post"


@pytest.mark.parametrize("who", [
    thing(9.0, 0.5, (0.5, 0.4, 1.7), kind=ObjectType.PEDESTRIAN),
    thing(9.0, 0.5, (4.5, 1.8, 1.5), kind=ObjectType.VEHICLE),
    thing(9.0, 0.5, (1.7, 0.6, 1.7), kind=ObjectType.CYCLIST),
])
def test_it_keeps_waiting_for_people_and_vehicles_however_long(who):
    w = WaitingIsPointless()
    for t in range(0, 120, 3):
        assert w.update(True, 10.0, who, now=float(t)) is None


def test_an_unnamed_thing_that_moves_is_waited_for():
    w = WaitingIsPointless()
    box = thing(9.0, 0.5, (0.8, 0.6, 1.0), stationary=False)
    for t in range(0, 60, 3):
        assert w.update(True, 10.0, box, now=float(t)) is None


def test_an_unnamed_thing_that_sits_in_the_way_20_s_is_not_going_anywhere():
    """A post inside the parking lane is never labelled static -- the lane is road."""
    w = WaitingIsPointless()
    post = thing(9.0, 1.6, (0.4, 0.4, 3.9))
    assert w.update(True, 12.0, post, now=0.0) is None
    assert w.update(True, 12.0, post, now=15.0) is None
    assert w.update(True, 12.0, post, now=20.5) == "fixed object"


def test_a_flicker_of_clear_does_not_restart_the_clock():
    """Seen live: blocked 4.2 s, clear 1.1 s while creeping 1.2 m, blocked again."""
    w = WaitingIsPointless()
    kerb = thing(9.0, 1.5, (1.2, 0.2, 0.15))
    w.update(True, 12.0, kerb, now=0.0)
    w.update(False, 12.0, None, now=4.2)
    w.update(False, 12.0, None, now=5.3)
    assert w.update(True, 12.0, kerb, now=6.2) == "kerb"


def test_far_from_the_spot_it_keeps_waiting():
    w = WaitingIsPointless()
    kerb = thing(9.0, 1.5, (1.2, 0.2, 0.15))
    for t in range(0, 30, 3):
        assert w.update(True, 45.0, kerb, now=float(t)) is None


def test_the_clock_restarts_when_the_way_really_clears():
    w = WaitingIsPointless()
    kerb = thing(9.0, 1.5, (1.2, 0.2, 0.15))
    w.update(True, 12.0, kerb, now=0.0)
    w.update(False, 12.0, None, now=4.0)
    w.update(False, 12.0, None, now=6.5)            # clear for 2.5 s: that is the way opening
    assert w.update(True, 12.0, kerb, now=8.0) is None, "a fresh 6 s after it cleared"
    assert w.update(True, 12.0, kerb, now=14.5) == "kerb"


def test_a_bent_piece_of_bay_behind_the_slot_is_no_approach():
    """Slots are skipped where the bay bends round something; the turn-in must not cross it.
    Seen live: slot k=4 chosen with k=1..3 skipped, and the van met a kerb."""
    slots = planner()._slice_run_into_slots(run(50))           # k = 0..6
    for s in slots:
        s["occupied"] = False
    del slots[2:4]                                   # k=2 and k=3: a bend in the bay
    k = {s["k"]: i for i, s in enumerate(slots)}
    assert not RoutePlanner.slot_reachable(slots, k[4]), "sweeps through the bend at k=2, 3"
    assert not RoutePlanner.slot_reachable(slots, k[5]), "still sweeps through k=3"
    assert RoutePlanner.slot_reachable(slots, k[6]), "k=4 and k=5 are straight, free bay"
    assert RoutePlanner.choose_free_slot(slots) == k[6]
