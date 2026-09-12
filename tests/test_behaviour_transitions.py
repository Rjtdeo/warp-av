"""Planning V2, P3: every change of what the van is doing says why, from a fixed short list.

The behaviour layer always wrote a sentence. A sentence cannot be counted and cannot be
tested for, so the planner was given a closed set of reason codes first (P0). This is the
same for the state machine: one code from transitions.ALL_WHY on every decision, the changes
kept in order, and a drive readable afterwards as "what it was doing, what it changed to,
why".
"""
import re
import time
from pathlib import Path

import pytest

from warp_av.behavior import transitions as T
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.localization.localization import Pose
from warp_av.perception.perception import PerceptionOutput, ObjectType


def van():
    b = BehaviorSystem()
    b.set_mission()
    return b


def clear():
    return PerceptionOutput()


def blocked_by(kind, distance=6.0, speed=0.0):
    return PerceptionOutput(closest_obstacle_distance=distance, closest_obstacle_type=kind,
                            closest_obstacle_speed=speed, path_blocked=True)


def drive(b, perception, **kw):
    kw.setdefault("pose", Pose(healthy=True))
    kw.setdefault("destination_distance", 200.0)
    kw.setdefault("safety_ok", True)
    return b.update(perception=perception, **kw)


# ---- every decision carries a code -------------------------------------------------------

def test_every_decision_in_the_code_hands_over_a_reason():
    """No branch may answer without one: the list is only closed if nothing escapes it."""
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "behavior" / "behavior.py").read_text()
    calls = [m for m in re.finditer(r"self\._decide\(", src)]
    assert len(calls) >= 20
    for m in calls:
        depth, i = 1, m.end()
        while depth:
            depth += (src[i] == "(") - (src[i] == ")")
            i += 1
        args = src[m.end():i - 1]
        # a splatted answer (the light rule) carries its own code inside the tuple
        assert "why=" in args or args.strip().startswith("*"), f"a decision with no reason: {args[:60]}"


def test_the_light_branch_hands_its_own_code_up():
    b = van()
    perception = clear()
    perception.traffic_light = "red"
    perception.traffic_light_distance_m = 30.0
    out = drive(b, perception, stop_line_m=25.0, light_id=7)
    assert out.why == T.LIGHT_ROLL_UP
    out = drive(b, perception, stop_line_m=0.2, light_id=7)
    assert out.why == T.LIGHT_HOLD and out.behavior == DrivingBehavior.STOPPED_RED_LIGHT


@pytest.mark.parametrize("kind, want", [
    (ObjectType.PEDESTRIAN, T.VRU_IN_PATH),
    (ObjectType.CYCLIST, T.VRU_IN_PATH),
    (ObjectType.VEHICLE, T.VEHICLE_IN_PATH),
    (ObjectType.OBSTACLE, T.OBSTACLE_IN_PATH),
])
def test_what_is_in_the_way_names_itself(kind, want):
    assert drive(van(), blocked_by(kind)).why == want


def test_a_clear_road_a_lead_car_and_the_spot_all_name_themselves():
    b = van()
    assert drive(b, clear()).why == T.ROUTE_CLEAR
    lead = PerceptionOutput(closest_obstacle_distance=20.0, closest_obstacle_type=ObjectType.VEHICLE,
                            closest_obstacle_speed=4.0, path_blocked=False)   # moving, not in the way
    assert drive(b, lead).why == T.FOLLOWING_LEAD
    assert drive(b, clear(), destination_distance=20.0).why == T.DESTINATION_NEAR
    assert drive(b, clear(), destination_distance=6.0).why == T.PARKING_PULL_IN
    parked = drive(b, clear(), destination_distance=0.3, pose=Pose(healthy=True, speed=0.0))
    assert parked.why == T.PARKED and parked.behavior == DrivingBehavior.MISSION_COMPLETE


def test_stopping_for_the_van_itself_names_itself():
    assert drive(van(), clear(), safety_ok=False).why == T.SAFETY_HOLD
    assert drive(van(), clear(), pose=Pose(healthy=False)).why == T.LOCALIZATION_LOST
    sick = clear()
    sick.healthy = False
    assert drive(van(), sick).why == T.PERCEPTION_LOST
    idle = BehaviorSystem()
    assert drive(idle, clear()).why == T.NO_MISSION


def test_every_code_used_is_one_of_the_listed_ones():
    b = van()
    for perception in (clear(), blocked_by(ObjectType.VEHICLE), blocked_by(ObjectType.OBSTACLE),
                       blocked_by(ObjectType.PEDESTRIAN)):
        assert drive(b, perception).why in T.ALL_WHY


# ---- the changes, in order ---------------------------------------------------------------

def test_only_a_real_change_is_recorded():
    b = van()
    for _ in range(5):
        drive(b, clear())
    assert len(b.transitions.changes) == 1                  # five identical ticks, one change
    drive(b, blocked_by(ObjectType.VEHICLE))
    drive(b, blocked_by(ObjectType.VEHICLE))
    assert len(b.transitions.changes) == 2
    last = b.transitions.last
    assert (last.was, last.now, last.why) == ("following_route", "stopped_vehicle", T.VEHICLE_IN_PATH)
    assert last.stopping and last.speed_mps == 0.0 and "VEHICLE" in last.said


def test_the_same_state_for_a_different_reason_is_a_change():
    """Holding at a red light and then holding for a car are not the same thing."""
    log = T.TransitionLog()
    log.note("following_route", "stopped_red_light", T.LIGHT_HOLD, "red", 0.0, True)
    again = log.note("stopped_red_light", "stopped_red_light", T.VEHICLE_IN_PATH, "a car", 0.0, True)
    assert again is not None and log.total == 2


def test_a_made_up_reason_is_refused():
    with pytest.raises(ValueError):
        T.TransitionLog().note("idle", "following_route", "because", "prose")


def test_the_log_counts_and_reads_back():
    b = van()
    drive(b, clear())
    drive(b, blocked_by(ObjectType.OBSTACLE))
    drive(b, clear())
    d = b.transitions.as_dict()
    assert d["total"] == 3 and d["counts"][T.OBSTACLE_IN_PATH] == 1
    assert [c["why"] for c in d["recent"]] == [T.ROUTE_CLEAR, T.OBSTACLE_IN_PATH, T.ROUTE_CLEAR]
    assert str(b.transitions.last).startswith("stopped_obstacle -> following_route (route_clear)")


def test_the_van_publishes_the_changes():
    """main.py must hand them to the operator page and the mission log, or nobody sees them."""
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert '"behavior_why": behavior_output.why' in src
    assert '"behavior_changes": self.behavior.transitions.as_dict()' in src
    assert 'self.logger.log_event("behaviour", str(change))' in src


# ---- the order the questions are asked in (P3 step 2) ------------------------------------

def test_the_order_is_written_down_and_is_the_one_we_mean():
    names = [name for name, _rule, _place in BehaviorSystem.RULES]
    assert names == [
        "safety", "no_mission", "localization", "perception", "parked", "blocked_too_long",
        "vru_in_path", "vehicle_in_path", "obstacle_in_path", "confirming_clear",
        "predicted_crosser", "traffic_light", "road_sign", "junction_box", "junction", "parking",
        "following_lead",
        "object_ahead", "approaching", "cruise",
    ]
    for name, rule, place in BehaviorSystem.RULES:
        assert callable(rule) and place, f"{name} has no rule or no reason to sit there"


def test_what_matters_most_is_asked_first():
    order = {name: i for i, (name, _r, _p) in enumerate(BehaviorSystem.RULES)}
    assert order["safety"] == 0                                  # nothing beats the supervisor
    assert order["vru_in_path"] < order["vehicle_in_path"] < order["obstacle_in_path"]
    assert order["obstacle_in_path"] < order["traffic_light"]    # a thing in the way beats the law
    assert order["predicted_crosser"] < order["traffic_light"]   # ...and the light is read there
    assert order["parked"] < order["blocked_too_long"]           # a parked van is parked
    assert order["parking"] < order["object_ahead"]              # see the flapping below
    assert order["cruise"] == len(BehaviorSystem.RULES) - 1      # and the road is clear


def test_the_change_says_which_rule_answered():
    b = van()
    drive(b, clear())
    last = b.transitions.last
    assert (last.rule, last.rank) == ("cruise", len(BehaviorSystem.RULES))
    drive(b, blocked_by(ObjectType.PEDESTRIAN))
    assert b.transitions.last.rule == "vru_in_path"
    assert "[rule 7 vru_in_path]" in str(b.transitions.last)


def test_something_in_sight_no_longer_takes_the_state_away_from_a_pull_in():
    """Live 2026-09-11: through the last 15 m the state flapped parking -> object_ahead_slow
    -> stopped_obstacle -> parking five times. In sight is not in the way."""
    b = van()
    in_sight = PerceptionOutput(closest_obstacle_distance=9.0, closest_obstacle_type=ObjectType.OBSTACLE,
                                path_blocked=False)
    out = drive(b, in_sight, destination_distance=10.0)
    assert out.behavior == DrivingBehavior.PARKING and out.why == T.PARKING_PULL_IN
    assert out.desired_speed_mps <= b.slow_speed                 # but no faster for it
    assert "in sight at 9.0 m" in out.reason


def test_but_something_in_the_WAY_still_stops_a_pull_in():
    b = van()
    out = drive(b, blocked_by(ObjectType.OBSTACLE, distance=3.9), destination_distance=10.0)
    assert out.behavior == DrivingBehavior.STOPPED_OBSTACLE and out.should_stop


# ---- the manoeuvres are in the same story (P3 step 3) ------------------------------------

def test_a_move_is_not_a_change_of_state():
    """Going round a dead car is still "following the route"; the story needs it anyway."""
    log = T.TransitionLog()
    move = log.note_move(T.GO_AROUND_START, "dead vehicle at 9.0 m — passing on the left",
                         state="following_route")
    assert move.kind == T.MOVE and move.was == move.now == "following_route"
    assert str(move) == ("* go_around_start (while following_route): dead vehicle at 9.0 m "
                         "— passing on the left")
    assert log.as_dict()["counts"][T.GO_AROUND_START] == 1


def test_saying_the_same_thing_again_is_not_a_new_move():
    """The go-around repeats why it is still waiting every ten seconds."""
    log = T.TransitionLog()
    log.note_move(T.GO_AROUND_WAIT, "car coming up behind in the passing lane, 22 m back")
    assert log.note_move(T.GO_AROUND_WAIT, "car coming up behind in the passing lane, 22 m back") is None
    assert log.note_move(T.GO_AROUND_WAIT, "junction only 20 m ahead") is not None


def test_a_made_up_move_is_refused():
    with pytest.raises(ValueError):
        T.TransitionLog().note_move("teleport", "somewhere else")


def test_moves_and_states_keep_their_order():
    b = van()
    drive(b, clear())
    b.transitions.note_move(T.GO_AROUND_START, "passing on the left", state="following_route")
    drive(b, blocked_by(ObjectType.VEHICLE))
    story = [(c.kind, c.why) for c in b.transitions.recent(3)]
    assert story == [(T.STATE, T.ROUTE_CLEAR), (T.MOVE, T.GO_AROUND_START),
                     (T.STATE, T.VEHICLE_IN_PATH)]


def test_the_van_puts_every_manoeuvre_it_makes_into_the_story():
    """A drive read afterwards must not have holes where the big decisions were."""
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    for move in (T.GO_AROUND_START, T.GO_AROUND_WAIT, T.GO_AROUND_DONE, T.SPOT_CHOSEN,
                 T.SPOT_CONFIRMED, T.SPOT_RECHOSEN, T.SPOT_GIVEN_UP, T.GROUND_SEEN_FREE,
                 T.GROUND_BLOCKED):
        name = move.upper()
        assert f"self._note_move({name}" in src, f"{move} never reaches the story"


# ---- steadiness: one object may not change the van's mind nine times (P3 step 4) ---------

def in_sight(metres):
    return PerceptionOutput(closest_obstacle_distance=metres, closest_obstacle_type=ObjectType.OBSTACLE,
                            path_blocked=False)


def test_a_thing_hovering_on_the_slowing_line_is_not_argued_about():
    """Live 2026-09-11: route_clear -> object_ahead_slow -> route_clear nine times in a minute,
    4.0 -> 2.0 -> 4.0 m/s each time, as one object crossed the 20 m line back and forth."""
    b = van()
    for metres in (19.5, 20.5, 19.8, 21.0, 19.9, 20.4):
        drive(b, in_sight(metres))
    whys = [c.why for c in b.transitions.recent(10)]
    assert whys == [T.OBJECT_AHEAD_SLOW]                    # one decision, not six


def test_but_it_lets_go_once_the_thing_is_well_clear():
    b = van()
    drive(b, in_sight(19.0))
    assert b.transitions.last.why == T.OBJECT_AHEAD_SLOW
    out = drive(b, in_sight(b.slow_distance + b.slow_release_m + 1.0))
    assert out.why == T.ROUTE_CLEAR


def test_and_lets_go_after_a_moment_even_just_past_the_line():
    b = van()
    drive(b, in_sight(19.0))
    b._slowing_since -= b.slow_hold_s + 0.1                 # a moment later
    assert drive(b, in_sight(21.0)).why == T.ROUTE_CLEAR


def test_nothing_holds_a_stop_once_the_way_is_clear():
    """The other half of steadiness is deliberately NOT done: holding a stop after the path
    cleared cost 62% of the van's obstacle-stopped time on 2026-09-10 (see the release latch,
    which arms only for a blocker that has been there 0.6 s)."""
    b = van()
    assert drive(b, blocked_by(ObjectType.OBSTACLE, distance=5.0)).should_stop
    assert not drive(b, clear()).should_stop
