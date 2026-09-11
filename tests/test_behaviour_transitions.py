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
        assert "why=" in args or args.strip() == "*light", f"a decision with no reason: {args[:60]}"


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
