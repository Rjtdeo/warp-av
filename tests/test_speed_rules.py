"""How fast the van may go, and how quickly it may change its mind about it.

Three rules, all of them caps -- none of them can ever make the van go faster:
  * the limit on this piece of road, from the map;
  * never faster than it could stop inside the ground the laser has actually SEEN free
    (Planning V2, P2: the half of unknown_space that does something);
  * and easing off rather than stepping down, for comfort slowing only.

Live on 2026-09-11, before the last of these, one object crossing the 20 m line stepped the
van 4.0 -> 2.0 -> 4.0 m/s nine times in a minute.
"""
import math
from pathlib import Path

from warp_av.behavior.behavior import (BehaviorSystem, DrivingBehavior, EASE_OFF_REASONS,
                                       EASE_OFF_MPS, stopping_speed_for)
from warp_av.behavior import transitions as T
from warp_av.localization.localization import Pose
from warp_av.perception.perception import PerceptionOutput


def van():
    b = BehaviorSystem()
    b.set_mission()
    return b


def drive(b, **kw):
    kw.setdefault("perception", PerceptionOutput())
    kw.setdefault("pose", Pose(healthy=True))
    kw.setdefault("destination_distance", 300.0)
    kw.setdefault("safety_ok", True)
    return b.update(**kw)


def test_the_limit_on_this_road_is_never_exceeded():
    b = van()
    assert drive(b).desired_speed_mps == b.cruise_speed
    out = drive(b, speed_limit_mps=5.0)
    assert out.desired_speed_mps == 5.0 and "the limit here is 18 km/h" in out.reason


def test_a_limit_above_the_cruising_speed_changes_nothing():
    b = van()
    assert drive(b, speed_limit_mps=13.9).desired_speed_mps == b.cruise_speed


def test_it_never_goes_faster_than_it_can_stop_in_what_it_has_seen():
    b = van()
    out = drive(b, seen_ahead_m=3.0)
    assert out.desired_speed_mps == round(stopping_speed_for(3.0), 10) or \
        abs(out.desired_speed_mps - stopping_speed_for(3.0)) < 1e-9
    assert "only seen clear for 3.0 m" in out.reason
    assert drive(van(), seen_ahead_m=30.0).desired_speed_mps == b.cruise_speed


def test_the_tightest_of_the_caps_wins_and_none_of_them_can_speed_it_up():
    b = van()
    out = drive(b, speed_limit_mps=6.0, seen_ahead_m=4.0, blind_spot_m=20.0)
    assert out.desired_speed_mps == min(6.0, stopping_speed_for(4.0))
    assert drive(van(), speed_limit_mps=99.0).desired_speed_mps == b.cruise_speed


def test_easing_off_is_only_for_comfort_slowing():
    """A light, a junction, a yield, a parking run-in and every stop take effect at once."""
    assert set(EASE_OFF_REASONS) == {T.OBJECT_AHEAD_SLOW, T.FOLLOWING_LEAD, T.ROUTE_CLEAR}
    for why in (T.LIGHT_ROLL_UP, T.LIGHT_HOLD, T.JUNCTION_ROLL_UP, T.PARKING_PULL_IN,
                T.DESTINATION_NEAR, T.PREDICTED_CROSSER_SLOW, T.VRU_IN_PATH):
        assert why not in EASE_OFF_REASONS


def test_the_van_eases_off_where_the_command_is_formed_not_in_the_decision():
    """The decision itself is never shaped: what it wanted is what the story records."""
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert "behavior_output.why in EASE_OFF_REASONS" in src
    assert "self._eased_speed - EASE_OFF_MPS" in src
    assert "0.0 if behavior_output.should_stop" in src
    behaviour = (Path(__file__).parents[1] / "src" / "warp_av" / "behavior" / "behavior.py").read_text()
    assert "_eased_speed" not in behaviour.split("EASE_OFF_MPS = ")[1]


def test_the_limit_and_the_seen_road_come_from_the_map_and_the_laser():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    assert 'get_landmarks_of_type(150.0, "274")' in src, "a speed limit is a sign on the map"
    assert "grid.free_distance(0.0" in src
    assert "speed_limit_mps=self._speed_limit_mps(pose)" in src
    assert "seen_ahead_m=self._seen_ahead_m()" in src
