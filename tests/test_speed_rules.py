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


# ---- stop and give-way signs, read from the map (2026-09-11) -----------------------------

from warp_av.perception.road_signs import (RoadSign, STOP, GIVE_WAY, signs_on_route,  # noqa: E402
                                           next_sign, MIN_LANE_WIDTH_M)
from warp_av.behavior.behavior import SIGN_DWELL_S, SIGN_STOPPED_MPS  # noqa: E402


class _WP:
    def __init__(self, x, y, road, lane):
        self.x, self.y, self.road_id, self.lane_id = x, y, road, lane


def route(n=40, road=11, lane=-1):
    return [_WP(float(i), 0.0, road, lane) for i in range(n)]


def test_a_sign_on_our_lane_is_found_where_it_stands():
    sign = RoadSign(kind=STOP, road_id=11, lane_id=-1, x=20.0, y=0.0, yaw_deg=0.0)
    found = signs_on_route([sign], route())
    assert found and abs(found[0][0] - 20.0) < 1e-6
    assert abs(next_sign(found, 5.0)[0] - 15.0) < 1e-6
    assert next_sign(found, 25.0) is None            # behind us


def test_a_sign_on_the_crossing_road_is_not_ours():
    crossing = RoadSign(kind=STOP, road_id=99, lane_id=1, x=20.0, y=0.0, yaw_deg=90.0)
    other_lane = RoadSign(kind=STOP, road_id=11, lane_id=-2, x=20.0, y=0.0, yaw_deg=0.0)
    assert signs_on_route([crossing, other_lane], route()) == []


def test_a_sign_on_our_lane_but_far_off_the_line_is_not_ours_either():
    away = RoadSign(kind=STOP, road_id=11, lane_id=-1, x=20.0, y=9.0, yaw_deg=0.0)
    assert signs_on_route([away], route()) == []


def test_a_stop_sign_is_a_full_stop_then_the_junction_decides():
    b = van()
    at = (11, -1, 20.0, 0.0)
    rolling = drive(b, sign_m=6.0, sign_kind=STOP, sign_at=at)
    assert rolling.why == T.SIGN_ROLL_UP and 0 < rolling.desired_speed_mps <= 3.0
    held = drive(b, sign_m=0.4, sign_kind=STOP, sign_at=at, pose=Pose(healthy=True, speed=0.0))
    assert held.why == T.SIGN_STOP_HOLD and held.should_stop
    b._sign_still_since -= SIGN_DWELL_S + 0.1                     # a second later, still still
    after = drive(b, sign_m=0.4, sign_kind=STOP, sign_at=at, pose=Pose(healthy=True, speed=0.0))
    assert after.why != T.SIGN_STOP_HOLD, "a sign is stopped for once, not for ever"


def test_rolling_over_the_line_does_not_count_as_stopping():
    b = van()
    at = (11, -1, 20.0, 0.0)
    for _ in range(12):
        out = drive(b, sign_m=0.3, sign_kind=STOP, sign_at=at,
                    pose=Pose(healthy=True, speed=SIGN_STOPPED_MPS + 0.5))
    assert out.why == T.SIGN_STOP_HOLD, "still rolling: the stop has not happened yet"


def test_a_give_way_sign_asks_for_a_crawl_not_a_stop():
    b = van()
    out = drive(b, sign_m=5.0, sign_kind=GIVE_WAY, sign_at=(843, -1, 43.8, 38.4))
    assert out.why == T.SIGN_GIVE_WAY and 0 < out.desired_speed_mps <= b.slow_speed


def test_lane_validities_sweep_up_kerbs_and_pavements():
    """A sign's validity range covers every lane id between two numbers, which on Town10HD
    includes 0.6 m kerb strips and 6 m pavements. Only real lanes are kept."""
    assert MIN_LANE_WIDTH_M >= 2.0
