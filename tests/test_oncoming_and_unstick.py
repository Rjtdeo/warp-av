"""
Two faults found by driving, 2026-09-14.

WATCHING WHAT IS COMING THE OTHER WAY. The corridor check only ever looks in OUR lane. That is
right while the van stays in it -- and nothing asked whether it still was. An ambulance closed
head-on from 58 m to 5.7 m at 8 m/s while the planner said "clear, 999 m"; the van, correcting
a 0.99 m drift, swung 1.35 m the other way and they met. `oncoming_conflict` asks the question
the corridor cannot: is there clear air between the two BODIES as they pass?

GETTING OUT OF THE ROAD AGAIN. After the hit the van stood 126 s at 1.35 m off centre and 36
degrees across the lane, saying "replan or operator action required", until the run ended --
a parked obstruction in a live lane. `_maybe_unstick` backs it up and straightens it.

Van frame: x ahead, y to the RIGHT.
"""
import math

import pytest

from warp_av.planning.planner import (oncoming_conflict, ONCOMING_KEEP_M, ONCOMING_LOOK_S,
                                      ONCOMING_MIN_CLOSING)
from warp_av.perception.perception import ObjectType, DetectedObject

SWEPT_HALF_WIDTH = 1.29          # the Sprinter's body plus its safety margin


def coming_at_us(x, y, speed=8.0, width=1.9, kind=ObjectType.VEHICLE, length=4.8):
    """Something driving towards us: in OUR frame it is ahead, and its world velocity points
    back along our heading (we face +x world, so it moves in -x)."""
    o = DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed,
                       width_m=width, length_m=length, height_m=1.7, id=7)
    o.stationary = False
    o.vx_world, o.vy_world = -speed, 0.0
    return o


def going_our_way(x, y, speed=8.0):
    o = coming_at_us(x, y, speed=speed)
    o.vx_world, o.vy_world = +speed, 0.0
    return o


# ---- what is coming the other way -------------------------------------------------------

def test_a_car_in_the_next_lane_over_says_nothing():
    """3.5 m across: 3.5 - (0.95 + 1.29) = 1.26 m of clear air. An ordinary street."""
    assert oncoming_conflict([coming_at_us(30.0, -3.5)], SWEPT_HALF_WIDTH, 0.0, 4.0) is None


def test_the_same_car_with_the_van_drifted_towards_it_does_not():
    """The van 1.4 m off its line: the gap between the two bodies is gone."""
    met = oncoming_conflict([coming_at_us(30.0, -2.1)], SWEPT_HALF_WIDTH, 0.0, 4.0)
    assert met is not None
    why, meets_in, clearance = met
    assert clearance < ONCOMING_KEEP_M and clearance < 0.2
    assert 0.0 < meets_in <= ONCOMING_LOOK_S
    assert "coming the other way" in why and "between the two bodies" in why


def test_the_ambulance_that_actually_hit_us():
    """t=155.6 in the recording: 8.2 m ahead, 1.0 m across, closing at 8 m/s while we did 4.8."""
    met = oncoming_conflict([coming_at_us(8.2, 1.0, speed=8.0)], SWEPT_HALF_WIDTH, 0.0, 4.8)
    assert met is not None, "this is the one the van drove into"
    assert met[1] < 1.0, "and it was less than a second away"


def test_something_going_our_way_is_not_oncoming():
    assert oncoming_conflict([going_our_way(10.0, 0.3)], SWEPT_HALF_WIDTH, 0.0, 4.0) is None


def test_something_behind_us_is_not_oncoming():
    assert oncoming_conflict([coming_at_us(-10.0, 0.0)], SWEPT_HALF_WIDTH, 0.0, 4.0) is None


def test_a_parked_car_is_not_oncoming():
    parked = coming_at_us(10.0, 0.0, speed=0.0)
    parked.stationary = True
    parked.vx_world = 0.0
    assert oncoming_conflict([parked], SWEPT_HALF_WIDTH, 0.0, 4.0) is None


def test_something_far_enough_ahead_to_be_dealt_with_later_says_nothing_yet():
    far = coming_at_us(200.0, 1.0, speed=8.0)
    assert oncoming_conflict([far], SWEPT_HALF_WIDTH, 0.0, 4.0) is None


def test_a_van_standing_still_still_sees_what_is_coming_at_it():
    met = oncoming_conflict([coming_at_us(20.0, 1.0, speed=8.0)], SWEPT_HALF_WIDTH, 0.0, 0.0)
    assert met is not None and met[1] == pytest.approx(20.0 / 8.0, abs=0.1)


def test_the_nearest_one_is_the_one_reported():
    near = coming_at_us(10.0, 1.0)
    far = coming_at_us(30.0, 1.0)
    met = oncoming_conflict([far, near], SWEPT_HALF_WIDTH, 0.0, 4.0)
    assert met is not None and "10 m ahead" in met[0]


def test_a_crawl_and_not_a_stop_is_what_it_asks_for():
    """A stop zeroes the steering (VehicleController), and what the van needs while something
    comes the other way is to keep coming back to its own side while it slows."""
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text(encoding="utf-8")
    i = src.index("met = oncoming_conflict(")
    body = src[i:src.index("curve_cap = None", i)]
    assert "ONCOMING_CRAWL_MPS" in body and "should_stop = True" not in body
    assert "crawling until it is past" in body


# ---- getting out of the road again -------------------------------------------------------

def test_the_unstick_asks_the_same_questions_the_reverse_always_did():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text(encoding="utf-8")
    i = src.index("def _maybe_unstick")
    body = src[i:src.index("def _start_backing_out", i)]
    assert "self._overtake_point is not None" in body, "never mid-pass"
    assert "UNSTICK_AFTER_S" in body and "UNSTICK_OVER_EDGE_M" in body
    assert "_start_backing_out" in body, "it reuses the reverse, which checks the ground behind"
    # Whenever it is STANDING in the way -- not only when it calls itself blocked. It sat 0.88 m
    # off its line as `following_route`, held by the lane-change gap rule, while a car passed
    # 5 cm from its flank; a check that looked at the behaviour state never considered that.
    assert 'getattr(pose, "speed", 0.0) or 0.0) > 0.3' in body, "the question is whether it is moving"
    assert "STOPPED_RED_LIGHT" in body and "WAITING_AT_JUNCTION" in body, \
        "never back up at a light or a junction -- it is meant to be standing there"
    assert "DrivingBehavior.PARKING" in body, "nor half-way into a pull-in"
    # and the reverse it calls still refuses when the ground behind is not seen clear
    j = src.index("def _start_backing_out")
    assert "_rear_is_clear" in src[j:j + 600]


def test_how_far_over_the_line_counts_as_across_it():
    """A 3.5 m lane, a van 0.99 m half-width: its middle may be 0.76 m off before the body
    touches the line. The check is on the BODY, not the middle."""
    half_lane, half_van = 3.5 / 2.0, 0.99
    over = lambda off: abs(off) + half_van - half_lane
    assert over(0.0) < 0 and over(0.70) < 0                  # inside its lane
    assert over(1.35) > 0.15, "the pose it was left in after the crash is over the line"
    assert round(over(1.35), 2) == 0.59


def test_something_passing_close_is_recorded_even_when_the_van_cannot_slow_down():
    """A van already standing still cannot slow down -- and "it passed me by 5 cm" is exactly
    the thing that has to show up in the record afterwards (2026-09-14)."""
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text(encoding="utf-8")
    i = src.index("self._oncoming = None")
    body = src[i:src.index("curve_cap = None", i)]
    set_at = body.index("self._oncoming = {")
    gate_at = body.index("if not behavior_output.should_stop")
    assert set_at < gate_at, "the record is written before anything is decided about speed"
    assert 'log_event("oncoming"' in body, "and it goes in the drive's log either way"


def test_a_van_angled_across_the_road_reaches_much_further_than_its_half_width():
    """Live on 2026-09-14 the guard called 0.44 m of clearance where CARLA measured 0.19 m and
    stayed silent by four centimetres while a car went past at 8 m/s. It was measuring half
    widths. A van 5.92 m long, angled 15 degrees, reaches 2.1 m to its side."""
    HL = 3.26
    car = coming_at_us(1.0, -2.6, speed=8.0, width=1.75)     # the tick it missed
    square = oncoming_conflict([car], SWEPT_HALF_WIDTH, 0.0, 3.0, swept_half_length_m=HL,
                               off_axis_rad=0.0)
    across = oncoming_conflict([car], SWEPT_HALF_WIDTH, 0.0, 3.0, swept_half_length_m=HL,
                               off_axis_rad=math.radians(15))
    assert square is None, "square to the road there really is room"
    assert across is not None, "...and angled across it there is not"
    assert across[2] < 0.0, "the bodies would overlap"
    assert "across the road" in across[0]


def test_the_ordinary_pass_is_still_quiet_whatever_the_angle():
    car = coming_at_us(30.0, -3.5)
    for deg in (0, 5, 10, 15):
        assert oncoming_conflict([car], SWEPT_HALF_WIDTH, 0.0, 4.0, swept_half_length_m=3.26,
                                 off_axis_rad=math.radians(deg)) is None, \
            "a car a lane over must not set it off at %d deg" % deg


def test_the_angle_it_allows_for_is_capped():
    """Sideways-on, the half length would say the van reaches 3.3 m to its side, which would
    stop it for anything on the road. It is capped at 45 degrees."""
    car = coming_at_us(20.0, -3.4)
    wide = oncoming_conflict([car], SWEPT_HALF_WIDTH, 0.0, 4.0, swept_half_length_m=3.26,
                             off_axis_rad=math.radians(80))
    capped = oncoming_conflict([car], SWEPT_HALF_WIDTH, 0.0, 4.0, swept_half_length_m=3.26,
                               off_axis_rad=math.radians(45))
    assert (wide is None) == (capped is None)
