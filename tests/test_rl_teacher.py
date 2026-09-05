"""The round-9 teacher: a rule-based parker, exercised in a kinematic paper model."""
import math

import pytest

import rl.teacher as T
from rl.parking_math import feelers


SCENARIOS = [
    ("empty bay from 16 m", (-16.0, -3.1, 0.0), ()),
    ("empty bay from 36 m", (-36.0, -3.1, 0.0), ()),
    ("car three bays back, 29 m", (-29.0, -3.1, 0.0), (T.car_box(3),)),
    ("car two bays back, 22 m", (-22.0, -3.1, 0.0), (T.car_box(2),)),
    ("car two bays back, 16 m", (-16.0, -3.1, 0.0), (T.car_box(2),)),
    ("car in the bay ahead, 16 m", (-16.0, -3.1, 0.0), (T.car_box(-1),)),
    ("car right behind the bay, 16 m", (-16.0, -3.1, 0.0), (T.car_box(1),)),
    ("car right behind the bay, 22 m", (-22.0, -3.1, 0.0), (T.car_box(1),)),
    ("car right behind, knocked off course", (-18.0, -3.5, 0.12), (T.car_box(1),)),
    ("two back, knocked outward", (-22.0, -3.6, 0.17), (T.car_box(2),)),
    ("two back, knocked inward", (-22.0, -2.6, -0.17), (T.car_box(2),)),
    ("empty, knocked inward", (-16.0, -2.7, -0.15), ()),
    ("empty, knocked outward", (-18.0, -3.5, 0.15), ()),
]


@pytest.mark.parametrize("name,start,cars", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_the_teacher_parks_in_the_paper_model(name, start, cars):
    r = T.simulate(*start, cars=cars)
    assert r["result"] == "parked", (name, r["result"], r["t"])
    assert r["m_len"] >= 0.0 and r["m_wid"] >= 0.0
    assert abs(r["herr_deg"]) < 6.0
    assert r["t"] < 30.0


def test_a_car_right_behind_the_bay_is_parked_by_reversing_in():
    r = T.simulate(-16.0, -3.1, 0.0, cars=(T.car_box(1),))
    assert r["result"] == "parked" and r["reverse_steps"] > 20
    forward = T.simulate(-16.0, -3.1, 0.0)
    assert forward["reverse_steps"] == 0


def test_boxed_in_the_teacher_stops_rather_than_forcing_it():
    # cars right behind AND in the bay ahead: the arena never builds this, but
    # the teacher must fail safely if it meets it
    r = T.simulate(-16.0, -3.1, 0.0, cars=(T.car_box(1), T.car_box(-1)))
    assert r["result"] != "collision"


def test_far_from_the_bay_the_teacher_holds_the_lane():
    steer, pedal, gear = T.teacher_action(-20.0, T.LANE_Y, 0.0, 2.5, [1.0, 1.0, 1.0, 1.0])
    assert abs(steer) < 0.05 and pedal > 0 and gear == 0.0
    # drifted toward the bay line while far out: steers back toward the lane (negative = toward -ay)
    steer, _, _ = T.teacher_action(-20.0, T.LANE_Y + 0.6, 0.0, 2.5, [1.0, 1.0, 1.0, 1.0])
    assert steer < 0


def test_a_car_beside_or_blocking_keeps_the_hold_inside_the_turn_zone():
    free = T.teacher_action(-8.0, T.LANE_Y, 0.0, 1.8, [1.0, 1.0, 1.0, 1.0])
    assert free[0] > 0.2, "with nothing around, inside 12 m it turns toward the bay"
    beside = T.teacher_action(-8.0, T.LANE_Y, 0.0, 1.8, [1.0, 1.0, 1.0, 0.2])
    assert abs(beside[0]) < 0.05, "a car 2 m off the right flank: keep holding the lane"
    blocked = T.teacher_action(-11.0, T.LANE_Y, 0.0, 1.8, [1.0, 0.5, 1.0, 1.0])
    assert abs(blocked[0]) < 0.05, "a car 5 m ahead-right at the turn start: the bay is blocked, hold"


def test_reversing_the_lateral_correction_is_mirrored_and_the_wheel_sign_flips():
    # backing straight along the bay line, tail needs to move toward +ay (van is at ay<0):
    # the nose must swing toward -ay, which while reversing takes a POSITIVE wheel
    steer = T._steer_to_path(0.0, -0.5, 0.0, 0.0, 0.0, reverse=True, lookahead=1.2, k_lateral=0.8)
    assert steer > 0
    fwd = T._steer_to_path(0.0, -0.5, 0.0, 0.0, 0.0, reverse=False, lookahead=1.2, k_lateral=0.8)
    assert fwd > 0                        # going forward the same correction is also a right turn


def test_actions_stay_in_the_brains_ranges():
    for ax in (-30.0, -12.0, -6.0, -1.0, 3.0, 8.0):
        for ay in (-3.1, -1.5, -0.2, 0.3):
            for h in (-0.5, 0.0, 0.5):
                for v in (-1.0, 0.0, 2.0):
                    s, p, g = T.teacher_action(ax, ay, h, v, [1.0, 0.6, 1.0, 0.3])
                    assert -1.0 <= s <= 1.0 and -1.0 <= p <= 1.0 and g in (0.0, 1.0)


def test_car_box_sits_in_the_neighbouring_bay():
    cx, cy, yaw, hl, hw = T.car_box(2)
    assert (cx, cy, yaw) == (-14.0, 0.0, 0.0) and hl == 2.35
    pts = T.box_points(*T.car_box(1))
    assert len(pts) == 9 and min(x for x, _ in pts) == pytest.approx(-9.35)


def test_the_teacher_can_read_the_brains_own_observation():
    from rl.parking_math import observation
    ax, ay, herr, speed = -14.0, -3.1, 0.05, 2.0
    obs = observation(0.0 + ax, ay, herr, speed, 0.0, 0.0, 0.0, 0.0) + [1.0, 0.6, 1.0, 0.4]
    direct = T.teacher_action(ax, ay, herr, speed, [1.0, 0.6, 1.0, 0.4])
    via_obs = T.teacher_from_obs(obs)
    assert all(abs(x - y) < 1e-6 for x, y in zip(direct, via_obs))
