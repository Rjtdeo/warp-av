"""The RL parking reward must be honest: students are brilliant cheaters."""
import math

from rl.parking_math import (observation, to_slot_frame, van_corners_in_slot,
                             step_outcome)


def test_slot_frame_is_bay_independent():
    # the SAME relative pose at two differently-facing bays must produce
    # identical observations (this is what makes the skill transferable)
    ax, ay, herr = -2.2, 0.4, 0.07
    obs = []
    for sx, sy, syaw in ((12.0, 6.0, 0.1), (-28.0, 89.0, 2.2)):
        c, s = math.cos(syaw), math.sin(syaw)
        vx = sx + c * ax - s * ay
        vy = sy + s * ax + c * ay
        obs.append(observation(vx, vy, syaw + herr, 2.0, 0.0, sx, sy, syaw))
    for x, y in zip(*obs):
        assert abs(x - y) < 1e-6


def test_perfect_park_scores_high():
    r, done, info = step_outcome(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 8.0, False)
    assert done and info["result"] == "parked"
    assert r > 180.0, "centred, straight, stopped = the full prize"


def test_crooked_or_offset_park_scores_less():
    perfect, _, _ = step_outcome(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 8.0, False)
    offset, done, info = step_outcome(0.0, 0.24, 0.0, 0.0, 0.5, 0.0, 0.0, 8.0, False)
    assert done and info["result"] == "parked" and offset < perfect - 30
    crooked, done2, info2 = step_outcome(0.0, 0.0, math.radians(5), 0.0, 0.5,
                                         0.0, 0.0, 8.0, False)
    assert done2 and crooked < perfect


def test_stopping_outside_the_box_is_not_parked():
    # van centre 0.4 m across: a corner pokes out -> keep trying, no prize
    r, done, info = step_outcome(0.0, 0.42, 0.0, 0.0, 0.5, 0.0, 0.0, 8.0, False)
    assert not done and info["result"] == "driving"


def test_ramming_in_fast_is_not_parked():
    r, done, info = step_outcome(0.0, 0.0, 0.0, 2.5, 0.5, 0.0, 0.0, 8.0, False)
    assert not done, "must be stopped, not flying through the box"


def test_collision_is_the_worst_outcome():
    r, done, info = step_outcome(0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 8.0, True)
    assert done and info["result"] == "collision" and r <= -200.0


def test_progress_pays_and_wandering_costs():
    closer, _, _ = step_outcome(4.0, 1.0, 0.1, 1.0, 5.0, 0.1, 0.1, 3.0, False)
    away, _, _ = step_outcome(6.0, 1.0, 0.1, 1.0, 5.0, 0.1, 0.1, 3.0, False)
    assert closer > 0 > away


def test_corner_math_matches_geometry():
    inside, m_len, m_wid = van_corners_in_slot(0.0, 0.0, 0.0)
    assert inside and abs(m_len - 1.15) < 0.01 and abs(m_wid - 0.27) < 0.01
    inside2, _, m_wid2 = van_corners_in_slot(0.0, 0.3, 0.0)
    assert not inside2 and m_wid2 < 0.0


def test_spawn_distance_does_not_end_the_episode():
    """Smoke-test bug: the van STARTS ~11.5 m from the slot — that must be a
    normal driving step, not instant disqualification."""
    r, done, info = step_outcome(-11.0, -3.2, 0.0, 0.0, 11.5, 0.0, 0.0, 0.1, False)
    assert not done and info["result"] == "driving"


def test_driving_past_the_slot_ends_the_attempt():
    r, done, info = step_outcome(5.5, 0.5, 0.0, 2.0, 5.0, 0.0, 0.0, 6.0, False)
    assert done and info["result"] == "overshoot"


def test_being_inside_the_box_pays_even_while_moving():
    """Round-2 curriculum fix: entering the box must itself be rewarding —
    round 1's student never tasted success in 1,347 attempts."""
    inside_moving, done, _ = step_outcome(0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 5.0, False)
    outside_moving, _, _ = step_outcome(4.0, 0.0, 0.0, 1.0, 4.5, 0.0, 0.0, 5.0, False)
    assert not done
    assert inside_moving > outside_moving + 1.0
