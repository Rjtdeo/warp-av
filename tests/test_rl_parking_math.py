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


# ---------------------------------------------------------------------------
# Round 4: honest distance limits, honest time, and a ladder that moves
# ---------------------------------------------------------------------------
import random

from rl.parking_math import (Curriculum, LANE_START_M, OUT_OF_BOUNDS_M,
                             bounds_for, spawn_pose, timeout_for)


def test_far_start_is_not_disqualified_for_barely_moving():
    """Rounds 1-3 bug: a full-distance attempt begins ~16.3 m out and the flat
    18 m limit ended it after under 2 m of drift. The room must be measured
    from where THIS attempt started."""
    start = 16.3
    _, done_old, info_old = step_outcome(-19.0, 0.5, 0.0, 1.0, 18.5, 0.0, 0.0,
                                         3.0, False)
    assert done_old and info_old["result"] == "out_of_bounds", "old flat limit"
    _, done_new, info_new = step_outcome(-19.0, 0.5, 0.0, 1.0, 18.5, 0.0, 0.0,
                                         3.0, False, bounds_m=bounds_for(start))
    assert not done_new and info_new["result"] == "driving"


def test_bounds_never_shrink_below_the_old_floor():
    assert bounds_for(2.0) == OUT_OF_BOUNDS_M
    assert bounds_for(LANE_START_M) > LANE_START_M + 5.0


def test_time_allowed_grows_with_the_distance_to_cover():
    assert timeout_for(0.0) < timeout_for(8.0) < timeout_for(LANE_START_M)
    assert timeout_for(999.0) <= 60.0, "but not unlimited"


def test_spawn_pose_slides_from_the_box_to_the_lane():
    lane = (10.0, 0.0, 0.0)
    slot = (26.0, -3.0, 0.0)
    x1, y1, _ = spawn_pose(1.0, *lane, *slot)
    assert (round(x1, 6), round(y1, 6)) == slot[:2], "p=1 starts in the box"
    x0, y0, _ = spawn_pose(0.0, *lane, *slot)
    assert (round(x0, 6), round(y0, 6)) == lane[:2], "p=0 is the full exam"
    xm, ym, _ = spawn_pose(0.5, *lane, *slot)
    assert lane[0] < xm < slot[0] and slot[1] < ym < lane[1]


def test_the_exam_default_is_the_hardest_rung():
    assert Curriculum.LEVELS[0] == 1.0 and Curriculum.LEVELS[-1] == 0.0


def test_ladder_climbs_when_the_student_keeps_winning():
    c = Curriculum(window=4)
    assert c.focus == 0
    moved = [c.record(c.focus_p, True) for _ in range(4)]
    assert moved[-1] == 1 and c.focus == 1


def test_ladder_eases_back_when_the_student_keeps_losing():
    c = Curriculum(window=4, start_level=2)
    moved = [c.record(c.focus_p, False) for _ in range(4)]
    assert moved[-1] == 1 and c.focus == 1


def test_a_middling_score_holds_the_level():
    c = Curriculum(window=4, start_level=2)
    for parked in (True, False, True, False):
        c.record(c.focus_p, parked)
    assert c.focus == 2


def test_review_runs_never_move_the_ladder():
    """Easy revision attempts must not be counted as mastering the hard rung."""
    c = Curriculum(window=4, start_level=3)
    easier = Curriculum.LEVELS[0]
    for _ in range(20):
        assert c.record(easier, True) is None
    assert c.focus == 3


def test_the_ladder_stops_at_both_ends():
    top = Curriculum(window=4, start_level=len(Curriculum.LEVELS) - 1)
    for _ in range(20):
        top.record(top.focus_p, True)
    assert top.focus == len(Curriculum.LEVELS) - 1
    bottom = Curriculum(window=4)
    for _ in range(20):
        bottom.record(bottom.focus_p, False)
    assert bottom.focus == 0


def test_the_easiest_rung_has_nothing_to_revise():
    c = Curriculum(start_level=0, review_p=1.0)
    rng = random.Random(1)
    assert all(c.next_p(rng) == 1.0 for _ in range(20))


def test_revision_always_picks_an_easier_rung_than_the_focus():
    c = Curriculum(start_level=3, review_p=1.0)
    rng = random.Random(1)
    focus = c.focus_p
    for _ in range(50):
        p = c.next_p(rng)
        assert p != focus and p > focus, "revision must be easier, never harder"
