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
    assert done and info["result"] == "collision" and r == COLLISION_PENALTY
    lost, _, _ = step_outcome(-30.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 1.0, False)
    over, _, _ = step_outcome(6.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0, False)
    late, _, _ = step_outcome(-3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 31.0, False)
    assert r < min(lost, over, late), "nothing may be cheaper than hitting something"


def test_a_crash_no_longer_dwarfs_the_prize():
    """Round 5: -200 against a 100-220 prize froze the van a metre short of the
    box — the last metre paid +8 and risked -200. Worst outcome, yes; ten times
    the reward for trying, no."""
    assert 50.0 < -COLLISION_PENALTY < 100.0


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
    assert inside_moving > outside_moving, "still pays — deliberately only a little"


def test_crawling_the_box_never_beats_the_smallest_park():
    """The van has no reverse gear, so the longest it can stay inside the box
    while still 'moving' is crawling its full length at the parked-speed limit.
    Even that must earn less than the poorest genuine park (100), or the day it
    learns to enter the box is the day it learns never to stop."""
    steps_inside = SLOT_LEN / SUCCESS_SPEED / 0.1        # 0.1 s per step
    assert INSIDE_PAY * steps_inside < 100.0


def test_getting_lined_up_pays_and_getting_crooked_costs():
    """Round 5 paid only for distance, so the van reached the bay and hovered
    there crooked. Straightening and centring must pay on their own."""
    crooked = lateral_error(0.4, math.radians(5))
    straight = lateral_error(0.0, 0.0)
    assert crooked > straight == 0.0
    better, _, _ = step_outcome(-2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, False,
                                align_prev=crooked)
    worse, _, _ = step_outcome(-2.0, 0.4, math.radians(5), 1.0, 2.0, 0.0, 0.0,
                               3.0, False, align_prev=straight)
    same, _, _ = step_outcome(-2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, False,
                              align_prev=straight)
    assert better > same > worse
    assert abs(better - same - ALIGN_PAY * crooked) < 1e-9


def test_hovering_lined_up_earns_nothing_extra():
    """Both shaping terms are differences: sitting in a good pose must pay the
    same as sitting in a bad one, so the only way to earn is to improve."""
    # both poses 4 m short of the box, so neither is inside and neither parks
    good, _, _ = step_outcome(-4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 3.0, False,
                              align_prev=0.0)
    bad_pose = lateral_error(0.5, math.radians(8))
    bad, _, _ = step_outcome(-4.0, 0.5, math.radians(8), 0.0, math.hypot(4.0, 0.5),
                             0.0, 0.0, 3.0, False, align_prev=bad_pose)
    assert abs(good - bad) < 1e-9


def test_alignment_shaping_is_off_unless_asked_for():
    with_it, _, _ = step_outcome(-2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, False)
    off, _, _ = step_outcome(-2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, False,
                             align_prev=None)
    assert with_it == off


# ---------------------------------------------------------------------------
# Round 4: honest distance limits, honest time, and a ladder that moves
# ---------------------------------------------------------------------------
import random

from rl.parking_math import (ALIGN_PAY, APPROACH_PAY, COLLISION_PENALTY,
                             INSIDE_PAY, SLOT_LEN, SUCCESS_SPEED, Curriculum,
                             LANE_START_M, OUT_OF_BOUNDS_M, bounds_for,
                             lateral_error, spawn_pose, timeout_for)


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
    assert Curriculum.LEVELS[-1] == 0.0


def test_no_rung_ever_spawns_the_van_already_parked():
    """The bug that wasted rounds 1-4. A p=1.0 rung puts the van inside the box
    at a standstill, where doing nothing scores ~220 at once and driving would
    only overshoot. Measured 2026-09-03: 10/10 parked at p=1.0, 0/30 at p=0.0 —
    the student had learned to sit still and nothing else."""
    assert max(Curriculum.LEVELS) <= 0.8, "the easiest rung must still need driving"
    lane, slot = (-16.0, -3.0, 0.0), (0.0, 0.0, 0.0)
    for p in Curriculum.LEVELS:
        x, y, _ = spawn_pose(p, *lane, *slot)
        assert math.hypot(x, y) > 3.0, f"rung p={p} starts too close to be a lesson"


def test_approaching_the_slot_is_worth_more_than_standing_still():
    """Standing still for a 30 s episode costs about -45. Closing the full
    distance must beat that comfortably, or not moving stays the rational play."""
    approach = APPROACH_PAY * 16.3
    idle_cost = 0.05 * 300 + 30.0
    assert approach > idle_cost * 2, "driving must clearly beat doing nothing"


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
    assert all(c.next_p(rng) == Curriculum.LEVELS[0] for _ in range(20))


def test_revision_always_picks_an_easier_rung_than_the_focus():
    c = Curriculum(start_level=3, review_p=1.0)
    rng = random.Random(1)
    focus = c.focus_p
    for _ in range(50):
        p = c.next_p(rng)
        assert p != focus and p > focus, "revision must be easier, never harder"


def test_the_ladder_log_says_why_it_moved():
    """A move is the only thing ever logged, and it also empties the window —
    so the line must report the score that triggered it, not an empty window."""
    c = Curriculum(window=4)
    for _ in range(4):
        c.record(c.focus_p, True)
    line = c.describe()
    assert "recent at this level" not in line, "the window is empty at this moment"
    assert "no attempts yet" not in line, "but it is not a fresh ladder either"
    assert "promoted" in line and "100% parked" in line and "over 4 attempts" in line


def test_the_ladder_log_says_when_it_eased_back():
    c = Curriculum(window=4, start_level=2)
    for _ in range(4):
        c.record(c.focus_p, False)
    assert "eased back" in c.describe()


def test_the_ladder_names_a_rung_the_same_way_the_flag_sets_it():
    """describe() counts rungs from 1 for humans; --start-level counts from 0.
    Reading a number out of the log and passing it back must not land one rung
    too hard, so the line has to state the flag's own number."""
    for level in range(len(Curriculum.LEVELS)):
        line = Curriculum(start_level=level).describe()
        assert f"--start-level {level}" in line
        assert f"level {level + 1}/{len(Curriculum.LEVELS)}" in line


def test_a_fresh_ladder_does_not_pretend_to_have_results():
    assert "no attempts yet" in Curriculum().describe()


from rl.parking_math import bay_is_clear


def _world_point(ax, ay, slot):
    """Slot-frame (along, across) -> world (x, y) for a slot at (x, y, yaw)."""
    sx, sy, syaw = slot
    c, s = math.cos(syaw), math.sin(syaw)
    return (sx + c * ax - s * ay, sy + s * ax + c * ay)


def test_an_empty_map_leaves_every_bay_clear():
    assert bay_is_clear(10.0, 5.0, 0.4, [])


def test_a_vehicle_in_the_bay_itself_rules_it_out():
    slot = (30.0, -8.0, 1.1)
    assert not bay_is_clear(*slot, [[_world_point(0.0, 0.0, slot)]])


def test_the_measured_crash_spots_all_rule_a_bay_out():
    """Round-6 exam, seed 2026: where the van actually touched static vehicles,
    in the slot frame. Every one of these must be inside the keep-clear zone."""
    slot = (-12.0, 40.0, 2.6)
    for along, across in ((-6.34, -0.26), (-14.20, -1.56), (0.05, -0.03),
                          (-4.73, -0.14), (-11.80, -0.90), (-15.37, -2.29)):
        assert not bay_is_clear(*slot, [[_world_point(along, across, slot)]]), (along, across)


def test_vehicles_well_ahead_or_across_the_road_do_not_matter():
    slot = (0.0, 0.0, 0.0)
    ahead = [[_world_point(9.0, 0.0, slot)]]        # the bay after next, in front
    far_side = [[_world_point(-5.0, 7.5, slot)]]    # other side of the road
    assert bay_is_clear(*slot, ahead) and bay_is_clear(*slot, far_side)


def test_one_corner_inside_is_enough():
    slot = (0.0, 0.0, 0.0)
    outline = [_world_point(-25.0, 0.0, slot), _world_point(-17.5, 0.0, slot)]  # centre out, corner in
    assert not bay_is_clear(*slot, [outline])


from rl.parking_math import (FEELER_REACH_M, FEELER_SECTORS, feelers, neighbour_pose,
                             neighbour_behind_fits, NEIGHBOUR_BAY_PITCH_M)


def test_feelers_read_full_reach_when_nothing_is_near():
    assert feelers(3.0, -4.0, 1.2, []) == [1.0] * len(FEELER_SECTORS)
    far = [(3.0 + 30.0, -4.0)]
    assert feelers(3.0, -4.0, 0.0, far) == [1.0] * len(FEELER_SECTORS)


def test_something_dead_ahead_shows_in_both_ahead_feelers():
    ahead_left, ahead_right, left, right = feelers(0.0, 0.0, 0.0, [(3.0, 0.0)])
    assert abs(ahead_left - 0.3) < 1e-9 and abs(ahead_right - 0.3) < 1e-9
    assert left == 1.0 and right == 1.0


def test_side_feelers_see_only_their_side_and_nothing_sees_behind():
    al, ar, left, right = feelers(0.0, 0.0, 0.0, [(0.0, 5.0)])     # +y = the van's right
    assert right == 0.5 and left == 1.0 and al == 1.0 and ar == 1.0
    al, ar, left, right = feelers(0.0, 0.0, 0.0, [(0.0, -2.0)])
    assert left == 0.2 and right == 1.0
    assert feelers(0.0, 0.0, 0.0, [(-4.0, 0.0)]) == [1.0] * 4, "no reverse gear, no rear feeler"


def test_feelers_turn_with_the_van():
    """The same world point must read as 'ahead' when the van faces it and as
    'right' when the van has turned 90 degrees left of it."""
    pt = [(4.0, 0.0)]
    assert feelers(0.0, 0.0, 0.0, pt)[0] < 1.0
    turned = feelers(0.0, 0.0, math.radians(-90), pt)      # now the point is on the van's right
    assert turned[3] == 0.4 and turned[0] == 1.0 and turned[1] == 1.0


def test_the_nearest_point_wins_in_a_sector():
    al, ar, _, _ = feelers(0.0, 0.0, 0.0, [(6.0, 0.5), (2.0, 0.5), (9.0, 0.5)])
    assert abs(ar - math.hypot(2.0, 0.5) / FEELER_REACH_M) < 1e-9


def test_neighbour_bays_sit_one_slot_length_along_the_slot():
    x, y, yaw = neighbour_pose(10.0, 20.0, math.radians(90), -1)
    assert abs(x - 10.0) < 1e-9 and abs(y - (20.0 - NEIGHBOUR_BAY_PITCH_M)) < 1e-9
    assert yaw == math.radians(90)


def test_a_car_behind_is_only_placed_when_the_van_starts_clear_of_it():
    """The bay behind spans -9.3..-4.7 m along; a van starting at 11 m has its
    front bumper at about -8.7 m, inside that car. It must start farther back."""
    assert not neighbour_behind_fits(11.0)
    assert neighbour_behind_fits(13.5)


from rl.parking_math import IDLE_COST, neighbour_ahead_fits


def test_standing_still_outside_the_box_costs_more_than_creeping():
    """Round 7a froze within 1,200 attempts once a car ahead made overshoots
    into crashes: sitting still was the cheap option. It must not be."""
    still, _, _ = step_outcome(-4.0, -0.7, 0.0, 0.0, 4.06, 0.0, 0.0, 3.0, False)
    creeping, _, _ = step_outcome(-4.0, -0.7, 0.0, 0.5, 4.06, 0.0, 0.0, 3.0, False)
    assert creeping - still >= IDLE_COST - 1e-9


def test_a_whole_attempt_of_standing_still_costs_about_a_crash():
    """Easiest rung: ~200 steps. (0.05 + IDLE_COST) * 200 + 30 should land near
    the -80 a crash costs, so freezing is never the safe play."""
    total = -(0.05 + IDLE_COST) * 200 - 30.0
    assert total <= -75.0


def test_being_inside_the_box_is_exempt_from_the_idle_charge():
    inside_still, _, _ = step_outcome(0.0, 0.0, 0.0, 0.31, 0.5, 0.0, 0.0, 3.0, False)
    outside_still, _, _ = step_outcome(-4.0, -0.7, 0.0, 0.1, 4.06, 0.0, 0.0, 3.0, False)
    assert inside_still > outside_still


def test_a_car_ahead_only_appears_once_there_is_an_approach_to_practise():
    assert not neighbour_ahead_fits(4.1)
    assert not neighbour_ahead_fits(6.5)
    assert neighbour_ahead_fits(9.0)


from rl.parking_math import proximity_penalty, PROXIMITY_CLOSE, NEIGHBOUR_BEHIND_BAYS


def test_nothing_near_costs_nothing():
    assert proximity_penalty([1.0, 1.0, 1.0, 1.0]) == 0.0
    assert proximity_penalty([PROXIMITY_CLOSE, 0.5, 1.0, 1.0]) == 0.0


def test_the_warning_grows_as_something_gets_closer():
    far = proximity_penalty([0.18, 1.0, 1.0, 1.0])
    near = proximity_penalty([0.05, 1.0, 1.0, 1.0])
    assert 0.0 > far > near


def test_the_warning_reaches_the_reward_before_any_crash():
    """Same step, same pose: with a feeler reading 'very close' the step must
    score lower than with nothing near, while the attempt goes on."""
    clear, done1, _ = step_outcome(-8.0, -2.0, 0.1, 1.5, 8.3, 0.0, 0.0, 4.0, False,
                                   feeler_readings=[1.0, 1.0, 1.0, 1.0])
    close, done2, _ = step_outcome(-8.0, -2.0, 0.1, 1.5, 8.3, 0.0, 0.0, 4.0, False,
                                   feeler_readings=[1.0, 0.08, 1.0, 0.1])
    assert not done1 and not done2 and close < clear


def test_the_practice_car_sits_two_bays_back():
    """One bay back was a wall: parallel bays, no reverse gear, 0 parks in ~1,800
    tries. Two bays back is where the pull-in path really clips a car."""
    assert NEIGHBOUR_BEHIND_BAYS == 2
    x, y, _ = neighbour_pose(0.0, 0.0, 0.0, -NEIGHBOUR_BEHIND_BAYS)
    assert x == -14.0 and y == 0.0


def test_side_feelers_cover_the_rear_quarters():
    """Round 7c's crashes were rear-quarter clips the feelers never saw."""
    al, ar, left, right = feelers(0.0, 0.0, 0.0, [(-2.0, 2.0)])       # behind-right
    assert right < 1.0 and left == 1.0 and al == 1.0 and ar == 1.0
    al, ar, left, right = feelers(0.0, 0.0, 0.0, [(-2.0, -2.0)])      # behind-left
    assert left < 1.0 and right == 1.0
    assert feelers(0.0, 0.0, 0.0, [(-4.0, 0.0)]) == [1.0] * 4, "straight behind still unseen"


from rl.parking_math import Stages, ALIGN_ZONE_M, REVERSE_COST


def test_progress_along_the_bay_ignores_a_sideways_cut_in():
    """Round 7: centre-distance progress paid for cutting in early - the
    crash path. Along the bay, closing sideways earns nothing by itself."""
    cut_in, _, _ = step_outcome(-12.0, -1.0, 0.0, 1.0, 12.4, 0.0, 0.0, 3.0, False, ax_prev=-12.0)
    ahead, _, _ = step_outcome(-11.0, -3.0, 0.0, 1.0, 12.4, 0.0, 0.0, 3.0, False, ax_prev=-12.0)
    assert ahead > cut_in + 5.0


def test_reversing_into_the_slot_is_paid_like_driving_into_it():
    fwd, _, _ = step_outcome(-2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 3.0, False, ax_prev=-3.0)
    rev, _, _ = step_outcome(2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 3.0, False, ax_prev=3.0, reversing=True)
    assert abs(fwd - rev - REVERSE_COST) < 1e-9


def test_lining_up_only_pays_near_the_bay():
    far, _, _ = step_outcome(-14.0, -2.0, 0.1, 1.0, 14.1, 0.0, 0.0, 3.0, False, ax_prev=-14.0, align_prev=2.5)
    near, _, _ = step_outcome(-6.0, -2.0, 0.1, 1.0, 6.3, 0.0, 0.0, 3.0, False, ax_prev=-6.0, align_prev=2.5)
    assert near > far + 1.0 and ALIGN_ZONE_M < 10.0


def test_a_reversing_parker_may_pull_past_the_slot():
    _, done_fwd, info_fwd = step_outcome(8.0, -3.0, 0.0, 1.0, 8.5, 0.0, 0.0, 3.0, False)
    _, done_rev, info_rev = step_outcome(8.0, -3.0, 0.0, 1.0, 8.5, 0.0, 0.0, 3.0, False, overshoot_m=12.0)
    assert done_fwd and info_fwd["result"] == "overshoot"
    assert not done_rev


def test_hazard_stages_unlock_with_success_and_fall_back_with_failure():
    st = Stages(window=4)
    assert st.level == 0
    for _ in range(4):
        st.record(True)
    assert st.level == 1
    for _ in range(4):
        st.record(False)
    assert st.level == 0
    top = Stages(window=4, start=2)
    for _ in range(8):
        top.record(True)
    assert top.level == 2 and "right behind" in top.describe()
