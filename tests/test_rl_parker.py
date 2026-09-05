"""The learned parker inside the van: hand-over, action mapping, hand-back."""
import math
import time

import numpy as np

from warp_av.planning.rl_parker import (RLParker, HANDOVER_M, STEER_GAIN, THROTTLE_GAIN,
                                        REVERSE_MAX_SPEED, box_outline_points)


class Stub:
    def __init__(self, action):
        self.action = np.array(action, dtype=np.float32)
        self.seen = []

    def predict(self, obs, deterministic=True):
        self.seen.append(np.array(obs))
        return self.action, None


SLOT = dict(x=100.0, y=50.0, yaw=0.0, length=7.0, width=2.5)


def test_takes_over_only_within_reach_and_only_once():
    p = RLParker(model=Stub([0.0, 0.5]))
    assert not p.should_take_over(SLOT, 100.0 - 20.0, 50.0)
    assert p.should_take_over(SLOT, 100.0 - 15.0, 50.0)
    p.act(85.0, 50.0, 0.0, 1.0, SLOT)
    assert p.engaged and not p.should_take_over(SLOT, 85.0, 50.0)


def test_action_mapping_matches_the_training_arena():
    p = RLParker(model=Stub([0.5, 0.5]))
    out = p.act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert abs(out["steering"] - 0.5 * STEER_GAIN) < 1e-6
    assert abs(out["throttle"] - 0.5 * THROTTLE_GAIN) < 1e-6 and out["brake"] == 0.0
    p2 = RLParker(model=Stub([-0.2, -0.8]))
    out = p2.act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert out["throttle"] == 0.0 and abs(out["brake"] - 0.8) < 1e-6   # float32 brain output
    fast = RLParker(model=Stub([0.0, 1.0]))
    assert fast.act(90.0, 47.0, 0.0, 3.5, SLOT)["throttle"] == 0.0, "no throttle above the trained speed cap"


def test_the_brain_sees_the_five_numbers_in_the_slot_frame():
    stub = Stub([0.0, 0.0]); p = RLParker(model=stub)
    p.act(90.0, 47.0, 0.0, 2.0, SLOT)
    obs = stub.seen[0]
    assert obs.shape == (5,)
    assert abs(obs[0] - (-10.0 / 15.0)) < 1e-6 and abs(obs[1] - (-3.0 / 6.0)) < 1e-6
    assert abs(obs[3] - 2.0 / 5.0) < 1e-6


def test_inside_and_stopped_is_parked_and_hands_back():
    p = RLParker(model=Stub([0.0, 0.0]))
    out = p.act(100.0, 50.0, 0.0, 0.1, SLOT)
    assert out["parked"] and out["brake"] == 1.0 and p.done
    assert "parked by the learned parker" in out["reason"]


def test_overshoot_and_timeout_hand_back_to_the_rules():
    p = RLParker(model=Stub([0.0, 0.5]))
    p.act(99.0, 50.3, 0.0, 1.0, SLOT)                 # reached the bay ...
    out = p.act(106.0, 50.0, 0.0, 1.0, SLOT)          # ... then 6 m past the centre
    assert out["gave_up"] and p.done and "overshot" in out["reason"]
    q = RLParker(model=Stub([0.0, 0.5]))
    q.act(90.0, 47.0, 0.0, 1.0, SLOT)
    q.active_s = 100.0                                # a hundred seconds AT THE WHEEL
    out = q.act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert out["gave_up"] and "not parked after" in out["reason"]


def test_a_target_that_jumps_behind_the_van_is_not_an_overshoot():
    """Arm C, run 2: a re-scan moved the target and the parker reported an
    83 m overshoot on the spot. A new target restarts the approach."""
    p = RLParker(model=Stub([0.0, 0.5]))
    p.act(90.0, 47.0, 0.0, 1.0, SLOT)
    far_behind = dict(SLOT, x=10.0)
    out = p.act(90.0, 47.0, 0.0, 1.0, far_behind)
    assert not out["gave_up"] and not p.done


def test_time_held_by_an_obstacle_does_not_count_against_the_parker():
    p = RLParker(model=Stub([0.0, 0.5]))
    p.act(90.0, 47.0, 0.0, 1.0, SLOT)
    p._last_act -= 30.0                               # 30 s with no call: an obstacle stop
    p.act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert p.active_s < 1.0



# ---------------- the round-8 brain: four feelers and a reverse gear ----------------

def test_a_nine_input_brain_gets_the_feelers_from_the_vans_surroundings():
    stub = Stub([0.0, 0.5, 0.0]); p = RLParker(model=stub, n_obs=9, n_act=3)
    # a parked car 2 m ahead and 1.5 m to the right (van frame: x forward, y right)
    out = p.act(90.0, 47.0, 0.0, 2.0, SLOT, obstacle_points=[(2.0, 1.5)])
    obs = stub.seen[0]
    assert obs.shape == (9,)
    assert abs(obs[0] - (-10.0 / 15.0)) < 1e-6           # the five old numbers first
    ahead_left, ahead_right, left, right = obs[5:9]
    assert abs(ahead_right - 0.25) < 1e-6                 # 2.5 m away, on a 10 m reach
    assert ahead_left == 1.0 and left == 1.0 and right == 1.0
    assert "nearest ahead-right 2.5 m" in out["reason"]
    assert out["reverse"] is False


def test_nothing_around_reads_as_all_clear():
    stub = Stub([0.0, 0.5, 0.0]); p = RLParker(model=stub, n_obs=9, n_act=3)
    p.act(90.0, 47.0, 0.0, 2.0, SLOT, obstacle_points=[])
    assert list(stub.seen[0][5:9]) == [1.0, 1.0, 1.0, 1.0]
    p.act(90.0, 47.0, 0.0, 2.0, SLOT)                    # no points given at all
    assert list(stub.seen[1][5:9]) == [1.0, 1.0, 1.0, 1.0]


def test_the_third_control_is_the_reverse_gear_with_the_arenas_slow_cap():
    p = RLParker(model=Stub([0.0, 0.5, 0.9]), n_obs=9, n_act=3)
    out = p.act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert out["reverse"] is True and abs(out["throttle"] - 0.5 * THROTTLE_GAIN) < 1e-6
    fast = p.act(90.0, 47.0, 0.0, REVERSE_MAX_SPEED + 0.1, SLOT)
    assert fast["reverse"] is True and fast["throttle"] == 0.0, "reversing is capped at 1.5 m/s"
    fwd = RLParker(model=Stub([0.0, 0.5, 0.2]), n_obs=9, n_act=3).act(90.0, 47.0, 0.0, 1.0, SLOT)
    assert fwd["reverse"] is False


def test_the_round_six_brain_still_sees_five_numbers_and_never_reverses():
    stub = Stub([0.0, 0.5]); p = RLParker(model=stub)
    out = p.act(90.0, 47.0, 0.0, 2.0, SLOT, obstacle_points=[(2.0, 1.5)])
    assert stub.seen[0].shape == (5,) and out["reverse"] is False
    assert "round6" in p.describe() or "5 inputs" in p.describe()


def test_the_brains_shape_is_read_from_the_model_itself():
    from types import SimpleNamespace
    fake = Stub([0.0, 0.5, 0.0])
    fake.observation_space = SimpleNamespace(shape=(9,))
    fake.action_space = SimpleNamespace(shape=(3,))
    p = RLParker(model=fake)
    p.act(90.0, 47.0, 0.0, 2.0, SLOT, obstacle_points=[(2.0, -1.5)])
    assert p.n_obs == 9 and p.n_act == 3 and fake.seen[0].shape == (9,)
    assert abs(fake.seen[0][5] - 0.25) < 1e-6              # ahead-LEFT this time (y negative = left)
    assert "9 inputs incl. 4 obstacle feelers, 3 controls incl. reverse gear" in p.describe()


def test_a_reversing_brain_may_pull_past_the_slot_before_giving_up():
    p = RLParker(model=Stub([0.0, 0.5, 0.0]), n_obs=9, n_act=3)
    p.act(99.0, 50.0, 0.0, 1.0, SLOT)                     # been near the slot
    out = p.act(108.0, 50.0, 0.0, 1.0, SLOT)              # 8 m past: fine for a reversing brain
    assert not out["gave_up"]
    out = p.act(113.0, 50.0, 0.0, 1.0, SLOT)              # 13 m past: that is an overshoot
    assert out["gave_up"] and "overshot" in out["reason"]


def test_box_outline_has_centre_corners_and_edge_midpoints():
    pts = box_outline_points(10.0, 5.0, 0.0, 2.0, 1.0)
    assert len(pts) == 9 and pts[0] == (10.0, 5.0)
    assert (12.0, 6.0) in [(round(x, 6), round(y, 6)) for x, y in pts]     # a corner
    assert (8.0, 5.0) in [(round(x, 6), round(y, 6)) for x, y in pts]      # rear midpoint
    turned = box_outline_points(0.0, 0.0, math.pi / 2, 2.0, 1.0)
    xs = [round(x, 6) for x, _ in turned]; ys = [round(y, 6) for _, y in turned]
    assert max(ys) == 2.0 and max(xs) == 1.0                                # length now along y


def test_the_hand_over_distance_is_a_setting():
    far = RLParker(model=Stub([0.0, 0.5, 0.0]), n_obs=9, n_act=3, handover_m=30.0)
    assert not far.should_take_over(SLOT, 100.0 - 32.0, 50.0)
    assert far.should_take_over(SLOT, 100.0 - 29.0, 50.0)
    default = RLParker(model=Stub([0.0, 0.5]))
    assert default.handover_m == HANDOVER_M and not default.should_take_over(SLOT, 100.0 - 29.0, 50.0)



def test_hand_over_needs_the_van_lined_up_with_the_bay_when_the_heading_is_known():
    from warp_av.planning.rl_parker import ALIGN_SIDE_M
    p = RLParker(model=Stub([0.0, 0.5, 0.0]), n_obs=9, n_act=3, handover_m=30.0)
    assert p.should_take_over(SLOT, 100.0 - 25.0, 50.0 - 3.1, 0.0)          # in the lane, straight, 25 m out
    assert not p.should_take_over(SLOT, 100.0 - 25.0, 50.0 - 10.5, 0.0)    # 10.5 m off the line (run 3)
    assert not p.should_take_over(SLOT, 100.0 - 25.0, 50.0 - 3.1, math.radians(60))   # pointing across the road
    assert not p.should_take_over(SLOT, 100.0 + 5.0, 50.0 - 3.1, 0.0)      # already past the slot
    assert p.should_take_over(SLOT, 100.0 - 25.0, 50.0 - 10.5)            # no heading given: distance only, as before
    assert ALIGN_SIDE_M >= 3.1


def test_a_behaviour_stop_overrides_the_brain_only_for_people_movers_and_the_very_close():
    from warp_av.planning.rl_parker import stop_overrides_brain
    assert stop_overrides_brain("pedestrian", 0.0, 8.0)
    assert stop_overrides_brain("vehicle", 1.2, 8.0)             # moving car
    assert stop_overrides_brain("vehicle", 0.0, 1.5)             # parked, but right on the nose
    assert not stop_overrides_brain("vehicle", 0.0, 8.4)         # the parked car of run 2: brain drives on
    assert not stop_overrides_brain("obstacle", 0.0, 3.0)
    assert stop_overrides_brain("unknown", 0.0, 9.0)             # when perception cannot say, stop wins
