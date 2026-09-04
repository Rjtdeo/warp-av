"""The learned parker inside the van: hand-over, action mapping, hand-back."""
import math
import time

import numpy as np

from warp_av.planning.rl_parker import RLParker, HANDOVER_M, STEER_GAIN, THROTTLE_GAIN


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
