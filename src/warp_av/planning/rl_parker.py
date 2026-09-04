"""
The learned parker, inside the van.

Round 6's brain learned the last 16 m into a bay: five numbers in (where it
is and how fast, relative to the slot), two out (steer, pedal). Here it takes
the wheel from the hand-written parker once the chosen slot - from the map
or from the lidar - is within reach, and hands it back if it overshoots,
wanders, or runs out of time. The safety supervisor and the behaviour layer's
stops still win: this only runs when the stack is in its parking phase and
not being told to stop for something.

The action mapping is the one the brain trained with (rl/parking_env.py):
steer x 0.8, pedal > 0 is throttle x 0.7 below 3 m/s, pedal < 0 is brake.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, Optional

import numpy as np

from rl.parking_math import (observation, to_slot_frame, van_corners_in_slot,
                             SUCCESS_SPEED, OVERSHOOT_M, LATERAL_LOST_M)

HANDOVER_M = 16.5        # take the wheel when the slot centre is this close
TIMEOUT_S = 45.0         # then give it back if not parked by then
MAX_SPEED = 3.0
STEER_GAIN = 0.8
THROTTLE_GAIN = 0.7
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rl", "models",
                             "parking_ppo_round6.zip")


class RLParker:
    def __init__(self, model_path: str = DEFAULT_MODEL, model=None):
        self.model_path = os.path.abspath(model_path)
        self._model = model          # a stub can be injected for tests
        self.reset()

    def reset(self):
        self.engaged = False
        self.done = False            # parked or gave up: hand-written parker owns the rest
        self.prev_steer = 0.0
        self.t0: Optional[float] = None
        self.active_s = 0.0          # seconds actually at the wheel (stops for obstacles do not count)
        self._last_act: Optional[float] = None
        self._slot_ref = None        # (x, y) of the slot we engaged on
        self._closest_ax = None      # nearest we have been to the slot centre, along the bay
        self.result = ""

    def _load(self):
        if self._model is None:
            from stable_baselines3 import PPO   # slow import: only when first needed
            self._model = PPO.load(self.model_path, device="cpu")
        return self._model

    @staticmethod
    def distance_to(slot: Dict, x: float, y: float) -> float:
        return math.hypot(slot["x"] - x, slot["y"] - y)

    def should_take_over(self, slot: Dict, x: float, y: float) -> bool:
        return (not self.done and not self.engaged
                and self.distance_to(slot, x, y) <= HANDOVER_M)

    def act(self, x: float, y: float, yaw: float, speed: float, slot: Dict) -> Dict:
        """One step at the wheel. Returns steering/throttle/brake plus
        parked / gave_up flags and a reason."""
        now = time.time()
        if self.t0 is None:
            self.t0 = now
        sx, sy, syaw = slot["x"], slot["y"], slot["yaw"]
        if self._slot_ref is None or math.hypot(sx - self._slot_ref[0], sy - self._slot_ref[1]) > 1.0:
            # a new target (a re-scan moved it): start the approach afresh
            self._slot_ref = (sx, sy)
            self._closest_ax = None
            self.prev_steer = 0.0
        if self._last_act is not None and now - self._last_act < 1.0:
            self.active_s += now - self._last_act
        self._last_act = now
        self.engaged = True
        ax, ay, herr = to_slot_frame(x, y, yaw, sx, sy, syaw)
        inside, m_len, m_wid = van_corners_in_slot(ax, ay, herr)
        if self._closest_ax is None or abs(ax) < abs(self._closest_ax):
            self._closest_ax = ax
        elapsed = self.active_s

        if inside and abs(speed) < SUCCESS_SPEED:
            self.done = True
            self.result = (f"parked by the learned parker in {elapsed:.0f} s: margins "
                           f"{m_len:.2f} m front/back, {m_wid:.2f} m side, heading off {math.degrees(herr):.1f} deg")
            return dict(steering=0.0, throttle=0.0, brake=1.0, parked=True, gave_up=False,
                        reason=self.result)
        why = None
        # overshoot only counts once we have actually been near the slot -
        # a target that jumps behind us is not an 83 m overshoot
        if ax > OVERSHOOT_M and self._closest_ax is not None and self._closest_ax < 3.0:
            why = f"overshot the slot by {ax:.1f} m"
        elif abs(ay) > LATERAL_LOST_M:
            why = f"wandered {abs(ay):.1f} m off the slot line"
        elif elapsed > TIMEOUT_S:
            why = f"not parked after {elapsed:.0f} s"
        if why:
            self.done = True
            self.result = f"learned parker gave up ({why}) - hand-written parker resumes"
            return dict(steering=0.0, throttle=0.0, brake=1.0, parked=False, gave_up=True,
                        reason=self.result)

        obs = np.array(observation(x, y, yaw, speed, self.prev_steer, sx, sy, syaw), dtype=np.float32)
        action, _ = self._load().predict(obs, deterministic=True)
        steer = float(np.clip(action[0], -1.0, 1.0))
        pedal = float(np.clip(action[1], -1.0, 1.0))
        throttle = max(0.0, pedal) * THROTTLE_GAIN if speed < MAX_SPEED else 0.0
        brake = max(0.0, -pedal)
        self.prev_steer = steer
        return dict(steering=steer * STEER_GAIN, throttle=throttle, brake=brake,
                    parked=False, gave_up=False,
                    reason=f"learned parker: {-ax:.1f} m to go, {abs(ay):.2f} m across, {math.degrees(herr):.0f} deg")
