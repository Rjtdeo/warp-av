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
                             SUCCESS_SPEED, OVERSHOOT_M, LATERAL_LOST_M, feelers)

HANDOVER_M = 16.5        # take the wheel when the slot centre is this close
TIMEOUT_S = 45.0         # then give it back if not parked by then
MAX_SPEED = 3.0
STEER_GAIN = 0.8
THROTTLE_GAIN = 0.7
DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rl", "models",
                             "parking_ppo_round6.zip")
ROUND8_MODEL = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rl", "models",
                            "parking_ppo_round8.zip")
ALIGN_SIDE_M = 4.5        # hand over only within this of the bay's line (the arena starts 3.1 m out)
ALIGN_YAW_DEG = 25.0      # ... and roughly heading along it
STOP_OVERRIDE_M = 2.0     # a stationary vehicle closer than this still stops the brain
MOVING_MPS = 0.3          # anything moving faster is not a parked car
REVERSE_MAX_SPEED = 1.5   # the arena's cap when the gear is reverse
REVERSE_OVERSHOOT_M = 12.0  # a reversing brain may pull past the slot first (arena rule)


def stop_overrides_brain(obstacle_type: str, obstacle_speed: float, obstacle_distance: float) -> bool:
    """While the learned parker is at the wheel, does a behaviour-layer stop
    still win? Yes for a pedestrian, for anything moving, or for anything
    within STOP_OVERRIDE_M. A parked car farther out is the brain's business:
    its feelers see it and it was trained around them. Sensor test 2026-09-05
    run 2: the stop for a parked car 8 m ahead froze the van for 200 s."""
    if obstacle_type in ("pedestrian", "unknown"):
        return True
    if obstacle_speed is not None and abs(obstacle_speed) > MOVING_MPS:
        return True
    return obstacle_distance is not None and obstacle_distance < STOP_OVERRIDE_M


def box_outline_points(cx, cy, yaw, half_len, half_wid):
    """Nine points of a box: centre, four corners, four edge midpoints - the
    same outline the arena gives its feelers for a parked car."""
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(cx, cy)]
    for fx, fy in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)):
        lx, ly = fx * half_len, fy * half_wid
        pts.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return pts


class RLParker:
    def __init__(self, model_path: str = DEFAULT_MODEL, model=None, n_obs=None, n_act=None,
                 handover_m: float = HANDOVER_M):
        self.model_path = os.path.abspath(model_path)
        # How far from the slot the brain takes the wheel. 16.5 m is the
        # arena's exam distance; the round-8 brain parked 97% from 29 m with a
        # car on the approach, so 30 m (about where the lidar first finds a
        # bay) is a fair setting for it.
        self.handover_m = float(handover_m)
        self._model = model          # a stub can be injected for tests
        # What this brain expects: 5 inputs / 2 controls (round 6) or 9 inputs
        # (4 obstacle feelers) / 3 controls (reverse gear, round 8). Read from
        # the loaded brain unless given.
        self.n_obs = n_obs
        self.n_act = n_act
        self.reset()

    def describe(self) -> str:
        n_obs = self.n_obs or 5
        n_act = self.n_act or 2
        return (f"{os.path.basename(self.model_path)}: {n_obs} inputs"
                f"{' incl. 4 obstacle feelers' if n_obs >= 9 else ''}, {n_act} controls"
                f"{' incl. reverse gear' if n_act >= 3 else ''}")

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
        if self.n_obs is None:
            space = getattr(self._model, "observation_space", None)
            self.n_obs = int(space.shape[0]) if space is not None and getattr(space, "shape", None) else 5
        if self.n_act is None:
            space = getattr(self._model, "action_space", None)
            self.n_act = int(space.shape[0]) if space is not None and getattr(space, "shape", None) else 2
        return self._model

    @staticmethod
    def distance_to(slot: Dict, x: float, y: float) -> float:
        return math.hypot(slot["x"] - x, slot["y"] - y)

    def should_take_over(self, slot: Dict, x: float, y: float, yaw: float = None) -> bool:
        """Within reach, and - when the heading is known - lined up with the
        bay's line the way the arena's starts are: at most ALIGN_SIDE_M off
        the line and ALIGN_YAW_DEG off its heading. Sensor test 2026-09-05 run 3:
        handed over at 29.8 m on a bend, the brain was 10.5 m off the line and
        gave up on its first step."""
        if self.done or self.engaged or self.distance_to(slot, x, y) > self.handover_m:
            return False
        if yaw is None:
            return True
        ax, ay, herr = to_slot_frame(x, y, yaw, slot["x"], slot["y"], slot["yaw"])
        return ax < 0.0 and abs(ay) <= ALIGN_SIDE_M and abs(herr) <= math.radians(ALIGN_YAW_DEG)

    def act(self, x: float, y: float, yaw: float, speed: float, slot: Dict,
            obstacle_points=None) -> Dict:
        """One step at the wheel. Returns steering/throttle/brake (+ reverse
        flag) plus parked / gave_up flags and a reason.

        obstacle_points: outline points of everything solid nearby, in the
        VAN's frame (x forward, y right, metres) - lidar clusters or actor
        boxes. Only a 9-input brain looks at them (its four feelers)."""
        self._load()
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
        overshoot_m = REVERSE_OVERSHOOT_M if (self.n_act or 2) >= 3 else OVERSHOOT_M
        if ax > overshoot_m and self._closest_ax is not None and self._closest_ax < 3.0:
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

        obs_list = observation(x, y, yaw, speed, self.prev_steer, sx, sy, syaw)
        near = ""
        if (self.n_obs or 5) >= 9:
            # the arena's feelers, measured from the van's centre in its own frame
            f = feelers(0.0, 0.0, 0.0, list(obstacle_points or []))
            obs_list = obs_list + f
            if min(f) < 1.0:
                names = ("ahead-left", "ahead-right", "left", "right")
                i = min(range(4), key=lambda k: f[k])
                near = f", nearest {names[i]} {f[i] * 10.0:.1f} m"
        obs = np.array(obs_list, dtype=np.float32)
        action, _ = self._load().predict(obs, deterministic=True)
        steer = float(np.clip(action[0], -1.0, 1.0))
        pedal = float(np.clip(action[1], -1.0, 1.0))
        gear_rev = bool((self.n_act or 2) >= 3 and len(action) > 2 and float(action[2]) > 0.5)
        cap = REVERSE_MAX_SPEED if gear_rev else MAX_SPEED
        throttle = max(0.0, pedal) * THROTTLE_GAIN if abs(speed) < cap else 0.0
        brake = max(0.0, -pedal)
        self.prev_steer = steer
        return dict(steering=steer * STEER_GAIN, throttle=throttle, brake=brake, reverse=gear_rev,
                    parked=False, gave_up=False,
                    reason=f"learned parker{' (reversing)' if gear_rev else ''}: {-ax:.1f} m to go, "
                           f"{abs(ay):.2f} m across, {math.degrees(herr):.0f} deg{near}")
