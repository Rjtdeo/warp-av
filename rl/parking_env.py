"""
CARLA practice arena for the RL parking student.

Each episode: the van appears on the driving lane ~9-14 m before a randomly
chosen real parking bay, and must end fully inside a 7 m slot there, parallel
and stopped. Runs CARLA in synchronous fast-forward while training and puts
the world back to normal on exit.

IMPORTANT: run this INSTEAD of the main van program, never alongside it —
both would fight over the simulator clock.
"""
from __future__ import annotations

import math
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import carla
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.parking_math import (observation, to_slot_frame, step_outcome,
                             SLOT_LEN, SLOT_WID)

FIXED_DT = 0.1
MAX_SPEED = 3.0


class CarlaParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=2000, seed=7):
        super().__init__()
        self.rng = random.Random(seed)
        self.client = carla.Client(host, port)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        self._original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        self.world.apply_settings(settings)

        self.cmap = self.world.get_map()
        self.bays = self._scan_bays()
        if len(self.bays) < 5:
            raise RuntimeError(f"only {len(self.bays)} usable bays found")
        print(f"[ParkingEnv] {len(self.bays)} practice bays ready")

        bp = self.world.get_blueprint_library().filter("vehicle.mercedes.sprinter")[0]
        bp.set_attribute("role_name", "warp_rl")
        self.van = None
        self.van_bp = bp
        self.col_sensor = None
        self._collided = False

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(5,), dtype=np.float32)
        self.slot = None
        self._t = 0.0
        self._steer_prev = 0.0
        self._dist_prev = 0.0

    # ------------------------------------------------------------------
    def _scan_bays(self):
        """(start_transform, slot_x, slot_y, slot_yaw) per usable bay slot:
        straight driving-lane sections with a Parking/Shoulder lane on the
        right, far from junctions."""
        bays = []
        for wp in self.cmap.generate_waypoints(4.0):
            if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
                continue
            r = wp.get_right_lane()
            if r is None or r.lane_type not in (carla.LaneType.Parking, carla.LaneType.Shoulder):
                continue
            if r.lane_width < 1.8:
                continue
            back = wp.previous(12.0)
            if not back or back[0].is_junction:
                continue
            dyaw = abs((back[0].transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
            if dyaw > 6.0:
                continue                     # approach must be straight
            t = r.transform
            bays.append((wp, t.location.x, t.location.y,
                         math.radians(t.rotation.yaw)))
        return bays

    def _destroy(self):
        for a in (self.col_sensor, self.van):
            try:
                if a is not None:
                    a.destroy()
            except Exception:
                pass
        self.col_sensor = None
        self.van = None

    def _on_col(self, _event):
        self._collided = True

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self._collided = False
        self._t = 0.0
        self._steer_prev = 0.0

        for _ in range(30):
            drive_wp, sx, sy, syaw = self.rng.choice(self.bays)
            # Curriculum: some starts almost at the slot (success reachable
            # by luck -> the prize enters experience), some at exam range.
            dist_back = self.rng.choice([3.0, 4.5, 6.0, 8.0, 10.0, 12.0])
            back = drive_wp.previous(dist_back)
            if not back:
                continue
            start_tf = back[0].transform
            tf = carla.Transform(
                carla.Location(start_tf.location.x, start_tf.location.y,
                               start_tf.location.z + 0.3),
                carla.Rotation(yaw=start_tf.rotation.yaw + self.rng.uniform(-5, 5)))
            self.van = self.world.try_spawn_actor(self.van_bp, tf)
            if self.van is not None:
                self.slot = (sx, sy, syaw)
                break
        if self.van is None:
            raise RuntimeError("could not spawn the practice van anywhere")

        col_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(),
                                                 attach_to=self.van)
        self.col_sensor.listen(self._on_col)

        self.world.tick()
        obs = self._obs()
        ax, ay, _ = self._slot_frame()
        self._dist_prev = math.hypot(ax, ay)
        return np.array(obs, dtype=np.float32), {}

    def _pose(self):
        tf = self.van.get_transform()
        v = self.van.get_velocity()
        return (tf.location.x, tf.location.y, math.radians(tf.rotation.yaw),
                math.hypot(v.x, v.y))

    def _slot_frame(self):
        x, y, yaw, _ = self._pose()
        return to_slot_frame(x, y, yaw, *self.slot)

    def _obs(self):
        x, y, yaw, speed = self._pose()
        return observation(x, y, yaw, speed, self._steer_prev, *self.slot)

    def step(self, action):
        steer = float(np.clip(action[0], -1, 1))
        accel = float(np.clip(action[1], -1, 1))
        _, _, _, speed = self._pose()
        throttle = max(0.0, accel) * 0.7 if speed < MAX_SPEED else 0.0
        brake = max(0.0, -accel)
        self.van.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer * 0.8, brake=brake))
        self.world.tick()
        self._t += FIXED_DT

        ax, ay, herr = self._slot_frame()
        _, _, _, speed = self._pose()
        reward, done, info = step_outcome(
            ax, ay, herr, speed, self._dist_prev, steer, self._steer_prev,
            self._t, self._collided)
        self._dist_prev = math.hypot(ax, ay)
        self._steer_prev = steer
        obs = np.array(self._obs(), dtype=np.float32)
        return obs, reward, done, False, info

    def close(self):
        self._destroy()
        try:
            self.world.apply_settings(self._original_settings)
            print("[ParkingEnv] world restored to normal mode")
        except Exception:
            pass
