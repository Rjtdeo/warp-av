"""
CARLA practice arena for the RL parking student.

Each episode the van appears somewhere along the ideal pull-in for a randomly
chosen real parking bay, and must end fully inside a 7 m slot there, parallel
and stopped. How far back it starts is the DIFFICULTY: a Curriculum walks the
student from "already in the box, just stop" out to the full 16 m lane start
as its success rate climbs. Runs CARLA in synchronous fast-forward while
training and puts the world back to normal on exit.

Pass exam_p to freeze the difficulty instead (exam_p=0.0 is the real exam:
every attempt from the full distance).

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
                             spawn_pose, bounds_for, timeout_for, Curriculum,
                             LANE_START_M, SLOT_LEN, SLOT_WID)

FIXED_DT = 0.1
MAX_SPEED = 3.0


class CarlaParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=2000, seed=7, exam_p=None,
                 curriculum=None):
        super().__init__()
        self.rng = random.Random(seed)
        self.exam_p = exam_p
        self.curriculum = None if exam_p is not None else (curriculum or Curriculum())
        self.client = carla.Client(host, port)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        self._original_settings = self.world.get_settings()

        # Synchronous mode is a SERVER setting and outlives this client, so if
        # anything below raises we must put the world back before propagating —
        # otherwise __init__ never returns, the caller's finally: env.close()
        # never runs, and CARLA is left frozen with nobody ticking it. Every
        # later client (trainer retry, exam, the main van program) then hangs.
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = FIXED_DT
            self.world.apply_settings(settings)

            self.cmap = self.world.get_map()
            self.bays = self._scan_bays()
            if len(self.bays) < 5:
                raise RuntimeError(f"only {len(self.bays)} usable bays found")
            print(f"[ParkingEnv] {len(self.bays)} practice bays ready")

            bps = self.world.get_blueprint_library().filter("vehicle.mercedes.sprinter")
            if not bps:
                raise RuntimeError("blueprint vehicle.mercedes.sprinter is not in "
                                   "this CARLA build")
            bp = bps[0]
            bp.set_attribute("role_name", "warp_rl")
        except BaseException:
            try:
                self.world.apply_settings(self._original_settings)
                print("[ParkingEnv] startup failed — world restored to normal mode")
            except Exception:
                pass
            raise
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
        self._p = 0.0
        self._start_dist = 0.0
        self._bounds_m = None
        self._timeout_s = 30.0

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

    def _next_p(self):
        if self.exam_p is not None:
            return self.exam_p
        return self.curriculum.next_p(self.rng)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self._collided = False
        self._t = 0.0
        self._steer_prev = 0.0

        p = (options or {}).get("p")
        if p is None:
            p = self._next_p()      # only draw when the caller did not fix it
        for _ in range(40):
            drive_wp, sx, sy, syaw = self.rng.choice(self.bays)
            back = drive_wp.previous(LANE_START_M)
            if not back:
                continue
            lane_tf = back[0].transform
            px, py, pyaw = spawn_pose(
                p, lane_tf.location.x, lane_tf.location.y,
                math.radians(lane_tf.rotation.yaw), sx, sy, syaw)
            tf = carla.Transform(
                carla.Location(px, py, lane_tf.location.z + 0.3),
                carla.Rotation(yaw=math.degrees(pyaw) + self.rng.uniform(-3, 3)))
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
        # Room to stray and time allowed both scale with where this attempt
        # began — a flat limit punished far starts for existing.
        self._p = p
        self._start_dist = self._dist_prev
        self._bounds_m = bounds_for(self._start_dist)
        self._timeout_s = timeout_for(self._start_dist)
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
            self._t, self._collided, timeout_s=self._timeout_s,
            bounds_m=self._bounds_m)
        self._dist_prev = math.hypot(ax, ay)
        self._steer_prev = steer
        if done:
            info["p"] = round(self._p, 2)
            info["start_dist"] = round(self._start_dist, 1)
            if self.curriculum is not None:
                moved = self.curriculum.record(self._p,
                                               info.get("result") == "parked")
                if moved is not None:
                    print(f"[curriculum] {self.curriculum.describe()}")
        obs = np.array(self._obs(), dtype=np.float32)
        return obs, reward, done, False, info

    def close(self):
        self._destroy()
        try:
            self.world.apply_settings(self._original_settings)
            print("[ParkingEnv] world restored to normal mode")
        except Exception:
            pass
