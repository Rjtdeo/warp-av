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
                             lateral_error, spawn_pose, bounds_for, timeout_for,
                             bay_is_clear, feelers, neighbour_pose,
                             neighbour_behind_fits, neighbour_ahead_fits, lane_start_for,
                             NEIGHBOUR_BEHIND_BAYS, FEELER_SECTORS, Curriculum, Stages,
                             LANE_START_M, SLOT_LEN, SLOT_WID)

FIXED_DT = 0.1
MAX_SPEED = 3.0


def static_vehicle_outlines(world):
    """2D outlines (centre + four box corners) of the DECORATIVE vehicles baked
    into the map's static layer. Same scan as the main stack's
    _static_vehicle_points(): they are not actors, so nothing else sees them."""
    pts, seen = [], set()
    try:
        for name in ("Vehicles", "Car", "Truck", "Bus", "Motorcycle", "Bicycle"):
            lbl = getattr(carla.CityObjectLabel, name, None)
            if lbl is None:
                continue
            for obj in world.get_environment_objects(lbl):
                if obj.id in seen:
                    continue
                seen.add(obj.id)
                bb = obj.bounding_box
                cx, cy, ext = bb.location.x, bb.location.y, bb.extent
                yaw = math.radians(bb.rotation.yaw)
                c, s = math.cos(yaw), math.sin(yaw)
                outline = [(cx, cy)]
                for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                    outline.append((cx + sx * c * ext.x - sy * s * ext.y,
                                    cy + sx * s * ext.x + sy * c * ext.y))
                pts.append(outline)
    except Exception as e:
        print(f"[ParkingEnv] static vehicle scan failed: {e}")
    return pts


class CarlaParkingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host="localhost", port=2000, seed=7, exam_p=None,
                 curriculum=None, lane_start_m=LANE_START_M, yaw_noise_deg=3.0,
                 lateral_noise_m=0.0, obs_noise=None, obs_dropout=0.0, obs_delay=0,
                 neighbour_p=0.5, neighbour_ahead_p=0.3, use_feelers=True,
                 reverse=False, obstacles=False, neighbour_behind_bays=NEIGHBOUR_BEHIND_BAYS,
                 lane_start_jitter_m=0.0,
                 stages=None):
        """Harder-exam knobs (all default to how training ran):
        lane_start_m     how far back down the lane a p=0 start is (train: 16)
        yaw_noise_deg    random heading error at spawn, +/- (train: 3)
        lateral_noise_m  random sideways offset of the spawn point, +/- (train: 0)
        obs_noise        (pos_m, yaw_deg, speed_mps) gaussian noise added to what
                         the student SEES each step - never to the physics or the
                         score. A preview of real sensors.
        obs_dropout      chance per step that the student's view FREEZES (it is
                         handed last step's numbers again) - a sensor dropout
        obs_delay        steps of lag between the world and what it sees
        neighbour_p      chance per attempt of a REAL parked car in the bay
                         behind the target (only when the van starts far enough
                         back not to be born touching it)
        neighbour_ahead_p  chance of one in the bay ahead
        use_feelers      False reproduces the round-6 five-number view, so an
                         older brain can sit the same exam as a newer one
        reverse          a third control: > 0.5 selects reverse gear (round 8)
        obstacles        hazards in stages (parking_math.Stages) instead of the
                         flat neighbour_p / neighbour_ahead_p chances
        neighbour_behind_bays  how many bays back the practice car sits (1 or 2)"""
        super().__init__()
        self.rng = random.Random(seed)
        self.exam_p = exam_p
        self.lane_start_m = float(lane_start_m)
        # Training only: spread every far start over [lane start, lane start +
        # jitter]. Rounds 6-8f were born at exactly 16 m (and, with a car, at
        # exactly bays*7+8 m), so the brain's habits are welded to those
        # spots: at 22 m with a car two bays back it still swerved right at
        # birth (0/360) while from 29 m with a car three bays back it held the
        # lane and parked 92%. A spread of starts teaches the same lesson
        # everywhere along the approach.
        self.lane_start_jitter_m = float(lane_start_jitter_m)
        self.yaw_noise_deg = float(yaw_noise_deg)
        self.lateral_noise_m = float(lateral_noise_m)
        self.obs_noise = tuple(obs_noise) if obs_noise else None
        self.obs_dropout = float(obs_dropout)
        self.obs_delay = int(obs_delay)
        self.neighbour_p = float(neighbour_p)
        self.neighbour_ahead_p = float(neighbour_ahead_p)
        self.neighbours = []
        self.use_feelers = bool(use_feelers)
        self._feelers_now = None
        self.reverse = bool(reverse)
        self.obstacles = bool(obstacles)
        self.neighbour_behind_bays = int(neighbour_behind_bays)
        self.stages = (stages or Stages()) if obstacles else None
        self._ax_prev = 0.0
        self._reverse_steps = 0
        self._reversing = False
        self._hazard = ""              # which neighbour this attempt got: behind1 / behind2 / ahead / ""
        self._obs_queue = []            # for obs_delay
        self._obs_last = None           # for obs_dropout
        self._spawn_yaw_off = 0.0
        self._spawn_lat_off = 0.0
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
            found = self._scan_bays()
            statics = static_vehicle_outlines(self.world)
            self.static_outlines = statics
            self._static_points = [pt for outline in statics for pt in outline]
            self.bays = [b for b in found if bay_is_clear(b[1], b[2], b[3], statics)]
            if len(self.bays) < 5:
                raise RuntimeError(f"only {len(self.bays)} usable bays found")
            print(f"[ParkingEnv] {len(self.bays)} practice bays ready "
                  f"({len(found) - len(self.bays)} dropped: a decorative parked "
                  f"vehicle in the bay, behind it, or on the approach; "
                  f"{len(statics)} such vehicles on this map)")

            bps = self.world.get_blueprint_library().filter("vehicle.mercedes.sprinter")
            if not bps:
                raise RuntimeError("blueprint vehicle.mercedes.sprinter is not in "
                                   "this CARLA build")
            bp = bps[0]
            bp.set_attribute("role_name", "warp_rl")
            # Nothing longer than a car may play the parked car: a 7 m minibus
            # (fusorosa) two or three bays back reaches the van's birth point
            # (exam 2026-09-04: one crash at birth, and training crashes that
            # were nobody's fault).
            big = ("bus", "truck", "carlamotors", "ambulance", "firetruck", "sprinter",
                   "fusorosa", "cybertruck", "t2")
            self.neighbour_bps = [
                b for b in self.world.get_blueprint_library().filter("vehicle.*")
                if b.has_attribute("number_of_wheels")
                and b.get_attribute("number_of_wheels").as_int() == 4
                and not any(k in b.id for k in big)]
            if not self.neighbour_bps:
                self.neighbour_bps = [bp]
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
        self._hit = None            # what the last collision was with (diagnostics)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(3 if self.reverse else 2,), dtype=np.float32)
        # 5 pose numbers + 4 feelers (see parking_math.feelers)
        n_obs = 5 + (len(FEELER_SECTORS) if self.use_feelers else 0)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(n_obs,),
                                            dtype=np.float32)
        self.slot = None
        self._t = 0.0
        self._steer_prev = 0.0
        self._dist_prev = 0.0
        self._align_prev = 0.0
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
        for a in [self.col_sensor, self.van] + list(self.neighbours):
            try:
                if a is not None:
                    a.destroy()
            except Exception:
                pass
        self.col_sensor = None
        self.van = None
        self.neighbours = []

    def _spawn_neighbour(self, drive_wp, bays_away):
        """A real, physics-off parked car in the bay `bays_away` slots along.
        Skipped quietly if the spot is taken."""
        sx, sy, syaw = self.slot
        nx, ny, nyaw = neighbour_pose(sx, sy, syaw, bays_away)
        tf = carla.Transform(
            carla.Location(nx, ny, drive_wp.transform.location.z + 0.3),
            carla.Rotation(yaw=math.degrees(nyaw)))
        actor = self.world.try_spawn_actor(self.rng.choice(self.neighbour_bps), tf)
        if actor is not None:
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass
            self.neighbours.append(actor)
        return actor is not None

    def _plan_hazard(self):
        """Decide which car this attempt gets BEFORE the van is placed, so the
        far start can be set back far enough to be born behind it. With
        obstacle stages: stage 0 none; stage 1 a car `bays_back` bays back
        (50%) or one ahead (30%), never both; stage 2 also a car RIGHT behind
        (30%, bay ahead free). Without stages: the flat chances.
        Returns ("behind", bays) / ("ahead", 1) / None."""
        if self.stages is not None:
            lvl = self.stages.level
            if lvl == 0:
                return None
            if lvl >= 2 and self.rng.random() < 0.3:
                return ("behind", 1)
            if self.rng.random() < 0.5:
                return ("behind", self.stages.bays_back)
            if self.rng.random() < 0.3:
                return ("ahead", 1)
            return None
        if self.neighbour_p and self.rng.random() < self.neighbour_p:
            return ("behind", self.neighbour_behind_bays)
        # Never both without a reverse gear: a gap between two parked cars is
        # not a lesson, it is a wall.
        if self.neighbour_ahead_p and self.rng.random() < self.neighbour_ahead_p:
            return ("ahead", 1)
        return None

    def _place_hazards(self, drive_wp, planned_start, plan):
        """Spawn the planned car, but only if this start is far enough back for
        it (a revision attempt that begins close to the bay gets no car)."""
        self._hazard = ""
        if plan is None:
            return
        kind, bays = plan
        if kind == "behind":
            if neighbour_behind_fits(planned_start, bays) and self._spawn_neighbour(drive_wp, -bays):
                self._hazard = f"behind{bays}"
        elif neighbour_ahead_fits(planned_start) and self._spawn_neighbour(drive_wp, +1):
            self._hazard = "ahead"

    def _obstacle_points(self):
        """Outline points of everything the feelers may touch: the real
        neighbours (centre, corners, edge midpoints) plus the map's decorative
        vehicles."""
        pts = list(self._static_points)
        for a in self.neighbours:
            try:
                tf = a.get_transform()
                ext = a.bounding_box.extent
            except Exception:
                continue
            cx, cy = tf.location.x, tf.location.y
            yaw = math.radians(tf.rotation.yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            for fx, fy in ((0, 0), (1, 1), (1, -1), (-1, -1), (-1, 1),
                           (1, 0), (-1, 0), (0, 1), (0, -1)):
                pts.append((cx + fx * c * ext.x - fy * s * ext.y,
                            cy + fx * s * ext.x + fy * c * ext.y))
        return pts

    def _on_col(self, event):
        self._collided = True
        try:
            self._hit = event.other_actor.type_id
        except Exception:
            self._hit = "unknown"

    def _next_p(self):
        if self.exam_p is not None:
            return self.exam_p
        return self.curriculum.next_p(self.rng)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self._collided = False
        self._hit = None
        self._obs_queue = []
        self._obs_last = None
        self._t = 0.0
        self._steer_prev = 0.0

        p = (options or {}).get("p")
        if p is None:
            p = self._next_p()      # only draw when the caller did not fix it
        plan = self._plan_hazard()
        lane_m = self.lane_start_m
        if plan is not None and plan[0] == "behind":
            lane_m = lane_start_for(plan[1], self.lane_start_m)   # born BEHIND the car
        if self.lane_start_jitter_m > 0:
            lane_m += self.rng.uniform(0.0, self.lane_start_jitter_m)
        for _ in range(40):
            drive_wp, sx, sy, syaw = self.rng.choice(self.bays)
            back = drive_wp.previous(lane_m)
            if not back:
                continue
            lane_tf = back[0].transform
            lyaw = math.radians(lane_tf.rotation.yaw)
            lx, ly = lane_tf.location.x, lane_tf.location.y
            off = 0.0
            if self.lateral_noise_m:
                off = self.rng.uniform(-self.lateral_noise_m, self.lateral_noise_m)
                lx += -math.sin(lyaw) * off
                ly += math.cos(lyaw) * off
            px, py, pyaw = spawn_pose(p, lx, ly, lyaw, sx, sy, syaw)
            yaw_off = self.rng.uniform(-self.yaw_noise_deg, self.yaw_noise_deg)
            tf = carla.Transform(
                carla.Location(px, py, lane_tf.location.z + 0.3),
                carla.Rotation(yaw=math.degrees(pyaw) + yaw_off))
            self._spawn_yaw_off, self._spawn_lat_off = yaw_off, off
            self.van = self.world.try_spawn_actor(self.van_bp, tf)
            if self.van is not None:
                self.slot = (sx, sy, syaw)
                planned_start = math.hypot(px - sx, py - sy)
                self._place_hazards(drive_wp, planned_start, plan)
                break
        if self.van is None:
            raise RuntimeError("could not spawn the practice van anywhere")

        col_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.col_sensor = self.world.spawn_actor(col_bp, carla.Transform(),
                                                 attach_to=self.van)
        self.col_sensor.listen(self._on_col)

        self.world.tick()
        obs = self._obs()
        ax, ay, herr = self._slot_frame()
        self._dist_prev = math.hypot(ax, ay)
        self._align_prev = lateral_error(ay, herr)
        self._ax_prev = ax
        self._reverse_steps = 0
        self._reversing = False
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
        if self.obs_noise:
            # What the student sees, not where the van is: the reward and the
            # exam still judge the true pose.
            sp, syaw, sv = self.obs_noise
            x += self.rng.gauss(0.0, sp)
            y += self.rng.gauss(0.0, sp)
            yaw += math.radians(self.rng.gauss(0.0, syaw))
            speed = max(0.0, speed + self.rng.gauss(0.0, sv))
        obs = observation(x, y, yaw, speed, self._steer_prev, *self.slot)
        if self.use_feelers:
            self._feelers_now = feelers(x, y, yaw, self._obstacle_points())
            obs = obs + self._feelers_now
        if self.obs_delay > 0:
            # The world moves on; the student sees where things WERE.
            self._obs_queue.append(obs)
            while len(self._obs_queue) < self.obs_delay + 1:
                self._obs_queue.insert(0, obs)      # first frames: nothing older to show
            obs = self._obs_queue.pop(0)
        if self.obs_dropout > 0 and self._obs_last is not None \
                and self.rng.random() < self.obs_dropout:
            obs = self._obs_last                    # the view froze this step
        self._obs_last = obs
        return obs

    def step(self, action):
        steer = float(np.clip(action[0], -1, 1))
        accel = float(np.clip(action[1], -1, 1))
        gear_rev = bool(self.reverse and len(action) > 2 and float(action[2]) > 0.5)
        _, _, _, speed = self._pose()
        cap = 1.5 if gear_rev else MAX_SPEED          # reversing is slow
        throttle = max(0.0, accel) * 0.7 if speed < cap else 0.0
        brake = max(0.0, -accel)
        self.van.apply_control(carla.VehicleControl(
            throttle=throttle, steer=steer * 0.8, brake=brake, reverse=gear_rev))
        self._reversing = gear_rev
        if gear_rev:
            self._reverse_steps += 1
        self.world.tick()
        self._t += FIXED_DT

        ax, ay, herr = self._slot_frame()
        _, _, _, speed = self._pose()
        reward, done, info = step_outcome(
            ax, ay, herr, speed, self._dist_prev, steer, self._steer_prev,
            self._t, self._collided, timeout_s=self._timeout_s,
            bounds_m=self._bounds_m, align_prev=self._align_prev,
            feeler_readings=self._feelers_now, ax_prev=self._ax_prev,
            overshoot_m=12.0 if self.reverse else None, reversing=self._reversing,
            lane_hold=True)
        self._dist_prev = math.hypot(ax, ay)
        self._align_prev = lateral_error(ay, herr)
        self._ax_prev = ax
        self._steer_prev = steer
        if done:
            info["p"] = round(self._p, 2)
            info["start_dist"] = round(self._start_dist, 1)
            info["spawn_yaw_off_deg"] = round(self._spawn_yaw_off, 1)
            info["spawn_lat_off_m"] = round(self._spawn_lat_off, 2)
            info["neighbours"] = len(self.neighbours)
            info["reverse_steps"] = self._reverse_steps
            info["hazard"] = self._hazard
            info["stage"] = self.stages.level if self.stages is not None else ""
            if self.stages is not None:
                # A stage is judged ONLY on attempts that contain its own hazard.
                # Counting the easy empty and car-ahead attempts too unlocked
                # "car right behind" while the two-bays-back skill was at 25%.
                lvl = self.stages.level
                counts = (lvl == 0) \
                    or (lvl == 1 and self._hazard == f"behind{self.stages.bays_back}") \
                    or (lvl >= 2 and self._hazard == "behind1")
                if counts:
                    moved = self.stages.record(info.get("result") == "parked")
                    if moved is not None:
                        print(f"[stages] {self.stages.describe()}")
            if info.get("result") == "collision":
                # Where it was and what it touched, so a crash is a fact, not a guess.
                info["hit"] = self._hit or "unknown"
                info["ax"] = round(ax, 2)
                info["ay"] = round(ay, 2)
                info["herr_deg"] = round(math.degrees(herr), 1)
                info["speed"] = round(speed, 2)
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
