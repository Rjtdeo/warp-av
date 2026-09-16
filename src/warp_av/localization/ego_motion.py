"""How far the van moved between two instants, from its own sensors and nothing else.

Phase L5. This exists because of one line in the LiDAR sweep: every delivery used to be
placed in the world with the pose CARLA stamped on it, so the assembled point cloud already
contained perfect knowledge of the van's motion. Scan matching on that cloud would have been
measuring the simulator, not the world -- the registration would have had nothing left to
find, and any heading it "recovered" would have been CARLA's own answer handed back.

WHAT THE SWEEP ACTUALLY NEEDS. The old code computed `inv(M_now) @ M_i` for each delivery.
The world frame cancels in that product: it is purely the RELATIVE motion between two capture
times about a tenth of a second apart. Absolute position was never required, which is why this
can be replaced honestly rather than approximated.

INPUTS, and why each is allowed:

  * GYRO yaw rate     -- an IMU reading. Rotation is the part that matters: at 10 m/s a van
                         travels 1 m in a sweep, but at 30 deg/s it turns 3 degrees, and a
                         3-degree smear at 40 m throws a point 2 m sideways.
  * WHEEL SPEED       -- a scalar, the same one the EKF and the dead reckoner already use.
                         A magnitude, never a direction: the velocity VECTOR would leak the
                         course, which is one of the things we are trying to measure.
  * STATIC EXTRINSIC  -- where the LiDAR is bolted to the van. Calibration, measured once with
                         a tape measure on a real vehicle, and not a dynamic quantity at all.
                         Kept explicitly separate from pose below.

NOT INPUTS: CARLA x/y/yaw, GNSS, the EKF's own estimate, or the true velocity vector. Nothing
in this module imports carla, and nothing in it can reach a simulator.

WHAT IT DOES NOT MODEL. Sideslip. The van is assumed to travel along its own nose, which
L2-GAP measured to be wrong by 0.756 degrees on average. Over a whole sweep at 10 m/s that
misplaces a point by about 13 mm -- below the LiDAR's own range noise, and far below the
2 m smear the rotation would cause if left uncorrected. It is deliberately NOT corrected
here: sideslip is a thing we want scan matching to reveal, so assuming a value for it in the
de-skew would be assuming the answer.
"""
from __future__ import annotations

import math
import threading
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

#: Where the LiDAR sits relative to the van's own origin. STATIC EXTRINSIC -- calibration,
#: not localization. Matches the spawn transform in adapters/carla_sensor_adapter.setup_sensors
#: (carla.Transform(carla.Location(x=0.0, z=2.5))). The horizontal offset is zero, so under
#: yaw-only motion the sensor and vehicle frames share an origin in the plane and this
#: transform cancels -- but it is applied properly anyway, so moving the mount stays correct.
LIDAR_MOUNT_XYZ = (0.0, 0.0, 2.5)

#: Samples older than this are dropped. One LiDAR rotation is 0.1 s; a second is generous.
DEFAULT_MAX_AGE_S = 1.0
#: Refuse to integrate across a gap longer than this between samples -- a rate sampled at the
#: ends of a long gap says nothing about the middle. Same reasoning as the EKF's MAX_PREDICT_S.
MAX_SAMPLE_GAP_S = 0.25


def _rigid_2d(dyaw: float, dx: float, dy: float) -> np.ndarray:
    """The 4x4 taking a point expressed in frame A into frame B, where B sits at (dx, dy) in
    A's coordinates and is rotated by `dyaw` relative to A.

        p_A = R(dyaw) p_B + d      =>      p_B = R(-dyaw) (p_A - d)
    """
    c, s = math.cos(dyaw), math.sin(dyaw)
    M = np.eye(4, dtype=np.float64)
    M[0, 0], M[0, 1] = c, s
    M[1, 0], M[1, 1] = -s, c
    M[0, 3] = -(c * dx + s * dy)
    M[1, 3] = -(-s * dx + c * dy)
    return M


def mount_matrix(xyz: Tuple[float, float, float] = LIDAR_MOUNT_XYZ) -> np.ndarray:
    """Sensor -> vehicle. Pure translation: the LiDAR is not tilted or yawed on its mount."""
    E = np.eye(4, dtype=np.float64)
    E[0, 3], E[1, 3], E[2, 3] = float(xyz[0]), float(xyz[1]), float(xyz[2])
    return E


class EgoMotionHistory:
    """Timestamped yaw rate and speed in; relative motion between two instants out.

    Fed from sensor callback threads and read from the LiDAR callback thread, so every
    touch of the two buffers is under one lock.
    """

    def __init__(self, extrinsic: Optional[np.ndarray] = None,
                 max_age_s: float = DEFAULT_MAX_AGE_S):
        self._gyro: Deque[Tuple[float, float]] = deque()      # (sim_time, yaw rate rad/s)
        self._speed: Deque[Tuple[float, float]] = deque()     # (sim_time, signed m/s)
        self._lock = threading.Lock()
        self.max_age_s = float(max_age_s)
        self.E = mount_matrix() if extrinsic is None else np.asarray(extrinsic, dtype=np.float64)
        self.E_inv = np.linalg.inv(self.E)
        # bookkeeping, all of it reported
        self.refused_no_data = 0
        self.refused_gap = 0
        self.served = 0

    # ---------------------------------------------------------------- inputs
    def add_gyro(self, sim_time: float, yaw_rate: float) -> None:
        if sim_time is None or not math.isfinite(yaw_rate):
            return
        with self._lock:
            self._gyro.append((float(sim_time), float(yaw_rate)))
            self._trim(self._gyro, float(sim_time))

    def add_speed(self, sim_time: float, speed_mps: float, reverse: bool = False) -> None:
        """A SCALAR speed with the time it was read. Sign comes from the gear, not from a
        measured direction of travel."""
        if sim_time is None or not math.isfinite(speed_mps):
            return
        v = -abs(float(speed_mps)) if reverse else abs(float(speed_mps))
        with self._lock:
            self._speed.append((float(sim_time), v))
            self._trim(self._speed, float(sim_time))

    def _trim(self, buf: Deque, now: float) -> None:
        cutoff = now - self.max_age_s
        while len(buf) > 2 and buf[0][0] < cutoff:
            buf.popleft()

    def reset(self) -> None:
        with self._lock:
            self._gyro.clear()
            self._speed.clear()

    # ---------------------------------------------------------------- output
    @staticmethod
    def _held(buf, t: float) -> Optional[float]:
        """The most recent value at or before `t` (zero-order hold), or the earliest value if
        `t` predates the buffer by less than one sample."""
        best = None
        for ts, v in buf:
            if ts <= t:
                best = v
            else:
                break
        if best is None and buf:
            return buf[0][1]
        return best

    def integrate(self, t_from: float, t_to: float) -> Optional[Tuple[float, float, float]]:
        """(dyaw, dx, dy) of the VEHICLE frame at t_to, expressed in the vehicle frame at
        t_from. None when the history cannot honestly cover the interval."""
        if t_to < t_from:
            flip = self.integrate(t_to, t_from)
            if flip is None:
                return None
            dyaw, dx, dy = flip
            c, s = math.cos(-dyaw), math.sin(-dyaw)     # invert the rigid transform
            return -dyaw, -(c * dx - s * dy), -(s * dx + c * dy)
        with self._lock:
            gyro = list(self._gyro)
            speed = list(self._speed)
        if not gyro:
            self.refused_no_data += 1
            return None
        if t_to - t_from <= 1e-9:
            self.served += 1
            return 0.0, 0.0, 0.0
        # every sample boundary inside the interval, so a rate change lands where it happened
        edges = {t_from, t_to}
        for ts, _ in gyro:
            if t_from < ts < t_to:
                edges.add(ts)
        for ts, _ in speed:
            if t_from < ts < t_to:
                edges.add(ts)
        marks = sorted(edges)
        # the interval must be covered by real samples, not extrapolated across a hole
        newest = gyro[-1][0]
        if newest < t_to - MAX_SAMPLE_GAP_S:
            self.refused_gap += 1
            return None
        for a, b in zip(marks, marks[1:]):
            if b - a > MAX_SAMPLE_GAP_S:
                self.refused_gap += 1
                return None
        yaw = 0.0
        dx = dy = 0.0
        for a, b in zip(marks, marks[1:]):
            dt = b - a
            if dt <= 0:
                continue
            w = self._held(gyro, a) or 0.0
            v = self._held(speed, a)
            v = 0.0 if v is None else v
            yaw_mid = yaw + 0.5 * w * dt          # midpoint heading over the sub-step
            step = v * dt
            dx += step * math.cos(yaw_mid)
            dy += step * math.sin(yaw_mid)
            yaw += w * dt
        self.served += 1
        return yaw, dx, dy

    def relative(self, t_from: float, t_to: float) -> Optional[np.ndarray]:
        """The 4x4 taking a point captured in the SENSOR frame at `t_from` into the SENSOR
        frame at `t_to`. None when the motion for that interval is not known.

        The static mount is applied around the dynamic motion:

            T_sensor = E^-1 @ T_vehicle @ E

        which is the whole separation the phase asked for, in one line -- E is calibration,
        T_vehicle is everything the sensors measured.
        """
        got = self.integrate(t_from, t_to)
        if got is None:
            return None
        dyaw, dx, dy = got
        return self.E_inv @ _rigid_2d(dyaw, dx, dy) @ self.E

    def state(self) -> dict:
        with self._lock:
            ng, ns = len(self._gyro), len(self._speed)
            newest = self._gyro[-1][0] if self._gyro else None
        return {"gyro_samples": ng, "speed_samples": ns, "newest_gyro_t": newest,
                "served": self.served, "refused_no_data": self.refused_no_data,
                "refused_gap": self.refused_gap}
