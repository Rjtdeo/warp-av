"""Where the van would think it is from its own wheels and a gyro, and nothing else.

Phase L2. This is the first thing in the project that actually ESTIMATES the van's pose
rather than being told it. It is deliberately the smallest such thing that can be honest:

    speed  from the vehicle itself  (VehicleState.speed_mps -- on a real van, wheel speed
                                     off CAN; in CARLA, the same number the simulator would
                                     give a speedometer)
    gear   from the vehicle itself  (which way the wheels are turning)
    turn   from the IMU gyro        (summed over every 20 Hz sample by the sensor adapter,
                                     because the driving loop only runs at 9-10 Hz)

No GNSS, no LiDAR, no camera, no map, no filter. Those are later phases. What this exists to
answer is one question that nothing in the project has ever measured: HOW FAST DOES IT DRIFT?
Everything after it has to beat that number, and without it there is nothing to beat.

IT DRIVES NOTHING. The estimator runs beside the stack, publishes to telemetry so a drive can
be scored against CARLA's own truth afterwards, and is read by no planner, no behaviour rule
and no controller. That is the whole point of the phase: measure before trusting.

SEEDING. Dead reckoning has no idea where it starts -- it only knows how it has moved since.
So it is seeded once, from the pose source, at the moment a mission begins. That is not
cheating, it is what dead reckoning IS; the interesting number is how far the estimate has
wandered from the truth N metres later. GNSS is what will eventually re-seed and correct it,
and that is L3's job, not this one.

WHAT IT CANNOT DO, and these are not oversights:
  * it cannot recover. Every error is permanent and they accumulate;
  * it has no idea it is wrong, beyond a covariance that grows with distance;
  * it ignores wheel slip, tyre radius error, gyro bias and scale error, all of which a real
    vehicle has and CARLA does not simulate.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from .localization import Pose, PoseCovariance

#: Drift growth per metre travelled and per second elapsed, as VARIANCE (m^2 and rad^2).
#: These are MEASURED in L2, not guessed: a dead reckoner's error grows roughly with distance
#: (heading error turns into sideways error the further you go) and with time (gyro bias
#: integrates). Until the live runs have been scored these stay at zero, which publishes an
#: honestly empty covariance rather than a made-up one.
DRIFT_VAR_PER_M = 0.0        # m^2 per metre travelled
DRIFT_VAR_PER_S = 0.0        # m^2 per second
YAW_VAR_PER_S = 0.0          # rad^2 per second

#: A step longer than this is a stall, a restart or a paused simulator, not motion.
MAX_STEP_S = 0.5


class DeadReckoning:
    """Speed in, turn in, a guess at where the van is out."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0                  # radians
        self.speed = 0.0
        self.seeded = False
        self.seeded_at = None
        self.distance_m = 0.0           # travelled since the seed
        self.elapsed_s = 0.0            # ... and how long that took
        self.steps = 0
        self.skipped = 0                # steps thrown away as too long or out of order
        self._last_t = None
        self._last_turn = None          # the adapter's running turn total, when last read

    # ---------------------------------------------------------------- seeding
    def seed(self, x: float, y: float, yaw: float, now: Optional[float] = None,
             turn_rad: Optional[float] = None) -> None:
        """Start again from a known pose. Everything accumulated so far is forgotten."""
        self.x, self.y, self.yaw = float(x), float(y), float(yaw)
        self.speed = 0.0
        self.seeded = True
        self.seeded_at = time.time() if now is None else now
        self.distance_m = 0.0
        self.elapsed_s = 0.0
        self.steps = 0
        self.skipped = 0
        self._last_t = self.seeded_at
        self._last_turn = turn_rad

    def forget(self) -> None:
        """No mission, no estimate. Stops a parked van accumulating gyro noise for an hour."""
        self.seeded = False
        self._last_t = None
        self._last_turn = None

    # ---------------------------------------------------------------- stepping
    def update(self, speed_mps: float, reverse: bool, turn_rad: Optional[float],
               now: Optional[float] = None) -> Optional[Pose]:
        """One step. `turn_rad` is the sensor adapter's RUNNING TOTAL of how far the van has
        turned, not this step's turn -- the difference since the last look is what is used, so
        that every 20 Hz gyro sample counts even though this is called at 9-10 Hz.

        Returns the estimate, or None before the first seed.
        """
        if not self.seeded:
            return None
        now = time.time() if now is None else float(now)

        dt = now - (self._last_t if self._last_t is not None else now)
        self._last_t = now
        if not (0.0 < dt < MAX_STEP_S):
            # A gap this long means the loop stalled or the clock jumped. Integrating across
            # it would invent a straight line through whatever really happened, so the step is
            # dropped and the turn total re-based -- losing the motion is honest, faking it is
            # not.
            if dt != 0.0:
                self.skipped += 1
            if turn_rad is not None:
                self._last_turn = float(turn_rad)
            return self.pose()

        # --- heading first, from the gyro ---
        yaw0 = self.yaw
        if turn_rad is not None:
            if self._last_turn is not None:
                self.yaw = _wrap(self.yaw + (float(turn_rad) - self._last_turn))
            self._last_turn = float(turn_rad)

        # --- then position, along the AVERAGE heading over the step ---
        # Using the new heading alone (or the old alone) cuts corners on every bend: over a
        # long drive that is a systematic error, not noise. The mid-point costs one atan2.
        v = -abs(float(speed_mps)) if reverse else abs(float(speed_mps))
        self.speed = v
        step = v * dt
        mid = _wrap(yaw0 + _wrap(self.yaw - yaw0) * 0.5)
        self.x += step * math.cos(mid)
        self.y += step * math.sin(mid)

        self.distance_m += abs(step)
        self.elapsed_s += dt
        self.steps += 1
        return self.pose()

    # ---------------------------------------------------------------- output
    def pose(self) -> Optional[Pose]:
        if not self.seeded:
            return None
        return Pose(x=self.x, y=self.y, z=0.0, yaw=self.yaw, speed=self.speed,
                    confidence=1.0, healthy=True, reason="DEAD_RECKONING",
                    cov=self.covariance())

    def covariance(self) -> PoseCovariance:
        """How unsure this estimate is. Grows with distance and time and never shrinks,
        because nothing here can correct it -- which is exactly what dead reckoning is.

        Zero while the growth rates are unmeasured: an empty covariance that says so is more
        use than an invented one that does not.
        """
        var = DRIFT_VAR_PER_M * self.distance_m + DRIFT_VAR_PER_S * self.elapsed_s
        return PoseCovariance(xx=var, yy=var, yaw=YAW_VAR_PER_S * self.elapsed_s)

    def error_against(self, truth: Pose) -> dict:
        """How wrong it is, right now, against a pose believed to be true. Scoring only."""
        if not self.seeded or truth is None:
            return {}
        dx, dy = self.x - truth.x, self.y - truth.y
        # ...and split the error into along-track and cross-track in the TRUE heading's frame,
        # because a metre sideways and a metre late are not the same mistake.
        c, s = math.cos(truth.yaw), math.sin(truth.yaw)
        return {
            "dx_m": dx, "dy_m": dy,
            "error_m": math.hypot(dx, dy),
            "along_m": dx * c + dy * s,
            "cross_m": -dx * s + dy * c,
            "yaw_err_deg": math.degrees(_wrap(self.yaw - truth.yaw)),
            "distance_m": self.distance_m,
            "elapsed_s": self.elapsed_s,
            "error_per_100m": (math.hypot(dx, dy) / self.distance_m * 100.0
                               if self.distance_m > 1.0 else None),
        }

    def state(self) -> dict:
        """For /api/state and the evidence recorder."""
        if not self.seeded:
            return {"seeded": False}
        return {"seeded": True, "x": round(self.x, 3), "y": round(self.y, 3),
                "yaw_deg": round(math.degrees(self.yaw), 3),
                "speed_mps": round(self.speed, 3),
                "distance_m": round(self.distance_m, 2),
                "elapsed_s": round(self.elapsed_s, 2),
                "steps": self.steps, "skipped": self.skipped,
                "sigma_x_m": round(self.covariance().sigma_x, 3),
                "sigma_yaw_deg": round(math.degrees(self.covariance().sigma_yaw), 3)}


def _wrap(a: float) -> float:
    """An angle, in radians, brought back to +/-pi."""
    return math.atan2(math.sin(a), math.cos(a))
