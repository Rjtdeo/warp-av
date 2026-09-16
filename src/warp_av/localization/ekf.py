"""The first fused estimate of where the van is: gyro and wheels to carry it, GNSS to hold it.

Phase L3. Three inputs, three states, one correction. Deliberately the smallest filter that
can be honest, because a filter nobody can debug is worse than dead reckoning.

    STATE      x, y, yaw           metres, metres, radians
    PREDICT    wheel speed + gyro yaw rate, over the ACTUAL time between measurements
    CORRECT    GNSS position, converted to local metres (localization/geo.py)

IT DRIVES NOTHING. Like the dead reckoner beside it, this runs in shadow: it publishes to
telemetry so a drive can be scored against CARLA's truth afterwards, and no planner, behaviour
rule, controller, perception path or traffic-light lookup reads a single number it produces.

WHY ONLY THREE STATES. Speed is measured directly and its error is small and roughly
multiplicative -- L2-GAP put sampled-speed integration within 0.14 % of the true path. Putting
it in the state would buy a scale estimate that GNSS can barely observe over a short route,
at the cost of a fourth row everywhere and a filter that is harder to reason about. Gyro bias
is the state most worth adding NEXT, because it is unobservable from dead reckoning and GNSS
does constrain it over time -- but adding it before the three-state version has been scored
would mean never knowing which part helped.

SIDESLIP IS NOT CORRECTED HERE. L2-GAP measured the van travelling 0.756 degrees off its own
heading on average and up to 3.151, and that mismatch is the largest single error in the dead
reckoner. It is entered as PROCESS NOISE across the direction of travel, not as a magic
correction term: the filter is told the motion model is uncertain sideways, and GNSS is left
to show whether that is enough. If it is not, the evidence will say so.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geo import GeoFrame, DEFAULT as DEFAULT_GEO
from .localization import Pose, PoseCovariance

# ---- process noise ---------------------------------------------------------------------
#: Speed is trusted to about this fraction of itself. L2-GAP measured tick-rate integration
#: inside 0.14 % of the true path, so 1 % leaves room for the wheel-slip and tyre-radius error
#: a real vehicle has and CARLA does not simulate.
SPEED_SCALE_SIGMA = 0.01
#: How far off its own nose the van may be travelling, as a standard deviation. L2-GAP:
#: mean |slip| 0.756 deg, worst 3.151, over 2 deg in a fifth of moving samples. This is the
#: honest way to carry that -- the model is uncertain sideways, and the filter is told so.
SLIP_SIGMA_RAD = math.radians(1.0)
#: Gyro white noise (rad/s/sqrt(Hz)-ish) and bias, matching the noisy_sim sensor profile.
GYRO_NOISE_RAD_S = 0.002
GYRO_BIAS_RAD_S = 5e-5
#: A floor, so a stationary filter still loosens slightly rather than freezing its covariance
#: and refusing every future correction.
POS_NOISE_FLOOR_M2_PER_S = 1e-4
YAW_NOISE_FLOOR_RAD2_PER_S = 1e-8

# ---- measurement noise -----------------------------------------------------------------
#: What the GNSS is believed to be worth, in metres, one standard deviation per axis. This is
#: the CONFIGURED sensor model (adapters/carla_sensor_adapter.GNSS_NOISE_M), not a number
#: tuned until the score looked good.
GNSS_SIGMA_M = 0.02

#: ...and what the compass is believed to be worth. Matches COMPASS_NOISE_DEG in the sensor
#: profile. It is deliberately NOT told about the compass BIAS: a filter cannot subtract an
#: offset it does not estimate, so the bias becomes the floor under the heading accuracy.
#: Widening R to cover the bias would only make the filter ignore a sensor that is telling the
#: truth on average; estimating the bias is a state, and that is a later decision.
COMPASS_SIGMA_RAD = math.radians(1.0)

#: Below this speed a GNSS fix may move x and y but MUST NOT rotate the heading.
#:
#: L3 measured an 11.3 degree heading error at t=3.1 s, pulling away from rest at 1.5-2.1 m/s,
#: while the gyro alone was 0.15 degrees out. Heading is only observable through MOTION: a fix
#: says where the van is, not which way it points, and the filter can only infer heading from
#: the direction it appears to have travelled. Barely moving, a few centimetres of position
#: residual look exactly like a large heading error, and the position-yaw cross-covariance
#: dutifully rotates the estimate to explain it.
#:
#: So below this speed the yaw row of the Kalman gain is zeroed. Position is still corrected --
#: the fix is good and there is no reason to throw it away -- only its indirect pull on the
#: heading is suppressed. The covariance update stays consistent because the Joseph form is
#: valid for ANY gain, not only the optimal one.
GNSS_YAW_MIN_SPEED_MPS = 2.0

# ---- time handling ---------------------------------------------------------------------
#: Longer than this between measurements and the filter predicts but says it lost time; the
#: motion across such a gap is not knowable from a rate and a speed sampled at its ends.
MAX_PREDICT_S = 0.5
#: Anything at or before the filter's own clock is a duplicate or an overtake. Dropped, and
#: counted, rather than integrated backwards.
MIN_STEP_S = 1e-6


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class LocalizationEKF:
    """x, y, yaw -- carried by the wheels and the gyro, held in place by GNSS."""

    def __init__(self, geo: Optional[GeoFrame] = None, gnss_sigma_m: float = GNSS_SIGMA_M):
        self.geo = geo or DEFAULT_GEO
        self.gnss_sigma_m = float(gnss_sigma_m)
        self.x = np.zeros(3, dtype=float)          # [x, y, yaw]
        self.P = np.zeros((3, 3), dtype=float)
        self.seeded = False
        self.t = None                              # the filter's own clock, in SIM time
        self.speed = 0.0
        self.reverse = False
        self.speed_t = None
        # bookkeeping, all of it reported
        self.predicts = 0
        self.corrections = 0
        self.rejected_old = 0
        self.rejected_gap = 0
        self.gnss_rejected = 0
        self.heading_corrections = 0
        self.gnss_yaw_suppressed = 0
        self.distance_m = 0.0
        self.last_gnss_t = None

    # ------------------------------------------------------------------ seeding
    def seed(self, x: float, y: float, yaw: float, sim_time: Optional[float],
             pos_sigma_m: float = 0.5, yaw_sigma_deg: float = 2.0) -> None:
        """Start from a known pose.

        INITIALISATION ONLY. This is the one and only place the filter is told anything by the
        authoritative pose source; after it, the estimate lives on speed, gyro and GNSS alone
        and is never nudged back towards truth. The starting covariance is deliberately loose
        rather than zero -- a filter that begins certain refuses the corrections that would
        have told it otherwise.
        """
        self.x = np.array([float(x), float(y), _wrap(float(yaw))], dtype=float)
        self.P = np.diag([pos_sigma_m ** 2, pos_sigma_m ** 2,
                          math.radians(yaw_sigma_deg) ** 2]).astype(float)
        self.seeded = True
        self.t = sim_time
        self.speed = 0.0
        self.reverse = False
        self.speed_t = sim_time
        self.predicts = self.corrections = 0
        self.rejected_old = self.rejected_gap = self.gnss_rejected = 0
        self.heading_corrections = self.gnss_yaw_suppressed = 0
        self.distance_m = 0.0
        self.last_gnss_t = None

    def forget(self) -> None:
        self.seeded = False
        self.t = None

    # ------------------------------------------------------------------ inputs
    def set_speed(self, speed_mps: float, reverse: bool, sim_time: Optional[float]) -> None:
        """The wheels, with the time they were read. Held until the next prediction uses it."""
        self.speed = abs(float(speed_mps))
        self.reverse = bool(reverse)
        if sim_time is not None:
            self.speed_t = float(sim_time)

    def predict_to(self, sim_time: float, yaw_rate: float) -> bool:
        """Carry the estimate forward to `sim_time`, turning at `yaw_rate` rad/s.

        Returns False when the step was refused, which is not a failure -- it is the filter
        declining to invent motion it cannot know about.
        """
        if not self.seeded:
            return False
        if self.t is None:
            self.t = sim_time
            return False
        dt = sim_time - self.t
        if dt < MIN_STEP_S:
            self.rejected_old += 1              # duplicate, or a measurement that overtook
            return False
        if dt > MAX_PREDICT_S:
            self.rejected_gap += 1
            self.t = sim_time                   # skip the gap; do not integrate across it
            return False

        v = -self.speed if self.reverse else self.speed
        yaw = self.x[2]
        yaw_mid = _wrap(yaw + 0.5 * yaw_rate * dt)
        step = v * dt
        c, s = math.cos(yaw_mid), math.sin(yaw_mid)

        # --- state ---
        self.x[0] += step * c
        self.x[1] += step * s
        self.x[2] = _wrap(yaw + yaw_rate * dt)
        self.distance_m += abs(step)

        # --- jacobian of that motion with respect to the state ---
        F = np.array([[1.0, 0.0, -step * s],
                      [0.0, 1.0, step * c],
                      [0.0, 0.0, 1.0]], dtype=float)

        # --- process noise, built in the frame the van is MOVING in, then rotated ---
        # along  : the speed could be a percent out
        # across : the van may not be going where its nose points (sideslip)
        along = (SPEED_SCALE_SIGMA * abs(step)) ** 2 + POS_NOISE_FLOOR_M2_PER_S * dt
        cross = (abs(step) * SLIP_SIGMA_RAD) ** 2 + POS_NOISE_FLOOR_M2_PER_S * dt
        R = np.array([[c, -s], [s, c]], dtype=float)
        Qpos = R @ np.diag([along, cross]) @ R.T
        qyaw = (GYRO_NOISE_RAD_S ** 2) * dt + (GYRO_BIAS_RAD_S * dt) ** 2 \
            + YAW_NOISE_FLOOR_RAD2_PER_S * dt
        Q = np.zeros((3, 3), dtype=float)
        Q[:2, :2] = Qpos
        Q[2, 2] = qyaw

        self.P = F @ self.P @ F.T + Q
        self.t = sim_time
        self.predicts += 1
        return True

    def correct_gnss(self, lat: float, lon: float, sim_time: float,
                     sigma_m: Optional[float] = None) -> bool:
        """Pull the estimate towards a satellite fix.

        Below GNSS_YAW_MIN_SPEED_MPS the fix still moves x and y but is not allowed to rotate
        the heading -- see the constant for why.
        """
        if not self.seeded:
            return False
        gx, gy = self.geo.to_xy(lat, lon)
        if not (math.isfinite(gx) and math.isfinite(gy)):
            self.gnss_rejected += 1
            return False
        sig = self.gnss_sigma_m if sigma_m is None else float(sigma_m)
        Rm = np.diag([sig ** 2, sig ** 2]).astype(float)
        H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        z = np.array([gx, gy], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.gnss_rejected += 1
            return False
        if abs(self.speed) < GNSS_YAW_MIN_SPEED_MPS:
            K = K.copy()
            K[2, :] = 0.0                      # position yes, heading no
            self.gnss_yaw_suppressed += 1
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        # Joseph form: stays symmetric and positive-definite over a long run, where the short
        # form quietly does not.
        A = np.eye(3) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.corrections += 1
        self.last_gnss_t = sim_time
        return True

    def correct_heading(self, yaw_meas: float, sim_time: float,
                        sigma_rad: Optional[float] = None) -> bool:
        """Tell the filter which way the van is FACING.

        This is what makes yaw observable instead of inferred. Without it the only evidence
        about heading is the direction the van appears to be travelling, which is no evidence
        at all when it is barely moving -- and which is also wrong by the sideslip angle when
        it is.

        `yaw_meas` is already in the stack's frame: the caller converts the compass with
        geo.bearing_to_yaw, which encodes the measured compass = yaw + 90 degrees. The
        innovation is WRAPPED before use, so a measurement at +179 and a state at -179 are two
        degrees apart rather than three hundred and fifty eight.
        """
        if not self.seeded:
            return False
        if not math.isfinite(yaw_meas):
            return False
        sig = COMPASS_SIGMA_RAD if sigma_rad is None else float(sigma_rad)
        H = np.array([[0.0, 0.0, 1.0]], dtype=float)
        Rm = np.array([[sig ** 2]], dtype=float)
        innov = _wrap(float(yaw_meas) - float(self.x[2]))
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        self.x = self.x + (K @ np.array([innov], dtype=float))
        self.x[2] = _wrap(self.x[2])
        A = np.eye(3) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.heading_corrections += 1
        return True

    # ------------------------------------------------------------------ output
    def pose(self) -> Optional[Pose]:
        if not self.seeded:
            return None
        return Pose(x=float(self.x[0]), y=float(self.x[1]), z=0.0, yaw=float(self.x[2]),
                    speed=(-self.speed if self.reverse else self.speed),
                    confidence=1.0, healthy=True, reason="EKF",
                    cov=self.covariance(), sim_time=self.t)

    def covariance(self) -> PoseCovariance:
        """Straight from the filter's own P. Not a growth model, not a fitted curve --
        the first uncertainty in this project that is computed rather than assumed."""
        P = self.P
        return PoseCovariance(xx=float(P[0, 0]), yy=float(P[1, 1]), yaw=float(P[2, 2]),
                              xy=float(P[0, 1]), x_yaw=float(P[0, 2]), y_yaw=float(P[1, 2]))

    def error_against(self, truth: Pose) -> dict:
        if not self.seeded or truth is None:
            return {}
        dx, dy = self.x[0] - truth.x, self.x[1] - truth.y
        c, s = math.cos(truth.yaw), math.sin(truth.yaw)
        cov = self.covariance()
        err = math.hypot(dx, dy)
        sig = math.sqrt(max(1e-12, 0.5 * (cov.xx + cov.yy)))
        yaw_err = math.degrees(_wrap(float(self.x[2]) - truth.yaw))
        return {"error_m": err, "along_m": dx * c + dy * s, "cross_m": -dx * s + dy * c,
                "yaw_err_deg": yaw_err,
                "sigma_m": sig, "sigma_yaw_deg": math.degrees(cov.sigma_yaw),
                "err_over_sigma": err / sig if sig > 0 else None,
                "yaw_over_sigma": (abs(yaw_err) / math.degrees(cov.sigma_yaw)
                                   if cov.sigma_yaw > 0 else None),
                "distance_m": self.distance_m}

    def state(self) -> dict:
        if not self.seeded:
            return {"seeded": False}
        cov = self.covariance()
        return {"seeded": True, "x": round(float(self.x[0]), 3), "y": round(float(self.x[1]), 3),
                "yaw_deg": round(math.degrees(float(self.x[2])), 3),
                "sigma_x_m": round(cov.sigma_x, 4), "sigma_y_m": round(cov.sigma_y, 4),
                "sigma_yaw_deg": round(math.degrees(cov.sigma_yaw), 4),
                "distance_m": round(self.distance_m, 2),
                "predicts": self.predicts, "corrections": self.corrections,
                "rejected_old": self.rejected_old, "rejected_gap": self.rejected_gap,
                "gnss_rejected": self.gnss_rejected,
                "heading_corrections": self.heading_corrections,
                "gnss_yaw_suppressed": self.gnss_yaw_suppressed,
                "gnss_age_s": (round(self.t - self.last_gnss_t, 2)
                               if (self.t is not None and self.last_gnss_t is not None) else None)}
