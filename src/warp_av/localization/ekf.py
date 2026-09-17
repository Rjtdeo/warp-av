"""The first fused estimate of where the van is: gyro and wheels to carry it, GNSS to hold it.

Phase L3. Three inputs, three states, one correction. Deliberately the smallest filter that
can be honest, because a filter nobody can debug is worse than dead reckoning.

    STATE      x, y, yaw, gyro_bias, compass_bias
               metres, metres, radians, radians/second, radians
    PREDICT    wheel speed + gyro yaw rate, over the ACTUAL time between measurements
    CORRECT    GNSS position, converted to local metres (localization/geo.py)

IT DRIVES NOTHING. Like the dead reckoner beside it, this runs in shadow: it publishes to
telemetry so a drive can be scored against CARLA's truth afterwards, and no planner, behaviour
rule, controller, perception path or traffic-light lookup reads a single number it produces.

WHY THE FOURTH STATE IS THE GYRO BIAS (L6), AND WHY IT USED TO BE THE COMPASS BIAS.

L4.1 put the COMPASS offset in the state, and it failed instructively. The compass measures
yaw + offset; GNSS and the motion model say where the van actually WENT, which is its heading
plus the sideslip angle. Asked to reconcile two things that disagree for two different
reasons, the offset state absorbed both: it converged on 0.55-0.68 degrees against an injected
0.30, close to offset-minus-sideslip, and heading got WORSE. There was no third opinion to
break the tie.

L5 built that third opinion. Registering one LiDAR sweep against the next measures rotation
against the standing world -- it owes nothing to the compass and nothing to the direction of
travel -- and it is essentially unbiased: a signed mean of +0.0001 degrees against 0.147 of
noise, over 3,879 live measurements. THAT is what makes gyro bias observable: the gyro says
the van turned by omega*dt, the LiDAR says it turned by something else, and the difference
accumulated over many sweeps is the bias.

So the compass is gone from the fusion entirely -- main.py still computes and reports it for
comparison, and nothing in here reads it. Its 0.30-degree offset is twice the whole heading
budget, and L4.1 proved that estimating that offset makes matters worse rather than better.
What replaces it is a quantity a second, independent instrument can actually pin down.

WHY SPEED IS STILL NOT A STATE. It is measured directly and its error is small and roughly
multiplicative -- L2-GAP put sampled-speed integration within 0.14 % of the true path.

SIDESLIP IS NOT CORRECTED HERE. L2-GAP measured the van travelling 0.756 degrees off its own
heading on average and up to 3.151, and that mismatch is the largest single error in the dead
reckoner. It is entered as PROCESS NOISE across the direction of travel, not as a magic
correction term: the filter is told the motion model is uncertain sideways, and GNSS is left
to show whether that is enough. If it is not, the evidence will say so.
"""
from __future__ import annotations

import math
import os
from collections import deque
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
#: Gyro white noise, matching the noisy_sim sensor profile. The gyro's BIAS used to be a
#: process-noise term here as well. It is not any more, and must not come back: since L6 the
#: bias is a STATE with its own uncertainty and its own random walk, so adding it to the yaw
#: process noise would charge for the same doubt twice -- and would put the injected value
#: back inside the filter, which is exactly what it must not know.
GYRO_NOISE_RAD_S = 0.002
#: A floor, so a stationary filter still loosens slightly rather than freezing its covariance
#: and refusing every future correction.
POS_NOISE_FLOOR_M2_PER_S = 1e-4
YAW_NOISE_FLOOR_RAD2_PER_S = 1e-8

# ---- measurement noise -----------------------------------------------------------------
#: What the GNSS is believed to be worth, in metres, one standard deviation per axis. This is
#: the CONFIGURED sensor model (adapters/carla_sensor_adapter.GNSS_NOISE_M), not a number
#: tuned until the score looked good.
GNSS_SIGMA_M = 0.02

#: ...and the part of the GNSS error that does NOT average away. The CONFIGURED receiver bias
#: (adapters/carla_sensor_adapter.GNSS_BIAS_M), drawn once and held for the drive.
#:
#: This is the whole reason the position covariance has been dishonest since L3, and the reason
#: is worth writing down because it is invisible from inside the filter. Fed a stream of fixes
#: that all lean the same way, the filter converges onto the LEANED position and its
#: innovations go quiet -- the measurements agree with the estimate, because the estimate has
#: moved to where the measurements are. The normalised innovation squared stays at 1.83 with a
#: centimetre of offset present, exactly as it does with none: measured, not assumed.
#:
#: So P collapses towards the white-noise floor while the true error keeps the offset in it.
#: A filter cannot estimate this away -- nothing else in the van measures absolute position --
#: but it must not claim to be surer than the offset it cannot see.
GNSS_BIAS_M = 0.01

#: What the compass is worth as an ABSOLUTE heading anchor, and why it is back.
#:
#: L6 first removed it outright. That was wrong, and the scoring said so: heading on
#: turn-heavy routes came out at 2.5-3.9 degrees mean against L4's 0.35, because nothing was
#: left holding the absolute angle down. LiDAR measures ROTATION -- how far the van swung --
#: and a rotation carries no information about where the swing started. GNSS anchors heading
#: only through the direction of travel, and that coupling is deliberately gated off below
#: 2 m/s, which is exactly when a van turns. Seed, then drift.
#:
#: It is an ANCHOR, not a heading source: at this weight, once GNSS has pinned the heading,
#: a single reading barely moves the estimate, and what it buys is that the angle cannot walk
#: away over a route.
#:
#: L7 narrows this back to the compass's NOISE alone. L6 widened it to 1.2 degrees to cover an
#: offset the filter could not remove; the offset is now state 4, so paying for it twice would
#: only throw away a sensor that is right on average.
COMPASS_SIGMA_RAD = math.radians(1.0)

#: What the filter believes about the compass offset before it has seen anything -- WHEN the
#: state is switched on at all, which by default it is not.
#:
#: L7 added this state and measured it doing harm. On live routes it settled between -3.15 and
#: +1.20 degrees against an injected +0.30, differing run to run, and heading came out 2.4x
#: worse than without it (0.77 degrees mean against 0.63). The cause is not tuning. The compass
#: measures yaw + offset; GNSS measures yaw + sideslip; that is two equations in three
#: unknowns. LiDAR constrains how yaw CHANGES, not where it is, so it supplies no third -- and
#: the estimate lands on offset-minus-sideslip whether LiDAR is feeding it or not, to four
#: decimal places.
#:
#: The state is kept, switched off, so the experiment can be re-run rather than re-argued.
#: WARP_EKF_COMPASS_BIAS=1 turns it on. Off, its prior and its process noise are both zero, so
#: it cannot move and the filter behaves exactly as four states.
COMPASS_BIAS_SIGMA0_RAD = math.radians(2.0)


def compass_bias_enabled(env=None) -> bool:
    """Whether to ESTIMATE the compass offset. Off by default -- see above for the measurement
    that decided it, and tests/test_ekf.py for the proof that it cannot be separated from the
    sideslip with the instruments this van carries."""
    env = os.environ if env is None else env
    return str(env.get("WARP_EKF_COMPASS_BIAS", "0")).strip().lower() in ("1", "on", "true", "yes")


#: How fast that offset may wander. A magnetometer's offset moves with temperature and with
#: what is parked next to it, over minutes rather than frames. At this rate an unobserved
#: offset loosens by about 0.06 degrees over 100 seconds.
COMPASS_BIAS_RW_RAD_PER_SQRT_S = math.radians(0.006)

#: What the filter believes about the GYRO's offset before it has seen anything: not much,
#: to within half a milliradian per second. Comfortably wider than anything the simulated
#: sensor does, so the filter has room to find a value rather than being handed one. The
#: injected figure is deliberately not written down anywhere in this module, and a test
#: asserts it does not appear.
GYRO_BIAS_SIGMA0_RAD_S = 5e-4

#: How fast that offset may wander, as a random walk per square-root second. A MEMS gyro's
#: bias drifts with temperature over minutes, not frames. At this rate an unobserved bias
#: loosens by 1e-5 rad/s over 100 seconds -- enough to follow a slow drift, far too slow to
#: absorb a single bad sweep.
GYRO_BIAS_RW_RAD_S_PER_SQRT_S = 1e-6

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

    def __init__(self, geo: Optional[GeoFrame] = None, gnss_sigma_m: float = GNSS_SIGMA_M,
                 estimate_compass_bias: Optional[bool] = None):
        self.geo = geo or DEFAULT_GEO
        self.estimate_compass_bias = (compass_bias_enabled() if estimate_compass_bias is None
                                      else bool(estimate_compass_bias))
        self.gnss_sigma_m = float(gnss_sigma_m)
        # [x, y, yaw, gyro bias rad/s, compass bias rad]
        self.x = np.zeros(5, dtype=float)
        self.P = np.zeros((5, 5), dtype=float)
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
        self.gnss_yaw_suppressed = 0
        self.lidar_corrections = 0
        self.lidar_rejected = 0
        self.compass_corrections = 0
        # Filter consistency, measured WITHOUT truth (L7). The normalised innovation squared
        # is the standard test: if the filter's idea of its own uncertainty is right, a 2-D
        # GNSS innovation scores 2 on average. Much more than 2 and it is overconfident about
        # something -- and unlike an error-versus-sigma comparison, this needs no simulator.
        self.nis_gnss = deque(maxlen=400)
        self.innov_gnss = deque(maxlen=400)
        self.distance_m = 0.0
        self.last_gnss_t = None
        self.last_lidar_t = None
        #: the newest RAW gyro reading, and the running integral of it with the times at which
        #: each sample landed. The LiDAR correction needs the angle the gyro turned through
        #: over the SAME window the scan match covers -- see correct_lidar_yaw_rate.
        self._last_gyro = 0.0
        self._gyro_angle = 0.0
        self._gyro_hist = deque(maxlen=64)

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
        # The gyro offset starts at ZERO and unknown to within GYRO_BIAS_SIGMA0. It is never
        # seeded from truth, and nothing here knows what the simulated offset is.
        self.x = np.array([float(x), float(y), _wrap(float(yaw)), 0.0, 0.0], dtype=float)
        self.P = np.diag([pos_sigma_m ** 2, pos_sigma_m ** 2,
                          math.radians(yaw_sigma_deg) ** 2,
                          GYRO_BIAS_SIGMA0_RAD_S ** 2,
                          (COMPASS_BIAS_SIGMA0_RAD ** 2
                           if self.estimate_compass_bias else 0.0)]).astype(float)
        self.seeded = True
        self.t = sim_time
        self.speed = 0.0
        self.reverse = False
        self.speed_t = sim_time
        self.predicts = self.corrections = 0
        self.rejected_old = self.rejected_gap = self.gnss_rejected = 0
        self.gnss_yaw_suppressed = 0
        self.lidar_corrections = self.lidar_rejected = 0
        self.compass_corrections = 0
        self.nis_gnss.clear()
        self.innov_gnss.clear()
        self.distance_m = 0.0
        self.last_gnss_t = None
        self.last_lidar_t = None
        self._last_gyro = 0.0
        self._gyro_angle = 0.0
        self._gyro_hist.clear()

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
        """Carry the estimate forward to `sim_time`, turning at the gyro's `yaw_rate` rad/s.

        `yaw_rate` is the RAW reading. The filter's own estimate of the gyro's offset is
        subtracted here, which is the only place that state does any work:

            turn actually taken  =  what the gyro said  -  what we think it is out by

        Returns False when the step was refused, which is not a failure -- it is the filter
        declining to invent motion it cannot know about.
        """
        if not self.seeded:
            return False
        # The raw gyro, integrated. Trapezoid over the step, so a rate that is changing inside
        # the step is accounted for rather than held at its newest value.
        if self.t is not None and math.isfinite(yaw_rate):
            step_dt = float(sim_time) - self.t
            if 0.0 < step_dt <= MAX_PREDICT_S:
                self._gyro_angle += 0.5 * (self._last_gyro + float(yaw_rate)) * step_dt
                self._gyro_hist.append((float(sim_time), self._gyro_angle))
        self._last_gyro = float(yaw_rate)
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
        rate = float(yaw_rate) - float(self.x[3])          # the gyro, less its estimated offset
        yaw_mid = _wrap(yaw + 0.5 * rate * dt)
        step = v * dt
        c, s = math.cos(yaw_mid), math.sin(yaw_mid)

        # --- state ---
        self.x[0] += step * c
        self.x[1] += step * s
        self.x[2] = _wrap(yaw + rate * dt)
        self.distance_m += abs(step)

        # --- jacobian of that motion with respect to the state ---
        # Unlike the compass offset it replaced, the gyro offset DOES move the van: every
        # degree per second of it is a degree per second the heading is carried the wrong way,
        # and the position follows the heading. So its column is not inert.
        #   d(yaw)/d(bias) = -dt        the offset is subtracted from the rate
        #   d(x)/d(bias)   = +step*s*dt/2 , d(y)/d(bias) = -step*c*dt/2
        #                               through the midpoint heading used for the step
        # The compass offset moves nothing: it is a property of an instrument, not of the
        # van, so its row and column really are inert here. The gyro offset's are not.
        F = np.array([[1.0, 0.0, -step * s, 0.5 * dt * step * s, 0.0],
                      [0.0, 1.0, step * c, -0.5 * dt * step * c, 0.0],
                      [0.0, 0.0, 1.0, -dt, 0.0],
                      [0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

        # --- process noise, built in the frame the van is MOVING in, then rotated ---
        # along  : the speed could be a percent out
        # across : the van may not be going where its nose points (sideslip)
        along = (SPEED_SCALE_SIGMA * abs(step)) ** 2 + POS_NOISE_FLOOR_M2_PER_S * dt
        cross = (abs(step) * SLIP_SIGMA_RAD) ** 2 + POS_NOISE_FLOOR_M2_PER_S * dt
        R = np.array([[c, -s], [s, c]], dtype=float)
        Qpos = R @ np.diag([along, cross]) @ R.T
        qyaw = (GYRO_NOISE_RAD_S ** 2) * dt + YAW_NOISE_FLOOR_RAD2_PER_S * dt
        Q = np.zeros((5, 5), dtype=float)
        Q[:2, :2] = Qpos
        Q[2, 2] = qyaw
        Q[3, 3] = (GYRO_BIAS_RW_RAD_S_PER_SQRT_S ** 2) * dt
        Q[4, 4] = ((COMPASS_BIAS_RW_RAD_PER_SQRT_S ** 2) * dt
                   if self.estimate_compass_bias else 0.0)

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
        H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]], dtype=float)
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
            # Position yes, heading no -- and that means BOTH heading states. Zeroing only the
            # yaw row would leave the same bad inference a door: a few centimetres of residual
            # at a crawl would flow into the compass offset instead, where it would persist
            # long after the van sped up. The offset is a heading quantity; it is gated with
            # the heading.
            K[2, :] = 0.0
            K[3, :] = 0.0
            K[4, :] = 0.0
            self.gnss_yaw_suppressed += 1
        try:
            self.nis_gnss.append(float(y @ np.linalg.inv(S) @ y))
            self.innov_gnss.append((float(sim_time), float(y[0]), float(y[1])))
        except np.linalg.LinAlgError:
            pass
        self.x = self.x + K @ y
        self.x[2] = _wrap(self.x[2])
        # Joseph form: stays symmetric and positive-definite over a long run, where the short
        # form quietly does not.
        A = np.eye(5) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.corrections += 1
        self.last_gnss_t = sim_time
        return True

    def correct_lidar_yaw_rate(self, delta_yaw: float, dt: float, sim_time: float,
                               sigma_rad: float) -> bool:
        """How fast the van REALLY turned, measured against the standing world.

        LiDAR scan matching is used as a RELATIVE rotation, never as a heading. It has no idea
        which way is north and no opinion about it; what it knows is how far the van swung
        between two sweeps a tenth of a second apart.

        Entered as an ANGLE over the same window the scan match covers:

            measured        delta_yaw
            model           the angle the GYRO turned through over that window,
                            less the offset we are estimating, times how long it lasted
            H               [0, 0, 0, -dt]

        WHY THE WINDOW MATTERS, and it does. The first version compared the LiDAR's average
        rate against the newest single gyro sample. Standing still or cruising that is nearly
        the same thing; in a turn it is not remotely, because the rate is changing inside the
        very interval being measured. Scoring found the consequence: on turn-heavy routes the
        gyro-offset estimate -- a quantity that should creep -- swung between -0.20 and +0.08
        deg/s and dragged the heading seven degrees with it. At 14.6 deg/s of turn, the gap
        between the instantaneous sample and the window average is several deg/s against a
        sigma of 0.6, which is an eight-sigma kick straight into the offset. Comparing angle
        with angle over the same window removes the whole effect.

        so the residual falls on the GYRO OFFSET, which is the thing LiDAR can genuinely see
        and the gyro cannot. Heading itself is then corrected through the yaw-to-offset
        correlation the prediction has been building all along: every step has carried yaw
        forward using (gyro - offset), so the two are coupled, and pinning one moves the other.

        WHY A RATE, AND NOT A HEADING INCREMENT ON THE STATE. The obvious alternative is to
        keep the previous yaw in the state and measure the difference directly (stochastic
        cloning). It was not chosen, and the reason is arithmetic rather than taste. Such an
        update lowers the heading variance only when the measurement is sharper than the
        prediction it is correcting, and over one sweep it is not: LiDAR's calibrated sigma of
        0.06 degrees is 1.1e-6 rad^2, while the gyro accumulates 4e-7 rad^2 over the same tenth
        of a second. Over ONE interval the gyro is the better instrument by roughly a factor of
        three. What the gyro cannot do is audit itself over a long run, and that -- not
        short-term sharpness -- is what LiDAR is here for.

        A consequence worth stating plainly: this can never shrink the ABSOLUTE heading
        uncertainty, because a relative measurement carries no information about where the
        heading started. It slows the growth. Bounding absolute heading is GNSS's job.

        `sigma_rad` is the CALIBRATED uncertainty from the measurement
        (lidar_odometry.calibrated_sigma_yaw_rad), not the registration's own optimistic one.
        The caller is responsible for having refused a measurement that failed its gates; this
        method assumes it is being handed something the odometer was willing to stand behind.
        """
        if not self.seeded:
            return False
        if not (math.isfinite(delta_yaw) and math.isfinite(dt) and dt > 1e-6):
            self.lidar_rejected += 1
            return False
        if not (math.isfinite(sigma_rad) and sigma_rad > 0.0):
            self.lidar_rejected += 1
            return False
        gyro_angle = self._gyro_angle_over(float(sim_time) - float(dt), float(sim_time))
        if gyro_angle is None:
            self.lidar_rejected += 1
            return False
        z = _wrap(float(delta_yaw))
        H = np.array([[0.0, 0.0, 0.0, -float(dt), 0.0]], dtype=float)
        Rm = np.array([[float(sigma_rad) ** 2]], dtype=float)
        innov = z - (gyro_angle - float(self.x[3]) * float(dt))
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.lidar_rejected += 1
            return False
        self.x = self.x + (K @ np.array([innov], dtype=float))
        self.x[2] = _wrap(self.x[2])
        A = np.eye(5) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.lidar_corrections += 1
        self.last_lidar_t = sim_time
        return True

    # ------------------------------------------------------------------ output
    def correct_compass(self, yaw_meas: float, sim_time: float,
                        sigma_rad: Optional[float] = None) -> bool:
        """The one thing that knows which way is north, fused weakly on purpose.

        This is an ABSOLUTE heading measurement: H touches yaw alone. It is not trusted to say
        where the van is pointing -- its offset is twice the heading budget -- only to stop the
        estimate walking away over a route, which is a job nothing else here can do. The offset
        is deliberately NOT estimated: L4.1 tried that and the offset state absorbed the
        sideslip instead, making heading worse.
        """
        if not self.seeded or not math.isfinite(yaw_meas):
            return False
        sig = COMPASS_SIGMA_RAD if sigma_rad is None else float(sigma_rad)
        H = np.array([[0.0, 0.0, 1.0, 0.0, 1.0]], dtype=float)
        Rm = np.array([[sig ** 2]], dtype=float)
        innov = _wrap(float(yaw_meas) - (float(self.x[2]) + float(self.x[4])))
        S = H @ self.P @ H.T + Rm
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        self.x = self.x + (K @ np.array([innov], dtype=float))
        self.x[2] = _wrap(self.x[2])
        self.x[4] = _wrap(self.x[4])
        A = np.eye(5) - K @ H
        self.P = A @ self.P @ A.T + K @ Rm @ K.T
        self.compass_corrections += 1
        return True

    def _gyro_angle_over(self, t0: float, t1: float) -> Optional[float]:
        """How far the RAW gyro says the van turned between two instants, interpolated out of
        the running integral. None when the history does not cover the window, which is the
        honest answer rather than an extrapolation."""
        h = self._gyro_hist
        if len(h) < 2:
            return None
        # A few milliseconds of slack at each end: a gyro sample landing on the same simulator
        # tick as the scan is refused by predict_to as a duplicate, so the history can stop a
        # hair short of the window without anything being wrong. More than that is a real gap
        # and the answer is None rather than an extrapolation.
        slack = 5e-3
        if t0 < h[0][0] - slack or t1 > h[-1][0] + slack:
            return None
        t0 = min(max(t0, h[0][0]), h[-1][0])
        t1 = min(max(t1, h[0][0]), h[-1][0])

        def at(t):
            lo, hi = 0, len(h) - 1
            while lo < hi:                       # the history is short; a plain walk is fine
                mid = (lo + hi) // 2
                if h[mid][0] < t:
                    lo = mid + 1
                else:
                    hi = mid
            if lo == 0:
                return h[0][1]
            a, b = h[lo - 1], h[lo]
            span = b[0] - a[0]
            f = 0.0 if span <= 0 else (t - a[0]) / span
            return a[1] + (b[1] - a[1]) * f
        return at(t1) - at(t0)

    def pose(self) -> Optional[Pose]:
        if not self.seeded:
            return None
        return Pose(x=float(self.x[0]), y=float(self.x[1]), z=0.0, yaw=float(self.x[2]),
                    speed=(-self.speed if self.reverse else self.speed),
                    confidence=1.0, healthy=True, reason="EKF",
                    cov=self.covariance(), sim_time=self.t)

    def pose_at(self, now: float) -> Optional[Pose]:
        """The same pose, with its AGE charged to the uncertainty. What a consumer should ask
        for, because what it gets is where the van was when the last measurement landed."""
        p = self.pose()
        if p is not None:
            p.cov = self.covariance(now=now)
        return p

    @property
    def compass_bias_rad(self) -> float:
        """The compass offset the filter has worked out for itself, in radians."""
        return float(self.x[4])

    @property
    def compass_bias_sigma_rad(self) -> float:
        v = float(self.P[4, 4])
        return math.sqrt(v) if v > 0 else float("nan")

    @property
    def gyro_bias_rad_s(self) -> float:
        """The gyro offset the filter has worked out for itself, in radians per second."""
        return float(self.x[3])

    @property
    def gyro_bias_sigma_rad_s(self) -> float:
        """How sure it is of that. Starts at GYRO_BIAS_SIGMA0_RAD_S and comes down only when
        LiDAR is agreeing or disagreeing with the gyro -- without it, nothing observes this."""
        v = float(self.P[3, 3])
        return math.sqrt(v) if v > 0 else float("nan")

    def covariance(self, now: Optional[float] = None) -> PoseCovariance:
        """The filter's own P, plus the two things P structurally cannot contain.

        Both are physical, both come from configured values rather than from scoring, and
        neither is a multiplier chosen until the containment looked better:

        THE GNSS OFFSET. See GNSS_BIAS_M. An error every fix shares does not average away, and
        the filter converges onto it without ever seeing it.

        AGE. The state stands at the last measurement's timestamp, not at the instant somebody
        reads it. A consumer reading a pose `now` is reading where the van WAS, and at 4 m/s a
        tenth of a second is 0.4 m. Pass `now` and that is charged for honestly; leave it out
        and the answer describes the pose at its own timestamp, which is what it always did.
        """
        P = self.P
        extra = GNSS_BIAS_M ** 2
        if now is not None and self.t is not None:
            age = float(now) - float(self.t)
            if age > 0.0:
                extra += (abs(self.speed) * age) ** 2
        return PoseCovariance(xx=float(P[0, 0]) + extra, yy=float(P[1, 1]) + extra,
                              yaw=float(P[2, 2]),
                              xy=float(P[0, 1]), x_yaw=float(P[0, 2]), y_yaw=float(P[1, 2]))

    def error_against(self, truth: Pose) -> dict:
        """Scored against truth AT THE TRUTH'S OWN MOMENT, and the age charged for.

        The estimate stands at the last measurement it was given; the truth it is compared
        against was read at the tick. Those differ by about twelve milliseconds here, which
        sounds like nothing and is four centimetres at 4 m/s -- the largest single contributor
        to the position error this project has been reporting. It is a real error for anyone
        reading the pose, so it is charged to the uncertainty rather than explained away.
        """
        if not self.seeded or truth is None:
            return {}
        dx, dy = self.x[0] - truth.x, self.x[1] - truth.y
        c, s = math.cos(truth.yaw), math.sin(truth.yaw)
        cov = self.covariance(now=truth.sim_time)
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
                # The fourth state and how sure it is. Nothing told the filter this number.
                "gyro_bias_deg_s": round(math.degrees(self.gyro_bias_rad_s), 5),
                "sigma_gyro_bias_deg_s": round(math.degrees(self.gyro_bias_sigma_rad_s), 5),
                "compass_bias_deg": round(math.degrees(self.compass_bias_rad), 4),
                "sigma_compass_bias_deg": round(math.degrees(self.compass_bias_sigma_rad), 4),
                "sigma_x_m": round(cov.sigma_x, 4), "sigma_y_m": round(cov.sigma_y, 4),
                "sigma_yaw_deg": round(math.degrees(cov.sigma_yaw), 4),
                "distance_m": round(self.distance_m, 2),
                "predicts": self.predicts, "corrections": self.corrections,
                "rejected_old": self.rejected_old, "rejected_gap": self.rejected_gap,
                "gnss_rejected": self.gnss_rejected,
                "lidar_corrections": self.lidar_corrections,
                "compass_corrections": self.compass_corrections,
                "lidar_rejected": self.lidar_rejected,
                "gnss_yaw_suppressed": self.gnss_yaw_suppressed,
                "lidar_age_s": (round(self.t - self.last_lidar_t, 2)
                                if (self.t is not None and self.last_lidar_t is not None) else None),
                # The filter's OWN clock, so the gap between it and the moment a consumer
                # reads the pose can be measured rather than guessed at.
                "clock": (round(float(self.t), 4) if self.t is not None else None),
                "nis_gnss": (round(sum(self.nis_gnss) / len(self.nis_gnss), 3)
                             if self.nis_gnss else None),
                "gnss_age_s": (round(self.t - self.last_gnss_t, 2)
                               if (self.t is not None and self.last_gnss_t is not None) else None)}
