"""
Localization Module

YOUR ROVER equivalent:
    sensor_node.py publishes GPS string to 'sensor/gps' topic.
    decision_node doesn't really use it yet.

THIS VERSION:
    Takes the van's pose from ONE source (localization/pose_source.py) and reports
    position, heading, speed, CONFIDENCE and an UNCERTAINTY.
    The safety supervisor watches confidence — if localization is bad,
    the vehicle must stop.

    The source is still CARLA's own answer today. What changed on 2026-09-16 is that
    it is asked in one place instead of three, and that a Pose now carries room for
    an uncertainty an estimator can fill in. Nothing here estimates anything yet.

    Future: fuse GNSS + IMU + odometry for real-world localization.
"""

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

#: A drifting fault reaches its full value over this long (see inject_fault).
FAULT_RAMP_S = 10.0


class LocalizationQuality(Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    LOST = "lost"


@dataclass(frozen=True)
class PoseCovariance:
    """How unsure the pose is, as the lower half of a 3x3 over (x, y, yaw).

    UNITS, because mixing these silently is the classic way to lose a week:

        xx, yy, xy          square metres          (m^2)
        yaw                 square radians         (rad^2)
        x_yaw, y_yaw        metre-radians          (m*rad)

    These are VARIANCES, not standard deviations. `sigma_x`, `sigma_y` and `sigma_yaw`
    below are the square roots, in metres and radians, for anyone who wants to read a
    number in the units they measure in.

    Nothing in planning, behaviour or control reads this yet, and that is deliberate:
    L1 puts the structure in place so a later estimator has somewhere honest to put its
    uncertainty, without inventing a fake one in the meantime.
    """
    xx: float = 0.0
    yy: float = 0.0
    yaw: float = 0.0
    xy: float = 0.0
    x_yaw: float = 0.0
    y_yaw: float = 0.0

    #: True when the pose came from the simulator rather than from an estimate. Zero
    #: variance is the honest description of ground truth, but a filter handed a zero
    #: covariance divides by it, so the flag says "these zeros mean truth, not certainty".
    is_truth: bool = False

    @classmethod
    def exact(cls) -> "PoseCovariance":
        """Ground truth: no error at all. Only CarlaTruthPoseSource may answer this."""
        return cls(is_truth=True)

    @classmethod
    def unknown(cls) -> "PoseCovariance":
        """No pose at all. Not zero error — unbounded error."""
        return cls(xx=float("inf"), yy=float("inf"), yaw=float("inf"))

    def as_matrix(self):
        """Row-major 3x3 over (x, y, yaw). Symmetric by construction."""
        return [[self.xx, self.xy, self.x_yaw],
                [self.xy, self.yy, self.y_yaw],
                [self.x_yaw, self.y_yaw, self.yaw]]

    @property
    def sigma_x(self) -> float:
        """Standard deviation in METRES."""
        return math.sqrt(self.xx) if self.xx >= 0 else float("nan")

    @property
    def sigma_y(self) -> float:
        """Standard deviation in METRES."""
        return math.sqrt(self.yy) if self.yy >= 0 else float("nan")

    @property
    def sigma_yaw(self) -> float:
        """Standard deviation in RADIANS."""
        return math.sqrt(self.yaw) if self.yaw >= 0 else float("nan")


@dataclass
class Pose:
    """Where the vehicle thinks it is."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0          # radians
    speed: float = 0.0        # m/s
    confidence: float = 1.0   # 0.0 to 1.0
    quality: LocalizationQuality = LocalizationQuality.GOOD
    timestamp: float = field(default_factory=time.time)
    healthy: bool = True
    reason: str = "OK"
    #: How unsure the above is. Zero-and-is_truth while the source is the simulator.
    cov: PoseCovariance = field(default_factory=PoseCovariance)
    #: The SIMULATOR's clock at the moment this pose (and its speed) was read, when the
    #: source can say. Fusion weights measurements by when they were taken, and L2-GAP found
    #: speed to be the one input that arrived with no time of its own. None off-simulator.
    sim_time: Optional[float] = None


class LocalizationSystem:
    """
    Tracks vehicle position and heading.

    Currently: reads directly from CARLA (perfect localization).
    Future: GNSS + IMU + wheel odometry fusion.
    """

    def __init__(self, source):
        """`source` is an EgoPoseSource (localization/pose_source.py).

        A raw CARLA actor is still accepted and wrapped, because the standalone demos in
        sim/ hand one straight over and there is no reason to break them for this.
        """
        if not hasattr(source, "pose"):
            from .pose_source import CarlaTruthPoseSource
            source = CarlaTruthPoseSource(source)
        self.source = source
        #: kept for anything that still reaches for the actor; the pose comes from `source`
        self.vehicle = getattr(source, "_actor", None)
        self._enabled = True
        self._last_pose: Pose = Pose()
        # Fault-injection hooks (see testing/fault_injector.py)
        self._fault = {"freeze": False, "stale_age_s": 0.0, "confidence": None, "ramp": None,
                       "offset_m": 0.0, "offset_mode": "jump", "offset_t0": 0.0, "crash": False,
                       # heading fault (2026-09-16). L0 could bend the van's idea of WHERE it
                       # was but not of WHICH WAY IT FACED, so heading tolerance was the one
                       # thing that phase could not measure.
                       "yaw_rad": 0.0, "yaw_mode": "jump", "yaw_t0": 0.0}

    def update(self) -> Pose:
        """
        Get current position estimate. Call every tick.
        """
        if not self._enabled:
            return Pose(
                healthy=False,
                reason="LOCALIZATION_DISABLED",
                quality=LocalizationQuality.LOST,
                confidence=0.0
            )

        if self._fault["crash"]:
            self._fault["crash"] = False
            raise RuntimeError("INJECTED_LOCALIZATION_CRASH")
        if self._fault["freeze"] and self._last_pose.healthy:
            return self._last_pose

        try:
            base = self.source.pose()
            if not base.healthy:
                return base
            speed = base.speed
            yaw = base.yaw

            # --- injected lateral offset (drift / jump) ---
            # Taken along the TRUE heading, before any heading fault below, so that a
            # yaw-only fault never moves x or y and the two can be read apart.
            off = self._fault["offset_m"]
            if off and self._fault["offset_mode"] == "drift":
                off = min(off, off * (time.time() - self._fault["offset_t0"]) / 10.0)  # reach full offset in 10 s
            ox = -math.sin(yaw) * off
            oy = math.cos(yaw) * off

            # --- injected heading error (drift / jump) ---
            yaw_err = self._fault["yaw_rad"]
            if yaw_err and self._fault["yaw_mode"] == "drift":
                # Ramp on the MAGNITUDE, so a negative angle ramps like a positive one.
                # (The offset ramp above is left exactly as it was: changing it would
                # change behaviour, and L1 is structural. Its min() does not ramp a
                # negative offset -- written down in the L1 report, not fixed here.)
                grown = min(1.0, max(0.0, (time.time() - self._fault["yaw_t0"]) / FAULT_RAMP_S))
                yaw_err = yaw_err * grown
            if yaw_err:
                yaw = math.atan2(math.sin(yaw + yaw_err), math.cos(yaw + yaw_err))

            # --- injected confidence (step or ramp) ---
            conf = 1.0
            if self._fault["confidence"] is not None:
                target = self._fault["confidence"]
                ramp = self._fault["ramp"]
                if ramp and ramp[1] > 0:
                    frac = min(1.0, (time.time() - ramp[0]) / ramp[1])
                    conf = 1.0 + (target - 1.0) * frac
                else:
                    conf = target
            quality = (LocalizationQuality.GOOD if conf >= 0.7 else
                       LocalizationQuality.DEGRADED if conf >= 0.3 else LocalizationQuality.LOST)

            pose = Pose(
                x=base.x + ox,
                y=base.y + oy,
                z=base.z,
                yaw=yaw,
                speed=speed,
                confidence=conf,
                quality=quality,
                timestamp=time.time() - self._fault["stale_age_s"],
                healthy=True,
                reason="OK" if conf >= 0.3 else "LOW_CONFIDENCE",
                # Whatever the source claimed. Under CARLA truth that is exact-and-flagged;
                # an injected fault does NOT widen it, and that is the point of L0's finding:
                # the van is wrong and still says it is certain. A later estimator is what
                # makes this number mean something.
                cov=base.cov,
            )
            self._last_pose = pose
            return pose

        except Exception as e:
            return Pose(
                healthy=False,
                reason=f"LOCALIZATION_ERROR: {e}",
                quality=LocalizationQuality.LOST,
                confidence=0.0
            )

    def get_last_pose(self) -> Pose:
        return self._last_pose

    def disable(self):
        """For testing Scenario 6."""
        self._enabled = False
        print("[Localization] DISABLED")

    def enable(self):
        self._enabled = True
        self._fault = {"freeze": False, "stale_age_s": 0.0, "confidence": None, "ramp": None,
                       "offset_m": 0.0, "offset_mode": "jump", "offset_t0": 0.0, "crash": False,
                       "yaw_rad": 0.0, "yaw_mode": "jump", "yaw_t0": 0.0}
        print("[Localization] Re-enabled")

    def inject_fault(self, action: str, **params):
        """freeze | stale(age_s) | low_confidence(value, ramp_s) | noise(offset_m, mode, confidence)
        | yaw(deg, mode) | crash.

        `yaw` bends the van's idea of which way it is FACING, leaving x and y alone
        (2026-09-16). Degrees in, because every other angle on this API is in degrees and a
        test asking for "two degrees of heading error" should say 2. Stored in radians.
        `mode` is jump (at once, and stays) or drift (grows to full over FAULT_RAMP_S).
        """
        if action == "freeze":
            self._fault["freeze"] = True
        elif action == "stale":
            self._fault["stale_age_s"] = float(params.get("age_s", 2.0))
        elif action == "low_confidence":
            self._fault["confidence"] = float(params.get("value", 0.0))
            ramp_s = float(params.get("ramp_s", 0.0))
            self._fault["ramp"] = (time.time(), ramp_s) if ramp_s > 0 else None
        elif action == "noise":
            self._fault["offset_m"] = float(params.get("offset_m", 1.0))
            self._fault["offset_mode"] = params.get("mode", "jump")
            self._fault["offset_t0"] = time.time()
            if "confidence" in params:
                self._fault["confidence"] = float(params["confidence"])
                self._fault["ramp"] = None
        elif action == "yaw":
            deg = float(params.get("deg", params.get("yaw_deg", 1.0)))
            mode = params.get("mode", "jump")
            if mode not in ("jump", "drift"):
                return False
            self._fault["yaw_rad"] = math.radians(deg)
            self._fault["yaw_mode"] = mode
            self._fault["yaw_t0"] = time.time()
        elif action == "crash":
            self._fault["crash"] = True
        else:
            return False
        print(f"[Localization] FAULT INJECTED: {action} {params}")
        return True
