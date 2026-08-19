"""
Localization Module

YOUR ROVER equivalent:
    sensor_node.py publishes GPS string to 'sensor/gps' topic.
    decision_node doesn't really use it yet.

THIS VERSION:
    Reads vehicle position from CARLA (simulation ground truth).
    Reports position, heading, speed, and CONFIDENCE.
    The safety supervisor watches confidence — if localization is bad,
    the vehicle must stop.

    Future: fuse GNSS + IMU + odometry for real-world localization.
"""

import time
import math
from dataclasses import dataclass, field
from enum import Enum


class LocalizationQuality(Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    LOST = "lost"


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


class LocalizationSystem:
    """
    Tracks vehicle position and heading.

    Currently: reads directly from CARLA (perfect localization).
    Future: GNSS + IMU + wheel odometry fusion.
    """

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self._enabled = True
        self._last_pose: Pose = Pose()
        # Fault-injection hooks (see testing/fault_injector.py)
        self._fault = {"freeze": False, "stale_age_s": 0.0, "confidence": None, "ramp": None,
                       "offset_m": 0.0, "offset_mode": "jump", "offset_t0": 0.0, "crash": False}

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
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            yaw = math.radians(transform.rotation.yaw)

            # --- injected lateral offset (drift / jump) ---
            off = self._fault["offset_m"]
            if off and self._fault["offset_mode"] == "drift":
                off = min(off, off * (time.time() - self._fault["offset_t0"]) / 10.0)  # reach full offset in 10 s
            ox = -math.sin(yaw) * off
            oy = math.cos(yaw) * off

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
                x=transform.location.x + ox,
                y=transform.location.y + oy,
                z=transform.location.z,
                yaw=yaw,
                speed=speed,
                confidence=conf,
                quality=quality,
                timestamp=time.time() - self._fault["stale_age_s"],
                healthy=True,
                reason="OK" if conf >= 0.3 else "LOW_CONFIDENCE"
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
                       "offset_m": 0.0, "offset_mode": "jump", "offset_t0": 0.0, "crash": False}
        print("[Localization] Re-enabled")

    def inject_fault(self, action: str, **params):
        """freeze | stale(age_s) | low_confidence(value, ramp_s) | noise(offset_m, mode, confidence) | crash."""
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
        elif action == "crash":
            self._fault["crash"] = True
        else:
            return False
        print(f"[Localization] FAULT INJECTED: {action} {params}")
        return True
