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

        try:
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            pose = Pose(
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
                yaw=math.radians(transform.rotation.yaw),
                speed=speed,
                confidence=1.0,  # Perfect in simulation
                quality=LocalizationQuality.GOOD,
                timestamp=time.time(),
                healthy=True,
                reason="OK"
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
        print("[Localization] Re-enabled")
