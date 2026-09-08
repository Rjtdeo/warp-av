"""
Which of the van's senses are working, and what to do when one is not.
(Perception V2, day 8.)

Before today the safety supervisor was handed a single yes-or-no for the whole
of perception. It could stop the van, but it could not say which eye had gone
dark, and the eyes it never heard about at all could fail in silence: with the
GPS switched off the van carried on at full speed for as long as it was left
that way.

This module gathers every sense into one report, and says for each one what
losing it means:

  * **stop**: the van cannot drive without it. The LiDAR finds the obstacles,
    and position tells the van where the road goes.
  * **slow**: the van can still drive, carefully. Losing the camera costs it
    the names of things, not the things themselves, and a van crawling to a
    halt is safer in a live lane than one that freezes mid-junction.
  * **note**: worth reporting and nothing more.

The verdict is a description, not an order. The safety supervisor decides.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Severity(Enum):
    STOP = "stop"        # bring the van to a halt now
    SLOW = "slow"        # keep going under a speed cap, and come to a halt if it persists
    NOTE = "note"        # tell the operator, carry on


#: what losing each sense means. The camera is "slow" rather than "stop" because the
#: LiDAR still finds everything solid; what is lost is the name on it.
SENSOR_POLICY: Dict[str, Severity] = {
    "lidar": Severity.STOP,
    "position": Severity.STOP,
    "camera": Severity.SLOW,
    "object_detection": Severity.SLOW,
    "imu": Severity.SLOW,
    "gps": Severity.NOTE,        # the position check above covers a lost fix
    "controller": Severity.STOP,
    "vehicle": Severity.STOP,
}

DEGRADED_SPEED_CAP_MPS = 2.0     # a walking pace while a sense is missing
DEGRADED_GRACE_S = 10.0          # ... and after this long, stop and wait for it to come back


@dataclass
class SensorState:
    name: str
    healthy: bool
    enabled: bool = True
    detail: str = ""
    age_s: Optional[float] = None

    @property
    def failed(self) -> bool:
        """Not working is not working.

        An earlier version excused a sensor that was switched off, on the grounds that
        somebody meant to switch it off. That is exactly how a failure is injected in this
        stack, and it meant a switched-off LiDAR came out as "all senses working" while the
        van drove on. A van that cannot see does not care why.
        """
        return not self.healthy


@dataclass
class HealthReport:
    """Every sense, and what the van should do about the worst of them."""

    sensors: List[SensorState] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    degraded_since: Optional[float] = None

    def failed(self) -> List[SensorState]:
        return [s for s in self.sensors if s.failed]

    def by_name(self, name: str) -> Optional[SensorState]:
        for s in self.sensors:
            if s.name == name:
                return s
        return None

    def worst(self) -> Severity:
        worst = Severity.NOTE
        for s in self.failed():
            sev = SENSOR_POLICY.get(s.name, Severity.NOTE)
            if sev is Severity.STOP:
                return Severity.STOP
            if sev is Severity.SLOW:
                worst = Severity.SLOW
        return worst

    def names(self, severity: Severity) -> List[str]:
        return [s.name for s in self.failed() if SENSOR_POLICY.get(s.name, Severity.NOTE) is severity]

    def reason(self) -> str:
        """A sentence naming what is wrong, for the operator and the log."""
        stop = self.names(Severity.STOP)
        slow = self.names(Severity.SLOW)
        if stop:
            return f"{', '.join(stop)} not working — stopping"
        if slow:
            return f"{', '.join(slow)} not working — driving slowly"
        note = self.names(Severity.NOTE)
        return f"{', '.join(note)} not working" if note else "all senses working"

    def speed_cap_mps(self, now: Optional[float] = None,
                      cap: float = DEGRADED_SPEED_CAP_MPS,
                      grace_s: float = DEGRADED_GRACE_S) -> Optional[float]:
        """How fast the van may go. None means no limit, 0.0 means stop.

        A sense the van can manage without earns a crawl, but not for ever: if it does
        not come back within the grace period the van stops and waits.
        """
        worst = self.worst()
        if worst is Severity.STOP:
            return 0.0
        if worst is not Severity.SLOW:
            return None
        now = time.time() if now is None else now
        if self.degraded_since is not None and (now - self.degraded_since) > grace_s:
            return 0.0
        return cap

    def as_dict(self) -> dict:
        return {"worst": self.worst().value, "reason": self.reason(),
                "failed": [s.name for s in self.failed()],
                "speed_cap_mps": self.speed_cap_mps(),
                "sensors": {s.name: {"healthy": s.healthy, "enabled": s.enabled,
                                     "detail": s.detail,
                                     "age_s": None if s.age_s is None else round(s.age_s, 2)}
                            for s in self.sensors}}


class HealthMonitor:
    """Builds the report each tick and remembers how long a sense has been missing."""

    def __init__(self):
        self._degraded_since: Optional[float] = None

    def update(self, sensors: List[SensorState], now: Optional[float] = None) -> HealthReport:
        now = time.time() if now is None else now
        report = HealthReport(sensors=list(sensors), timestamp=now)
        if report.worst() is Severity.SLOW:
            if self._degraded_since is None:
                self._degraded_since = now
        else:
            self._degraded_since = None
        report.degraded_since = self._degraded_since
        return report


def read_sensors(sensor_adapter, perception_healthy: bool, perception_reason: str,
                 pose, controller_healthy: bool, vehicle_alive: bool,
                 now: Optional[float] = None) -> List[SensorState]:
    """Ask each part of the van whether it is working. Anything missing counts as failed,
    never as fine, so a sense that disappears cannot pass unnoticed."""
    now = time.time() if now is None else now

    def ask(fn, default=False):
        try:
            return bool(fn())
        except Exception:
            return default

    def enabled(attr):
        return bool(getattr(sensor_adapter, attr, True)) if sensor_adapter is not None else True

    out = [
        SensorState("vehicle", healthy=bool(vehicle_alive), detail="the link to the van itself"),
        SensorState("controller", healthy=bool(controller_healthy), detail="steering and throttle"),
        SensorState("position", healthy=bool(getattr(pose, "healthy", False)),
                    detail=str(getattr(pose, "reason", "")),
                    age_s=max(0.0, now - float(getattr(pose, "timestamp", now) or now))),
        SensorState("object_detection", healthy=bool(perception_healthy), detail=perception_reason),
    ]
    if sensor_adapter is not None:
        # note: `enabled` is reported for the operator, never used to excuse a fault
        out += [
            SensorState("lidar", healthy=ask(sensor_adapter.is_lidar_healthy),
                        enabled=enabled("lidar_enabled"), detail="the laser that finds obstacles"),
            SensorState("camera", healthy=ask(sensor_adapter.is_camera_healthy),
                        enabled=enabled("camera_enabled"), detail="the picture that names them"),
            SensorState("gps", healthy=ask(sensor_adapter.is_gnss_healthy),
                        enabled=enabled("gnss_enabled"), detail="satellite fix"),
            SensorState("imu", healthy=ask(sensor_adapter.is_imu_healthy),
                        enabled=enabled("imu_enabled"), detail="motion sensing"),
        ]
    return out
