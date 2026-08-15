"""
Telemetry Logger

YOUR ROVER equivalent:
    self.get_logger().info(f"FORWARD F:{front:.0f} L:{left:.0f} R:{right:.0f}")
    That's logging, but only to the terminal and it's gone forever.

THIS VERSION:
    Writes a JSONL file every tick with ALL system state.
    After a mission you can open the file and see exactly
    what happened, when, and why.

    One line per tick. One file per mission.
"""

import json
import time
import os
from typing import Optional


class TelemetryLogger:
    """
    Writes structured logs. One JSON line per tick.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._file = None
        self._mission_id = None

    def start_mission_log(self, mission_id: str):
        self._mission_id = mission_id
        filename = os.path.join(self.log_dir, f"{mission_id}.jsonl")
        self._file = open(filename, "w")
        print(f"[Logger] Logging to {filename}")

    def log_tick(
        self,
        pose_x: float, pose_y: float, pose_yaw: float, pose_speed: float,
        behavior: str, behavior_reason: str,
        steering: float, throttle: float, brake: float,
        safety_state: str, safety_reason: str,
        perception_objects: int, closest_obstacle: float,
        mission_state: str,
        extra: dict = None
    ):
        """Write one line of telemetry."""
        if not self._file:
            return

        entry = {
            "t": time.time(),
            "pose": {"x": pose_x, "y": pose_y, "yaw": pose_yaw, "speed": pose_speed},
            "behavior": behavior,
            "behavior_reason": behavior_reason,
            "command": {"steer": round(steering, 3), "throttle": round(throttle, 3), "brake": round(brake, 3)},
            "safety": {"state": safety_state, "reason": safety_reason},
            "perception": {"objects": perception_objects, "closest": round(closest_obstacle, 1)},
            "mission": mission_state,
        }
        if extra:
            entry["extra"] = extra

        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def log_event(self, event_type: str, description: str, data: dict = None):
        """Log a discrete event (not every-tick data)."""
        if not self._file:
            return
        entry = {
            "t": time.time(),
            "event": event_type,
            "description": description,
        }
        if data:
            entry["data"] = data
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def stop_mission_log(self):
        if self._file:
            self._file.close()
            self._file = None
            print(f"[Logger] Log closed for {self._mission_id}")

    def __del__(self):
        self.stop_mission_log()
