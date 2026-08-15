"""
Mission Manager

YOUR ROVER equivalent:
    You type START in the terminal → rover goes.
    You type STOP → rover stops.
    No concept of "go to this place."

THIS VERSION:
    Manages the full lifecycle of a mission:
    1. Receive destination
    2. Plan route
    3. Execute (behavior + control loop)
    4. Complete / fail / cancel
    5. Log everything
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from enum import Enum


class MissionState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MissionEvent:
    """Something that happened during a mission."""
    timestamp: float
    event_type: str      # "started", "behavior_change", "obstacle_detected", "stopped", etc.
    description: str
    data: dict = field(default_factory=dict)


@dataclass
class Mission:
    mission_id: str
    destination_x: float
    destination_y: float
    state: MissionState = MissionState.IDLE
    start_time: float = 0.0
    end_time: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    events: List[MissionEvent] = field(default_factory=list)
    distance_traveled: float = 0.0
    reason_ended: str = ""


class MissionManager:
    """
    Tracks missions from start to finish.
    Stores history so you can review past missions.
    """

    def __init__(self):
        self.current_mission: Optional[Mission] = None
        self.mission_history: List[Mission] = []
        self._mission_counter = 0

    def start_mission(self, dest_x: float, dest_y: float, start_x: float, start_y: float) -> Mission:
        """Begin a new mission."""
        self._mission_counter += 1
        mission = Mission(
            mission_id=f"mission_{self._mission_counter:04d}",
            destination_x=dest_x,
            destination_y=dest_y,
            state=MissionState.PLANNING,
            start_time=time.time(),
            start_x=start_x,
            start_y=start_y,
        )
        mission.events.append(MissionEvent(
            timestamp=time.time(),
            event_type="started",
            description=f"Mission started: go to ({dest_x:.1f}, {dest_y:.1f})"
        ))
        self.current_mission = mission
        print(f"[Mission] {mission.mission_id} started → ({dest_x:.1f}, {dest_y:.1f})")
        return mission

    def set_executing(self):
        if self.current_mission:
            self.current_mission.state = MissionState.EXECUTING
            self.log_event("executing", "Route planned, execution started")

    def complete_mission(self, reason: str = "Arrived at destination"):
        if self.current_mission:
            self.current_mission.state = MissionState.COMPLETED
            self.current_mission.end_time = time.time()
            self.current_mission.reason_ended = reason
            self.log_event("completed", reason)
            self.mission_history.append(self.current_mission)
            print(f"[Mission] {self.current_mission.mission_id} COMPLETED: {reason}")
            self.current_mission = None

    def fail_mission(self, reason: str):
        if self.current_mission:
            self.current_mission.state = MissionState.FAILED
            self.current_mission.end_time = time.time()
            self.current_mission.reason_ended = reason
            self.log_event("failed", reason)
            self.mission_history.append(self.current_mission)
            print(f"[Mission] {self.current_mission.mission_id} FAILED: {reason}")
            self.current_mission = None

    def cancel_mission(self):
        if self.current_mission:
            self.current_mission.state = MissionState.CANCELLED
            self.current_mission.end_time = time.time()
            self.current_mission.reason_ended = "Cancelled by operator"
            self.log_event("cancelled", "Cancelled by operator")
            self.mission_history.append(self.current_mission)
            print(f"[Mission] {self.current_mission.mission_id} CANCELLED")
            self.current_mission = None

    def pause_mission(self):
        if self.current_mission and self.current_mission.state == MissionState.EXECUTING:
            self.current_mission.state = MissionState.PAUSED
            self.log_event("paused", "Mission paused by operator")

    def resume_mission(self):
        if self.current_mission and self.current_mission.state == MissionState.PAUSED:
            self.current_mission.state = MissionState.EXECUTING
            self.log_event("resumed", "Mission resumed")

    def log_event(self, event_type: str, description: str, data: dict = None):
        if self.current_mission:
            event = MissionEvent(
                timestamp=time.time(),
                event_type=event_type,
                description=description,
                data=data or {}
            )
            self.current_mission.events.append(event)

    def get_status(self) -> dict:
        """Current status for the operator console."""
        if self.current_mission:
            m = self.current_mission
            return {
                "mission_id": m.mission_id,
                "state": m.state.value,
                "destination": {"x": m.destination_x, "y": m.destination_y},
                "duration_sec": time.time() - m.start_time,
                "event_count": len(m.events),
                "reason_ended": m.reason_ended
            }
        return {"mission_id": None, "state": "idle"}

    def get_history(self) -> List[dict]:
        """Past missions for operator console."""
        return [
            {
                "mission_id": m.mission_id,
                "state": m.state.value,
                "destination": {"x": m.destination_x, "y": m.destination_y},
                "duration_sec": m.end_time - m.start_time if m.end_time else 0,
                "event_count": len(m.events),
                "reason_ended": m.reason_ended,
            }
            for m in self.mission_history
        ]
