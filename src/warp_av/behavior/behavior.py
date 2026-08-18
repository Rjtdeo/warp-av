"""
Behavior / Decision Module

YOUR ROVER equivalent:
    decision_node.py checks front < 50, right < 20, etc.
    If front is blocked, turn right. If all clear, go forward.

THIS VERSION:
    Same idea but with named states and REASONS for every decision.
    This is the most important thing Warp is testing:
    "If the vehicle stops, we should be able to determine why."

    Every behavior change publishes a REASON STRING.
    That single feature answers half the observability questions.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..perception.perception import PerceptionOutput, ObjectType
from ..localization.localization import Pose, LocalizationQuality


class DrivingBehavior(Enum):
    IDLE = "idle"
    FOLLOWING_ROUTE = "following_route"
    APPROACHING_DESTINATION = "approaching_destination"
    STOPPED_OBSTACLE = "stopped_obstacle"
    STOPPED_PEDESTRIAN = "stopped_pedestrian"
    STOPPED_VEHICLE = "stopped_vehicle"
    STOPPED_BLOCKED = "stopped_blocked"
    STOPPED_SAFETY = "stopped_safety"
    STOPPED_ESTOP = "stopped_estop"
    MISSION_COMPLETE = "mission_complete"
    NO_MISSION = "no_mission"


@dataclass
class BehaviorOutput:
    """What the behavior layer decided to do and WHY."""
    behavior: DrivingBehavior = DrivingBehavior.IDLE
    reason: str = ""                    # THE KEY FIELD — human-readable explanation
    desired_speed_mps: float = 0.0
    should_stop: bool = False
    timestamp: float = field(default_factory=time.time)


class BehaviorSystem:
    """
    Decides what the vehicle should do based on perception + localization + mission.

    Think of it as your decision_node.make_decision() but it outputs
    a named behavior + reason instead of just "FORWARD" or "TURN_LEFT".
    """

    def __init__(self):
        self.current_behavior = DrivingBehavior.NO_MISSION
        self.current_reason = "No mission assigned"
        self.has_mission = False
        self.mission_complete = False

        # Tuning
        self.cruise_speed = 8.0          # m/s (~18 mph, good for cargo van)
        self.slow_speed = 3.0            # m/s when approaching obstacle
        self.stop_distance = 5.0         # meters — stop if object closer
        self.slow_distance = 15.0        # meters — slow down
        self.destination_threshold = 5.0 # meters — "close enough" to destination

        # If the path stays blocked for this long,
        # treat it as a blocked route instead of a temporary obstacle.
        self.blocked_timeout = 3.0
        self._blocked_since = None

    def update(
        self,
        perception: PerceptionOutput,
        pose: Pose,
        destination_distance: Optional[float],
        safety_ok: bool
    ) -> BehaviorOutput:
        """
        One decision cycle.

        Your rover did:
            if front < 20: STOP
            elif front < 50: TURN
            else: FORWARD

        This does the same thing but with richer states and always a reason.
        """

        # --- Safety override (highest priority) ---
        if not safety_ok:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                "Safety supervisor commanded stop",
                speed=0.0, stop=True
            )

        # --- No mission ---
        if not self.has_mission:
            return self._decide(
                DrivingBehavior.NO_MISSION,
                "No mission assigned — waiting for destination",
                speed=0.0, stop=True
            )

        # --- Localization lost ---
        if not pose.healthy or pose.quality == LocalizationQuality.LOST:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Localization unhealthy: {pose.reason}",
                speed=0.0, stop=True
            )

        # --- Perception unhealthy ---
        if not perception.healthy:
            return self._decide(
                DrivingBehavior.STOPPED_SAFETY,
                f"Perception unhealthy: {perception.reason}",
                speed=0.0, stop=True
            )

        # --- Arrived at destination ---
        if destination_distance is not None and destination_distance < self.destination_threshold:
            self.mission_complete = True
            self.has_mission = False
            return self._decide(
                DrivingBehavior.MISSION_COMPLETE,
                f"Arrived at destination (distance: {destination_distance:.1f}m)",
                speed=0.0, stop=True
            )

        # --- Persistent blocked route ---
        #
        # A pedestrian or stopped vehicle is a temporary road situation,
        # not automatically a "blocked route".
        #
        # Only static/other obstacles can become a persistent blocked road.
        if (
            perception.path_blocked
            and perception.closest_obstacle_type
            not in (
                ObjectType.PEDESTRIAN,
                ObjectType.VEHICLE,
            )
        ):
            if self._blocked_since is None:
                self._blocked_since = time.time()

            blocked_duration = time.time() - self._blocked_since

            if blocked_duration >= self.blocked_timeout:
                return self._decide(
                    DrivingBehavior.STOPPED_BLOCKED,
                    (
                        f"Route blocked for {blocked_duration:.1f}s by "
                        f"{perception.closest_obstacle_type.value} "
                        f"at {perception.closest_obstacle_distance:.1f}m "
                        f"— replan or operator action required"
                    ),
                    speed=0.0,
                    stop=True
                )
        else:
            self._blocked_since = None

        # --- Path blocked by pedestrian (ALWAYS stop for pedestrians) ---
        if perception.path_blocked and perception.closest_obstacle_type == ObjectType.PEDESTRIAN:
            return self._decide(
                DrivingBehavior.STOPPED_PEDESTRIAN,
                f"PEDESTRIAN in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Path blocked by vehicle ---
        if perception.path_blocked and perception.closest_obstacle_type == ObjectType.VEHICLE:
            return self._decide(
                DrivingBehavior.STOPPED_VEHICLE,
                f"VEHICLE blocking path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Path blocked by obstacle ---
        if perception.path_blocked:
            return self._decide(
                DrivingBehavior.STOPPED_OBSTACLE,
                f"OBSTACLE in path at {perception.closest_obstacle_distance:.1f}m — stopped",
                speed=0.0, stop=True
            )

        # --- Object ahead, slow down ---
        if perception.closest_obstacle_distance < self.slow_distance:
            return self._decide(
                DrivingBehavior.FOLLOWING_ROUTE,
                f"Object detected at {perception.closest_obstacle_distance:.1f}m — slowing to {self.slow_speed:.1f} m/s",
                speed=self.slow_speed, stop=False
            )

        # --- Approaching destination ---
        if destination_distance is not None and destination_distance < 20.0:
            return self._decide(
                DrivingBehavior.APPROACHING_DESTINATION,
                f"Approaching destination ({destination_distance:.1f}m) — slowing",
                speed=self.slow_speed, stop=False
            )

        # --- All clear, drive normally ---
        return self._decide(
            DrivingBehavior.FOLLOWING_ROUTE,
            f"Route clear — cruising at {self.cruise_speed:.1f} m/s",
            speed=self.cruise_speed, stop=False
        )

    def _decide(self, behavior, reason, speed, stop) -> BehaviorOutput:
        # Log when behavior CHANGES (important for debugging)
        if behavior != self.current_behavior:
            print(f"[Behavior] {self.current_behavior.value} → {behavior.value}: {reason}")
        self.current_behavior = behavior
        self.current_reason = reason
        return BehaviorOutput(
            behavior=behavior,
            reason=reason,
            desired_speed_mps=speed,
            should_stop=stop
        )

    def set_mission(self):
        self.has_mission = True
        self.mission_complete = False
        self.current_behavior = DrivingBehavior.IDLE

    def cancel_mission(self):
        self.has_mission = False
