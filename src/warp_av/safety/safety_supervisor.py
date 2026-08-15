"""
Safety Supervisor

YOUR ROVER equivalent:
    In decision_node.py: if front < 20: STOP (that's your only safety check)

THIS VERSION:
    A dedicated system that watches EVERYTHING and decides:
    "Is it safe to keep driving?"

    Warp said: "We care a lot about this."
    Warp said: "Failure behavior is a first class part of the assignment."

    This is separate from behavior. Behavior decides WHAT to do.
    Safety decides WHETHER the vehicle is ALLOWED to keep doing it.
"""

import time
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class SafetyState(Enum):
    OK = "ok"                          # All systems healthy, driving allowed
    WARNING = "warning"                # Something degraded but still safe
    INTERVENTION = "intervention"      # Safety taking control — stopping vehicle
    EMERGENCY_STOP = "emergency_stop"  # Hard stop — something very wrong


@dataclass
class SafetyCheck:
    """One thing the safety supervisor checked."""
    name: str
    passed: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SafetyOutput:
    """The safety supervisor's verdict this tick."""
    state: SafetyState
    driving_allowed: bool
    reason: str
    checks: List[SafetyCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class SafetySupervisor:
    """
    Runs every tick. Checks all systems.
    If ANYTHING is wrong enough → stops the vehicle.

    Think of this as a paranoid copilot who can grab the wheel.
    """

    def __init__(self):
        self._estop_active = False
        self._estop_reason = ""

        # Staleness thresholds (seconds)
        self.max_perception_age = 1.0
        self.max_localization_age = 1.0
        self.max_command_age = 0.5

        # Tracking
        self._last_perception_time = time.time()
        self._last_localization_time = time.time()
        self._last_command_time = time.time()

        self._warnings: List[str] = []
        self._errors: List[str] = []

    def update(
        self,
        perception_healthy: bool,
        perception_timestamp: float,
        localization_healthy: bool,
        localization_confidence: float,
        localization_timestamp: float,
        controller_healthy: bool,
        vehicle_alive: bool,
        current_speed: float,
    ) -> SafetyOutput:
        """
        Run all safety checks. Returns whether driving is allowed.
        """
        checks = []
        self._warnings = []
        self._errors = []

        # --- E-STOP (highest priority, latching) ---
        if self._estop_active:
            checks.append(SafetyCheck("emergency_stop", False, self._estop_reason))
            return SafetyOutput(
                state=SafetyState.EMERGENCY_STOP,
                driving_allowed=False,
                reason=f"EMERGENCY STOP: {self._estop_reason}",
                checks=checks
            )

        # --- Vehicle connection ---
        if not vehicle_alive:
            checks.append(SafetyCheck("vehicle_connection", False, "Vehicle connection lost"))
            self._errors.append("Vehicle connection lost")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason="Vehicle connection lost — stopping",
                checks=checks
            )
        checks.append(SafetyCheck("vehicle_connection", True, "Connected"))

        # --- Perception ---
        perception_age = time.time() - perception_timestamp
        if not perception_healthy:
            checks.append(SafetyCheck("perception", False, "Perception system unhealthy"))
            self._errors.append("Perception unhealthy")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason="Perception system unhealthy — stopping",
                checks=checks
            )
        elif perception_age > self.max_perception_age:
            checks.append(SafetyCheck("perception_staleness", False,
                f"Perception data stale ({perception_age:.1f}s old)"))
            self._errors.append(f"Perception stale: {perception_age:.1f}s")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason=f"Perception data stale ({perception_age:.1f}s) — stopping",
                checks=checks
            )
        checks.append(SafetyCheck("perception", True, "Healthy"))

        # --- Localization ---
        localization_age = time.time() - localization_timestamp
        if not localization_healthy:
            checks.append(SafetyCheck("localization", False, "Localization system unhealthy"))
            self._errors.append("Localization unhealthy")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason="Localization unhealthy — stopping",
                checks=checks
            )
        elif localization_confidence < 0.3:
            checks.append(SafetyCheck("localization_confidence", False,
                f"Low confidence: {localization_confidence:.2f}"))
            self._errors.append(f"Localization confidence low: {localization_confidence:.2f}")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason=f"Localization confidence too low ({localization_confidence:.2f}) — stopping",
                checks=checks
            )
        elif localization_age > self.max_localization_age:
            checks.append(SafetyCheck("localization_staleness", False,
                f"Localization stale ({localization_age:.1f}s)"))
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason=f"Localization stale ({localization_age:.1f}s) — stopping",
                checks=checks
            )
        checks.append(SafetyCheck("localization", True, "Healthy"))

        # --- Controller ---
        if not controller_healthy:
            checks.append(SafetyCheck("controller", False, "Controller unhealthy"))
            self._errors.append("Controller unhealthy")
            return SafetyOutput(
                state=SafetyState.INTERVENTION,
                driving_allowed=False,
                reason="Controller unhealthy — stopping",
                checks=checks
            )
        checks.append(SafetyCheck("controller", True, "Healthy"))

        # --- All checks passed ---
        return SafetyOutput(
            state=SafetyState.OK,
            driving_allowed=True,
            reason="All systems healthy",
            checks=checks
        )

    def trigger_estop(self, reason: str = "Operator commanded"):
        """Latch emergency stop. Requires explicit clear."""
        self._estop_active = True
        self._estop_reason = reason
        print(f"[Safety] *** EMERGENCY STOP: {reason} ***")

    def clear_estop(self):
        """Clear emergency stop. Vehicle goes to MANUAL, not AUTONOMOUS."""
        self._estop_active = False
        self._estop_reason = ""
        print("[Safety] E-STOP cleared")

    @property
    def is_estop(self) -> bool:
        return self._estop_active

    @property
    def warnings(self) -> List[str]:
        return self._warnings

    @property
    def errors(self) -> List[str]:
        return self._errors
