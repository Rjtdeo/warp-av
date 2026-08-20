"""
Vehicle Controller

YOUR ROVER equivalent:
    Your Arduino does forward(150) with a fixed PWM speed.
    No steering control — just left motor vs right motor.

THIS VERSION:
    PID speed controller (like cruise control).
    Pure pursuit steering (steers toward the next waypoint).
    Outputs VehicleCommand (steering, throttle, brake).
"""

import math
import time
from dataclasses import dataclass, field
from ..vehicle_interface import VehicleCommand, GearState


class PIDController:
    """Simple PID for speed control."""
    def __init__(self, kp=0.5, ki=0.05, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = time.time()

    def update(self, error):
        now = time.time()
        dt = now - self._prev_time
        if dt <= 0:
            dt = 0.01
        self._prev_time = now

        self._integral += error * dt
        self._integral = max(-5.0, min(5.0, self._integral))  # anti-windup

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0


class VehicleController:
    """
    Converts behavior decisions into actual vehicle commands.

    Two jobs:
    1. Steer toward the next waypoint (pure pursuit)
    2. Control speed to match desired speed (PID)
    """

    # --- Steering tuning (Troy fix #5: oscillation) ---
    # Gain scheduling: full authority at parking speed, gentle at cruise.
    STEER_GAIN_LOW = 1.5      # at <= GAIN_SPEED_LO m/s
    STEER_GAIN_HIGH = 0.55    # at >= GAIN_SPEED_HI m/s
    GAIN_SPEED_LO = 3.0
    GAIN_SPEED_HI = 10.0
    STEER_FILTER_ALPHA = 0.5  # low-pass: new = old + alpha*(raw-old) per tick (10 Hz)
    STEER_RATE_FAST = 0.30    # max steering change per tick above 5 m/s
    STEER_RATE_SLOW = 0.60    # max steering change per tick at low speed
    CT_GAIN = 0.12            # centreline correction (per metre of offset)
    CT_MAX = 0.30             # cap of that correction

    # --- Speed tuning (no more random brake taps) ---
    COAST_BAND_MPS = 0.8      # up to this much over target: coast, do not brake
    BRAKE_GAIN = 0.35         # proportional brake beyond the coast band
    SERVICE_BRAKE_CAP = 0.6   # normal slowing never exceeds this (~3.6 m/s^2);
                              # should_stop / e-stop paths still use full brake
    SPEED_SLEW_UP = 0.15      # max increase of the speed target per tick
                              # (1.5 m/s^2): stops it flooring the throttle
                              # mid-corner-exit; slowing down is never limited

    def __init__(self):
        self.speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1)
        self._enabled = True
        self._last_steer = 0.0       # low-pass / rate-limit state
        self._desired_eff = 0.0      # slew-limited speed target
        self._fault_nan = False      # inject NaN steering (command-validation test)
        self._fault_stale_s = 0.0    # back-date command timestamps

    def compute_command(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_speed: float,
        target_x: float,
        target_y: float,
        desired_speed: float,
        should_stop: bool,
        cross_track_m: float = 0.0,
    ) -> VehicleCommand:
        """
        Compute steering + throttle + brake.

        current_*: where the vehicle is now
        target_*: the waypoint to steer toward
        desired_speed: how fast behavior wants to go
        should_stop: behavior says stop NOW
        """
        if not self._enabled:
            return VehicleCommand(brake=1.0)  # fail-safe: brake

        # --- STOP ---
        if should_stop or desired_speed <= 0:
            self.speed_pid.reset()
            self._last_steer = 0.0
            self._desired_eff = 0.0
            return VehicleCommand(
                steering=0.0,
                throttle=0.0,
                brake=1.0,
                gear=GearState.DRIVE
            )

        # --- STEERING (pure pursuit) ---
        # Calculate angle to target waypoint
        dx = target_x - current_x
        dy = target_y - current_y
        target_angle = math.atan2(dy, dx)

        # Angle error (how far off we are from pointing at the target)
        angle_error = target_angle - current_yaw
        # Normalize to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Convert to steering command [-1, 1].
        # Gain is speed-scheduled: strong at parking speed, gentle at cruise —
        # a fixed 1.5 made the van overshoot and weave above ~6 m/s.
        v = max(0.0, current_speed)
        if v <= self.GAIN_SPEED_LO:
            gain = self.STEER_GAIN_LOW
        elif v >= self.GAIN_SPEED_HI:
            gain = self.STEER_GAIN_HIGH
        else:
            frac = (v - self.GAIN_SPEED_LO) / (self.GAIN_SPEED_HI - self.GAIN_SPEED_LO)
            gain = self.STEER_GAIN_LOW + (self.STEER_GAIN_HIGH - self.STEER_GAIN_LOW) * frac
        raw_steer = angle_error * gain
        # Centreline correction: pull back toward the lane centre. Pure pursuit
        # alone tolerates a steady offset in bends (kerb clipping); this term
        # cancels it. cross_track_m > 0 = left of path -> steer right (negative).
        raw_steer += max(-self.CT_MAX, min(self.CT_MAX, -self.CT_GAIN * cross_track_m))
        raw_steer = max(-1.0, min(1.0, raw_steer))

        # Low-pass + rate limit: kills tick-to-tick steering chatter without
        # touching the stop path (braking above never goes through this).
        filtered = self._last_steer + self.STEER_FILTER_ALPHA * (raw_steer - self._last_steer)
        max_step = self.STEER_RATE_FAST if v > 5.0 else self.STEER_RATE_SLOW
        steering = max(self._last_steer - max_step, min(self._last_steer + max_step, filtered))
        steering = max(-1.0, min(1.0, steering))
        self._last_steer = steering

        # --- SPEED ---
        # Slew-limit target increases (never decreases): smooth pull-away after
        # corners/stops instead of full throttle while still turning.
        if desired_speed <= self._desired_eff:
            self._desired_eff = desired_speed
        else:
            self._desired_eff = min(desired_speed,
                                    max(self._desired_eff, current_speed) + self.SPEED_SLEW_UP)

        # Throttle from the PID; braking is separate with a coast band, so a
        # small overshoot means "lift off", not "tap the brakes".
        speed_error = self._desired_eff - current_speed

        if speed_error >= 0:
            pid_output = self.speed_pid.update(speed_error)
            throttle = max(0.0, min(1.0, pid_output))
            brake = 0.0
        elif -speed_error <= self.COAST_BAND_MPS:
            self.speed_pid.reset()          # avoid integral wind-up while coasting
            throttle = 0.0
            brake = 0.0
        else:
            self.speed_pid.reset()
            throttle = 0.0
            brake = min(self.SERVICE_BRAKE_CAP, self.BRAKE_GAIN * (-speed_error - self.COAST_BAND_MPS))

        cmd = VehicleCommand(
            steering=steering,
            throttle=throttle,
            brake=brake,
            gear=GearState.DRIVE
        )
        if self._fault_nan:
            cmd.steering = float("nan")
        if self._fault_stale_s:
            cmd.timestamp -= self._fault_stale_s
        return cmd

    def inject_fault(self, action: str, **params):
        if action == "nan_command":
            self._fault_nan = True
        elif action == "stale":
            self._fault_stale_s = float(params.get("age_s", 1.0))
        else:
            return False
        print(f"[Controller] FAULT INJECTED: {action}")
        return True

    def emergency_brake(self) -> VehicleCommand:
        """Immediate full brake."""
        self.speed_pid.reset()
        self._last_steer = 0.0
        self._desired_eff = 0.0
        return VehicleCommand(steering=0.0, throttle=0.0, brake=1.0)

    def disable(self):
        self._enabled = False
        print("[Controller] DISABLED")

    def enable(self):
        self._enabled = True
        self._fault_nan = False
        self._fault_stale_s = 0.0
        print("[Controller] Re-enabled")
