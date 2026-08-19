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

    def __init__(self):
        self.speed_pid = PIDController(kp=0.5, ki=0.05, kd=0.1)
        self._enabled = True
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
        should_stop: bool
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

        # Convert to steering command [-1, 1]
        # Gain of 1.5 means ~38 degrees of angle error = full lock
        steering = max(-1.0, min(1.0, angle_error * 1.5))

        # --- SPEED (PID) ---
        speed_error = desired_speed - current_speed
        pid_output = self.speed_pid.update(speed_error)

        if pid_output > 0:
            throttle = min(1.0, pid_output)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, abs(pid_output))

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
        return VehicleCommand(steering=0.0, throttle=0.0, brake=1.0)

    def disable(self):
        self._enabled = False
        print("[Controller] DISABLED")

    def enable(self):
        self._enabled = True
        self._fault_nan = False
        self._fault_stale_s = 0.0
        print("[Controller] Re-enabled")
