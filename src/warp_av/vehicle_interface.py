"""
Vehicle Interface — the contract between autonomy software and any vehicle.

YOUR ROVER equivalent:
    motor_node.py sends "FORWARD", "STOP" etc to Arduino via serial.
    That's a vehicle interface — just a simple one.

THIS VERSION:
    Any vehicle (CARLA sim or future real cargo van) must implement these methods.
    The autonomy stack calls these methods and doesn't care what's underneath.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time


class GearState(Enum):
    PARK = "park"
    DRIVE = "drive"
    REVERSE = "reverse"
    NEUTRAL = "neutral"


class AutonomyState(Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    EMERGENCY_STOP = "emergency_stop"
    DISENGAGED = "disengaged"


@dataclass
class VehicleCommand:
    """What the autonomy stack tells the vehicle to do. Like your 'FORWARD' / 'STOP' commands but with real values."""
    steering: float = 0.0      # -1.0 (full left) to 1.0 (full right)
    throttle: float = 0.0      # 0.0 to 1.0
    brake: float = 0.0         # 0.0 to 1.0
    gear: GearState = GearState.DRIVE
    timestamp: float = field(default_factory=time.time)


@dataclass
class VehicleState:
    """What the vehicle reports back. Like reading encoder ticks but for a car."""
    speed_mps: float = 0.0           # meters per second
    steering_angle_rad: float = 0.0  # current steering angle in radians
    gear: GearState = GearState.PARK
    autonomy_state: AutonomyState = AutonomyState.MANUAL
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0                # heading in radians
    timestamp: float = field(default_factory=time.time)


class VehicleInterface(ABC):
    """
    Abstract vehicle interface.

    CARLA implements this now.
    A real cargo van implements this later.
    The autonomy stack never imports CARLA directly — only this interface.
    """

    @abstractmethod
    def send_command(self, cmd: VehicleCommand) -> bool:
        """Send a drive command. Returns True if accepted."""
        ...

    @abstractmethod
    def get_state(self) -> VehicleState:
        """Read current vehicle state."""
        ...

    @abstractmethod
    def engage_autonomy(self) -> bool:
        """Enable autonomous mode. Returns True if successful."""
        ...

    @abstractmethod
    def disengage_autonomy(self) -> None:
        """Return to manual mode."""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediate full brake. Must always work."""
        ...

    @abstractmethod
    def is_alive(self) -> bool:
        """Check if vehicle connection is healthy."""
        ...


class PhysicalVehicleAdapter(VehicleInterface):
    """
    STUB — Future adapter for real cargo van via CAN bus.
    Not implemented during 7-day trial.
    Exists to prove the interface is designed for physical hardware.
    """

    def send_command(self, cmd: VehicleCommand) -> bool:
        raise NotImplementedError("Physical vehicle adapter not yet implemented. Requires CAN bus integration.")

    def get_state(self) -> VehicleState:
        raise NotImplementedError("Physical vehicle adapter not yet implemented.")

    def engage_autonomy(self) -> bool:
        raise NotImplementedError("Physical vehicle adapter not yet implemented.")

    def disengage_autonomy(self) -> None:
        raise NotImplementedError("Physical vehicle adapter not yet implemented.")

    def emergency_stop(self) -> None:
        raise NotImplementedError("Physical vehicle adapter not yet implemented.")

    def is_alive(self) -> bool:
        raise NotImplementedError("Physical vehicle adapter not yet implemented.")
