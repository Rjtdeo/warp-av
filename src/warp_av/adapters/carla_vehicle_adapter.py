"""
CARLA Vehicle Adapter

This file connects the generic Warp VehicleInterface to CARLA.

Autonomy software
        ↓
VehicleCommand
        ↓
CarlaVehicleAdapter
        ↓
CARLA simulated vehicle

Later, CarlaVehicleAdapter can be replaced by a physical vehicle adapter
without changing the rest of the autonomy stack.
"""

import carla
import time
import math

from ..vehicle_interface import (
    VehicleInterface,
    VehicleCommand,
    VehicleState,
    GearState,
    AutonomyState,
)


class CarlaVehicleAdapter(VehicleInterface):

    def __init__(
        self,
        host="localhost",
        port=2000,
        vehicle_filter="vehicle.mercedes.sprinter",
    ):
        """
        Connect to CARLA and spawn a simulated vehicle.
        """

        print("[CarlaVehicleAdapter] Connecting to CARLA...")

        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Find requested vehicle.
        matching_blueprints = self.blueprint_library.filter(vehicle_filter)

        if not matching_blueprints:
            raise RuntimeError(
                f"Vehicle blueprint not found: {vehicle_filter}"
            )

        blueprint = matching_blueprints[0]

        # Find a free spawn point.
        spawn_points = self.world.get_map().get_spawn_points()

        self.vehicle = None
        self.spawn_point = None

        for spawn_point in spawn_points:
            vehicle = self.world.try_spawn_actor(
                blueprint,
                spawn_point,
            )

            if vehicle is not None:
                self.vehicle = vehicle
                self.spawn_point = spawn_point
                break

        if self.vehicle is None:
            raise RuntimeError(
                "Could not find a free CARLA spawn point."
            )

        # Give CARLA a short moment to initialize actor state.
        # We intentionally do NOT use wait_for_tick() here because
        # we do not want the adapter constructor to block.
        time.sleep(0.5)

        self._autonomy_state = AutonomyState.MANUAL
        self._alive = True
        self._simulate_connection_loss = False   # fault injection (testing/fault_injector.py)
        self.max_command_age_s = 0.5
        self.last_command_rejected = ""

        print(
            f"[CarlaVehicleAdapter] Spawned "
            f"{vehicle_filter} at {self.spawn_point.location}"
        )

    def send_command(self, cmd: VehicleCommand) -> bool:
        """
        Send steering, throttle and brake commands to CARLA.
        """

        # Emergency stop always overrides normal commands.
        if self._autonomy_state == AutonomyState.EMERGENCY_STOP:

            control = carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=1.0,
                hand_brake=True,
            )

            self.vehicle.apply_control(control)

            return False

        # Normal driving commands are only accepted
        # when autonomy is engaged.
        if self._autonomy_state != AutonomyState.AUTONOMOUS:
            return False

        # --- command validation: a physical DBW gateway must do exactly this ---
        if not self._command_valid(cmd):
            self.vehicle.apply_control(carla.VehicleControl(brake=1.0))
            return False

        try:

            # Keep commands inside valid CARLA ranges.
            steering = max(
                -1.0,
                min(1.0, cmd.steering)
            )

            throttle = max(
                0.0,
                min(1.0, cmd.throttle)
            )

            brake = max(
                0.0,
                min(1.0, cmd.brake)
            )

            control = carla.VehicleControl(
                steer=steering,
                throttle=throttle,
                brake=brake,
                reverse=(cmd.gear == GearState.REVERSE),
            )

            self.vehicle.apply_control(control)

            return True

        except Exception as error:

            print(
                f"[CarlaVehicleAdapter] "
                f"Command failed: {error}"
            )

            self._alive = False

            return False

    def get_state(self) -> VehicleState:
        """
        Read the current vehicle state from CARLA.
        """

        try:

            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            control = self.vehicle.get_control()

            # Calculate vehicle speed.
            speed_mps = math.sqrt(
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            )

            # CARLA gives yaw in degrees.
            # Warp VehicleState stores radians.
            yaw_rad = math.radians(
                transform.rotation.yaw
            )

            # CARLA steering feedback is normalized roughly
            # between -1 and +1.
            safe_steer = max(
                -1.0,
                min(1.0, control.steer)
            )

            # Approximate physical steering angle.
            steering_angle_rad = (
                safe_steer * math.radians(70)
            )

            gear = (
                GearState.REVERSE
                if control.reverse
                else GearState.DRIVE
            )

            return VehicleState(
                speed_mps=speed_mps,
                steering_angle_rad=steering_angle_rad,
                gear=gear,
                autonomy_state=self._autonomy_state,
                x=transform.location.x,
                y=transform.location.y,
                z=transform.location.z,
                yaw=yaw_rad,
                timestamp=time.time(),
            )

        except Exception as error:

            print(
                f"[CarlaVehicleAdapter] "
                f"State read failed: {error}"
            )

            self._alive = False

            return VehicleState(
                autonomy_state=self._autonomy_state
            )

    def engage_autonomy(self) -> bool:
        """
        Enable autonomous commands.
        """

        if self._autonomy_state == AutonomyState.EMERGENCY_STOP:

            print(
                "[CarlaVehicleAdapter] "
                "Cannot engage autonomy: E-STOP active"
            )

            return False

        self._autonomy_state = AutonomyState.AUTONOMOUS

        # Disable CARLA's built-in autopilot.
        # Warp software will provide the commands.
        self.vehicle.set_autopilot(False)

        print(
            "[CarlaVehicleAdapter] "
            "AUTONOMOUS mode engaged"
        )

        return True

    def disengage_autonomy(self) -> None:
        """
        Stop autonomous operation and safely brake.
        """

        self._autonomy_state = AutonomyState.DISENGAGED

        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
        )

        self.vehicle.apply_control(control)

        print(
            "[CarlaVehicleAdapter] "
            "DISENGAGED — vehicle stopped"
        )

    def emergency_stop(self) -> None:
        """
        Immediately stop the vehicle.
        """

        self._autonomy_state = (
            AutonomyState.EMERGENCY_STOP
        )

        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
        )

        self.vehicle.apply_control(control)

        print(
            "[CarlaVehicleAdapter] "
            "*** EMERGENCY STOP ***"
        )

    def clear_emergency_stop(self) -> None:
        """
        Reset E-stop.

        After clearing, the vehicle returns to MANUAL.
        Autonomy must be explicitly engaged again.
        """

        self._autonomy_state = AutonomyState.MANUAL

        control = carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
        )

        self.vehicle.apply_control(control)

        print(
            "[CarlaVehicleAdapter] "
            "E-STOP cleared -> MANUAL"
        )

    def _command_valid(self, cmd: VehicleCommand) -> bool:
        """Reject non-finite or stale commands (fail-safe: caller brakes)."""
        vals = (cmd.steering, cmd.throttle, cmd.brake)
        if any(v != v or v in (float("inf"), float("-inf")) for v in vals):   # NaN / inf
            self.last_command_rejected = "INVALID_COMMAND: non-finite value"
            print("[CarlaVehicleAdapter] REJECTED command (non-finite) -> brake")
            return False
        age = time.time() - cmd.timestamp
        if age > self.max_command_age_s:
            self.last_command_rejected = f"STALE_COMMAND: {age:.2f}s old"
            print(f"[CarlaVehicleAdapter] REJECTED stale command ({age:.2f}s) -> brake")
            return False
        self.last_command_rejected = ""
        return True

    def simulate_connection_loss(self, lost: bool):
        """Fault injection: make is_alive() report False without touching CARLA."""
        self._simulate_connection_loss = lost
        print(f"[CarlaVehicleAdapter] simulated connection loss = {lost}")

    def is_alive(self) -> bool:
        """
        Check whether communication with the vehicle still works.
        """
        if self._simulate_connection_loss:
            return False

        try:

            if self.vehicle is None:
                return False

            if not self.vehicle.is_alive:
                self._alive = False
                return False

            self.vehicle.get_transform()

            self._alive = True

            return True

        except Exception:

            self._alive = False

            return False

    def get_spawn_points(self):
        """
        Return CARLA map spawn points.
        Useful for choosing mission destinations.
        """

        return self.world.get_map().get_spawn_points()

    def get_map(self):
        """
        Return the current CARLA map.
        """

        return self.world.get_map()

    def destroy(self):
        """
        Remove the simulated vehicle from CARLA.
        """

        if self.vehicle is not None:

            try:

                if self.vehicle.is_alive:
                    self.vehicle.destroy()

                print(
                    "[CarlaVehicleAdapter] "
                    "Vehicle destroyed"
                )

            except Exception as error:

                print(
                    "[CarlaVehicleAdapter] "
                    f"Destroy failed: {error}"
                )

            finally:

                self.vehicle = None
                self._alive = False
