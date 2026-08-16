"""
Vehicle Interface A-to-B Architecture Test

Purpose:
    Prove that driving commands pass through Warp's VehicleInterface.

Flow:

    CARLA BasicAgent
          |
          v
    VehicleCommand
          |
          v
    CarlaVehicleAdapter
          |
          v
    Mercedes Sprinter in CARLA

BasicAgent is used only as an open-source baseline planner/controller.
Vehicle commands are sent through our vehicle abstraction.
"""

import time
import carla

from agents.navigation.basic_agent import BasicAgent

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.vehicle_interface import VehicleCommand


TARGET_SPEED_KMH = 20.0
MISSION_TIMEOUT_SECONDS = 240


def main():

    print("")
    print("========================================")
    print(" WARP VEHICLE INTERFACE A-to-B TEST")
    print("========================================")

    adapter = None

    try:

        # --------------------------------------------------
        # 1. Create simulated vehicle through our adapter
        # --------------------------------------------------

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        vehicle = adapter.vehicle
        world = adapter.world

        print("\nVehicle Interface connected.")

        # --------------------------------------------------
        # 2. Read initial vehicle state through our interface
        # --------------------------------------------------

        initial_state = adapter.get_state()

        print(
            f"Initial position: "
            f"x={initial_state.x:.1f}, "
            f"y={initial_state.y:.1f}"
        )

        # --------------------------------------------------
        # 3. Pick destination
        # --------------------------------------------------

        spawn_points = adapter.get_spawn_points()

        start_location = vehicle.get_location()

        destination_transform = max(
            spawn_points,
            key=lambda point:
                point.location.distance(start_location)
        )

        destination = destination_transform.location

        print(
            f"Destination: "
            f"x={destination.x:.1f}, "
            f"y={destination.y:.1f}"
        )

        print(
            f"Straight-line distance: "
            f"{start_location.distance(destination):.1f} m"
        )

        # --------------------------------------------------
        # 4. Create CARLA open-source baseline agent
        # --------------------------------------------------

        agent = BasicAgent(
            vehicle,
            target_speed=TARGET_SPEED_KMH
        )

        agent.set_destination(destination)

        print("\nCARLA BasicAgent route created.")
        print(
            "BasicAgent is the OPEN-SOURCE baseline "
            "planner/controller."
        )

        # --------------------------------------------------
        # 5. Engage OUR vehicle interface
        # --------------------------------------------------

        print("\nEngaging Warp Vehicle Interface...")

        if not adapter.engage_autonomy():
            raise RuntimeError(
                "Could not engage autonomous mode."
            )

        print("")
        print("========================================")
        print(" MISSION STARTED")
        print("========================================")

        mission_start_time = time.time()
        last_print_time = 0.0

        # --------------------------------------------------
        # 6. Mission loop
        # --------------------------------------------------

        while True:

            # Check mission timeout.
            elapsed = time.time() - mission_start_time

            if elapsed > MISSION_TIMEOUT_SECONDS:

                print("\nMISSION TIMEOUT")

                adapter.emergency_stop()

                break

            # Has BasicAgent reached destination?
            if agent.done():

                # Normal stop command through our interface.
                adapter.send_command(
                    VehicleCommand(
                        throttle=0.0,
                        steering=0.0,
                        brake=1.0,
                    )
                )

                time.sleep(1.0)

                print("")
                print("========================================")
                print(" MISSION COMPLETE")
                print("========================================")

                break

            # --------------------------------------------------
            # BasicAgent calculates requested control.
            # It DOES NOT directly command the van here.
            # --------------------------------------------------

            requested_control = agent.run_step()

            # --------------------------------------------------
            # Convert CARLA control request into OUR
            # simulator-independent VehicleCommand.
            # --------------------------------------------------

            command = VehicleCommand(
                throttle=requested_control.throttle,
                steering=requested_control.steer,
                brake=requested_control.brake,
            )

            # --------------------------------------------------
            # Send command through OUR Vehicle Interface.
            # --------------------------------------------------

            command_accepted = adapter.send_command(command)

            if not command_accepted:

                print(
                    "\nVehicle Interface rejected command."
                )

                adapter.emergency_stop()

                break

            # --------------------------------------------------
            # Read vehicle state through OUR interface.
            # --------------------------------------------------

            state = adapter.get_state()

            current_location = carla.Location(
                x=state.x,
                y=state.y,
                z=state.z,
            )

            distance_to_destination = (
                current_location.distance(destination)
            )

            # Print telemetry once per second.
            now = time.time()

            if now - last_print_time >= 1.0:

                print(
                    f"Speed: {state.speed_mps * 3.6:5.1f} km/h | "
                    f"Distance: {distance_to_destination:6.1f} m | "
                    f"Throttle: {command.throttle:.2f} | "
                    f"Steer: {command.steering:+.2f} | "
                    f"Brake: {command.brake:.2f} | "
                    f"State: {state.autonomy_state.value}"
                )

                last_print_time = now

            # CARLA is running asynchronously.
            # Small sleep prevents unnecessary CPU usage.
            time.sleep(0.05)

        # --------------------------------------------------
        # 7. End mission safely
        # --------------------------------------------------

        print("\nDisengaging autonomy...")

        adapter.disengage_autonomy()

        final_state = adapter.get_state()

        print(
            f"Final speed: "
            f"{final_state.speed_mps * 3.6:.2f} km/h"
        )

    except KeyboardInterrupt:

        print("\nUser interrupted mission.")

        if adapter is not None:
            adapter.emergency_stop()

    except Exception as error:

        print(f"\nERROR: {error}")

        if adapter is not None:
            adapter.emergency_stop()

        raise

    finally:

        if adapter is not None:

            print("\nCleaning up vehicle...")
            adapter.destroy()

    print("")
    print("========================================")
    print(" VEHICLE INTERFACE A-to-B TEST FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()
