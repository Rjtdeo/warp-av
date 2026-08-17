"""
Warp AV - Mission Manager + Vehicle Interface A-to-B Test

This test proves:

Mission Manager
      ↓
CARLA BasicAgent
      ↓
VehicleCommand
      ↓
CarlaVehicleAdapter
      ↓
Mercedes Sprinter
      ↓
Destination

BasicAgent is CARLA's open-source baseline planner/controller.
Warp's Mission Manager tracks the mission lifecycle.
Warp's Vehicle Interface sends commands to the vehicle.
"""

import time
import carla

from agents.navigation.basic_agent import BasicAgent

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.vehicle_interface import VehicleCommand
from warp_av.mission.mission_manager import MissionManager


TARGET_SPEED_KMH = 20.0
MISSION_TIMEOUT_SECONDS = 240


def main():

    print("")
    print("========================================")
    print(" WARP MISSION + VEHICLE INTERFACE TEST")
    print("========================================")

    adapter = None
    mission_manager = MissionManager()
    mission_finished = False

    try:

        # --------------------------------------------------
        # 1. Create vehicle through our Vehicle Interface
        # --------------------------------------------------

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        vehicle = adapter.vehicle

        print("\nVehicle Interface connected.")

        # --------------------------------------------------
        # 2. Read starting vehicle position
        # --------------------------------------------------

        initial_state = adapter.get_state()

        start_x = initial_state.x
        start_y = initial_state.y

        print(
            f"Start position: "
            f"x={start_x:.1f}, "
            f"y={start_y:.1f}"
        )

        # --------------------------------------------------
        # 3. Choose destination
        # --------------------------------------------------

        spawn_points = adapter.get_spawn_points()

        start_location = carla.Location(
            x=start_x,
            y=start_y,
            z=initial_state.z
        )

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

        # --------------------------------------------------
        # 4. START MISSION
        # --------------------------------------------------

        print("\nCreating mission...")

        mission = mission_manager.start_mission(
            dest_x=destination.x,
            dest_y=destination.y,
            start_x=start_x,
            start_y=start_y
        )

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # At this point:
        #
        # IDLE -> PLANNING

        # --------------------------------------------------
        # 5. Create route using CARLA BasicAgent
        # --------------------------------------------------

        print("\nPlanning route...")

        agent = BasicAgent(
            vehicle,
            target_speed=TARGET_SPEED_KMH
        )

        agent.set_destination(destination)

        print("CARLA BasicAgent route created.")
        print(
            "BasicAgent = OPEN-SOURCE "
            "baseline planner/controller"
        )

        # Route exists, so mission can execute.
        mission_manager.set_executing()

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # At this point:
        #
        # PLANNING -> EXECUTING

        # --------------------------------------------------
        # 6. Engage vehicle autonomy
        # --------------------------------------------------

        print("\nEngaging Vehicle Interface...")

        if not adapter.engage_autonomy():

            mission_manager.fail_mission(
                "Vehicle autonomy could not be engaged"
            )

            raise RuntimeError(
                "Could not engage vehicle autonomy"
            )

        print("")
        print("========================================")
        print(" MISSION EXECUTING")
        print("========================================")

        mission_start_time = time.time()
        last_print_time = 0.0

        # --------------------------------------------------
        # 7. Main mission loop
        # --------------------------------------------------

        while True:

            elapsed = (
                time.time() - mission_start_time
            )

            # ----------------------------------------------
            # Mission timeout
            # ----------------------------------------------

            if elapsed > MISSION_TIMEOUT_SECONDS:

                print("\nMISSION TIMEOUT")

                adapter.emergency_stop()

                mission_manager.fail_mission(
                    "Mission timeout"
                )

                mission_finished = True

                break

            # ----------------------------------------------
            # Destination reached
            # ----------------------------------------------

            if agent.done():

                # Normal stop command through
                # our Vehicle Interface.
                adapter.send_command(
                    VehicleCommand(
                        throttle=0.0,
                        steering=0.0,
                        brake=1.0
                    )
                )

                time.sleep(1.0)

                mission_manager.complete_mission(
                    "Arrived at destination"
                )

                mission_finished = True

                print("")
                print("========================================")
                print(" MISSION COMPLETE")
                print("========================================")

                break

            # ----------------------------------------------
            # BasicAgent calculates requested controls
            # ----------------------------------------------

            requested_control = agent.run_step()

            # Convert BasicAgent output into
            # our generic VehicleCommand.
            command = VehicleCommand(
                throttle=requested_control.throttle,
                steering=requested_control.steer,
                brake=requested_control.brake
            )

            # ----------------------------------------------
            # Send through OUR Vehicle Interface
            # ----------------------------------------------

            accepted = adapter.send_command(command)

            if not accepted:

                adapter.emergency_stop()

                mission_manager.fail_mission(
                    "Vehicle Interface rejected command"
                )

                mission_finished = True

                break

            # ----------------------------------------------
            # Read vehicle state
            # ----------------------------------------------

            state = adapter.get_state()

            current_location = carla.Location(
                x=state.x,
                y=state.y,
                z=state.z
            )

            distance = current_location.distance(
                destination
            )

            # ----------------------------------------------
            # Show status once per second
            # ----------------------------------------------

            now = time.time()

            if now - last_print_time >= 1.0:

                mission_status = (
                    mission_manager.get_status()
                )

                print(
                    f"Mission: "
                    f"{mission_status['mission_id']} | "
                    f"State: "
                    f"{mission_status['state']} | "
                    f"Speed: "
                    f"{state.speed_mps * 3.6:5.1f} km/h | "
                    f"Distance: "
                    f"{distance:6.1f} m | "
                    f"Throttle: "
                    f"{command.throttle:.2f} | "
                    f"Steer: "
                    f"{command.steering:+.2f} | "
                    f"Brake: "
                    f"{command.brake:.2f}"
                )

                last_print_time = now

            time.sleep(0.05)

        # --------------------------------------------------
        # 8. Safely disengage vehicle
        # --------------------------------------------------

        print("\nDisengaging autonomy...")

        adapter.disengage_autonomy()

        time.sleep(0.5)

        final_state = adapter.get_state()

        print(
            f"Final vehicle speed: "
            f"{final_state.speed_mps * 3.6:.2f} km/h"
        )

        # --------------------------------------------------
        # 9. Show mission history
        # --------------------------------------------------

        print("\nMission history:")

        for completed_mission in mission_manager.get_history():

            print(
                f"  {completed_mission['mission_id']} | "
                f"{completed_mission['state']} | "
                f"Reason: "
                f"{completed_mission['reason_ended']}"
            )

        print(
            "\nCurrent mission status:",
            mission_manager.get_status()
        )

    except KeyboardInterrupt:

        print("\nMission interrupted by operator.")

        if adapter is not None:
            adapter.emergency_stop()

        if (
            mission_manager.current_mission
            is not None
        ):
            mission_manager.cancel_mission()

    except Exception as error:

        print(f"\nERROR: {error}")

        if adapter is not None:
            adapter.emergency_stop()

        if (
            not mission_finished
            and mission_manager.current_mission
            is not None
        ):
            mission_manager.fail_mission(
                str(error)
            )

        raise

    finally:

        if adapter is not None:

            print("\nCleaning up vehicle...")
            adapter.destroy()

    print("")
    print("========================================")
    print(" MISSION MANAGEMENT TEST FINISHED")
    print("========================================")


if __name__ == "__main__":
    main()
