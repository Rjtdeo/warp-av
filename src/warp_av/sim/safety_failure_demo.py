"""
Warp AV - Safety Supervisor Failure Injection Demo

Purpose:
    Prove that the Safety Supervisor can detect a critical failure
    while the vehicle is moving and command a safe stop.

Test flow:

    Mission starts
        ↓
    Safety = OK
        ↓
    Vehicle drives normally
        ↓
    After 15 seconds:
    SIMULATED PERCEPTION FAILURE
        ↓
    Safety Supervisor detects failure
        ↓
    SafetyState.INTERVENTION
        ↓
    driving_allowed = False
        ↓
    Vehicle Interface commands full brake
        ↓
    Vehicle stops
        ↓
    Mission = FAILED

IMPORTANT:
    The perception failure in this test is deliberately injected.
    This is a safety integration test, not a real perception failure.
"""

import time
import carla

from agents.navigation.basic_agent import BasicAgent

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.vehicle_interface import VehicleCommand
from warp_av.mission.mission_manager import MissionManager
from warp_av.safety.safety_supervisor import SafetySupervisor


TARGET_SPEED_KMH = 20.0

# Inject a fake perception failure after this many seconds.
FAILURE_AFTER_SECONDS = 8.0

MISSION_TIMEOUT_SECONDS = 120.0


def main():

    print("")
    print("================================================")
    print(" WARP SAFETY SUPERVISOR FAILURE INJECTION TEST")
    print("================================================")

    adapter = None

    mission_manager = MissionManager()
    safety = SafetySupervisor()

    try:

        # ==================================================
        # 1. Create vehicle
        # ==================================================

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        vehicle = adapter.vehicle

        print("\nVehicle Interface connected.")

        initial_state = adapter.get_state()

        print(
            f"Start position: "
            f"x={initial_state.x:.1f}, "
            f"y={initial_state.y:.1f}"
        )

        # ==================================================
        # 2. Choose destination
        # ==================================================

        spawn_points = adapter.get_spawn_points()

        start_location = carla.Location(
            x=initial_state.x,
            y=initial_state.y,
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

        # ==================================================
        # 3. Start mission
        # ==================================================

        mission_manager.start_mission(
            dest_x=destination.x,
            dest_y=destination.y,
            start_x=initial_state.x,
            start_y=initial_state.y
        )

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # ==================================================
        # 4. Create baseline route
        # ==================================================

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

        mission_manager.set_executing()

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # ==================================================
        # 5. Engage Vehicle Interface
        # ==================================================

        if not adapter.engage_autonomy():

            mission_manager.fail_mission(
                "Could not engage vehicle autonomy"
            )

            raise RuntimeError(
                "Vehicle autonomy engagement failed"
            )

        print("")
        print("================================================")
        print(" VEHICLE DRIVING - SAFETY MONITOR ACTIVE")
        print("================================================")

        mission_start_time = time.time()
        last_print_time = 0.0

        perception_failure_injected = False

        # ==================================================
        # 6. Main driving + safety loop
        # ==================================================

        while True:

            now = time.time()

            elapsed = now - mission_start_time

            # ----------------------------------------------
            # Mission timeout protection
            # ----------------------------------------------

            if elapsed > MISSION_TIMEOUT_SECONDS:

                print("\nMission timeout.")

                adapter.emergency_stop()

                mission_manager.fail_mission(
                    "Mission timeout"
                )

                break

            # ----------------------------------------------
            # Check normal destination completion
            # ----------------------------------------------

            if agent.done():

                adapter.send_command(
                    VehicleCommand(
                        throttle=0.0,
                        steering=0.0,
                        brake=1.0
                    )
                )

                time.sleep(1.0)

                mission_manager.complete_mission(
                    "Arrived before failure injection"
                )

                print("\nMission reached destination.")

                break

            # ----------------------------------------------
            # Inject perception failure after 15 seconds
            # ----------------------------------------------

            if elapsed >= FAILURE_AFTER_SECONDS:

                if not perception_failure_injected:

                    print("")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(" SIMULATED PERCEPTION FAILURE INJECTED")
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("")

                    perception_failure_injected = True

                perception_healthy = False

            else:

                perception_healthy = True

            # ----------------------------------------------
            # Read current vehicle state
            # ----------------------------------------------

            vehicle_state = adapter.get_state()

            # ----------------------------------------------
            # Run Safety Supervisor
            # ----------------------------------------------

            safety_output = safety.update(

                perception_healthy=perception_healthy,

                # We keep timestamp fresh because this test
                # specifically tests an unhealthy component,
                # not stale data.
                perception_timestamp=time.time(),

                localization_healthy=True,
                localization_confidence=0.95,
                localization_timestamp=time.time(),

                controller_healthy=True,

                vehicle_alive=adapter.is_alive(),

                current_speed=vehicle_state.speed_mps
            )

            # ----------------------------------------------
            # SAFETY INTERVENTION
            # ----------------------------------------------

            if not safety_output.driving_allowed:

                print("")
                print("================================================")
                print(" SAFETY INTERVENTION")
                print("================================================")

                print(
                    "Safety state:",
                    safety_output.state.value
                )

                print(
                    "Reason:",
                    safety_output.reason
                )

                print(
                    "Driving allowed:",
                    safety_output.driving_allowed
                )

                print("\nApplying full brake through Vehicle Interface...")

                # Do NOT use E-stop here.
                #
                # Perception failure causes a controlled
                # safety intervention / safe stop.
                #
                # E-stop is reserved for emergency-stop cases.

                while True:

                    adapter.send_command(
                        VehicleCommand(
                            throttle=0.0,
                            steering=0.0,
                            brake=1.0
                        )
                    )

                    state = adapter.get_state()

                    speed_kmh = state.speed_mps * 3.6

                    print(
                        f"Stopping... "
                        f"Speed: {speed_kmh:.2f} km/h"
                    )

                    if state.speed_mps < 0.10:
                        break

                    time.sleep(0.1)

                print("\nVehicle reached safe stop.")

                # End autonomous control.
                adapter.disengage_autonomy()

                # Mark mission FAILED and store the reason.
                mission_manager.fail_mission(
                    safety_output.reason
                )

                break

            # ----------------------------------------------
            # Safety says driving is allowed
            # ----------------------------------------------

            requested_control = agent.run_step()

            command = VehicleCommand(
                throttle=requested_control.throttle,
                steering=requested_control.steer,
                brake=requested_control.brake
            )

            accepted = adapter.send_command(command)

            if not accepted:

                print("\nVehicle command rejected.")

                adapter.emergency_stop()

                mission_manager.fail_mission(
                    "Vehicle Interface rejected command"
                )

                break

            # ----------------------------------------------
            # Print status once per second
            # ----------------------------------------------

            if now - last_print_time >= 1.0:

                state = adapter.get_state()

                mission_status = (
                    mission_manager.get_status()
                )

                print(
                    f"Time: {elapsed:5.1f}s | "
                    f"Mission: {mission_status['state']} | "
                    f"Safety: {safety_output.state.value} | "
                    f"Allowed: {safety_output.driving_allowed} | "
                    f"Speed: {state.speed_mps * 3.6:5.1f} km/h | "
                    f"Throttle: {command.throttle:.2f} | "
                    f"Steer: {command.steering:+.2f} | "
                    f"Brake: {command.brake:.2f}"
                )

                last_print_time = now

            time.sleep(0.05)

        # ==================================================
        # 7. Final results
        # ==================================================

        final_state = adapter.get_state()

        print("")
        print("================================================")
        print(" FINAL RESULT")
        print("================================================")

        print(
            f"Final vehicle speed: "
            f"{final_state.speed_mps * 3.6:.2f} km/h"
        )

        print(
            "Current mission:",
            mission_manager.get_status()
        )

        print("\nMission history:")

        for mission in mission_manager.get_history():

            print(
                f"  {mission['mission_id']} | "
                f"State: {mission['state']} | "
                f"Reason: {mission['reason_ended']}"
            )

    except KeyboardInterrupt:

        print("\nOperator interrupted test.")

        if adapter is not None:

            adapter.emergency_stop()

        if mission_manager.current_mission is not None:

            mission_manager.cancel_mission()

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        if adapter is not None:

            adapter.emergency_stop()

        if mission_manager.current_mission is not None:

            mission_manager.fail_mission(
                str(error)
            )

        raise

    finally:

        if adapter is not None:

            print("\nCleaning up vehicle...")
            adapter.destroy()

    print("")
    print("================================================")
    print(" SAFETY FAILURE INJECTION TEST FINISHED")
    print("================================================")


if __name__ == "__main__":
    main()
