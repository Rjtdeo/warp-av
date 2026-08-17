"""
Warp AV - Safety Supervisor + Telemetry Failure Injection Demo

PURPOSE:
    Prove that the system can:

    1. Start a mission
    2. Drive the simulated Mercedes Sprinter
    3. Monitor safety continuously
    4. Inject a simulated perception failure
    5. Detect the failure
    6. Stop the moving vehicle safely
    7. Mark the mission FAILED
    8. Record everything into a JSONL telemetry log

ARCHITECTURE:

    Mission Manager
          |
          v
    CARLA BasicAgent
          |
          v
    VehicleCommand
          |
          v
    CarlaVehicleAdapter
          |
          v
    Mercedes Sprinter


    Safety Supervisor
          |
          +---- watches system health
          |
          +---- can stop driving


    Telemetry Logger
          |
          v
    logs/mission_0001.jsonl


NOTE:
    CARLA BasicAgent is being used as an OPEN-SOURCE
    baseline planner/controller.

    The perception failure in this test is deliberately injected.
"""

import time
import carla

from agents.navigation.basic_agent import BasicAgent

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.vehicle_interface import VehicleCommand
from warp_av.mission.mission_manager import MissionManager
from warp_av.safety.safety_supervisor import SafetySupervisor
from warp_av.telemetry.logger import TelemetryLogger


# ============================================================
# Configuration
# ============================================================

TARGET_SPEED_KMH = 20.0

# Inject perception failure while vehicle is moving.
FAILURE_AFTER_SECONDS = 8.0

MISSION_TIMEOUT_SECONDS = 120.0


def main():

    print("")
    print("================================================")
    print(" WARP SAFETY + TELEMETRY FAILURE TEST")
    print("================================================")

    adapter = None

    mission_manager = MissionManager()
    safety = SafetySupervisor()

    # Creates logs/ automatically if it does not exist.
    logger = TelemetryLogger(log_dir="logs")

    mission_log_started = False

    try:

        # ====================================================
        # 1. Create simulated vehicle
        # ====================================================

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

        # ====================================================
        # 2. Choose destination
        # ====================================================

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

        # ====================================================
        # 3. Create mission
        # ====================================================

        mission = mission_manager.start_mission(
            dest_x=destination.x,
            dest_y=destination.y,
            start_x=initial_state.x,
            start_y=initial_state.y
        )

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # ====================================================
        # 4. Start telemetry log
        # ====================================================

        logger.start_mission_log(
            mission.mission_id
        )

        mission_log_started = True

        logger.log_event(
            "mission_started",
            (
                f"Mission started from "
                f"({initial_state.x:.1f}, {initial_state.y:.1f}) "
                f"to "
                f"({destination.x:.1f}, {destination.y:.1f})"
            )
        )

        print(
            f"\nTelemetry logging started: "
            f"logs/{mission.mission_id}.jsonl"
        )

        # ====================================================
        # 5. Create route
        # ====================================================

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

        logger.log_event(
            "route_planned",
            "CARLA BasicAgent route successfully created"
        )

        # Mission:
        #
        # PLANNING -> EXECUTING

        mission_manager.set_executing()

        logger.log_event(
            "mission_executing",
            "Mission entered EXECUTING state"
        )

        print(
            "Mission state:",
            mission_manager.get_status()["state"]
        )

        # ====================================================
        # 6. Engage vehicle autonomy
        # ====================================================

        print("\nEngaging Vehicle Interface...")

        if not adapter.engage_autonomy():

            reason = "Could not engage vehicle autonomy"

            logger.log_event(
                "mission_failed",
                reason
            )

            mission_manager.fail_mission(
                reason
            )

            raise RuntimeError(reason)

        logger.log_event(
            "autonomy_engaged",
            "Vehicle autonomy successfully engaged"
        )

        print("")
        print("================================================")
        print(" VEHICLE DRIVING - SAFETY MONITOR ACTIVE")
        print("================================================")

        mission_start_time = time.time()

        last_print_time = 0.0

        perception_failure_injected = False

        # ====================================================
        # 7. Main driving + safety loop
        # ====================================================

        while True:

            now = time.time()

            elapsed = (
                now - mission_start_time
            )

            # ------------------------------------------------
            # Mission timeout
            # ------------------------------------------------

            if elapsed > MISSION_TIMEOUT_SECONDS:

                reason = "Mission timeout"

                print(
                    "\nMISSION TIMEOUT"
                )

                logger.log_event(
                    "mission_timeout",
                    reason
                )

                adapter.emergency_stop()

                logger.log_event(
                    "emergency_stop",
                    "Emergency stop caused by mission timeout"
                )

                mission_manager.fail_mission(
                    reason
                )

                logger.log_event(
                    "mission_failed",
                    reason
                )

                break

            # ------------------------------------------------
            # Normal destination completion
            # ------------------------------------------------

            if agent.done():

                print(
                    "\nDestination reached."
                )

                stop_command = VehicleCommand(
                    throttle=0.0,
                    steering=0.0,
                    brake=1.0
                )

                adapter.send_command(
                    stop_command
                )

                time.sleep(1.0)

                final_state = (
                    adapter.get_state()
                )

                logger.log_tick(
                    pose_x=final_state.x,
                    pose_y=final_state.y,
                    pose_yaw=final_state.yaw,
                    pose_speed=final_state.speed_mps,

                    behavior="mission_complete",
                    behavior_reason="Arrived at destination",

                    steering=0.0,
                    throttle=0.0,
                    brake=1.0,

                    safety_state="ok",
                    safety_reason="All systems healthy",

                    perception_objects=0,
                    closest_obstacle=999.0,

                    mission_state="completed",

                    extra={
                        "failure_injected":
                            perception_failure_injected
                    }
                )

                logger.log_event(
                    "mission_completed",
                    "Arrived at destination"
                )

                mission_manager.complete_mission(
                    "Arrived at destination"
                )

                break

            # ------------------------------------------------
            # Inject perception failure after 8 seconds
            # ------------------------------------------------

            if elapsed >= FAILURE_AFTER_SECONDS:

                if not perception_failure_injected:

                    print("")
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    )
                    print(
                        " SIMULATED PERCEPTION FAILURE INJECTED"
                    )
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                    )
                    print("")

                    perception_failure_injected = True

                    logger.log_event(
                        "perception_failure",
                        (
                            "Simulated perception "
                            "system failure injected"
                        ),
                        {
                            "elapsed_sec": round(
                                elapsed,
                                2
                            )
                        }
                    )

                perception_healthy = False

            else:

                perception_healthy = True

            # ------------------------------------------------
            # Read current vehicle state
            # ------------------------------------------------

            vehicle_state = (
                adapter.get_state()
            )

            # ------------------------------------------------
            # Run Safety Supervisor
            # ------------------------------------------------

            safety_output = safety.update(

                perception_healthy=
                    perception_healthy,

                # Fresh timestamp:
                # We are testing HEALTH failure,
                # not stale-data failure.
                perception_timestamp=
                    time.time(),

                localization_healthy=True,

                localization_confidence=0.95,

                localization_timestamp=
                    time.time(),

                controller_healthy=True,

                vehicle_alive=
                    adapter.is_alive(),

                current_speed=
                    vehicle_state.speed_mps
            )

            # ------------------------------------------------
            # SAFETY INTERVENTION
            # ------------------------------------------------

            if not safety_output.driving_allowed:

                print("")
                print(
                    "================================================"
                )
                print(
                    " SAFETY INTERVENTION"
                )
                print(
                    "================================================"
                )

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

                # Save the important event.
                logger.log_event(
                    "safety_intervention",
                    safety_output.reason,
                    {
                        "safety_state":
                            safety_output.state.value,

                        "driving_allowed":
                            safety_output.driving_allowed,

                        "speed_mps":
                            vehicle_state.speed_mps,

                        "speed_kmh":
                            vehicle_state.speed_mps
                            * 3.6
                    }
                )

                print(
                    "\nApplying full brake "
                    "through Vehicle Interface..."
                )

                # ============================================
                # SAFE STOP LOOP
                # ============================================

                while True:

                    stop_command = VehicleCommand(
                        throttle=0.0,
                        steering=0.0,
                        brake=1.0
                    )

                    adapter.send_command(
                        stop_command
                    )

                    state = (
                        adapter.get_state()
                    )

                    speed_kmh = (
                        state.speed_mps * 3.6
                    )

                    # ----------------------------------------
                    # Log EVERY braking step
                    # ----------------------------------------

                    logger.log_tick(
                        pose_x=state.x,
                        pose_y=state.y,
                        pose_yaw=state.yaw,
                        pose_speed=state.speed_mps,

                        behavior="safety_stop",

                        behavior_reason=
                            safety_output.reason,

                        steering=0.0,
                        throttle=0.0,
                        brake=1.0,

                        safety_state=
                            safety_output.state.value,

                        safety_reason=
                            safety_output.reason,

                        # Placeholder values.
                        # Real perception is not connected
                        # to this demo yet.
                        perception_objects=0,
                        closest_obstacle=999.0,

                        mission_state="executing",

                        extra={
                            "failure_injected":
                                True,

                            "safe_stop_active":
                                True,

                            "elapsed_sec":
                                round(
                                    time.time()
                                    - mission_start_time,
                                    2
                                )
                        }
                    )

                    print(
                        f"Stopping... "
                        f"Speed: {speed_kmh:.2f} km/h"
                    )

                    # Vehicle considered stopped
                    # below 0.10 m/s.
                    if state.speed_mps < 0.10:

                        break

                    time.sleep(0.1)

                print(
                    "\nVehicle reached safe stop."
                )

                logger.log_event(
                    "vehicle_safe_stop",
                    "Vehicle reached safe stop",
                    {
                        "final_speed_mps":
                            state.speed_mps,

                        "final_speed_kmh":
                            state.speed_mps * 3.6
                    }
                )

                # --------------------------------------------
                # Disengage autonomous control
                # --------------------------------------------

                adapter.disengage_autonomy()

                logger.log_event(
                    "autonomy_disengaged",
                    (
                        "Autonomy disengaged "
                        "after safety intervention"
                    )
                )

                # --------------------------------------------
                # Record failed mission
                # --------------------------------------------

                logger.log_event(
                    "mission_failed",
                    safety_output.reason
                )

                mission_manager.fail_mission(
                    safety_output.reason
                )

                # Final telemetry entry showing FAILED.
                final_state = (
                    adapter.get_state()
                )

                logger.log_tick(
                    pose_x=final_state.x,
                    pose_y=final_state.y,
                    pose_yaw=final_state.yaw,
                    pose_speed=final_state.speed_mps,

                    behavior="stopped",

                    behavior_reason=
                        safety_output.reason,

                    steering=0.0,
                    throttle=0.0,
                    brake=1.0,

                    safety_state=
                        safety_output.state.value,

                    safety_reason=
                        safety_output.reason,

                    perception_objects=0,
                    closest_obstacle=999.0,

                    mission_state="failed",

                    extra={
                        "failure_injected": True,
                        "safe_stop_complete": True
                    }
                )

                break

            # ------------------------------------------------
            # Safety says driving is allowed
            # ------------------------------------------------

            requested_control = (
                agent.run_step()
            )

            # Convert BasicAgent output into
            # OUR generic VehicleCommand.
            command = VehicleCommand(
                throttle=
                    requested_control.throttle,

                steering=
                    requested_control.steer,

                brake=
                    requested_control.brake
            )

            # Send through OUR Vehicle Interface.
            accepted = (
                adapter.send_command(
                    command
                )
            )

            if not accepted:

                reason = (
                    "Vehicle Interface "
                    "rejected command"
                )

                print(
                    f"\n{reason}"
                )

                logger.log_event(
                    "vehicle_command_rejected",
                    reason
                )

                adapter.emergency_stop()

                logger.log_event(
                    "emergency_stop",
                    (
                        "Emergency stop after "
                        "vehicle command rejection"
                    )
                )

                mission_manager.fail_mission(
                    reason
                )

                logger.log_event(
                    "mission_failed",
                    reason
                )

                break

            # ------------------------------------------------
            # Read updated state
            # ------------------------------------------------

            state = (
                adapter.get_state()
            )

            # ------------------------------------------------
            # LOG NORMAL DRIVING TELEMETRY
            # ------------------------------------------------

            logger.log_tick(
                pose_x=state.x,
                pose_y=state.y,
                pose_yaw=state.yaw,
                pose_speed=state.speed_mps,

                behavior="baseline_drive",

                behavior_reason=(
                    "Following CARLA "
                    "BasicAgent route"
                ),

                steering=
                    command.steering,

                throttle=
                    command.throttle,

                brake=
                    command.brake,

                safety_state=
                    safety_output.state.value,

                safety_reason=
                    safety_output.reason,

                # Placeholder until real perception
                # is connected to this demo.
                perception_objects=0,

                closest_obstacle=999.0,

                mission_state=
                    mission_manager.get_status()[
                        "state"
                    ],

                extra={
                    "failure_injected":
                        perception_failure_injected,

                    "elapsed_sec":
                        round(
                            elapsed,
                            2
                        )
                }
            )

            # ------------------------------------------------
            # Terminal display once per second
            # ------------------------------------------------

            if now - last_print_time >= 1.0:

                mission_status = (
                    mission_manager.get_status()
                )

                print(
                    f"Time: {elapsed:5.1f}s | "
                    f"Mission: "
                    f"{mission_status['state']} | "
                    f"Safety: "
                    f"{safety_output.state.value} | "
                    f"Allowed: "
                    f"{safety_output.driving_allowed} | "
                    f"Speed: "
                    f"{state.speed_mps * 3.6:5.1f} km/h | "
                    f"Throttle: "
                    f"{command.throttle:.2f} | "
                    f"Steer: "
                    f"{command.steering:+.2f} | "
                    f"Brake: "
                    f"{command.brake:.2f}"
                )

                last_print_time = now

            time.sleep(0.05)

        # ====================================================
        # 8. Final results
        # ====================================================

        final_state = (
            adapter.get_state()
        )

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

        for completed_mission in (
            mission_manager.get_history()
        ):

            print(
                f"  "
                f"{completed_mission['mission_id']} | "
                f"State: "
                f"{completed_mission['state']} | "
                f"Reason: "
                f"{completed_mission['reason_ended']}"
            )

    # ========================================================
    # Keyboard interrupt
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\nOperator interrupted test."
        )

        if mission_log_started:

            logger.log_event(
                "operator_interrupt",
                "Test interrupted by operator"
            )

        if adapter is not None:

            adapter.emergency_stop()

            if mission_log_started:

                logger.log_event(
                    "emergency_stop",
                    (
                        "Emergency stop caused "
                        "by operator interrupt"
                    )
                )

        if (
            mission_manager.current_mission
            is not None
        ):

            mission_manager.cancel_mission()

            if mission_log_started:

                logger.log_event(
                    "mission_cancelled",
                    "Mission cancelled by operator"
                )

    # ========================================================
    # Unexpected error
    # ========================================================

    except Exception as error:

        print(
            f"\nERROR: {error}"
        )

        if mission_log_started:

            logger.log_event(
                "system_error",
                str(error)
            )

        if adapter is not None:

            adapter.emergency_stop()

            if mission_log_started:

                logger.log_event(
                    "emergency_stop",
                    (
                        "Emergency stop caused "
                        "by system error"
                    )
                )

        if (
            mission_manager.current_mission
            is not None
        ):

            mission_manager.fail_mission(
                str(error)
            )

            if mission_log_started:

                logger.log_event(
                    "mission_failed",
                    str(error)
                )

        raise

    # ========================================================
    # Cleanup
    # ========================================================

    finally:

        # Close log before removing vehicle.
        logger.stop_mission_log()

        if adapter is not None:

            print(
                "\nCleaning up vehicle..."
            )

            adapter.destroy()

    print("")
    print("================================================")
    print(" SAFETY + TELEMETRY TEST FINISHED")
    print("================================================")


if __name__ == "__main__":
    main()
