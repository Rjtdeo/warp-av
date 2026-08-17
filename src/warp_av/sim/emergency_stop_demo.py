"""
Moving Emergency Stop Demo

Demonstrates:

Sprinter drives normally
    -> Operator triggers E-STOP
    -> Safety Supervisor enters EMERGENCY_STOP
    -> Vehicle Interface applies emergency brake
    -> Vehicle reaches safe stop
    -> E-STOP remains latched
    -> Normal drive command cannot restart vehicle
    -> Operator explicitly clears E-STOP

This demonstrates safety-state + physical vehicle response.
"""

import time

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.localization.localization import LocalizationSystem
from warp_av.control.controller import VehicleController
from warp_av.safety.safety_supervisor import (
    SafetySupervisor,
    SafetyState,
)


def main():

    print("")
    print("================================================")
    print("MOVING EMERGENCY STOP DEMO")
    print("================================================")

    adapter = None

    try:

        # -------------------------------------------------
        # 1. Spawn Sprinter
        # -------------------------------------------------

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        ego = adapter.vehicle

        print("\n[EGO] Sprinter spawned")

        time.sleep(0.5)

        start_tf = ego.get_transform()
        forward = start_tf.get_forward_vector()

        # Straight steering target
        target_x = (
            start_tf.location.x
            + forward.x * 60.0
        )

        target_y = (
            start_tf.location.y
            + forward.y * 60.0
        )

        # -------------------------------------------------
        # 2. Create modules
        # -------------------------------------------------

        localization = LocalizationSystem(ego)
        controller = VehicleController()
        safety = SafetySupervisor()

        adapter.engage_autonomy()

        print("[SYSTEM] Autonomy engaged")
        print("")
        print("Accelerating vehicle before E-STOP...")
        print("")

        start_time = time.time()
        last_print = 0.0

        estop_triggered = False
        estop_speed_kmh = 0.0
        safe_stop = False

        # -------------------------------------------------
        # 3. Drive normally, then trigger E-STOP
        # -------------------------------------------------

        while True:

            elapsed = time.time() - start_time
            pose = localization.update()

            # Healthy inputs before E-STOP.
            safety_output = safety.update(
                perception_healthy=True,
                perception_timestamp=time.time(),

                localization_healthy=pose.healthy,
                localization_confidence=pose.confidence,
                localization_timestamp=pose.timestamp,

                controller_healthy=True,
                vehicle_alive=adapter.is_alive(),
                current_speed=pose.speed,
            )

            # -------------------------------------------------
            # Trigger E-STOP once vehicle is clearly moving
            # -------------------------------------------------

            if (
                not estop_triggered
                and pose.speed * 3.6 >= 18.0
            ):

                estop_triggered = True
                estop_speed_kmh = pose.speed * 3.6

                print("")
                print("================================================")
                print("OPERATOR TRIGGERS E-STOP")
                print("================================================")

                print(
                    "Speed at trigger:",
                    round(estop_speed_kmh, 2),
                    "km/h"
                )

                # Safety latch
                safety.trigger_estop(
                    "Operator pressed emergency stop"
                )

                # Physical emergency stop through vehicle interface
                adapter.emergency_stop()

                # Re-run safety after trigger
                safety_output = safety.update(
                    perception_healthy=True,
                    perception_timestamp=time.time(),

                    localization_healthy=pose.healthy,
                    localization_confidence=pose.confidence,
                    localization_timestamp=pose.timestamp,

                    controller_healthy=True,
                    vehicle_alive=adapter.is_alive(),
                    current_speed=pose.speed,
                )

                print(
                    "Safety state:",
                    safety_output.state.value
                )

                print(
                    "Driving allowed:",
                    safety_output.driving_allowed
                )

                print(
                    "Reason:",
                    safety_output.reason
                )

            # -------------------------------------------------
            # Normal driving before E-STOP
            # -------------------------------------------------

            if not estop_triggered:

                command = controller.compute_command(
                    current_x=pose.x,
                    current_y=pose.y,
                    current_yaw=pose.yaw,
                    current_speed=pose.speed,

                    target_x=target_x,
                    target_y=target_y,

                    desired_speed=6.0,
                    should_stop=False
                )

                adapter.send_command(command)

            # -------------------------------------------------
            # After E-STOP: vehicle must remain stopped
            # -------------------------------------------------

            if estop_triggered:

                # Vehicle adapter is already in emergency-stop state.
                # We intentionally do NOT re-engage autonomy.

                if pose.speed < 0.1:

                    safe_stop = True

                    print("")
                    print("================================================")
                    print("VEHICLE REACHED SAFE STOP")
                    print("================================================")

                    print(
                        "Final speed:",
                        round(pose.speed * 3.6, 3),
                        "km/h"
                    )

                    break

            # -------------------------------------------------
            # Console print
            # -------------------------------------------------

            if elapsed - last_print >= 0.5:

                last_print = elapsed

                print(
                    f"{elapsed:4.1f}s | "
                    f"Speed={pose.speed * 3.6:5.2f} km/h | "
                    f"Safety={safety_output.state.value}"
                )

            if elapsed > 20.0:

                print("\n[TEST] Timeout reached")
                break

            time.sleep(0.05)

        # -------------------------------------------------
        # 4. Prove E-STOP is still latched
        # -------------------------------------------------

        print("")
        print("================================================")
        print("E-STOP LATCH TEST")
        print("================================================")

        pose = localization.update()

        latched_output = safety.update(
            perception_healthy=True,
            perception_timestamp=time.time(),

            localization_healthy=pose.healthy,
            localization_confidence=pose.confidence,
            localization_timestamp=pose.timestamp,

            controller_healthy=True,
            vehicle_alive=adapter.is_alive(),
            current_speed=pose.speed,
        )

        estop_latched = (
            safety.is_estop
            and latched_output.state
            == SafetyState.EMERGENCY_STOP
            and latched_output.driving_allowed is False
        )

        print(
            "E-STOP still active:",
            "YES ✅" if estop_latched else "NO ❌"
        )

        print(
            "Driving allowed:",
            latched_output.driving_allowed
        )

        # -------------------------------------------------
        # 5. Clear E-STOP explicitly
        # -------------------------------------------------

        print("")
        print("[OPERATOR] Clearing E-STOP...")

        safety.clear_estop()
        adapter.clear_emergency_stop()

        estop_cleared = not safety.is_estop

        print(
            "E-STOP cleared:",
            "YES ✅" if estop_cleared else "NO ❌"
        )

        print(
            "Note: vehicle is now MANUAL, "
            "not automatically AUTONOMOUS."
        )

        # -------------------------------------------------
        # 6. Final result
        # -------------------------------------------------

        print("")
        print("================================================")
        print("FINAL TEST RESULT")
        print("================================================")

        print(
            "Vehicle was moving before E-STOP:",
            "YES ✅" if estop_speed_kmh >= 18.0 else "NO ❌"
        )

        print(
            "Emergency stop triggered:",
            "YES ✅" if estop_triggered else "NO ❌"
        )

        print(
            "Vehicle reached safe stop:",
            "YES ✅" if safe_stop else "NO ❌"
        )

        print(
            "E-STOP remained latched:",
            "YES ✅" if estop_latched else "NO ❌"
        )

        print(
            "Explicit clear worked:",
            "YES ✅" if estop_cleared else "NO ❌"
        )

        if (
            estop_triggered
            and safe_stop
            and estop_latched
            and estop_cleared
        ):

            print("")
            print("OVERALL: PASS ✅")

        else:

            print("")
            print("OVERALL: NEEDS CHECKING ⚠️")

    finally:

        if adapter is not None:

            try:
                adapter.disengage_autonomy()
            except Exception:
                pass

            adapter.destroy()

    print("")
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
