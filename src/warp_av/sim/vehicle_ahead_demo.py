"""
Vehicle Ahead Autonomous Reaction Demo

Demonstrates:

CARLA Sprinter
    -> Perception detects stationary vehicle
    -> Behavior slows when vehicle is within 15 m
    -> Behavior stops when vehicle is within 5 m
    -> Controller generates brake command
    -> Vehicle Interface applies command
    -> Sprinter reaches safe stop

Important:
Perception currently uses CARLA ground-truth actors.
It is NOT camera/YOLO perception.
"""

import carla
import time

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.perception.perception import PerceptionSystem
from warp_av.localization.localization import LocalizationSystem
from warp_av.behavior.behavior import BehaviorSystem
from warp_av.control.controller import VehicleController


def main():

    print("")
    print("================================================")
    print("VEHICLE-AHEAD AUTONOMOUS REACTION DEMO")
    print("================================================")

    adapter = None
    target_vehicle = None

    try:

        # =================================================
        # 1. SPAWN EGO SPRINTER
        # =================================================

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        world = adapter.world
        ego = adapter.vehicle

        print("\n[EGO] Sprinter spawned")

        time.sleep(0.5)

        ego_transform = ego.get_transform()
        forward = ego_transform.get_forward_vector()

        # =================================================
        # 2. SPAWN STATIONARY VEHICLE 25 M AHEAD
        # =================================================

        target_location = carla.Location(
            x=ego_transform.location.x + forward.x * 25.0,
            y=ego_transform.location.y + forward.y * 25.0,
            z=ego_transform.location.z + 0.3
        )

        target_transform = carla.Transform(
            target_location,
            ego_transform.rotation
        )

        target_bp = world.get_blueprint_library().find(
            "vehicle.audi.a2"
        )

        target_vehicle = world.try_spawn_actor(
            target_bp,
            target_transform
        )

        if target_vehicle is None:
            raise RuntimeError(
                "Could not spawn target vehicle ahead"
            )

        # Keep target stationary for repeatable testing.
        target_vehicle.set_simulate_physics(False)

        print("[TARGET] Audi placed 25 m ahead")

        # =================================================
        # 3. CREATE AV MODULES
        # =================================================

        perception = PerceptionSystem(
            world,
            ego
        )

        localization = LocalizationSystem(
            ego
        )

        behavior = BehaviorSystem()

        controller = VehicleController()

        behavior.set_mission()

        # =================================================
        # 4. STRAIGHT-AHEAD STEERING TARGET
        # =================================================

        target_x = (
            ego_transform.location.x
            + forward.x * 60.0
        )

        target_y = (
            ego_transform.location.y
            + forward.y * 60.0
        )

        # =================================================
        # 5. ENGAGE AUTONOMY
        # =================================================

        adapter.engage_autonomy()

        print("")
        print("[SYSTEM] Autonomy engaged")
        print("")
        print(
            "Time | Speed | Distance | Behavior | "
            "Throttle | Brake"
        )
        print(
            "------------------------------------------------------------"
        )

        start_time = time.time()
        last_print = 0.0

        saw_slow = False
        saw_stop = False
        safe_stop = False

        # =================================================
        # 6. AUTONOMOUS LOOP
        # =================================================

        while True:

            elapsed = time.time() - start_time

            # ---------------------------------------------
            # Localization
            # ---------------------------------------------

            pose = localization.update()

            # ---------------------------------------------
            # Perception
            # ---------------------------------------------

            perception_output = perception.update()

            # ---------------------------------------------
            # Behavior
            # ---------------------------------------------

            behavior_output = behavior.update(
                perception=perception_output,
                pose=pose,
                destination_distance=100.0,
                safety_ok=True
            )

            # ---------------------------------------------
            # Controller
            # ---------------------------------------------

            command = controller.compute_command(
                current_x=pose.x,
                current_y=pose.y,
                current_yaw=pose.yaw,
                current_speed=pose.speed,

                target_x=target_x,
                target_y=target_y,

                desired_speed=(
                    behavior_output.desired_speed_mps
                ),

                should_stop=(
                    behavior_output.should_stop
                )
            )

            # ---------------------------------------------
            # Vehicle Interface
            # ---------------------------------------------

            adapter.send_command(command)

            distance = (
                perception_output.closest_obstacle_distance
            )

            # ---------------------------------------------
            # Did behavior enter slowdown region?
            # ---------------------------------------------

            if (
                distance < 15.0
                and distance >= 5.0
                and not behavior_output.should_stop
            ):
                saw_slow = True

            # ---------------------------------------------
            # Did behavior command a stop?
            # ---------------------------------------------

            if behavior_output.should_stop:
                saw_stop = True

            # ---------------------------------------------
            # Console output
            # ---------------------------------------------

            if elapsed - last_print >= 0.5:

                last_print = elapsed

                speed_kmh = pose.speed * 3.6

                if distance >= 900:
                    distance_text = "CLEAR"
                else:
                    distance_text = f"{distance:5.2f}m"

                print(
                    f"{elapsed:4.1f}s | "
                    f"{speed_kmh:5.2f} km/h | "
                    f"{distance_text:>7} | "
                    f"{behavior_output.behavior.value:18} | "
                    f"T={command.throttle:.2f} | "
                    f"B={command.brake:.2f}"
                )

            # ---------------------------------------------
            # Verify physical safe stop
            # ---------------------------------------------

            if (
                saw_stop
                and pose.speed < 0.1
            ):

                safe_stop = True

                print("")
                print("================================================")
                print("VEHICLE STOPPED")
                print("================================================")

                print(
                    "Final speed:",
                    round(pose.speed * 3.6, 3),
                    "km/h"
                )

                print(
                    "Final object distance:",
                    round(distance, 2),
                    "m"
                )

                print(
                    "Final behavior:",
                    behavior_output.behavior.value
                )

                print(
                    "Reason:",
                    behavior_output.reason
                )

                break

            # ---------------------------------------------
            # Timeout protection
            # ---------------------------------------------

            if elapsed > 25.0:

                print("")
                print("[TEST] Timeout reached")
                break

            time.sleep(0.05)

        # =================================================
        # 7. FINAL RESULT
        # =================================================

        print("")
        print("================================================")
        print("FINAL TEST RESULT")
        print("================================================")

        print(
            "Detected slow-down zone:",
            "YES ✅" if saw_slow else "NO ❌"
        )

        print(
            "Behavior commanded stop:",
            "YES ✅" if saw_stop else "NO ❌"
        )

        print(
            "Vehicle reached safe stop:",
            "YES ✅" if safe_stop else "NO ❌"
        )

        if saw_slow and saw_stop and safe_stop:

            print("")
            print("OVERALL: PASS ✅")
            print("")
            print(
                "Perception -> Behavior -> "
                "Controller -> Vehicle"
            )

        else:

            print("")
            print("OVERALL: NEEDS CHECKING ⚠️")

    finally:

        # Always stop ego first.
        if adapter is not None:
            try:
                adapter.disengage_autonomy()
            except Exception:
                pass

        # Remove target vehicle.
        if target_vehicle is not None:
            try:
                target_vehicle.destroy()
                print("\n[TARGET] Vehicle destroyed")
            except Exception:
                pass

        # Remove ego vehicle.
        if adapter is not None:
            adapter.destroy()

    print("")
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
