"""
Static Obstacle Autonomous Reaction Demo

CARLA Sprinter
    -> detects a street barrier
    -> slows down
    -> stops when barrier is too close
    -> explains why it stopped

Perception currently uses CARLA ground-truth actors.
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
    print("STATIC OBSTACLE AUTONOMOUS REACTION DEMO")
    print("================================================")

    adapter = None
    obstacle = None

    try:

        # 1. Spawn Sprinter
        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        world = adapter.world
        ego = adapter.vehicle

        print("\n[EGO] Sprinter spawned")

        time.sleep(0.5)

        ego_transform = ego.get_transform()
        forward = ego_transform.get_forward_vector()

        # 2. Put barrier 25 m directly ahead
        obstacle_location = carla.Location(
            x=ego_transform.location.x + forward.x * 25.0,
            y=ego_transform.location.y + forward.y * 25.0,
            z=ego_transform.location.z
        )

        obstacle_transform = carla.Transform(
            obstacle_location,
            ego_transform.rotation
        )

        obstacle_bp = world.get_blueprint_library().find(
            "static.prop.streetbarrier"
        )

        obstacle = world.try_spawn_actor(
            obstacle_bp,
            obstacle_transform
        )

        if obstacle is None:
            raise RuntimeError(
                "Could not spawn street barrier"
            )

        print("[TARGET] Street barrier placed 25 m ahead")

        # 3. Create AV modules
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

        # Keep van steering straight
        target_x = (
            ego_transform.location.x
            + forward.x * 60.0
        )

        target_y = (
            ego_transform.location.y
            + forward.y * 60.0
        )

        # 4. Engage autonomy
        adapter.engage_autonomy()

        print("")
        print("[SYSTEM] Autonomy engaged")
        print("")
        print(
            "Time | Speed | Distance | Type | "
            "Behavior | Throttle | Brake"
        )
        print(
            "------------------------------------------------------------"
        )

        start_time = time.time()
        last_print = 0.0

        saw_obstacle = False
        saw_slow = False
        saw_stop = False
        safe_stop = False

        # 5. Main loop
        while True:

            elapsed = time.time() - start_time

            # Localization
            pose = localization.update()

            # Perception
            perception_output = perception.update()

            distance = (
                perception_output.closest_obstacle_distance
            )

            object_type = (
                perception_output.closest_obstacle_type.value
            )

            if object_type == "obstacle":
                saw_obstacle = True

            # Behavior
            behavior_output = behavior.update(
                perception=perception_output,
                pose=pose,
                destination_distance=100.0,
                safety_ok=True
            )

            # Controller
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

            # Send command to vehicle
            adapter.send_command(command)

            # Did we enter slowdown zone?
            if (
                object_type == "obstacle"
                and distance < 15.0
                and distance >= 5.0
                and not behavior_output.should_stop
            ):
                saw_slow = True

            # Did behavior stop for obstacle?
            if (
                behavior_output.behavior.value
                == "stopped_obstacle"
            ):
                saw_stop = True

            # Print every 0.5 sec
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
                    f"{object_type:10} | "
                    f"{behavior_output.behavior.value:20} | "
                    f"T={command.throttle:.2f} | "
                    f"B={command.brake:.2f}"
                )

            # Did vehicle actually stop?
            if (
                saw_stop
                and pose.speed < 0.1
            ):

                safe_stop = True

                print("")
                print("================================================")
                print("VEHICLE STOPPED FOR STATIC OBSTACLE")
                print("================================================")

                print(
                    "Final speed:",
                    round(pose.speed * 3.6, 3),
                    "km/h"
                )

                print(
                    "Obstacle distance:",
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

            # Timeout
            if elapsed > 25.0:

                print("")
                print("[TEST] Timeout reached")
                break

            time.sleep(0.05)

        # 6. Results
        print("")
        print("================================================")
        print("FINAL TEST RESULT")
        print("================================================")

        print(
            "Static obstacle detected:",
            "YES ✅" if saw_obstacle else "NO ❌"
        )

        print(
            "Detected slow-down zone:",
            "YES ✅" if saw_slow else "NO ❌"
        )

        print(
            "Behavior commanded obstacle stop:",
            "YES ✅" if saw_stop else "NO ❌"
        )

        print(
            "Vehicle reached safe stop:",
            "YES ✅" if safe_stop else "NO ❌"
        )

        if (
            saw_obstacle
            and saw_slow
            and saw_stop
            and safe_stop
        ):

            print("")
            print("OVERALL: PASS ✅")
            print("")
            print(
                "Obstacle -> Perception -> Behavior "
                "-> Controller -> Vehicle"
            )

        else:

            print("")
            print("OVERALL: NEEDS CHECKING ⚠️")

    finally:

        if adapter is not None:
            try:
                adapter.disengage_autonomy()
            except Exception:
                pass

        if obstacle is not None:
            try:
                obstacle.destroy()
                print("\n[TARGET] Street barrier destroyed")
            except Exception:
                pass

        if adapter is not None:
            adapter.destroy()

    print("")
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
