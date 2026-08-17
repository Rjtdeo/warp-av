"""
Blocked Route Demo

Demonstrates:

Sprinter drives toward a static barrier
    -> Perception detects barrier
    -> Behavior slows
    -> Behavior stops for obstacle
    -> Barrier remains in path
    -> After 3 seconds, behavior changes to STOPPED_BLOCKED
    -> Vehicle remains safely stopped

Important:
This demo reports that replanning/operator action is required.
It does NOT yet automatically calculate a new route around the barrier.
"""

import time
import carla

from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
from warp_av.perception.perception import PerceptionSystem
from warp_av.localization.localization import LocalizationSystem
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.control.controller import VehicleController


def main():

    print("")
    print("================================================")
    print("BLOCKED ROUTE AUTONOMOUS DEMO")
    print("================================================")

    adapter = None
    obstacle = None

    try:

        # -------------------------------------------------
        # 1. Spawn Sprinter
        # -------------------------------------------------

        adapter = CarlaVehicleAdapter(
            vehicle_filter="vehicle.mercedes.sprinter"
        )

        world = adapter.world
        ego = adapter.vehicle

        print("\n[EGO] Sprinter spawned")

        time.sleep(0.5)

        ego_tf = ego.get_transform()
        forward = ego_tf.get_forward_vector()

        # -------------------------------------------------
        # 2. Place street barrier 25 m ahead
        # -------------------------------------------------

        obstacle_location = carla.Location(
            x=ego_tf.location.x + forward.x * 25.0,
            y=ego_tf.location.y + forward.y * 25.0,
            z=ego_tf.location.z
        )

        obstacle_transform = carla.Transform(
            obstacle_location,
            ego_tf.rotation
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

        print("[TARGET] Persistent street barrier placed 25 m ahead")

        # -------------------------------------------------
        # 3. Create AV modules
        # -------------------------------------------------

        perception = PerceptionSystem(world, ego)
        localization = LocalizationSystem(ego)
        behavior = BehaviorSystem()
        controller = VehicleController()

        behavior.set_mission()

        # Drive straight
        target_x = (
            ego_tf.location.x
            + forward.x * 60.0
        )

        target_y = (
            ego_tf.location.y
            + forward.y * 60.0
        )

        # -------------------------------------------------
        # 4. Engage autonomy
        # -------------------------------------------------

        adapter.engage_autonomy()

        print("")
        print("[SYSTEM] Autonomy engaged")
        print("")
        print(
            "Time | Speed | Distance | Behavior | Brake"
        )
        print(
            "------------------------------------------------------"
        )

        start_time = time.time()
        last_print = 0.0

        saw_obstacle_stop = False
        reached_safe_stop = False
        saw_blocked_route = False

        # -------------------------------------------------
        # 5. Main loop
        # -------------------------------------------------

        while True:

            elapsed = time.time() - start_time

            # Localization
            pose = localization.update()

            # Perception
            perception_output = perception.update()

            distance = (
                perception_output.closest_obstacle_distance
            )

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

            # Vehicle Interface
            adapter.send_command(command)

            # -------------------------------------------------
            # Track normal obstacle stop
            # -------------------------------------------------

            if (
                behavior_output.behavior
                == DrivingBehavior.STOPPED_OBSTACLE
            ):
                saw_obstacle_stop = True

            # -------------------------------------------------
            # Track physical stop
            # -------------------------------------------------

            if (
                saw_obstacle_stop
                and pose.speed < 0.1
            ):
                reached_safe_stop = True

            # -------------------------------------------------
            # Track persistent blocked-route decision
            # -------------------------------------------------

            if (
                behavior_output.behavior
                == DrivingBehavior.STOPPED_BLOCKED
            ):

                saw_blocked_route = True

                print("")
                print("================================================")
                print("ROUTE DECLARED BLOCKED")
                print("================================================")

                print(
                    "Vehicle speed:",
                    round(pose.speed * 3.6, 3),
                    "km/h"
                )

                print(
                    "Obstacle distance:",
                    round(distance, 2),
                    "m"
                )

                print(
                    "Behavior:",
                    behavior_output.behavior.value
                )

                print(
                    "Reason:",
                    behavior_output.reason
                )

                break

            # -------------------------------------------------
            # Print status every 0.5 sec
            # -------------------------------------------------

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
                    f"B={command.brake:.2f}"
                )

            # Timeout protection
            if elapsed > 30.0:
                print("")
                print("[TEST] Timeout reached")
                break

            time.sleep(0.05)

        # -------------------------------------------------
        # 6. Final result
        # -------------------------------------------------

        print("")
        print("================================================")
        print("FINAL TEST RESULT")
        print("================================================")

        print(
            "Initial obstacle stop:",
            "YES ✅" if saw_obstacle_stop else "NO ❌"
        )

        print(
            "Vehicle safely stopped:",
            "YES ✅" if reached_safe_stop else "NO ❌"
        )

        print(
            "Persistent route blockage detected:",
            "YES ✅" if saw_blocked_route else "NO ❌"
        )

        if (
            saw_obstacle_stop
            and reached_safe_stop
            and saw_blocked_route
        ):

            print("")
            print("OVERALL: PASS ✅")
            print("")
            print("Temporary obstacle")
            print("        ↓")
            print("STOPPED_OBSTACLE")
            print("        ↓")
            print("Obstacle remains")
            print("        ↓")
            print("STOPPED_BLOCKED")

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
                print("\n[TARGET] Barrier destroyed")
            except Exception:
                pass

        if adapter is not None:
            adapter.destroy()

    print("")
    print("DEMO COMPLETE")


if __name__ == "__main__":
    main()
