import time
import carla

from agents.navigation.basic_agent import BasicAgent


HOST = "localhost"
PORT = 2000
TARGET_SPEED = 20  # km/h


def main():
    print("Connecting to CARLA...")

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Connected to:", carla_map.name)

    spawn_points = carla_map.get_spawn_points()

    if len(spawn_points) < 2:
        raise RuntimeError("Not enough spawn points in map")

    # Pick a common vehicle
    blueprints = world.get_blueprint_library()

    vehicle_bp = None

    preferred = [
        "vehicle.mercedes.sprinter",
        "vehicle.tesla.model3",
    ]

    for name in preferred:
        matches = blueprints.filter(name)

        if matches:
            vehicle_bp = matches[0]
            break

    if vehicle_bp is None:
        vehicle_bp = blueprints.filter("vehicle.*")[0]

    vehicle = None

    try:
        # Find an available spawn point
        for spawn in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn)

            if vehicle is not None:
                start_transform = spawn
                break

        if vehicle is None:
            raise RuntimeError("Could not spawn vehicle")

        print("Vehicle spawned:", vehicle.type_id)

        start_location = start_transform.location

        # Choose the spawn point farthest away as destination
        destination_transform = max(
            spawn_points,
            key=lambda x: x.location.distance(start_location)
        )

        destination = destination_transform.location

        print(
            f"Start: x={start_location.x:.1f}, "
            f"y={start_location.y:.1f}"
        )

        print(
            f"Destination: x={destination.x:.1f}, "
            f"y={destination.y:.1f}"
        )

        # CARLA built-in baseline autonomy
        agent = BasicAgent(
            vehicle,
            target_speed=TARGET_SPEED
        )

        agent.set_destination(destination)

        print("\n==========================")
        print("MISSION STARTED")
        print("==========================")

        mission_start = time.time()

        while True:
            world.wait_for_tick()

            if agent.done():
                print("\n\n==========================")
                print("MISSION COMPLETE")
                print("==========================")

                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=0.0,
                        brake=1.0
                    )
                )

                time.sleep(2)
                break

            control = agent.run_step()

            vehicle.apply_control(control)

            velocity = vehicle.get_velocity()

            speed_mps = (
                velocity.x ** 2
                + velocity.y ** 2
                + velocity.z ** 2
            ) ** 0.5

            speed_kmh = speed_mps * 3.6

            current_location = vehicle.get_location()

            distance = current_location.distance(destination)

            elapsed = time.time() - mission_start

            print(
                f"\rSpeed: {speed_kmh:5.1f} km/h | "
                f"Distance to destination: {distance:6.1f} m | "
                f"Time: {elapsed:5.1f}s",
                end="",
                flush=True
            )

    finally:
        if vehicle is not None:
            print("\nRemoving vehicle...")
            vehicle.destroy()


if __name__ == "__main__":
    main()
