import carla
import random
import time
import math


random.seed(123)

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()
carla_map = world.get_map()
bp_lib = world.get_blueprint_library()

tm = client.get_trafficmanager(8000)
tm_port = tm.get_port()


# ============================================================
# FIND WARP VAN
# ============================================================

sprinters = [
    a
    for a in world.get_actors().filter("vehicle.*")
    if "mercedes.sprinter" in a.type_id
]

if not sprinters:
    raise RuntimeError(
        "Warp Sprinter not found. Start Warp AV first."
    )

ego = sprinters[0]

velocity = ego.get_velocity()

speed = math.sqrt(
    velocity.x ** 2
    + velocity.y ** 2
    + velocity.z ** 2
)

print()
print("======================================")
print(" WARP DENSE VALIDATION TRAFFIC")
print("======================================")
print("Warp vehicle ID:", ego.id)
print(f"Warp speed: {speed:.2f} m/s")

if speed < 0.3:
    print()
    print("WARNING: Warp van is almost stopped.")
    print("Start the mission first for the best test.")


ego_loc = ego.get_location()

ego_wp = carla_map.get_waypoint(
    ego_loc,
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

if ego_wp is None:
    raise RuntimeError(
        "Could not find road waypoint under Warp."
    )


cars = []
walkers = []
walker_controls = []


# ============================================================
# VEHICLE BLUEPRINTS
# ============================================================

vehicle_bps = []

for bp in bp_lib.filter("vehicle.*"):

    if "mercedes.sprinter" in bp.id:
        continue

    if bp.has_attribute("number_of_wheels"):

        try:
            if int(
                bp.get_attribute("number_of_wheels")
            ) != 4:
                continue

        except Exception:
            pass

    vehicle_bps.append(bp)


random.shuffle(vehicle_bps)


# ============================================================
# CONTROLLED MOVING LEAD VEHICLE
# ============================================================

print()
print("--- CONTROLLED LEAD VEHICLE ---")

lead = None

for distance in [
    28.0,
    32.0,
    36.0
]:

    points = ego_wp.next(distance)

    if not points:
        continue

    transform = points[0].transform
    transform.location.z += 0.4

    for bp in vehicle_bps:

        lead = world.try_spawn_actor(
            bp,
            transform
        )

        if lead:
            break

    if lead:
        break


if lead:

    lead.set_autopilot(
        True,
        tm_port
    )

    try:
        # Positive number means slower than normal traffic.
        tm.vehicle_percentage_speed_difference(
            lead,
            35.0
        )

    except Exception:
        pass

    cars.append(lead)

    distance = (
        lead.get_location()
        .distance(
            ego.get_location()
        )
    )

    print(
        "Lead vehicle spawned:",
        f"{distance:.1f} m away"
    )

else:

    print(
        "Could not spawn controlled lead vehicle"
    )


# ============================================================
# MORE MOVING VEHICLES NEAR WARP
# ============================================================

print()
print("--- SURROUNDING MOVING TRAFFIC ---")

spawn_points = carla_map.get_spawn_points()

nearby = []

for spawn_point in spawn_points:

    distance = (
        spawn_point.location
        .distance(
            ego.get_location()
        )
    )

    if 15.0 <= distance <= 75.0:

        nearby.append(
            (
                distance,
                spawn_point
            )
        )


# Randomize nearby traffic locations.
random.shuffle(nearby)


for distance, transform in nearby:

    if len(cars) >= 6:
        break

    if (
        transform.location
        .distance(
            ego.get_location()
        )
        < 15.0
    ):
        continue

    bp = random.choice(
        vehicle_bps
    )

    npc = world.try_spawn_actor(
        bp,
        transform
    )

    if npc is None:
        continue

    npc.set_autopilot(
        True,
        tm_port
    )

    try:

        speed_difference = random.choice(
            [
                -15.0,
                -5.0,
                0.0,
                10.0
            ]
        )

        tm.vehicle_percentage_speed_difference(
            npc,
            speed_difference
        )

    except Exception:
        pass

    cars.append(npc)

    actual_distance = (
        npc.get_location()
        .distance(
            ego.get_location()
        )
    )

    print(
        f"Car {len(cars)}:",
        f"{actual_distance:.1f} m"
    )


# ============================================================
# PEDESTRIANS NEAR ROUTE
# ============================================================

print()
print("--- PEDESTRIANS ---")

walker_bps = list(
    bp_lib.filter(
        "walker.pedestrian.*"
    )
)

walker_distances = [
    12.0,
    18.0,
    24.0,
    32.0,
    42.0,
    52.0
]


for distance in walker_distances:

    points = ego_wp.next(
        distance
    )

    if not points:
        continue

    waypoint = points[0]

    transform = waypoint.transform

    right = transform.get_right_vector()
    forward = transform.get_forward_vector()

    walker = None

    # Put people beside the lane.
    for offset in [
        4.5,
        -4.5,
        5.0,
        -5.0,
        5.5,
        -5.5
    ]:

        location = carla.Location(
            x=(
                transform.location.x
                + right.x * offset
            ),
            y=(
                transform.location.y
                + right.y * offset
            ),
            z=(
                transform.location.z
                + 0.8
            )
        )

        bp = random.choice(
            walker_bps
        )

        if bp.has_attribute(
            "is_invincible"
        ):
            bp.set_attribute(
                "is_invincible",
                "false"
            )

        walker = world.try_spawn_actor(
            bp,
            carla.Transform(
                location,
                transform.rotation
            )
        )

        if walker:
            break


    if not walker:
        continue


    walkers.append(
        walker
    )

    # Alternate walking direction.
    direction_sign = (
        1.0
        if len(walkers) % 2
        else -1.0
    )

    walker_controls.append(
        (
            walker,
            carla.Vector3D(
                x=(
                    forward.x
                    * direction_sign
                ),
                y=(
                    forward.y
                    * direction_sign
                ),
                z=0.0
            )
        )
    )

    actual_distance = (
        walker.get_location()
        .distance(
            ego.get_location()
        )
    )

    print(
        f"Person {len(walkers)}:",
        f"{actual_distance:.1f} m"
    )


# ============================================================
# READY
# ============================================================

print()
print("======================================")
print(" DENSE TRAFFIC ACTIVE")
print("======================================")

print(
    "Cars:",
    len(cars)
)

print(
    "Pedestrians:",
    len(walkers)
)

print()
print(
    "Mission Control:"
)
print(
    "http://localhost:5000"
)

print()
print(
    "Front camera:"
)
print(
    "http://localhost:5000/camera"
)

print()
print(
    "Press Ctrl+C to remove test traffic."
)
print(
    "======================================"
)
print()


# ============================================================
# RUN
# ============================================================

last_print = 0.0


try:

    while True:

        # Keep pedestrians moving.
        for walker, direction in walker_controls:

            if walker.is_alive:

                walker.apply_control(
                    carla.WalkerControl(
                        direction=direction,
                        speed=1.0,
                        jump=False
                    )
                )


        now = time.time()


        if now - last_print >= 2.0:

            last_print = now

            ego_now = (
                ego.get_location()
            )


            car_distances = sorted(
                [
                    round(
                        car.get_location()
                        .distance(
                            ego_now
                        ),
                        1
                    )
                    for car in cars
                    if car.is_alive
                ]
            )


            walker_distances_now = sorted(
                [
                    round(
                        walker.get_location()
                        .distance(
                            ego_now
                        ),
                        1
                    )
                    for walker in walkers
                    if walker.is_alive
                ]
            )


            print(
                "Nearest cars:",
                car_distances[:4],
                "m | people:",
                walker_distances_now[:4],
                "m"
            )


        time.sleep(
            0.05
        )


except KeyboardInterrupt:

    print()
    print(
        "Stopping validation traffic..."
    )


finally:

    # Destroy ONLY actors created by this test.
    for actor in walkers + cars:

        try:

            if actor.is_alive:
                actor.destroy()

        except Exception:
            pass


    print(
        "All validation traffic removed"
    )

