import carla
import random
import time
import math


# ============================================================
# CONFIG
# ============================================================

random.seed(123)

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TM_PORT = 8000

MAX_TOTAL_CARS = 6

# Moving cars deliberately placed ahead of Warp
LEAD_CAR_DISTANCES = [24.0, 42.0]

# Pedestrians positioned beside the route
PEDESTRIAN_DISTANCES = [
    14.0,
    20.0,
    28.0,
    36.0,
    46.0,
]

# ============================================================
# CONNECT
# ============================================================

client = carla.Client(
    CARLA_HOST,
    CARLA_PORT
)

client.set_timeout(10.0)

world = client.get_world()
carla_map = world.get_map()
bp_lib = world.get_blueprint_library()

tm = client.get_trafficmanager(
    TM_PORT
)

tm_port = tm.get_port()


# ============================================================
# HELPERS
# ============================================================

def actor_speed(actor):
    """Return actor speed in m/s."""

    velocity = actor.get_velocity()

    return math.sqrt(
        velocity.x ** 2
        + velocity.y ** 2
        + velocity.z ** 2
    )


def angle_difference(a, b):
    """Smallest yaw difference in degrees."""

    diff = (a - b + 180.0) % 360.0 - 180.0

    return abs(diff)


def choose_forward_candidate(current_wp, candidates):
    """
    CARLA can return multiple branches at intersections.

    Prefer:
    1. same road
    2. same lane
    3. smallest heading change

    This makes traffic placement much more deterministic
    than simply using next(...)[0].
    """

    if not candidates:
        return None

    current_yaw = (
        current_wp.transform.rotation.yaw
    )

    def score(candidate):

        value = 0.0

        if candidate.road_id != current_wp.road_id:
            value += 20.0

        if candidate.lane_id != current_wp.lane_id:
            value += 10.0

        candidate_yaw = (
            candidate.transform.rotation.yaw
        )

        value += angle_difference(
            current_yaw,
            candidate_yaw
        )

        return value

    return min(
        candidates,
        key=score
    )


def waypoint_ahead(start_wp, target_distance):
    """
    Walk forward through the CARLA road network.

    We use small steps instead of:
        ego_wp.next(30)[0]

    so junction selection stays more consistent.
    """

    current_wp = start_wp
    travelled = 0.0

    while travelled < target_distance:

        step = min(
            2.0,
            target_distance - travelled
        )

        candidates = current_wp.next(
            step
        )

        next_wp = choose_forward_candidate(
            current_wp,
            candidates
        )

        if next_wp is None:
            return None

        travelled += (
            current_wp.transform.location
            .distance(
                next_wp.transform.location
            )
        )

        current_wp = next_wp

    return current_wp


# ============================================================
# FIND WARP VAN
# ============================================================

sprinters = [
    actor
    for actor
    in world.get_actors().filter("vehicle.*")
    if "mercedes.sprinter" in actor.type_id
]

if not sprinters:

    raise RuntimeError(
        "Warp Sprinter not found. "
        "Start Warp AV first."
    )


ego = sprinters[0]

ego_location = ego.get_location()

ego_speed = actor_speed(
    ego
)


print()
print("==========================================")
print(" WARP CONTROLLED DENSE TRAFFIC VALIDATION")
print("==========================================")

print(
    "Warp vehicle ID:",
    ego.id
)

print(
    f"Warp speed: {ego_speed:.2f} m/s"
)

print(
    f"Warp position: "
    f"x={ego_location.x:.1f}, "
    f"y={ego_location.y:.1f}"
)


if ego_speed < 0.3:

    print()
    print(
        "WARNING: Warp is currently almost stopped."
    )

    print(
        "Best test: start the mission first, "
        "then run this script."
    )


ego_wp = carla_map.get_waypoint(
    ego_location,
    project_to_road=True,
    lane_type=carla.LaneType.Driving
)

if ego_wp is None:

    raise RuntimeError(
        "Could not find driving lane under Warp."
    )


print(
    "Warp road:",
    ego_wp.road_id,
    "| lane:",
    ego_wp.lane_id
)


spawned_cars = []
spawned_walkers = []

walker_controls = []


# ============================================================
# VEHICLE BLUEPRINTS
# ============================================================

vehicle_bps = []

for bp in bp_lib.filter("vehicle.*"):

    # Do not create another Warp Sprinter.
    if "mercedes.sprinter" in bp.id:
        continue

    if bp.has_attribute(
        "number_of_wheels"
    ):

        try:

            wheels = int(
                bp.get_attribute(
                    "number_of_wheels"
                )
            )

            if wheels != 4:
                continue

        except Exception:
            pass

    vehicle_bps.append(
        bp
    )


random.shuffle(
    vehicle_bps
)


# ============================================================
# SPAWN VEHICLE FUNCTION
# ============================================================

def spawn_vehicle_at_waypoint(
    waypoint,
    speed_difference
):

    if waypoint is None:
        return None

    transform = waypoint.transform

    transform.location.z += 0.35

    # Try several vehicle blueprints.
    for bp in vehicle_bps:

        vehicle = world.try_spawn_actor(
            bp,
            transform
        )

        if vehicle is None:
            continue

        vehicle.set_autopilot(
            True,
            tm_port
        )

        try:

            # Positive percentage = slower.
            tm.vehicle_percentage_speed_difference(
                vehicle,
                speed_difference
            )

            # Keep lead vehicle behavior predictable.
            tm.auto_lane_change(
                vehicle,
                False
            )

        except Exception:
            pass

        return vehicle

    return None


# ============================================================
# 1. CONTROLLED LEAD TRAFFIC
# ============================================================

print()
print("--- MOVING VEHICLES AHEAD OF WARP ---")


lead_speed_settings = [
    70.0,   # slow lead vehicle
    55.0,   # second vehicle
]


for index, distance in enumerate(
    LEAD_CAR_DISTANCES
):

    waypoint = waypoint_ahead(
        ego_wp,
        distance
    )

    if waypoint is None:

        print(
            f"Could not find road "
            f"{distance:.0f} m ahead"
        )

        continue


    vehicle = spawn_vehicle_at_waypoint(
        waypoint,
        lead_speed_settings[index]
    )


    if vehicle is None:

        print(
            f"Could not spawn vehicle "
            f"near {distance:.0f} m"
        )

        continue


    spawned_cars.append(
        vehicle
    )


    actual_distance = (
        vehicle.get_location()
        .distance(
            ego.get_location()
        )
    )


    print(
        f"CAR {len(spawned_cars)} "
        f"(controlled lead): "
        f"{actual_distance:.1f} m ahead"
    )


# ============================================================
# 2. AMBIENT CARS NEAR WARP
# ============================================================

print()
print("--- NEARBY AMBIENT CARS ---")


spawn_points = (
    carla_map.get_spawn_points()
)

ego_transform = ego.get_transform()

forward = (
    ego_transform.get_forward_vector()
)


candidate_points = []


for spawn_point in spawn_points:

    dx = (
        spawn_point.location.x
        - ego_location.x
    )

    dy = (
        spawn_point.location.y
        - ego_location.y
    )

    distance = math.sqrt(
        dx ** 2
        + dy ** 2
    )


    # Keep traffic genuinely close.
    if not (
        18.0
        <= distance
        <= 65.0
    ):
        continue


    # Positive dot product means generally
    # in front of Warp.
    dot = (
        dx * forward.x
        + dy * forward.y
    )


    if dot < -5.0:
        continue


    wp = carla_map.get_waypoint(
        spawn_point.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )


    if wp is None:
        continue


    # Avoid adding another car extremely close
    # on Warp's exact same lane.
    if (
        wp.road_id == ego_wp.road_id
        and
        wp.lane_id == ego_wp.lane_id
        and
        distance < 50.0
    ):
        continue


    candidate_points.append(
        (
            distance,
            spawn_point
        )
    )


# IMPORTANT:
# Sort nearest-first.
#
# Do NOT random.shuffle() this list.
candidate_points.sort(
    key=lambda item: item[0]
)


for distance, transform in candidate_points:

    if len(spawned_cars) >= MAX_TOTAL_CARS:
        break


    vehicle = None


    for bp in vehicle_bps:

        vehicle = world.try_spawn_actor(
            bp,
            transform
        )

        if vehicle:
            break


    if vehicle is None:
        continue


    vehicle.set_autopilot(
        True,
        tm_port
    )


    try:

        speed_difference = random.choice(
            [
                20.0,
                30.0,
                40.0,
            ]
        )


        tm.vehicle_percentage_speed_difference(
            vehicle,
            speed_difference
        )

    except Exception:
        pass


    spawned_cars.append(
        vehicle
    )


    actual_distance = (
        vehicle.get_location()
        .distance(
            ego.get_location()
        )
    )


    print(
        f"CAR {len(spawned_cars)} "
        f"(ambient): "
        f"{actual_distance:.1f} m away"
    )


# ============================================================
# 3. PEDESTRIANS BESIDE ROUTE
# ============================================================

print()
print("--- PEDESTRIANS NEAR WARP ROUTE ---")


walker_bps = list(
    bp_lib.filter(
        "walker.pedestrian.*"
    )
)


for index, distance in enumerate(
    PEDESTRIAN_DISTANCES
):

    waypoint = waypoint_ahead(
        ego_wp,
        distance
    )

    if waypoint is None:
        continue


    transform = waypoint.transform

    right = (
        transform.get_right_vector()
    )

    forward_direction = (
        transform.get_forward_vector()
    )


    lane_width = max(
        float(waypoint.lane_width),
        3.0
    )


    # Put pedestrian about 1 meter outside
    # the driving lane.
    base_offset = (
        lane_width / 2.0
        + 1.0
    )


    # Alternate left / right side.
    preferred_side = (
        1.0
        if index % 2 == 0
        else -1.0
    )


    offsets = [
        preferred_side * base_offset,
        preferred_side * (
            base_offset + 0.5
        ),
        -preferred_side * base_offset,
        -preferred_side * (
            base_offset + 0.5
        ),
    ]


    walker = None


    for offset in offsets:

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
                + 0.75
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


    if walker is None:

        print(
            f"Could not spawn pedestrian "
            f"near {distance:.0f} m"
        )

        continue


    spawned_walkers.append(
        walker
    )


    # Pedestrians walk along the road,
    # not intentionally across Warp's lane.
    direction_sign = (
        1.0
        if index % 2 == 0
        else -1.0
    )


    direction = carla.Vector3D(
        x=(
            forward_direction.x
            * direction_sign
        ),
        y=(
            forward_direction.y
            * direction_sign
        ),
        z=0.0
    )


    walker_controls.append(
        (
            walker,
            direction
        )
    )


    actual_distance = (
        walker.get_location()
        .distance(
            ego.get_location()
        )
    )


    print(
        f"PERSON {len(spawned_walkers)}: "
        f"{actual_distance:.1f} m away"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print(" CONTROLLED DENSE TRAFFIC ACTIVE")
print("==========================================")

print(
    "Cars:",
    len(spawned_cars)
)

print(
    "Pedestrians:",
    len(spawned_walkers)
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
    "Front RGB Camera:"
)
print(
    "http://localhost:5000/camera"
)

print()
print(
    "Watch Camera + LiDAR perception."
)

print(
    "Press Ctrl+C here to remove traffic."
)

print(
    "=========================================="
)
print()


# ============================================================
# LIVE TEST LOOP
# ============================================================

last_print = 0.0


try:

    while True:

        # Keep pedestrians walking.
        for walker, direction in walker_controls:

            if not walker.is_alive:
                continue


            walker.apply_control(
                carla.WalkerControl(
                    direction=direction,
                    speed=1.0,
                    jump=False
                )
            )


        now = time.time()


        if (
            now - last_print
            >= 2.0
        ):

            last_print = now

            ego_now = (
                ego.get_location()
            )


            car_info = []

            for car in spawned_cars:

                if not car.is_alive:
                    continue


                distance = (
                    car.get_location()
                    .distance(
                        ego_now
                    )
                )


                speed_now = (
                    actor_speed(
                        car
                    )
                )


                car_info.append(
                    (
                        distance,
                        speed_now
                    )
                )


            car_info.sort(
                key=lambda item: item[0]
            )


            people = []

            for walker in spawned_walkers:

                if not walker.is_alive:
                    continue


                people.append(
                    walker.get_location()
                    .distance(
                        ego_now
                    )
                )


            people.sort()


            print(
                f"Warp={actor_speed(ego):.1f}m/s",
                "| Cars:",
                [
                    (
                        round(d, 1),
                        round(s, 1)
                    )
                    for d, s
                    in car_info[:4]
                ],
                "| People:",
                [
                    round(d, 1)
                    for d
                    in people[:4]
                ]
            )


        time.sleep(
            0.05
        )


except KeyboardInterrupt:

    print()
    print(
        "Removing validation traffic..."
    )


finally:

    # Destroy ONLY actors created
    # by this script.
    for actor in (
        spawned_walkers
        + spawned_cars
    ):

        try:

            if actor.is_alive:
                actor.destroy()

        except Exception:
            pass


    print(
        "All validation traffic removed."
    )

