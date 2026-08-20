"""Junction give-way: pause before turning, wait out moving cross traffic, then go."""
import math
import time

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.localization.localization import Pose


def turn_route(direction="right", junction_at=20.0):
    """Straight, junction span with a 90-degree yaw change, straight out.
    CARLA convention: positive yaw change = right turn."""
    sgn = 1.0 if direction == "right" else -1.0
    wps = [Waypoint(x=i * 2.0, y=0.0, yaw=0.0) for i in range(int(junction_at / 2))]
    for i in range(1, 5):
        a = sgn * (math.pi / 2) * i / 4
        wps.append(Waypoint(x=junction_at + 4 * math.sin(abs(a)),
                            y=sgn * (4 - 4 * math.cos(a)), yaw=a, is_junction=True))
    for i in range(1, 10):
        wps.append(Waypoint(x=wps[-1].x, y=wps[-1].y + sgn * i * 2.0, yaw=sgn * math.pi / 2))
    return Route(waypoints=wps)


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def test_upcoming_turn_detection():
    p = planner()
    r = turn_route("right", junction_at=20.0)
    info = p.upcoming_turn(r, 4.0, 0.0)
    assert info and info["direction"] == "right" and 12 < info["distance_m"] < 20
    info = p.upcoming_turn(turn_route("left"), 4.0, 0.0)
    assert info and info["direction"] == "left"
    # far junction (outside horizon) and straight road: no turn reported
    assert p.upcoming_turn(r, -30.0, 0.0) is None
    straight = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(40)])
    assert p.upcoming_turn(straight, 0.0, 0.0) is None


def _b(dwell=0.05, timeout=0.6):
    b = BehaviorSystem(); b.set_mission()
    b.junction_dwell_s = dwell
    b.junction_wait_timeout_s = timeout
    return b


def moving_cross_car(dist=15.0):
    return DetectedObject(object_type=ObjectType.VEHICLE, x=10.0, y=dist - 10.0,
                          distance=dist, speed=6.0)


def test_pause_then_go_when_clear():
    b = _b()
    pose = Pose(healthy=True)
    # far from the crossing: roll up, don't freeze early
    roll = b.update(PerceptionOutput(), pose, 500, True, junction={"distance_m": 8.0, "direction": "right"})
    assert roll.behavior == DrivingBehavior.WAITING_AT_JUNCTION and not roll.should_stop
    assert 0.8 <= roll.desired_speed_mps <= 3.0 and "rolling up" in roll.reason
    # at the crossing: hold and look
    j = {"distance_m": 3.0, "direction": "right"}
    out = b.update(PerceptionOutput(), pose, 500, True, junction=j)
    assert out.behavior == DrivingBehavior.WAITING_AT_JUNCTION and out.should_stop
    assert "pausing to check" in out.reason
    time.sleep(0.07)
    out = b.update(PerceptionOutput(), pose, 500, True, junction=j)
    assert out.behavior != DrivingBehavior.WAITING_AT_JUNCTION, "clear junction must release after the dwell"
    # and it must NOT re-trigger for the same junction on the next tick
    out = b.update(PerceptionOutput(), pose, 500, True, junction=j)
    assert out.behavior != DrivingBehavior.WAITING_AT_JUNCTION


def test_waits_for_moving_cross_traffic_then_goes():
    b = _b()
    pose = Pose(healthy=True)
    j = {"distance_m": 3.0, "direction": "right"}
    conflict = PerceptionOutput(objects=[moving_cross_car()])
    b.update(conflict, pose, 500, True, junction=j)
    time.sleep(0.07)
    out = b.update(conflict, pose, 500, True, junction=j)
    assert out.behavior == DrivingBehavior.WAITING_AT_JUNCTION and out.should_stop
    assert "Giving way" in out.reason and "15" in out.reason
    # traffic gone -> proceed
    out = b.update(PerceptionOutput(), pose, 500, True, junction=j)
    assert out.behavior != DrivingBehavior.WAITING_AT_JUNCTION


def test_own_lane_lead_and_parked_cars_are_not_conflicts():
    b = _b()
    pose = Pose(healthy=True)
    j = {"distance_m": 3.0, "direction": "left"}
    lead_in_lane = DetectedObject(object_type=ObjectType.VEHICLE, x=12.0, y=0.3, distance=12.0, speed=5.0)
    parked_side = DetectedObject(object_type=ObjectType.VEHICLE, x=6.0, y=8.0, distance=10.0, speed=0.0)
    p_out = PerceptionOutput(objects=[lead_in_lane, parked_side])
    b.update(p_out, pose, 500, True, junction=j)
    time.sleep(0.07)
    out = b.update(p_out, pose, 500, True, junction=j)
    assert out.behavior != DrivingBehavior.WAITING_AT_JUNCTION, "lead car / parked cars must not block the turn"


def test_timeout_creeps_instead_of_deadlock():
    b = _b(dwell=0.02, timeout=0.15)
    pose = Pose(healthy=True)
    j = {"distance_m": 3.0, "direction": "right"}
    conflict = PerceptionOutput(objects=[moving_cross_car()])
    b.update(conflict, pose, 500, True, junction=j)
    time.sleep(0.2)
    out = b.update(conflict, pose, 500, True, junction=j)
    assert out.behavior == DrivingBehavior.WAITING_AT_JUNCTION
    assert not out.should_stop and out.desired_speed_mps == b.junction_creep_mps
    assert "timeout" in out.reason.lower()


def test_stops_still_outrank_junction_wait():
    b = _b()
    pose = Pose(healthy=True)
    j = {"distance_m": 3.0, "direction": "right"}
    ped = PerceptionOutput(path_blocked=True, closest_obstacle_type=ObjectType.PEDESTRIAN,
                           closest_obstacle_distance=5.0)
    out = b.update(ped, pose, 500, True, junction=j)
    assert out.behavior == DrivingBehavior.STOPPED_PEDESTRIAN
    red = PerceptionOutput(traffic_light="red")
    out = b.update(red, pose, 500, True, junction=j)
    assert out.behavior == DrivingBehavior.STOPPED_RED_LIGHT


def test_distance_to_next_junction():
    p = planner()
    r = turn_route("right", junction_at=20.0)
    d = p.distance_to_next_junction(r, 4.0, 0.0)
    assert d is not None and 12 < d < 20
    straight = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(40)])
    assert p.distance_to_next_junction(straight, 0.0, 0.0) is None
    # already inside the junction
    assert p.distance_to_next_junction(r, 22.0, 1.0) == 0.0
