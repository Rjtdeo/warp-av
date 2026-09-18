"""V2A: the REAL decision chain, end to end, offline (2026-09-17).

    controlled input -> localization -> perception -> planner -> safety -> behaviour
                     -> speed shaping (ease-off) -> controller -> VehicleCommand

WarpAV.tick() is driven exactly as the live van drives it, on a WarpAV built by its own
__init__. Only the outside world is replaced, at the module boundary where main.py imports
it: the CARLA vehicle adapter (world, map, vehicle body, command output), the CARLA truth
pose source, the CARLA sensor adapter (camera/LiDAR/GNSS/IMU delivery), the ground-truth
perception source (the controlled input), the telemetry file logger (a temp dir), the clock,
and the boot-time switch to camera mode (which would load YOLOX). Everything that decides is
the real code: LocalizationSystem, RoutePlanner.filter_to_route_corridor, the free-space
second opinion, build_world_model, HealthMonitor, SafetySupervisor, BehaviorSystem, the
ease-off, the curve/recovery caps, plan_trajectory, VehicleController.

Every case asserts the chain, not only the command: planner level/reason, safety state,
behaviour state/why/rule, raw and eased desired speed, safety_required, throttle, brake."""
import math
import sys
import time
import types
from types import SimpleNamespace

import pytest

from warp_av.localization.localization import Pose, LocalizationQuality
from warp_av.perception.perception import PerceptionOutput, DetectedObject, ObjectType
from warp_av.planning.planner import Route, Waypoint, WaitingIsPointless
from warp_av.planning.instrumentation import PATH_CLEAR, PATH_SLOW, PATH_BLOCKED, CLEAR
from warp_av.behavior.behavior import DrivingBehavior, distance_to_slow
from warp_av.behavior.transitions import (ROUTE_CLEAR, VRU_IN_PATH, LIGHT_ROLL_UP, LIGHT_HOLD,
                                          SAFETY_HOLD, OBSTACLE_IN_PATH, ROUTE_BLOCKED_TOO_LONG,
                                          NO_WAY_ROUND, OBJECT_AHEAD_SLOW)
from warp_av.mission.mission_manager import MissionState
from warp_av.vehicle_interface import AutonomyState
from warp_av.telemetry.logger import TelemetryLogger


# ---------------------------------------------------------------- the outside world, replaced
class FakeClock:
    """time.time() the tests can move. Starts at the real time so nothing looks stale."""
    def __init__(self):
        self.now = time.time()

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += float(s)


class StubVehicle:
    bounding_box = SimpleNamespace(location=SimpleNamespace(x=0.0),
                                   extent=SimpleNamespace(x=2.958, y=0.994))

    def get_speed_limit(self):
        return 0.0


class StubMap:
    """The CARLA map, absent: every question raises and the callers answer 'no answer'."""
    def get_waypoint(self, *a, **k):
        raise RuntimeError("no map in the harness")


class StubVehicleAdapter:
    """CarlaVehicleAdapter's surface as tick() and __init__ use it. Records every command."""
    def __init__(self, host="test", port=0):
        self.world = SimpleNamespace()          # anything asked of it raises AttributeError
        self.vehicle = StubVehicle()
        self._autonomy_state = AutonomyState.MANUAL
        self.last_command_rejected = ""
        self.sent = []

    def get_map(self):
        return StubMap()

    def is_alive(self):
        return True

    def send_command(self, cmd):
        self.sent.append(cmd)
        return True

    def engage_autonomy(self):
        self._autonomy_state = AutonomyState.AUTONOMOUS
        return True

    def disengage_autonomy(self):
        self._autonomy_state = AutonomyState.DISENGAGED


class StubPoseSource:
    """CarlaTruthPoseSource: the van is wherever the test says."""
    def __init__(self, vehicle=None):
        self._actor = None
        self.current = Pose(x=0.0, y=0.0, yaw=0.0, speed=0.0)

    def pose(self):
        return self.current


class StubSensorAdapter:
    """CarlaSensorAdapter: every sense healthy and enabled, no frames delivered."""
    def __init__(self, world=None, vehicle=None, pose_source=None):
        self.camera_enabled = self.lidar_enabled = self.gnss_enabled = self.imu_enabled = True
        self.latest_camera = None
        self.latest_lidar = None
        self.imu_turn_rad = 0.0
        self.fusion_q = []

    def setup_sensors(self):
        pass

    def is_camera_healthy(self, *a):
        return True

    def is_lidar_healthy(self, *a):
        return True

    def is_gnss_healthy(self, *a):
        return True

    def is_imu_healthy(self, *a):
        return True

    def camera_fault(self):
        return ""

    def lidar_fault(self):
        return ""


class StubPerception:
    """The ground-truth PerceptionSystem: hands the stack the controlled input, unchanged."""
    danger_distance = 8.0
    grid = None
    road_edges = None
    last_clusters = []

    def __init__(self, world=None, vehicle=None, pose_source=None):
        self.output = PerceptionOutput()

    def update(self):
        return self.output


def _stub_route_planner_import():
    """RoutePlanner.__init__ imports CARLA's GlobalRoutePlanner; give it a stand-in so the
    real constructor runs. Only plan_route / plan_route_avoiding use it, and no case here
    asks for a planned route: the route is the controlled input."""
    if "agents.navigation.global_route_planner" in sys.modules:
        return
    agents = types.ModuleType("agents")
    nav = types.ModuleType("agents.navigation")
    grp = types.ModuleType("agents.navigation.global_route_planner")

    class GlobalRoutePlanner:
        def __init__(self, carla_map, sampling_resolution=2.0):
            self.map = carla_map

    grp.GlobalRoutePlanner = GlobalRoutePlanner
    agents.navigation = nav
    nav.global_route_planner = grp
    sys.modules["agents"] = agents
    sys.modules["agents.navigation"] = nav
    sys.modules["agents.navigation.global_route_planner"] = grp


def straight_road(length_m=240.0, start_x=-20.0, step_m=2.0):
    n = int(length_m / step_m) + 1
    return Route(waypoints=[Waypoint(x=start_x + i * step_m, y=0.0, yaw=0.0, speed=8.0)
                            for i in range(n)], total_distance=length_m)


@pytest.fixture
def van(monkeypatch, tmp_path):
    """A real WarpAV, built by its own __init__, with the outside world replaced."""
    _stub_route_planner_import()
    clock = FakeClock()
    monkeypatch.setattr(time, "time", clock)
    import warp_av.main as main
    monkeypatch.setattr(main, "CarlaVehicleAdapter", StubVehicleAdapter)
    monkeypatch.setattr(main, "CarlaTruthPoseSource", StubPoseSource)
    monkeypatch.setattr(main, "CarlaSensorAdapter", StubSensorAdapter)
    monkeypatch.setattr(main, "PerceptionSystem", StubPerception)
    monkeypatch.setattr(main, "TelemetryLogger",
                        lambda log_dir="logs": TelemetryLogger(log_dir=str(tmp_path / "logs")))
    # boot default is camera mode, which loads the YOLOX model: stay on the controlled input
    monkeypatch.setattr(main.WarpAV, "api_set_perception_mode",
                        lambda self, mode: {"success": False, "reason": "V2A harness: ground truth"})
    v = main.WarpAV("test", 0)
    v.clock = clock
    v.pose_source.current = Pose(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    return v


def start_mission(van, route=None, dest=(220.0, 0.0)):
    """What WarpAV.start_mission does, minus the map: the route is handed in."""
    route = route or straight_road()
    pose = van.localization.update()
    van._parking_rechecked = False
    van._give_up = WaitingIsPointless()
    van._forget_the_manoeuvre()
    van.rl_parker.reset()
    mission = van.mission_manager.start_mission(dest[0], dest[1], pose.x, pose.y)
    van._route = route
    van.logger.start_mission_log(mission.mission_id)
    van.vehicle_adapter.engage_autonomy()
    van.behavior.set_mission()
    van.mission_manager.set_executing()
    return mission


def place(van, x=0.0, y=0.0, yaw=0.0, speed=0.0, **pose_kw):
    van.pose_source.current = Pose(x=x, y=y, yaw=yaw, speed=speed, **pose_kw)


def see(van, objects=(), light="none", light_m=None):
    """The controlled input: what perception hands the stack this tick."""
    objs = list(objects)
    closest = min(objs, key=lambda o: o.distance) if objs else None
    van.perception.output = PerceptionOutput(
        objects=objs,
        closest_obstacle_distance=closest.distance if closest else 999.0,
        closest_obstacle_type=closest.object_type if closest else ObjectType.UNKNOWN,
        traffic_light=light, traffic_light_distance_m=light_m,
        timestamp=time.time(), healthy=True, reason="OK")


def chain(van):
    """The intermediate decisions of the last tick, one dict, so a failure names its layer."""
    cmd = van.vehicle_adapter.sent[-1]
    so = van._last_safety_output
    sr = van._speed_request
    return {
        "planner_level": van._path.level, "planner_reason": van._path.reason,
        "closest_m": round(van._path.closest_distance_m, 2), "closest_kind": getattr(van._path.closest_kind, "value", None),
        "safety_state": so.state.value, "driving_allowed": so.driving_allowed, "safety_reason": so.reason,
        "behaviour": van._current_state["behavior"], "why": van._current_state["behavior_why"],
        "rule": van.behavior._rule_now[1],
        "raw_mps": sr["raw"], "eased_mps": sr["eased"], "safety_required": sr["safety"], "stop": sr["stop"],
        "throttle": round(cmd.throttle, 3), "brake": round(cmd.brake, 3), "steer": round(cmd.steering, 3),
        "mission": van.mission_manager.current_mission.state.value,
        "autonomy": van.vehicle_adapter._autonomy_state.value,
    }


def pedestrian(x, y=0.0):
    return DetectedObject(object_type=ObjectType.PEDESTRIAN, x=x, y=y, distance=math.hypot(x, y),
                          id=11, stationary=True, length_m=0.5, width_m=0.5, height_m=1.8)


def barrier(x, y=0.0):
    return DetectedObject(object_type=ObjectType.OBSTACLE, x=x, y=y, distance=math.hypot(x, y),
                          id=12, stationary=True, length_m=1.0, width_m=2.4, height_m=1.0,
                          box_length_m=1.0, box_width_m=2.4, motion_class="static")


def creeping_car(x, y=0.3, speed=0.6):
    """A vehicle in our lane, 8.7 m ahead, rolling at walking pace: in sight, not (yet) in the
    way. Not stationary, so the swept-path sweep (which turns any standing car inside 12 m into
    a hard block) leaves it to the corridor bands: beyond the 8 m danger band it is a SLOW."""
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y),
                          id=13, stationary=False, speed=speed, length_m=4.5, width_m=1.9,
                          height_m=1.5, box_length_m=4.5, box_width_m=1.9, motion_class="dynamic")


def parked_car(x, y=0.3):
    """A stationary car in our lane, beyond the swept-path sweep's 12 m reach: a SLOW for
    comfort's sake, well outside the stopping-distance zone."""
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y),
                          id=14, stationary=True, length_m=4.5, width_m=1.9, height_m=1.5,
                          box_length_m=4.5, box_width_m=1.9, motion_class="static")


# ---------------------------------------------------------------- the six cases

def test_case_1_clear_road_drives_on(van):
    start_mission(van)
    place(van, x=0.0, speed=5.0)
    see(van)
    van.tick()
    c = chain(van)
    assert c["planner_level"] == PATH_CLEAR and c["planner_reason"] == CLEAR, c
    assert c["safety_state"] == "ok" and c["driving_allowed"] is True, c
    assert c["behaviour"] == DrivingBehavior.FOLLOWING_ROUTE.value and c["why"] == ROUTE_CLEAR, c
    assert c["rule"] == "cruise", c
    assert c["raw_mps"] == pytest.approx(van.behavior.cruise_speed) and c["eased_mps"] == c["raw_mps"], c
    assert c["safety_required"] is False and c["stop"] is False, c
    assert c["throttle"] > 0.0 and c["brake"] == 0.0, c
    assert c["mission"] == MissionState.EXECUTING.value and c["autonomy"] == AutonomyState.AUTONOMOUS.value, c


def test_case_2_pedestrian_in_path_stops_the_van(van):
    start_mission(van)
    place(van, x=0.0, speed=4.0)
    see(van, [pedestrian(6.0)])
    van.tick()
    c = chain(van)
    assert c["planner_level"] == PATH_BLOCKED and c["closest_kind"] == "pedestrian", c
    assert 5.5 <= c["closest_m"] <= 6.5, c
    assert c["safety_state"] == "ok", c                                  # not a safety stop: a decision
    assert c["behaviour"] == DrivingBehavior.STOPPED_PEDESTRIAN.value and c["why"] == VRU_IN_PATH, c
    assert c["rule"] == "vru_in_path", c
    assert c["raw_mps"] == 0.0 and c["eased_mps"] == 0.0 and c["stop"] is True, c
    assert c["throttle"] == 0.0 and c["brake"] == 1.0, c


def test_case_3_red_light_rolls_up_then_holds_without_throttle(van):
    start_mission(van)
    # 20 m from the van's middle to the stop line, at 8 m/s: the rule rolls up, never over
    place(van, x=0.0, speed=8.0)
    see(van, light="red", light_m=20.0)
    van.tick()
    c1 = chain(van)
    assert c1["planner_level"] == PATH_CLEAR, c1
    assert c1["rule"] == "traffic_light" and c1["why"] == LIGHT_ROLL_UP, c1
    assert c1["behaviour"] == DrivingBehavior.FOLLOWING_ROUTE.value, c1
    assert 0.0 < c1["raw_mps"] <= 3.0 and c1["eased_mps"] == c1["raw_mps"], c1   # a light is never eased
    assert c1["throttle"] == 0.0 and c1["brake"] > 0.0, c1
    assert van.behavior.light_status["choice"] == "stop", van.behavior.light_status
    # at the line: 3.3 m from the middle is 0.34 m from the bumper, inside the hold gap
    place(van, x=16.7, speed=0.4)
    see(van, light="red", light_m=3.3)
    van.tick()
    c2 = chain(van)
    assert c2["behaviour"] == DrivingBehavior.STOPPED_RED_LIGHT.value and c2["why"] == LIGHT_HOLD, c2
    assert c2["rule"] == "traffic_light" and c2["stop"] is True and c2["raw_mps"] == 0.0, c2
    assert c2["throttle"] == 0.0 and c2["brake"] == 1.0, c2
    assert c2["safety_state"] == "ok" and c2["mission"] == MissionState.EXECUTING.value, c2


def test_case_4_lost_localization_stops_pauses_and_disengages(van):
    start_mission(van)
    place(van, x=10.0, speed=6.0)
    see(van)
    van.tick()
    assert chain(van)["throttle"] > 0.0                                   # driving before the fault
    place(van, x=12.0, speed=6.0, healthy=False, reason="LOCALIZATION_LOST",
          confidence=0.0, quality=LocalizationQuality.LOST)
    see(van)
    van.tick()
    c = chain(van)
    # the current policy: position is a STOP-severity sense -> the supervisor intervenes,
    # the mission is paused and autonomy disengaged; the behaviour's first rule stops the van
    assert c["safety_state"] == "intervention" and c["driving_allowed"] is False, c
    assert "position" in c["safety_reason"], c
    assert c["behaviour"] == DrivingBehavior.STOPPED_SAFETY.value and c["why"] == SAFETY_HOLD, c
    assert c["rule"] == "safety" and c["stop"] is True and c["raw_mps"] == 0.0, c
    assert c["throttle"] == 0.0 and c["brake"] == 1.0, c
    assert c["mission"] == MissionState.PAUSED.value, c
    assert c["autonomy"] == AutonomyState.DISENGAGED.value, c


def test_case_5_blocked_road_with_no_diversion_is_a_stated_refusal(van):
    start_mission(van)
    place(van, x=0.0, speed=3.0)
    see(van, [barrier(10.0)])
    van.tick()
    c1 = chain(van)
    assert c1["planner_level"] == PATH_BLOCKED and c1["closest_kind"] == "obstacle", c1
    assert c1["behaviour"] == DrivingBehavior.STOPPED_OBSTACLE.value and c1["why"] == OBSTACLE_IN_PATH, c1
    assert c1["rule"] == "obstacle_in_path" and c1["stop"] is True, c1
    assert c1["throttle"] == 0.0 and c1["brake"] == 1.0, c1
    assert van._reroute["code"] == "REROUTE_NOT_READY", van._reroute
    # standing at it: after the blocked timeout the behaviour says so in words
    place(van, x=0.0, speed=0.0)
    van.clock.advance(3.5)
    see(van, [barrier(10.0)])
    van.tick()
    c2 = chain(van)
    assert c2["behaviour"] == DrivingBehavior.STOPPED_BLOCKED.value and c2["why"] == ROUTE_BLOCKED_TOO_LONG, c2
    assert "operator action required" in van._current_state["behavior_reason"], van._current_state["behavior_reason"]
    assert c2["throttle"] == 0.0 and c2["brake"] == 1.0, c2
    # ...and once the road has stayed blocked long enough to ask for a way round, the
    # straight road has no junction to turn off at: an explicit refusal, not a failure
    van.clock.advance(3.0)
    see(van, [barrier(10.0)])
    van.tick()
    c3 = chain(van)
    assert van._reroute["code"] == "NO_DIVERSION_POINT", van._reroute
    assert "no junction" in van._reroute["reason"], van._reroute
    recent = van._current_state["behavior_changes"]["recent"]
    assert any(n["why"] == NO_WAY_ROUND for n in recent), recent
    assert c3["behaviour"] == DrivingBehavior.STOPPED_BLOCKED.value and c3["stop"] is True, c3
    assert c3["throttle"] == 0.0 and c3["brake"] == 1.0, c3
    assert c3["mission"] == MissionState.EXECUTING.value, c3          # refused, not failed
    assert c3["safety_state"] == "ok", c3


def test_case_6_a_safety_required_slow_down_is_not_eased_off(van):
    """V1.5 through the whole chain: a slow-down inside the stopping-distance zone is marked
    safety_required and the comfort ease-off must not touch it; a slow-down outside that zone
    is comfort, and that one IS eased -- the control, showing the difference is real."""
    start_mission(van)
    van.behavior.slow_speed = 2.0                  # the live van's camera-mode value (WAV-0615)
    place(van, x=0.0, speed=7.0)
    see(van)
    van.tick()
    assert chain(van)["eased_mps"] == pytest.approx(8.0)              # cruising: the ease-off's memory
    # WAV-0615's numbers: 6.65 m/s with a vehicle 8.7 m ahead, in sight but not in the way
    place(van, x=2.0, speed=6.65)
    see(van, [creeping_car(8.7)])
    van.tick()
    c = chain(van)
    assert c["planner_level"] == PATH_SLOW and c["planner_reason"] == CLEAR, c
    assert 8.5 <= c["closest_m"] <= 8.9, c
    assert c["rule"] == "object_ahead" and c["why"] == OBJECT_AHEAD_SLOW, c
    assert c["raw_mps"] == pytest.approx(2.0), c
    assert 8.7 < distance_to_slow(6.65, 2.0)                             # inside the zone
    assert c["safety_required"] is True, c
    assert c["eased_mps"] == pytest.approx(2.0), c                       # NOT eased to 7.65
    assert c["throttle"] == 0.0 and c["brake"] == pytest.approx(0.6), c  # service brake, at once
    # the control: a car 17 m out is a comfort slow-down, and that one IS eased. (It was 13 m
    # until the P-B05 follow-up of 2026-09-18: the early lane-edge slow now looks as far as
    # the slow needs at this speed -- distance_to_slow(6.65) 9.4 m plus the van's 5.9 m --
    # so a parked car inside 15.3 m is held and REQUIRED at 6.65 m/s; 17 m is still comfort.)
    place(van, x=2.0, speed=6.65)
    see(van)
    van.tick()                                                           # back to cruise: memory 8.0
    assert chain(van)["eased_mps"] == pytest.approx(8.0)
    see(van, [parked_car(17.0)])
    van.tick()
    c = chain(van)
    assert c["planner_level"] == PATH_SLOW and c["rule"] == "object_ahead", c
    assert c["raw_mps"] == pytest.approx(2.0) and 17.0 > distance_to_slow(6.65, 2.0), c
    assert c["safety_required"] is False, c
    assert c["eased_mps"] == pytest.approx(8.0 - 0.35), c
    assert c["throttle"] > 0.0 and c["brake"] == 0.0, c                  # the V1 failure mode, by design
