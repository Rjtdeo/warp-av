"""Safety supervisor, behaviour priority, command validation, fault injector — all without CARLA."""
import math, time
from types import SimpleNamespace

from warp_av.safety.safety_supervisor import SafetySupervisor, SafetyState
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.perception.perception import PerceptionOutput, ObjectType
from warp_av.localization.localization import Pose, LocalizationQuality
from warp_av.control.controller import VehicleController
from warp_av.vehicle_interface import VehicleCommand
from warp_av.testing.fault_injector import FaultInjector


def _ok(**kw):
    base = dict(perception_healthy=True, perception_timestamp=time.time(), localization_healthy=True,
                localization_confidence=1.0, localization_timestamp=time.time(), controller_healthy=True,
                vehicle_alive=True, current_speed=5.0)
    base.update(kw)
    return base


def test_supervisor_ok_and_each_failure():
    s = SafetySupervisor()
    assert s.update(**_ok()).driving_allowed
    assert s.update(**_ok(perception_healthy=False)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(perception_timestamp=time.time() - 2)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(localization_healthy=False)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(localization_confidence=0.2)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(localization_confidence=0.31)).state == SafetyState.OK
    assert s.update(**_ok(localization_timestamp=time.time() - 2)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(controller_healthy=False)).state == SafetyState.INTERVENTION
    assert s.update(**_ok(vehicle_alive=False)).state == SafetyState.INTERVENTION


def test_estop_latches_and_wins():
    s = SafetySupervisor()
    s.trigger_estop("operator")
    out = s.update(**_ok(perception_healthy=False))
    assert out.state == SafetyState.EMERGENCY_STOP and not out.driving_allowed
    assert s.update(**_ok()).state == SafetyState.EMERGENCY_STOP   # latched
    s.clear_estop()
    assert s.update(**_ok()).state == SafetyState.OK


def test_behavior_priority_order():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    ped = PerceptionOutput(path_blocked=True, closest_obstacle_type=ObjectType.PEDESTRIAN, closest_obstacle_distance=3.0)
    # safety beats pedestrian
    assert b.update(ped, pose, 100.0, safety_ok=False).behavior == DrivingBehavior.STOPPED_SAFETY
    # pedestrian stops, reason names it
    out = b.update(ped, pose, 100.0, safety_ok=True)
    assert out.behavior == DrivingBehavior.STOPPED_PEDESTRIAN and "PEDESTRIAN" in out.reason and out.should_stop
    # vehicle / obstacle
    veh = PerceptionOutput(path_blocked=True, closest_obstacle_type=ObjectType.VEHICLE, closest_obstacle_distance=4.0)
    assert b.update(veh, pose, 100.0, True).behavior == DrivingBehavior.STOPPED_VEHICLE
    obs = PerceptionOutput(path_blocked=True, closest_obstacle_type=ObjectType.OBSTACLE, closest_obstacle_distance=4.0)
    assert b.update(obs, pose, 100.0, True).behavior == DrivingBehavior.STOPPED_OBSTACLE
    # slow zone
    near = PerceptionOutput(closest_obstacle_distance=10.0)
    out = b.update(near, pose, 100.0, True)
    assert out.behavior == DrivingBehavior.FOLLOWING_ROUTE and out.desired_speed_mps == b.slow_speed
    # arrival: parked = within 1.5 m of the spot AND nearly stopped
    assert b.update(PerceptionOutput(), pose, 1.0, True).behavior == DrivingBehavior.MISSION_COMPLETE
    # lost localization
    b.set_mission()
    lost = Pose(healthy=False, quality=LocalizationQuality.LOST, reason="x")
    assert b.update(PerceptionOutput(), lost, 100.0, True).behavior == DrivingBehavior.STOPPED_SAFETY


def test_cruise_speed_setter_clamps():
    b = BehaviorSystem()
    b.set_cruise_speed(99); assert b.cruise_speed == 15.0
    b.set_cruise_speed(-3); assert b.cruise_speed == 0.0


def test_controller_nan_injection_and_reset():
    c = VehicleController()
    cmd = c.compute_command(0, 0, 0, 2.0, 10, 0, 5.0, False)
    assert math.isfinite(cmd.steering)
    c.inject_fault("nan_command")
    cmd = c.compute_command(0, 0, 0, 2.0, 10, 0, 5.0, False)
    assert math.isnan(cmd.steering)
    c.enable()
    assert math.isfinite(c.compute_command(0, 0, 0, 2.0, 10, 0, 5.0, False).steering)
    c.disable()
    assert c.compute_command(0, 0, 0, 2.0, 10, 0, 5.0, False).brake == 1.0


def test_vehicle_adapter_rejects_nan_and_stale():
    from warp_av.adapters.carla_vehicle_adapter import CarlaVehicleAdapter
    from warp_av.vehicle_interface import AutonomyState
    va = CarlaVehicleAdapter.__new__(CarlaVehicleAdapter)   # skip CARLA connection
    va.max_command_age_s = 0.5; va.last_command_rejected = ""
    assert va._command_valid(VehicleCommand(steering=0.1))
    assert not va._command_valid(VehicleCommand(steering=float("nan")))
    assert "INVALID" in va.last_command_rejected
    old = VehicleCommand(); old.timestamp -= 2.0
    assert not va._command_valid(old)
    assert "STALE" in va.last_command_rejected


def test_fault_injector_dispatch():
    class FakeSA:  camera_enabled = lidar_enabled = gnss_enabled = imu_enabled = True
    class FakeVA:
        lost = False
        def simulate_connection_loss(self, v): self.lost = v
    class FakePerception:
        def __init__(self): self.enabled = True; self.faults = []
        def disable(self): self.enabled = False
        def enable(self): self.enabled = True
        def inject_fault(self, a, **p): self.faults.append(a); return a in ("freeze", "stale", "latency", "crash")
    class FakeLogger:
        def __init__(self): self.events = []
        def log_event(self, *a, **k): self.events.append(a)
    sysm = SimpleNamespace(perception=FakePerception(), sensor_adapter=FakeSA(), vehicle_adapter=FakeVA(), logger=FakeLogger(),
                           localization=FakePerception(), controller=FakePerception(), planner=FakePerception())
    fi = FaultInjector(sysm)
    assert fi.inject("perception", "disable")["success"] and not sysm.perception.enabled
    assert "perception" in fi.active
    assert fi.inject("perception", "enable")["success"] and "perception" not in fi.active
    assert fi.inject("perception", "stale", age_s=2)["success"]
    assert not fi.inject("perception", "teleport")["success"]
    assert not fi.inject("flux_capacitor", "disable")["success"]
    assert fi.inject("camera", "disable")["success"] and not sysm.sensor_adapter.camera_enabled
    assert fi.inject("vehicle_connection", "disable")["success"] and sysm.vehicle_adapter.lost
    assert fi.inject("tick_latency", "latency", latency_s=0.4, mode="spike")["success"]
    assert fi.extra_tick_delay() == 0.4 and fi.extra_tick_delay() == 0.0
    assert len(sysm.logger.events) >= 6
