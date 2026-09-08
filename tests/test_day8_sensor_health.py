"""
Perception V2 day 8: when an eye goes blind.

Measured first, on the live van. Switching off the camera or the LiDAR stopped
it within two seconds, but the reason it gave was only "Safety supervisor
commanded stop", which does not say which eye. Switching off the GPS changed
nothing at all: the van drove on at full speed for as long as it was left off.

So each sense is now reported on its own, with a stated policy: the LiDAR and
the van's position are things it cannot drive without, the camera and the object
detector cost it names rather than obstacles, so it slows instead of freezing,
and it stops if they do not come back.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.safety.safety_supervisor import SafetyState, SafetySupervisor  # noqa: E402
from warp_av.sensor_health import (DEGRADED_GRACE_S, DEGRADED_SPEED_CAP_MPS, HealthMonitor,  # noqa: E402
                                   HealthReport, SENSOR_POLICY, SensorState, Severity,
                                   read_sensors)


def sensors(**states):
    """Every sense healthy except the ones named."""
    out = []
    for name in ("vehicle", "controller", "position", "object_detection", "lidar", "camera",
                 "gps", "imu"):
        out.append(SensorState(name, healthy=states.get(name, True)))
    return out


def report(**states):
    return HealthMonitor().update(sensors(**states))


def test_all_well_means_no_limit():
    r = report()
    assert r.failed() == []
    assert r.worst() is Severity.NOTE
    assert r.speed_cap_mps() is None
    assert r.reason() == "all senses working"


@pytest.mark.parametrize("sense", ["lidar", "position", "controller", "vehicle"])
def test_the_things_the_van_cannot_drive_without(sense):
    r = report(**{sense: False})
    assert r.worst() is Severity.STOP
    assert r.speed_cap_mps() == 0.0
    assert sense in r.reason() and "stopping" in r.reason()


@pytest.mark.parametrize("sense", ["camera", "object_detection", "imu"])
def test_the_things_it_can_manage_without_for_a_while(sense):
    r = report(**{sense: False})
    assert r.worst() is Severity.SLOW
    assert r.speed_cap_mps() == DEGRADED_SPEED_CAP_MPS
    assert sense in r.reason() and "slowly" in r.reason()


def test_a_missing_sense_that_does_not_come_back_stops_the_van():
    mon = HealthMonitor()
    r = mon.update(sensors(camera=False), now=100.0)
    assert r.speed_cap_mps(now=100.0) == DEGRADED_SPEED_CAP_MPS
    r = mon.update(sensors(camera=False), now=100.0 + DEGRADED_GRACE_S + 1.0)
    assert r.speed_cap_mps(now=100.0 + DEGRADED_GRACE_S + 1.0) == 0.0, \
        "a crawl is a way to come to a halt, not a way to keep going for ever"


def test_the_clock_restarts_when_the_sense_comes_back():
    mon = HealthMonitor()
    mon.update(sensors(camera=False), now=100.0)
    mon.update(sensors(), now=105.0)                      # restored
    r = mon.update(sensors(camera=False), now=106.0)      # and lost again
    assert r.degraded_since == 106.0
    assert r.speed_cap_mps(now=110.0) == DEGRADED_SPEED_CAP_MPS


def test_the_worst_fault_wins():
    r = report(camera=False, lidar=False)
    assert r.worst() is Severity.STOP and r.speed_cap_mps() == 0.0
    assert "lidar" in r.reason()


def test_a_sense_that_is_switched_off_still_counts_as_missing():
    """An earlier version excused a switched-off sensor. That is exactly how a failure is
    injected here, and it let a switched-off LiDAR read as 'all senses working' while the
    van drove on. A van that cannot see does not care why."""
    off = SensorState("camera", healthy=False, enabled=False)
    assert off.failed is True
    r = HealthMonitor().update([off])
    assert r.speed_cap_mps() == DEGRADED_SPEED_CAP_MPS
    assert "camera" in r.reason()


def test_a_sense_that_cannot_be_asked_counts_as_failed():
    """A sensor that throws when asked must never come out as fine."""
    class Broken:
        camera_enabled = lidar_enabled = gnss_enabled = imu_enabled = True

        def is_lidar_healthy(self):
            raise RuntimeError("no answer")

        def is_camera_healthy(self):
            return True

        def is_gnss_healthy(self):
            return True

        def is_imu_healthy(self):
            return True

    pose = types.SimpleNamespace(healthy=True, reason="OK", timestamp=0.0)
    got = read_sensors(Broken(), True, "OK", pose, True, True, now=0.0)
    lidar = [s for s in got if s.name == "lidar"][0]
    assert lidar.failed is True


# ---- what the safety supervisor does with the report ---------------------------------

def supervise(health=None):
    s = SafetySupervisor()
    return s.update(perception_healthy=True, perception_timestamp=__import__("time").time(),
                    localization_healthy=True, localization_confidence=1.0,
                    localization_timestamp=__import__("time").time(),
                    controller_healthy=True, vehicle_alive=True, current_speed=4.0,
                    health=health)


def test_safety_stops_for_a_lost_lidar_and_says_so():
    out = supervise(report(lidar=False))
    assert out.state is SafetyState.INTERVENTION
    assert out.driving_allowed is False and out.speed_cap_mps == 0.0
    assert "lidar" in out.reason, "the operator must be told which eye went dark"
    assert out.failed_sensors == ["lidar"]


def test_safety_slows_for_a_lost_camera_rather_than_freezing():
    out = supervise(report(camera=False))
    assert out.state is SafetyState.DEGRADED
    assert out.driving_allowed is True
    assert out.speed_cap_mps == DEGRADED_SPEED_CAP_MPS
    assert "camera" in out.reason


def test_safety_is_unchanged_when_every_sense_works():
    out = supervise(report())
    assert out.state is SafetyState.OK and out.driving_allowed is True
    assert out.speed_cap_mps is None


def test_without_a_report_the_supervisor_behaves_exactly_as_before():
    out = supervise(None)
    assert out.state is SafetyState.OK and out.driving_allowed is True


# ---- and what the behaviour layer does with the cap ------------------------------------

def test_the_cap_slows_the_van_but_never_speeds_it_up():
    from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
    from warp_av.localization.localization import Pose
    from warp_av.perception.perception import PerceptionOutput

    pose = Pose(x=0.0, y=0.0, yaw=0.0, speed=4.0)
    bs = BehaviorSystem()
    bs.set_mission()
    free = bs.update(perception=PerceptionOutput(), pose=pose, destination_distance=100.0,
                     safety_ok=True)
    slowed = bs.update(perception=PerceptionOutput(), pose=pose, destination_distance=100.0,
                       safety_ok=True, speed_cap_mps=2.0)
    assert free.desired_speed_mps > 2.0
    assert slowed.desired_speed_mps == 2.0 and "sensor is missing" in slowed.reason
    faster = bs.update(perception=PerceptionOutput(), pose=pose, destination_distance=100.0,
                       safety_ok=True, speed_cap_mps=99.0)
    assert faster.desired_speed_mps == free.desired_speed_mps, "a cap must never raise the speed"


def test_a_zero_cap_is_a_stop():
    from warp_av.behavior.behavior import BehaviorSystem
    from warp_av.localization.localization import Pose
    from warp_av.perception.perception import PerceptionOutput

    bs = BehaviorSystem()
    bs.set_mission()
    out = bs.update(perception=PerceptionOutput(), pose=Pose(speed=3.0), destination_distance=100.0,
                    safety_ok=True, speed_cap_mps=0.0)
    assert out.desired_speed_mps == 0.0 and out.should_stop is True


def test_every_sense_has_a_stated_policy():
    for name in ("lidar", "camera", "position", "object_detection", "gps", "imu",
                 "controller", "vehicle"):
        assert name in SENSOR_POLICY, f"{name} has no policy: it would fail in silence"


# ---- the pipeline itself, with an eye missing --------------------------------------

def test_a_lost_camera_leaves_the_van_seeing_but_unnamed():
    """The laser finds everything solid; what a missing camera costs is the name on it.
    Before day 8 a stale camera made the whole of perception unhealthy, so the van froze
    in the lane even though it could still see."""
    import numpy as np
    import time as _t
    import pytest as _p
    _p.importorskip("cv2")
    from warp_av.adapters.carla_sensor_adapter import CameraFrame, LidarScan
    from warp_av.perception.camera_lidar_perception import CameraLidarPerception

    class Detector:
        def detect(self, image):
            raise AssertionError("the camera is out: the model must not be asked")

    class Adapter:
        def __init__(self):
            pts = []
            for r in range(3, 30):
                for a in range(0, 360, 4):
                    import math as _m
                    pts.append((r * _m.cos(_m.radians(a)), r * _m.sin(_m.radians(a)), -2.5, 1.0, a % 32, 0.0))
            for k in range(60):
                pts.append((10.0 + 0.02 * k, 0.3 * (k % 3), -1.6 + 0.01 * k, 1.0, k % 32, 0.0))
            arr = np.array(pts, dtype=np.float32)
            self.latest_lidar = LidarScan(points=arr, timestamp=_t.time(), frames=4, span_s=0.1,
                                          sim_time=1.0, sensor_matrix=np.eye(4))
            self.latest_camera = CameraFrame(image=np.zeros((600, 800, 4), dtype=np.uint8),
                                             width=800, height=600, fov=90.0,
                                             timestamp=_t.time() - 5.0)     # five seconds old
            loc = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            rot = types.SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0)
            tf = types.SimpleNamespace(location=loc, rotation=rot)
            self.vehicle = types.SimpleNamespace(get_transform=lambda: tf)

    adapter = Adapter()
    perc = CameraLidarPerception(adapter, detector=Detector())
    perc.yolox_inline = True
    perc.inference_interval = 0.0
    perc.min_sweep_advance_s = 0.0
    out = None
    for k in range(3):
        adapter.latest_lidar = LidarScan(points=adapter.latest_lidar.points, timestamp=_t.time(),
                                         frames=4, span_s=0.1, sim_time=1.0 + 0.1 * k,
                                         sensor_matrix=np.eye(4))
        out = perc.update()
    perc.close()
    assert out.healthy is True, "the laser still works, so the van still sees"
    assert out.degraded is True and "CAMERA" in out.degraded_reason
    assert out.objects, "and it still finds the thing in front of it"
    assert all(o.object_type.value in ("obstacle", "vehicle") for o in out.objects), \
        "with no picture, nothing may be named a person"


def test_a_lost_lidar_is_still_a_full_stop():
    import numpy as np
    import time as _t
    import pytest as _p
    _p.importorskip("cv2")
    from warp_av.perception.camera_lidar_perception import CameraLidarPerception

    class Adapter:
        latest_lidar = None
        latest_camera = None
        vehicle = None

    perc = CameraLidarPerception(Adapter(), detector=types.SimpleNamespace(detect=lambda i: []))
    out = perc.update()
    perc.close()
    assert out.healthy is False and "LIDAR" in out.reason
