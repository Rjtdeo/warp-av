"""Troy fix #4: bigger safety buffer. Pins the new distances so they can't silently regress."""
from warp_av.perception.perception import PerceptionSystem, PerceptionOutput, ObjectType
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.localization.localization import Pose


def test_ground_truth_perception_buffer():
    p = PerceptionSystem.__new__(PerceptionSystem)  # skip CARLA constructor
    p.detection_range = 50.0
    # class defaults are set in __init__; read them from source contract instead:
    src = open("src/warp_av/perception/perception.py").read()
    assert "self.danger_distance = 8.0" in src
    assert "self.path_width = 3.5" in src   # lateral box unchanged on purpose


def test_camera_lidar_perception_buffer():
    src = open("src/warp_av/perception/camera_lidar_perception.py").read()
    assert "self.danger_distance = 8.0" in src, "camera+lidar perception must match the 8 m stop buffer"


def test_behavior_slows_at_20m_not_15m():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    assert b.slow_distance == 20.0
    # 18 m used to be "cruise"; now it must slow
    near = PerceptionOutput(closest_obstacle_distance=18.0)
    out = b.update(near, pose, 100.0, safety_ok=True)
    assert out.desired_speed_mps == b.slow_speed, "object at 18 m must trigger the slow zone"
    # 25 m is still free cruising
    far = PerceptionOutput(closest_obstacle_distance=25.0)
    out = b.update(far, pose, 100.0, safety_ok=True)
    assert out.behavior == DrivingBehavior.FOLLOWING_ROUTE and out.desired_speed_mps == b.cruise_speed


def test_stop_still_works_when_blocked():
    b = BehaviorSystem(); b.set_mission()
    blocked = PerceptionOutput(path_blocked=True, closest_obstacle_type=ObjectType.OBSTACLE,
                               closest_obstacle_distance=7.5)
    out = b.update(blocked, Pose(healthy=True), 100.0, safety_ok=True)
    assert out.should_stop and out.behavior == DrivingBehavior.STOPPED_OBSTACLE
