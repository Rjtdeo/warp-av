"""
Perception V2 day 5: giving a LiDAR blob the camera's name.

These tests drive the real pipeline with a fake adapter and scripted camera
boxes, and pin the three things that were wrong:

  * the detector's box was read as (left, top, right, bottom) when it is
    (left, top, width, height), so no label ever reached a blob;
  * the projection ignored the camera's own position and tilt;
  * whichever blob was nearest inside a box took the name, so a cone standing
    in front of a car was called a vehicle and the car became an obstacle.
"""
import math
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="the perception module needs OpenCV")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.adapters.carla_sensor_adapter import CameraFrame, LidarScan  # noqa: E402
from warp_av.perception.camera_lidar_perception import (CameraDetection, CameraLidarPerception,  # noqa: E402
                                                        PERSON_CLASS)
from warp_av.perception.camera_model import CameraModel  # noqa: E402

CAR_CLASS = 2
CAM = CameraModel()


class Detector:
    def __init__(self, detections=()):
        self.detections = list(detections)

    def detect(self, image):
        return list(self.detections)


def box_around(x, y, height_m, width_m=1.0, length_m=1.0, cam=CAM):
    """The image box a detector would draw around a thing standing at (x, y)."""
    us, vs = [], []
    for dx in (-length_m / 2, length_m / 2):
        for dy in (-width_m / 2, width_m / 2):
            for z in (-2.5, -2.5 + height_m):
                uv = cam.project(x + dx, y + dy, z)
                assert uv is not None
                us.append(uv[0]); vs.append(uv[1])
    return (min(us), min(vs), max(us) - min(us), max(vs) - min(vs))


def blob(x, y, height_m, n=40, spread=0.6):
    """LiDAR points making a solid thing at (x, y): a little patch at that height."""
    pts = []
    rng = np.random.default_rng(7)
    for _ in range(n):
        pts.append((x + rng.uniform(-spread, spread), y + rng.uniform(-spread / 2, spread / 2),
                    -2.5 + rng.uniform(height_m * 0.4, height_m)))
    return pts


class FakeAdapter:
    def __init__(self, points, width=800, height=600):
        arr = np.zeros((len(points), 6), dtype=np.float32)
        for i, (x, y, z) in enumerate(points):
            arr[i, 0], arr[i, 1], arr[i, 2] = x, y, z
            arr[i, 3] = 1.0
            arr[i, 4] = i % 32
        # a ring of road points so the ground filter has a road to find
        road = []
        for r in range(3, 40):
            for a in range(0, 360, 4):
                road.append((r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), -2.5, 1.0, a % 32, 0.0))
        arr = np.vstack([arr, np.array(road, dtype=np.float32)])
        self.latest_lidar = LidarScan(points=arr, timestamp=time.time(), frames=4, span_s=0.1,
                                      sim_time=1.0, sensor_matrix=np.eye(4))
        self.latest_camera = CameraFrame(image=np.zeros((height, width, 4), dtype=np.uint8),
                                         width=width, height=height, fov=90.0, timestamp=time.time())
        loc = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        rot = types.SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0)
        tf = types.SimpleNamespace(location=loc, rotation=rot)
        self.vehicle = types.SimpleNamespace(get_transform=lambda: tf)


def run(points, detections, ticks=3, **settings):
    adapter = FakeAdapter(points)
    perc = CameraLidarPerception(adapter, detector=Detector(detections))
    perc.yolox_inline = True
    perc.inference_interval = 0.0
    perc.min_sweep_advance_s = 0.0
    for k, v in settings.items():
        setattr(perc, k, v)
    out = None
    for i in range(ticks):
        adapter.latest_lidar = LidarScan(points=adapter.latest_lidar.points, timestamp=time.time(),
                                         frames=4, span_s=0.1, sim_time=1.0 + 0.1 * i, sensor_matrix=np.eye(4))
        out = perc.update()
    named = {c["cls"] for c in (perc.last_clusters or []) if c.get("cls")}
    perc.close()
    return out, perc, named


def test_a_person_gets_the_camera_s_name():
    pts = blob(10.0, 0.0, 1.8, spread=0.3)
    det = CameraDetection(PERSON_CLASS, "person", 0.9, box_around(10.0, 0.0, 1.8, 0.6, 0.6))
    out, _, named = run(pts, [det])
    assert "pedestrian" in named
    assert any(o.object_type.value == "pedestrian" for o in out.objects)


def test_without_the_camera_the_same_person_is_only_an_obstacle():
    out, _, named = run(blob(10.0, 0.0, 1.8, spread=0.3), [])
    assert "pedestrian" not in named
    assert all(o.object_type.value != "pedestrian" for o in out.objects)


def test_a_cone_in_front_of_a_car_does_not_steal_its_name():
    """The bug the day-5 scorecard found: the nearest blob inside a box took the label."""
    pts = blob(20.0, 0.0, 1.5, n=60, spread=1.6) + blob(13.0, 0.0, 0.9, n=20, spread=0.25)
    det = CameraDetection(CAR_CLASS, "car", 0.9, box_around(20.0, 0.0, 1.5, 2.0, 4.4))
    out, perc, _ = run(pts, [det])
    car = [c for c in perc.last_clusters if c["distance"] > 16]
    cone = [c for c in perc.last_clusters if 11 < c["distance"] < 15]
    assert car and cone, "both things should be found"
    assert car[0]["cls"] == "vehicle", "the box belongs to the car behind"
    assert cone[0]["cls"] != "vehicle", "the cone in front must not take the car's name"


def test_a_box_cannot_name_a_thing_of_the_wrong_size():
    """A person-sized box must not be pinned on a low kerb blob that happens to line up."""
    pts = blob(9.0, 0.0, 0.35, n=30, spread=0.4)
    det = CameraDetection(PERSON_CLASS, "person", 0.9, box_around(9.0, 0.0, 1.8, 0.6, 0.6))
    _, _, named = run(pts, [det])
    assert "pedestrian" not in named


def test_shape_alone_still_names_a_car_the_camera_cannot_see():
    """Behind and beside the van the front camera never looks, so shape decides."""
    pts = blob(-14.0, 0.0, 1.5, n=60, spread=2.0)          # behind the van
    _, perc, named = run(pts, [])
    assert "vehicle" in named


def test_a_stale_camera_falls_back_to_shape():
    """A blob that is car-ish but not plainly car-sized: with a working camera that saw
    nothing there it stays an obstacle, with no fresh picture the shape rule names it.

    The blob has to be car-shaped in all three directions now, not merely big: since the
    day-10 follow-up a building no longer passes for a car (see test_day10_vehicle_shape).
    """
    pts = blob(14.0, 0.0, 1.5, n=60, spread=1.3)
    _, _, fresh = run(pts, [])
    _, _, stale = run(pts, [], detection_max_age_s=-1.0)
    assert "vehicle" not in fresh, "a working camera that sees no car keeps it an obstacle"
    assert "vehicle" in stale, "with no fresh picture the van falls back to the shape rule"


def test_a_plainly_car_sized_blob_is_a_vehicle_even_if_the_camera_misses_it():
    pts = blob(16.0, 0.0, 1.5, n=60, spread=1.8)
    _, _, named = run(pts, [])
    assert "vehicle" in named
