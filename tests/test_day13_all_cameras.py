"""Perception V2 day 13: the side and rear cameras finally name things.

Measured live in Town10HD before any of this, standing a person in eight places round the
van: only the three in front of it were named. The other five came out as "obstacle" with
0.40 m of room -- a bin's worth -- when a person is meant to get 0.60 m.
"""
import time

import pytest

from warp_av.perception.camera_model import CameraModel, camera_models, VIEW_MOUNTS
from warp_av.perception.detection_worker import DetectionWorker


# ---------------------------------------------------------------- each camera's own view


def test_there_is_a_model_for_every_camera_on_the_van():
    cams = camera_models()
    assert set(cams) == {"front", "left", "right", "rear"} == set(VIEW_MOUNTS)


@pytest.mark.parametrize("where,x,y,expect", [
    ("straight ahead", 12.0, 0.0, "front"),
    ("out to the right", 0.0, 6.0, "right"),
    ("out to the left", 0.0, -6.0, "left"),
    ("straight behind", -12.0, 0.0, "rear"),
])
def test_each_camera_sees_its_own_side(where, x, y, expect):
    """A person's chest is about 1 m below the roof laser."""
    cams = camera_models()
    seen = [n for n, c in cams.items() if c.in_view(x, y, -1.0)]
    assert seen == [expect], f"{where}: expected only {expect}, got {seen}"


def test_nothing_is_in_view_of_a_camera_facing_away():
    cams = camera_models()
    assert cams["rear"].in_view(12.0, 0.0, -1.0) is False
    assert cams["front"].in_view(-12.0, 0.0, -1.0) is False


def test_turning_the_camera_does_not_disturb_the_front_one():
    """The front camera has no turn, so day 5's geometry must be untouched."""
    plain = CameraModel()
    assert plain.yaw_deg == 0.0
    assert plain.sensor_to_camera(10.0, 1.0, -1.0) == CameraModel(yaw_deg=0.0).sensor_to_camera(10.0, 1.0, -1.0)


# ---------------------------------------------------------------- one detector, taking turns


import numpy as np

VIEW_TAG = {"front": 1, "left": 2, "right": 3, "rear": 4}
TAG_VIEW = {v: k for k, v in VIEW_TAG.items()}


class FakeFrame:
    """A tiny picture that says which camera it came from, in its own pixels."""

    def __init__(self, tag):
        self.image = np.full((2, 2, 3), VIEW_TAG[tag], dtype=np.uint8)
        self.timestamp = time.time()
        self.width, self.height, self.fov = 480, 360, 90.0


def which_view(image):
    return TAG_VIEW[int(image[0, 0, 0])]


def test_the_detector_takes_the_cameras_in_turn():
    seen = []
    def detect(image):
        seen.append(which_view(image))
        return [f"box-from-{which_view(image)}"]
    views = {n: (lambda n=n: FakeFrame(n)) for n in ("front", "left", "right", "rear")}
    wk = DetectionWorker(detect, lambda: FakeFrame("front"), interval_s=0.0, views=views)
    wk.start()
    for _ in range(60):
        if len(set(seen)) == 4:
            break
        time.sleep(0.05)
    wk.stop()
    assert set(seen) == {"front", "left", "right", "rear"}, f"only looked at {set(seen)}"


def test_every_camera_keeps_its_own_answer():
    def detect(image):
        return [f"box-from-{which_view(image)}"]
    views = {n: (lambda n=n: FakeFrame(n)) for n in ("front", "left", "right", "rear")}
    wk = DetectionWorker(detect, lambda: FakeFrame("front"), interval_s=0.0, views=views)
    wk.start()
    for _ in range(60):
        if len(wk.latest_by_view(9.0)) == 4:
            break
        time.sleep(0.05)
    wk.stop()
    got = wk.latest_by_view(9.0)
    assert set(got) == {"front", "left", "right", "rear"}
    for view, (dets, age) in got.items():
        assert dets == [f"box-from-{view}"], f"{view} got another camera's answer"


def test_an_answer_that_has_gone_stale_is_dropped():
    wk = DetectionWorker(lambda img: ["box"], lambda: FakeFrame("front"), interval_s=0.0,
                         views={"rear": (lambda: FakeFrame("rear"))})
    wk.start()
    for _ in range(40):
        if wk.latest_by_view(9.0):
            break
        time.sleep(0.05)
    wk.stop()
    assert wk.latest_by_view(9.0)["rear"][0] == ["box"]
    assert wk.latest_by_view(0.0)["rear"][0] == [], "an old answer must not be used as if fresh"


def test_one_camera_still_works_exactly_as_before():
    """No views given = the old single-camera worker, which the replay scorer relies on."""
    wk = DetectionWorker(lambda img: ["box"], lambda: FakeFrame("front"), interval_s=0.0)
    wk.start()
    for _ in range(40):
        if wk.latest(9.0)[0]:
            break
        time.sleep(0.05)
    wk.stop()
    assert wk.latest(9.0)[0] == ["box"]
    assert wk.latest_by_view(9.0) == {}


def test_the_cheap_bearing_test_agrees_with_the_real_geometry():
    """could_see is a shortcut. It must never say no when the camera really can see."""
    for name, cam in camera_models().items():
        for x in range(-20, 21, 2):
            for y in range(-20, 21, 2):
                if x == 0 and y == 0:
                    continue
                if cam.in_view(float(x), float(y), -1.0):
                    assert cam.could_see(float(x), float(y)), \
                        f"{name} can see ({x}, {y}) but the shortcut said no"


def test_the_same_picture_is_never_looked_at_twice():
    """Taking turns broke the 'nothing new yet' check at first, so a camera whose picture
    had not changed was detected over and over, burning the time the van needs to think."""
    runs = []
    frozen = FakeFrame("rear")            # one frame, never updated
    wk = DetectionWorker(lambda img: runs.append(1) or ["box"],
                         lambda: frozen, interval_s=0.0,
                         views={"rear": (lambda: frozen)})
    wk.start()
    time.sleep(0.6)
    wk.stop()
    assert len(runs) == 1, f"looked at the same picture {len(runs)} times"
