"""
Perception V2 day 4: the van measures how big a thing is, not just where it is.

A cluster now reports the footprint it has actually seen (long side, short side,
heading), the track keeps the largest view of each side, and DetectedObject
carries length, width, height and heading through to whoever plans the drive.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.perception import DetectedObject, ObjectType  # noqa: E402
from warp_av.perception.tracking import ObjectTracker, cluster_points  # noqa: E402


def rect(cx, cy, length, width, yaw_deg=0.0, step=0.1):
    """Points on the outline of a rectangle, in the sensor frame."""
    a = math.radians(yaw_deg)
    pts = []
    n_l, n_w = int(length / step) + 1, int(width / step) + 1
    for i in range(n_l):
        u = -length / 2 + i * step
        for v in (-width / 2, width / 2):
            pts.append((cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)))
    for j in range(n_w):
        v = -width / 2 + j * step
        for u in (-length / 2, length / 2):
            pts.append((cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)))
    return pts


@pytest.mark.parametrize("yaw", [0.0, 30.0, -45.0, 80.0])
def test_footprint_of_a_known_rectangle(yaw):
    pts = rect(12.0, 1.0, 4.0, 1.8, yaw)
    c = cluster_points(pts, cell=1.0)[0]
    assert c["length_m"] == pytest.approx(4.0, abs=0.15)
    assert c["width_m"] == pytest.approx(1.8, abs=0.15)
    diff = (c["yaw_deg"] - yaw + 90.0) % 180.0 - 90.0
    assert abs(diff) < 6.0, f"heading {c['yaw_deg']:.1f} for a box at {yaw}"


def test_long_side_is_the_length():
    c = cluster_points(rect(10.0, 0.0, 1.0, 3.0, 0.0), cell=1.0)[0]
    assert c["length_m"] > c["width_m"]
    assert c["length_m"] == pytest.approx(3.0, abs=0.15)


def test_a_single_point_has_no_footprint():
    c = cluster_points([(8.0, 0.0), (8.05, 0.0), (8.1, 0.0)], cell=1.0)[0]
    assert c["length_m"] == pytest.approx(0.1, abs=0.05)
    assert c["width_m"] == pytest.approx(0.0, abs=0.05)


def test_track_keeps_the_largest_view():
    tr = ObjectTracker()
    obs = lambda l, w, h, yaw: [{"wx": 10.0, "wy": 0.0, "length_m": l, "width_m": w, "height_m": h, "yaw_deg": yaw}]
    tr.update(obs(1.0, 0.5, 1.2, 10.0), 0.0)
    tr.update(obs(3.5, 1.6, 1.5, 80.0), 0.1)
    tr.update(obs(2.0, 0.9, 1.1, 0.0), 0.2)
    t = tr.update(obs(2.0, 0.9, 1.1, 0.0), 0.3)[0]
    assert t.length_m == pytest.approx(3.5)
    assert t.width_m == pytest.approx(1.6)
    assert t.height_m == pytest.approx(1.5)
    assert t.yaw_deg == pytest.approx(80.0), "the heading should come from the biggest view"


def test_track_without_sizes_stays_zero():
    tr = ObjectTracker()
    tr.update([{"wx": 5.0, "wy": 0.0}], 0.0)
    t = tr.update([{"wx": 5.0, "wy": 0.0}], 0.1)[0]
    assert (t.length_m, t.width_m, t.height_m) == (0.0, 0.0, 0.0)


def test_detected_object_defaults_to_unmeasured():
    o = DetectedObject(object_type=ObjectType.OBSTACLE, x=5.0, y=0.0, distance=5.0)
    assert (o.length_m, o.width_m, o.height_m, o.yaw_deg) == (0.0, 0.0, 0.0, 0.0)


def test_smaller_cell_separates_two_things_a_metre_apart():
    pts = list(rect(10.0, 0.0, 0.4, 0.4)) + list(rect(10.0, 2.0, 0.4, 0.4))
    assert len(cluster_points(pts, cell=1.2)) == 1, "a coarse grid glues them together"
    assert len(cluster_points(pts, cell=0.8)) == 2, "the van's grid should keep them apart"
