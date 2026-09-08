"""
Perception V2 day 5: putting a LiDAR blob in the camera picture.

The camera sits ahead of the LiDAR, lower, and tilted ten degrees down. These
tests pin the geometry with values worked out by hand, and pin the box bug
that stopped every camera label from reaching a cluster.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.camera_model import (CameraModel, box_contains, box_edges,  # noqa: E402
                                             cluster_point)

CAM = CameraModel()          # 800 x 600, 90 deg, mounted (2.0, 0, 1.8), pitch -10


def test_focal_length_from_the_field_of_view():
    assert CAM.focal_px == pytest.approx(400.0)          # 90 deg over 800 px
    assert CameraModel(width=1600, fov_deg=90.0).focal_px == pytest.approx(800.0)
    assert CameraModel(fov_deg=60.0).focal_px == pytest.approx(400.0 / math.tan(math.radians(30)))


def test_straight_ahead_lands_on_the_centre_column():
    u, v = CAM.project(20.0, 0.0, -0.7)                  # 20 m ahead, at camera height
    assert u == pytest.approx(400.0)
    # tilted ten degrees down, the horizon moves up the picture by f * tan(10 deg)
    assert v == pytest.approx(300.0 - 400.0 * math.tan(math.radians(10.0)), abs=0.5)


def test_left_and_right_mirror_each_other():
    ur, _ = CAM.project(15.0, 2.0, -0.7)
    ul, _ = CAM.project(15.0, -2.0, -0.7)
    assert ur > 400.0 > ul
    assert (ur - 400.0) == pytest.approx(400.0 - ul)


def test_a_thing_on_the_road_is_lower_in_the_picture_than_a_tall_one():
    _, v_low = CAM.project(12.0, 0.0, -2.3)              # 20 cm above the road
    _, v_high = CAM.project(12.0, 0.0, -0.5)             # 2 m above the road
    assert v_low > v_high, "closer to the ground means further down the picture"


def test_the_mounts_matter():
    """Ignoring the offsets misplaces a near object by a long way."""
    naive_u = 400.0 + 400.0 * (1.0 / 6.0)                # the old angle-to-column guess
    u, _ = CAM.project(6.0, 1.0, -1.6)
    assert abs(u - naive_u) > 8.0


def test_behind_the_camera_is_not_in_the_picture():
    assert CAM.project(-5.0, 0.0, -1.0) is None
    assert not CAM.in_view(-5.0, 0.0, -1.0)


def test_in_view_edges():
    assert CAM.in_view(20.0, 0.0, -0.7)
    assert not CAM.in_view(5.0, 20.0, -0.7), "far to the side is out of a 90 degree view"
    assert CAM.in_view(5.0, 20.0, -0.7, margin_px=4000.0)


def test_box_edges_reads_width_and_height_not_corners():
    assert box_edges((400, 200, 100, 150)) == (400.0, 200.0, 500.0, 350.0)


def test_box_contains_uses_the_real_right_edge():
    box = (400, 200, 100, 150)                            # left 400, 100 px wide
    assert box_contains(box, 450, 275, margin_px=0.0)
    assert box_contains(box, 495, 340, margin_px=0.0), "inside, near the far corner"
    assert not box_contains(box, 600, 275, margin_px=0.0)
    # the day-5 bug: reading box[2] as the right edge gave the empty span 400..100
    assert not (box[0] <= 450 <= box[2]), "this is what the old code asked"


def test_cluster_point_aims_at_the_middle_of_a_thing():
    x, y, z = cluster_point({"x": 10.0, "y": 1.0, "height": 1.8}, lidar_height_m=2.5)
    assert (x, y) == (10.0, 1.0)
    assert z == pytest.approx(-2.5 + 0.9)
    # a cluster with no height still gets a sensible aim point
    assert cluster_point({"x": 5.0, "y": 0.0, "height": None})[2] == pytest.approx(-2.25)


def test_a_frame_of_a_different_size_keeps_the_mounting():
    big = CAM.with_frame(1600, 1200, 90.0)
    assert big.focal_px == pytest.approx(800.0)
    assert big.mount == CAM.mount and big.pitch_deg == CAM.pitch_deg
    u_small, v_small = CAM.project(12.0, 1.5, -1.0)
    u_big, v_big = big.project(12.0, 1.5, -1.0)
    assert u_big == pytest.approx(2 * u_small, abs=1e-6)
    assert v_big == pytest.approx(2 * v_small, abs=1e-6)
