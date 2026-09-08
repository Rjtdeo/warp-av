"""
Perception V2 day 2: the recorded fixture is sound, and the real LiDAR
pipeline (sweep gluing, ground filter, clustering) replays on it offline and
finds the objects that were placed. Day 3 turns this into the scored harness.
"""
import json
import math
import os

import numpy as np
import pytest

from warp_av.adapters.lidar_sweep import LidarSweepAccumulator, azimuth_coverage_bins
from warp_av.perception.ground_filter import GroundFilter, remove_road_edge_points
from warp_av.perception.tracking import cluster_points, MIN_POINTS_FAR, FAR_RANGE_M

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "perception", "town03_straight_a")
pytestmark = pytest.mark.skipif(not os.path.isdir(FIX), reason="fixture not present")


def load():
    d = np.load(os.path.join(FIX, "deliveries.npz"))
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    objects = json.load(open(os.path.join(FIX, "objects.json")))
    return d, meta, objects


def test_fixture_shape_and_ring_order():
    d, meta, objects = load()
    pts = d["points"]
    assert pts.shape[1] == 5 and pts.dtype == np.float32
    assert len(d["delivery_time"]) == len(d["delivery_matrix"]) >= 40
    # as dense as the van's own LiDAR (with drop-off), not the labelled one
    assert 6000 <= meta["lidar"]["points_per_rotation"] <= 8500
    # rings: one elevation each, falling from +10 to -30 degrees
    ring = pts[:, 4]
    el = np.degrees(np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))
    means = [float(el[ring == r].mean()) for r in range(32)]
    assert all(a > b for a, b in zip(means, means[1:]))
    assert abs(means[0] - 10.0) < 0.2 and abs(means[-1] + 30.0) < 0.2
    assert max(float(el[ring == r].max() - el[ring == r].min()) for r in range(32)) < 0.2
    assert os.path.exists(os.path.join(FIX, "labels.npz")) and len(objects) >= 5
    assert any(f.startswith("camera_") for f in os.listdir(FIX))


def test_pipeline_replays_on_the_fixture_and_finds_the_placed_objects():
    d, meta, objects = load()
    pts, idx = d["points"], d["delivery_index"]
    times, mats = d["delivery_time"], d["delivery_matrix"]
    acc = LidarSweepAccumulator()
    sweep = None
    for i in range(len(times)):
        sweep = acc.add(pts[idx == i], mats[i], float(times[i]))
    assert sweep is not None and sweep.shape[1] == 6
    assert azimuth_coverage_bins(sweep) == 36
    assert 6000 <= sweep.shape[0] <= 9000

    res = GroundFilter().apply(sweep)
    sel, heights, _ = remove_road_edge_points(sweep[res.keep], res.above[res.keep], cluster_points)
    clusters = cluster_points(sel[:, :2].tolist(), heights=heights.tolist(),
                              min_points_far=MIN_POINTS_FAR, far_range_m=FAR_RANGE_M)
    assert 5 <= len(clusters) <= 120

    # every placed object must have a cluster near it (van frame: x forward, y right)
    van = meta["van"]
    yaw = math.radians(van["yaw_deg"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    misses = []
    for o in objects:
        dx, dy = o["x"] - van["x"], o["y"] - van["y"]
        ox, oy = dx * cy + dy * sy, -dx * sy + dy * cy
        reach = max(1.5, math.hypot(o["extent"][0], o["extent"][1]) + 0.5)
        if not any(math.hypot(c["x"] - ox, c["y"] - oy) <= reach for c in clusters):
            misses.append((o["blueprint"], round(ox, 1), round(oy, 1)))
    assert not misses, f"objects without a cluster: {misses}"
