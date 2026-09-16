"""Two parked vehicles half a metre apart come out as one blob.

F_reroute, 2026-09-15: a stopped truck in the lane and a parked Nissan Patrol beside it, their
bodies 0.47 m apart across the road and overlapping about a metre along it. The van tracked
the pair as a single 7.2 m body whose centre sat between them -- five metres short of the
truck and 1.2 m toward the lane -- so it stopped 15.7 m back from a truck it should have been
reading at 13, and the truck itself was never detected as its own object at all.

A split was tried and measured against the four town03 fixtures. It removed none of their six
merges under any setting while doubling the blob count, because those merges are bodies whose
returns run continuously into each other, not two bodies with air between them. None of those
recordings contains this geometry, so there was nothing to develop a fix against.

This fixture is that geometry, recorded on Town10HD with the real LiDAR:

    tools/record_fixture.py --name town10_two_parked --seconds 1.2 --at=-18,140.3 \
        --place "vehicle.carlamotors.carlacola@16@0,vehicle.nissan.patrol_2021@11.7@3.04"

The truck sits in the lane, the 4x4 beside it, 0.65 m between their bodies and 1.08 m of
overlap along the road. Replayed through the real pipeline they come out as one blob of 118
points, 7.08 x 2.62 m, centred between them -- near enough to the 4x4 to count as finding it,
5.3 m from the truck, which is therefore lost.
"""
import json
import math
import os

import numpy as np
import pytest

from warp_av.adapters.lidar_sweep import LidarSweepAccumulator
from warp_av.perception.ground_filter import GroundFilter, remove_road_edge_points
from warp_av.perception.tracking import cluster_points, MIN_POINTS_FAR, FAR_RANGE_M

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "perception", "town10_two_parked")
pytestmark = pytest.mark.skipif(not os.path.isdir(FIX), reason="fixture not present")


def load():
    d = np.load(os.path.join(FIX, "deliveries.npz"))
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    objects = json.load(open(os.path.join(FIX, "objects.json")))
    return d, meta, objects


def in_van_frame(o, meta):
    van = meta["van"]
    yaw = math.radians(van["yaw_deg"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx, dy = o["x"] - van["x"], o["y"] - van["y"]
    return dx * cy + dy * sy, -dx * sy + dy * cy


def blobs():
    d, meta, objects = load()
    pts, idx = d["points"], d["delivery_index"]
    times, mats = d["delivery_time"], d["delivery_matrix"]
    acc = LidarSweepAccumulator()
    sweep = None
    for i in range(len(times)):
        sweep = acc.add(pts[idx == i], mats[i], float(times[i]))
    res = GroundFilter().apply(sweep)
    sel, heights, _ = remove_road_edge_points(sweep[res.keep], res.above[res.keep], cluster_points)
    cl = cluster_points(sel[:, :2].tolist(), heights=heights.tolist(),
                        min_points_far=MIN_POINTS_FAR, far_range_m=FAR_RANGE_M)
    return cl, meta, objects


def near(cl, o, meta):
    """Blobs whose centre is within reach of this object."""
    ox, oy = in_van_frame(o, meta)
    reach = max(1.5, math.hypot(o["extent"][0], o["extent"][1]) + 0.5)
    return [c for c in cl if math.hypot(c["x"] - ox, c["y"] - oy) <= reach]


# ---- the fixture itself ------------------------------------------------------------------

def test_the_fixture_holds_the_two_vehicles_it_was_recorded_for():
    d, meta, objects = load()
    assert meta["map"].endswith("Town10HD")
    kinds = sorted(o["blueprint"] for o in objects)
    assert kinds == ["vehicle.carlamotors.carlacola", "vehicle.nissan.patrol_2021"]
    assert meta["ring_order_ok"] is True
    assert d["points"].shape[1] == 5 and meta["deliveries"] >= 20


def test_they_are_close_enough_across_the_road_and_overlap_along_it():
    """The geometry is the whole point: bodies within a metre of each other sideways, with
    one beside the other rather than behind it."""
    _, meta, objects = load()
    box = {}
    for o in objects:
        ox, oy = in_van_frame(o, meta)
        box[o["blueprint"]] = (ox, oy, 2 * o["extent"][0], 2 * o["extent"][1])
    a = box["vehicle.carlamotors.carlacola"]
    b = box["vehicle.nissan.patrol_2021"]
    gap = abs(b[1] - a[1]) - a[3] / 2 - b[3] / 2
    overlap = min(a[0] + a[2] / 2, b[0] + b[2] / 2) - max(a[0] - a[2] / 2, b[0] - b[2] / 2)
    assert 0.3 < gap < 1.0, f"bodies {gap:.2f} m apart across the road"
    assert overlap > 0.5, f"they overlap {overlap:.2f} m along it"


def test_the_pipeline_replays_on_it():
    cl, meta, objects = blobs()
    assert 5 <= len(cl) <= 120


# ---- the fault it was recorded to hold -------------------------------------------------------

def test_neither_blob_swallows_the_other():
    """Until 2026-09-15 these two shared one blob of 118 points, 7.08 x 2.62 m, centred
    between them -- near enough to the 4x4 to count as finding it, 5.3 m from the truck,
    which was therefore lost. Now the points inside a blob are joined by distance rather than
    by grid cell, and half a metre of air is enough to keep them apart."""
    cl, meta, objects = blobs()
    both = [c for c in cl if all(near([c], o, meta) for o in objects)]
    assert both == [], "no blob covers both of them"
    for o in objects:
        for c in near(cl, o, meta):
            assert c["length_m"] < 6.5, \
                f"{o['blueprint']}'s blob is {c['length_m']:.2f} m long: it has swallowed its neighbour"


def test_each_parked_vehicle_gets_its_own_blob():
    """What the fixture was recorded to make true. It was a strict xfail from the day the
    fixture landed until the split worked, which is what made the change visible."""
    cl, meta, objects = blobs()
    for o in objects:
        got = near(cl, o, meta)
        assert got, f"{o['blueprint']} has no blob of its own"
        for c in got:
            assert c["length_m"] < 6.5, \
                f"{o['blueprint']}'s blob is {c['length_m']:.2f} m long: it has swallowed its neighbour"
