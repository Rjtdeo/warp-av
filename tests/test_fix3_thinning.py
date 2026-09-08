"""
Perception fix 3: no more one-in-three thinning; a far blob may count with 2
points but the tracker wants three sightings before it believes it.
"""
import math

import os

from warp_av.perception import tracking
from warp_av.perception.tracking import cluster_points, ObjectTracker, MIN_POINTS_FAR, FAR_RANGE_M, vehicle_shaped
from warp_av.perception.ground_filter import lidar_thin_step_from_env

PERC = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "perception", "camera_lidar_perception.py")


def test_thin_step_default_and_env():
    assert lidar_thin_step_from_env({}) == 1
    assert lidar_thin_step_from_env({"WARP_LIDAR_THIN": "3"}) == 3
    assert lidar_thin_step_from_env({"WARP_LIDAR_THIN": "nonsense"}) == 1
    assert lidar_thin_step_from_env({"WARP_LIDAR_THIN": "0"}) == 1


def test_far_blob_counts_with_two_points_near_blob_still_needs_three():
    near2 = [(10.0, 0.0), (10.2, 0.1)]                  # 2 points at 10 m: not enough
    far2 = [(18.0, 0.0), (18.2, 0.1)]                   # 2 points at 18 m: enough, but weak
    near3 = [(6.0, 1.0), (6.1, 1.1), (6.2, 0.9)]
    cl = cluster_points(near2 + far2 + near3, min_points_far=MIN_POINTS_FAR, far_range_m=FAR_RANGE_M)
    got = {round(c["x"]): c for c in cl}
    assert 10 not in got
    assert 18 in got and got[18]["weak"] is True and got[18]["n"] == 2
    assert 6 in got and got[6]["weak"] is False
    # without the far minimum (old callers) nothing changes
    old = cluster_points(near2 + far2 + near3)
    assert [round(c["x"]) for c in old] == [6]


def test_weak_sightings_need_three_in_a_row_strong_ones_two():
    tr = ObjectTracker()
    t = 100.0
    # a weak blob seen once, twice: not yet; three times: believed
    assert tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t) == []
    assert tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t + 0.1) == []
    out = tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t + 0.2)
    assert len(out) == 1 and abs(out[0].wx - 18.0) < 1e-9
    # a strong blob is believed on its second sighting, as before
    tr2 = ObjectTracker()
    assert tr2.update([{"wx": 8.0, "wy": 0.0}], t) == []
    assert len(tr2.update([{"wx": 8.0, "wy": 0.0}], t + 0.1)) == 1
    # strong then weak is not yet enough (1.67 of 2); one more weak sighting is
    tr3 = ObjectTracker()
    tr3.update([{"wx": 8.0, "wy": 0.0}], t)
    assert tr3.update([{"wx": 8.0, "wy": 0.0, "weak": True}], t + 0.1) == []
    assert len(tr3.update([{"wx": 8.0, "wy": 0.0, "weak": True}], t + 0.2)) == 1
    # an object coming closer: two weak far sightings, then one strong one: believed at once
    tr5 = ObjectTracker()
    tr5.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t)
    tr5.update([{"wx": 17.5, "wy": 0.0, "weak": True}], t + 0.1)
    assert len(tr5.update([{"wx": 17.0, "wy": 0.0}], t + 0.2)) == 1
    # a weak blob that vanishes for longer than the memory is forgotten, not accumulated
    tr4 = ObjectTracker()
    tr4.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t)
    tr4.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t + 0.1)
    assert tr4.update([], t + 2.0) == []
    assert tr4.update([{"wx": 18.0, "wy": 0.0, "weak": True}], t + 2.1) == []


def test_far_blob_with_three_points_is_not_weak_and_the_boundary_is_15_m():
    far3 = [(18.0, 0.0), (18.2, 0.1), (18.1, -0.1)]
    at_boundary = [(14.9, 3.0), (15.1, 3.0)]                  # centroid (15.0, 3.0): 15.3 m out, far, weak
    inside = [(14.5, -3.0), (14.6, -3.0)]                     # centroid 14.86 m out: near, dropped
    cl = cluster_points(far3 + at_boundary + inside, min_points_far=MIN_POINTS_FAR, far_range_m=FAR_RANGE_M)
    got = {(round(c["x"], 1), round(c["y"], 1)): c for c in cl}
    assert got[(18.1, 0.0)]["weak"] is False
    assert got[(15.0, 3.0)]["weak"] is True
    assert (14.6, -3.0) not in got and (14.5, -3.0) not in got


def test_cap_keeps_the_corridor_ahead_before_roadside_clutter():
    """160 clutter blobs beside and behind the van, nearer than a car 28 m
    ahead in the lane: the car must survive the cap."""
    pts = []
    for k in range(160):                                       # clutter: 3-point blobs, 5-25 m out, off to the sides
        x = -20.0 + (k % 40) * 1.1
        y = (6.0 + (k // 40) * 2.5) * (1 if k % 2 else -1)
        pts += [(x, y), (x + 0.2, y + 0.1), (x - 0.2, y - 0.1)]
    car = [(28.0, 0.0), (28.5, 0.4), (29.0, -0.3), (29.5, 0.2), (30.0, 0.0), (30.5, 0.3)]
    pts += car
    cl = cluster_points(pts, min_points_far=MIN_POINTS_FAR, far_range_m=FAR_RANGE_M, max_clusters=60)
    assert tracking.LAST_CLUSTER_TOTAL > 60
    assert len(cl) == 60
    assert any(27 < c["x"] < 31 and abs(c["y"]) < 1 for c in cl), "the car ahead was evicted by clutter"


def test_vehicle_shape_rule_wants_points_and_height():
    car = {"extent": 1.6, "n": 20, "height": 1.4}
    bench = {"extent": 1.0, "n": 14, "height": 0.45}
    planter = {"extent": 2.4, "n": 30, "height": 0.28}
    sparse_far_car = {"extent": 1.2, "n": 8, "height": 1.3}
    unknown_height = {"extent": 1.2, "n": 15, "height": None}
    assert vehicle_shaped(car) is True
    assert vehicle_shaped(bench) is False and vehicle_shaped(planter) is False
    assert vehicle_shaped(sparse_far_car) is False and vehicle_shaped(sparse_far_car, min_points=6) is True
    assert vehicle_shaped(unknown_height) is True


def test_tracker_forgets_before_it_associates():
    """A weak blob seen at 0, 0.1 s, then again 1.3 s later: the old track
    is gone by then, so the late sighting starts a new one (no confirmation)."""
    tr = ObjectTracker()
    tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], 0.0)
    tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], 0.1)
    assert tr.update([{"wx": 18.0, "wy": 0.0, "weak": True}], 1.4) == []
    # and a weak-only track is marked as such; one strong sighting clears the mark
    tr2 = ObjectTracker()
    for k in range(3):
        out = tr2.update([{"wx": 18.0, "wy": 0.0, "weak": True}], 0.1 * k)
    assert len(out) == 1 and out[0].weak_only is True
    out = tr2.update([{"wx": 18.0, "wy": 0.0}], 0.3)
    assert out[0].weak_only is False


def test_perception_wiring_for_fix3():
    src = open(PERC).read()
    assert "vehicle_shaped(c, min_points=12 if self.thin_step <= 1 else 6)" in src
    assert "self.last_clusters_before_cap = _tracking.LAST_CLUSTER_TOTAL" in src
    assert 'c.get("weak") and (c.get("height") is not None and c["height"] < 0.30)' in src
    assert '"weak": bool(c.get("weak", False))' in src
    assert 'if not getattr(tr, "weak_only", False) else 0.35' in src
