"""Camera/LiDAR perception v2 core: clustering + world-frame tracking."""
import math
import random

from warp_av.perception.tracking import cluster_points, ObjectTracker


def blob(cx, cy, n=12, r=0.8, seed=1):
    rng = random.Random(seed)
    return [(cx + rng.uniform(-r, r), cy + rng.uniform(-r, r)) for _ in range(n)]


def test_clustering_separates_two_cars_and_ignores_ego_returns():
    pts = blob(12.0, 0.0) + blob(12.0, 6.0, seed=2)          # two objects
    pts += [(1.5, 0.9), (2.4, 0.0), (1.6, -0.9)]             # our own body
    pts += [(0.5, 30.0)]                                     # lone stray ray
    clusters = cluster_points(pts)
    assert len(clusters) == 2
    assert abs(clusters[0]["y"]) < 1.0 and abs(clusters[1]["y"] - 6.0) < 1.0


def test_tracker_measures_speed_of_moving_object():
    tr = ObjectTracker()
    t = 100.0
    tracks = []
    for step in range(12):                    # object moving +5 m/s in x
        obs = [{"wx": 10.0 + 5.0 * step * 0.1, "wy": 2.0, "cls": "vehicle",
                "confidence": 0.9}]
        tracks = tr.update(obs, t + step * 0.1)
    assert len(tracks) == 1
    assert tracks[0].cls == "vehicle"
    assert abs(tr.reported_speed(tracks[0]) - 5.0) < 1.2


def test_tracker_parked_car_reports_zero_speed():
    tr = ObjectTracker()
    tracks = []
    for step in range(10):                    # jittering centroid, parked car
        obs = [{"wx": 30.0 + 0.05 * (step % 2), "wy": -1.0}]
        tracks = tr.update(obs, 200.0 + step * 0.1)
    assert len(tracks) == 1
    assert tr.reported_speed(tracks[0]) == 0.0, "jitter must not look like motion"


def test_tracker_keeps_ids_stable_and_drops_stale():
    tr = ObjectTracker()
    a = {"wx": 5.0, "wy": 0.0}
    b = {"wx": 25.0, "wy": 5.0}
    tr.update([a, b], 1.0)
    t2 = tr.update([{"wx": 5.3, "wy": 0.1}, {"wx": 25.2, "wy": 5.1}], 1.1)
    ids1 = sorted(x.tid for x in t2)
    tr.update([{"wx": 5.6, "wy": 0.2}], 1.2)             # b vanished
    later = tr.update([{"wx": 6.2, "wy": 0.3}], 2.35)    # a kept (gap 1.15 s),
    assert sorted(x.tid for x in later) == [ids1[0]], \
        "a keeps its id; b (unseen 1.25 s) must be forgotten"
