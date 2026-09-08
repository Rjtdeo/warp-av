"""
Perception V2 day 2: every LiDAR point carries its ring (beam) id and its age
inside the sweep; the scan carries the sensor pose. Nothing downstream changes.
"""
import math
import types

import numpy as np

from warp_av.adapters.carla_sensor_adapter import (CarlaSensorAdapter, LidarScan, LIDAR_COLUMNS,
                                                   decode_lidar)
from warp_av.adapters.lidar_sweep import LidarSweepAccumulator
from warp_av.perception.ground_filter import GroundFilter, flat_cut
from warp_av.perception.tracking import cluster_points


class FakeScan:
    """A CARLA LidarMeasurement look-alike: channel-major points with counts."""

    def __init__(self, per_channel, matrix=None, t=10.0, break_counts=False):
        rows = []
        for ch, n in enumerate(per_channel):
            for k in range(n):
                rows.append([5.0 + ch, 0.1 * k, -2.0 + 0.05 * ch, 0.5])
        self.raw_data = np.asarray(rows, dtype=np.float32).tobytes()
        self.channels = len(per_channel)
        self._counts = list(per_channel)
        if break_counts:
            self._counts[0] += 1
        self.timestamp = t
        m = np.eye(4) if matrix is None else matrix
        self.transform = types.SimpleNamespace(get_matrix=lambda: [list(r) for r in m])

    def get_point_count(self, ch):
        return self._counts[ch]


def test_decode_attaches_the_ring_of_every_point():
    pts = decode_lidar(FakeScan([2, 3, 0, 1]))
    assert pts.shape == (6, 5) and pts.dtype == np.float32
    assert pts[:, 4].tolist() == [0, 0, 1, 1, 1, 3]
    assert np.allclose(pts[:, 0], [5, 5, 6, 6, 6, 8])
    # counts that do not add up: ring unknown, points still delivered
    bad = decode_lidar(FakeScan([2, 3, 0, 1], break_counts=True))
    assert bad.shape == (6, 5) and (bad[:, 4] == -1).all()
    # an API without the channel helpers: same fallback
    plain = types.SimpleNamespace(raw_data=np.zeros(8, np.float32).tobytes())
    out = decode_lidar(plain)
    assert out.shape == (2, 5) and (out[:, 4] == -1).all()


def test_sweep_appends_the_age_of_every_point():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    a = np.c_[np.array([[10.0, 0.0, -1.0, 0.5], [10.5, 0.2, -1.0, 0.5]], np.float32), np.array([3, 4], np.float32)]
    b = np.c_[np.array([[-9.0, 0.0, -1.0, 0.5]], np.float32), np.array([7], np.float32)]
    acc.add(a, np.eye(4), sim_time=100.00)
    sweep = acc.add(b, np.eye(4), sim_time=100.06)
    assert sweep.shape == (3, 6)
    assert sweep[:, 4].tolist() == [3, 4, 7]                     # ring carried through the sweep
    assert np.allclose(sweep[:, 5], [-0.06, -0.06, 0.0], atol=1e-6)   # age: older delivery first


def test_adapter_publishes_six_columns_and_the_sensor_pose():
    on = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=True)
    m = np.eye(4); m[0, 3] = 12.5; m[1, 3] = -3.0
    on._on_lidar(FakeScan([4, 4, 4], matrix=m, t=20.00))
    on._on_lidar(FakeScan([4, 4, 4], matrix=m, t=20.04))
    scan = on.latest_lidar
    assert isinstance(scan, LidarScan) and scan.columns == LIDAR_COLUMNS
    assert scan.points.shape == (24, 6)
    assert set(scan.points[:, 4].tolist()) == {0.0, 1.0, 2.0}
    assert scan.points[:, 5].min() < 0 <= scan.points[:, 5].max() + 1e-9
    assert scan.sensor_matrix is not None and scan.sensor_matrix[0, 3] == 12.5
    assert scan.sim_time == 20.04 and on.lidar_sweep_errors == 0
    # raw (old) mode: still six columns, age zero, ring kept
    off = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=False)
    off._on_lidar(FakeScan([2, 2], t=1.0))
    assert off.latest_lidar.points.shape == (4, 6) and (off.latest_lidar.points[:, 5] == 0).all()


def test_downstream_reads_only_the_first_columns():
    """The ground filter, the flat cut and the clusterer must not care that
    points now have six columns."""
    rng = np.random.default_rng(2)
    n = 500
    pts6 = np.c_[rng.uniform(3, 40, n), rng.uniform(-6, 6, n), np.full(n, -2.5), rng.uniform(0, 1, n),
                 rng.integers(0, 32, n).astype(np.float32), rng.uniform(-0.1, 0, n)].astype(np.float32)
    pts6[:20, 2] = -2.5 + 0.8                                   # 20 object points 0.8 m up
    res6 = GroundFilter().apply(pts6)
    res4 = GroundFilter().apply(pts6[:, :4])
    assert (res6.keep == res4.keep).all() and res6.keep[:20].all()
    assert (flat_cut(pts6).keep == flat_cut(pts6[:, :4]).keep).all()
    sel = pts6[res6.keep]
    cl = cluster_points(sel[:, :2].tolist(), heights=res6.above[res6.keep].tolist())
    assert len(cl) >= 1


def test_ring_and_age_stay_aligned_through_the_seam_dedup():
    """Overlapping 120-point wedges: the de-dup removes rows from older frames,
    and every surviving point must still carry its own ring and age."""
    acc = LidarSweepAccumulator(rotation_hz=10.0)

    def wedge(a0, a1, ring_id, n=120):
        az = np.radians(np.linspace(a0, a1, n, endpoint=False))
        pts = np.zeros((n, 5), dtype=np.float32)
        pts[:, 0] = 10 * np.cos(az); pts[:, 1] = 10 * np.sin(az); pts[:, 2] = -1.0; pts[:, 3] = 0.5
        pts[:, 4] = ring_id
        return pts

    acc.add(wedge(0, 100, 1), np.eye(4), sim_time=1.00)
    acc.add(wedge(80, 180, 2), np.eye(4), sim_time=1.02)          # overlaps 80-100 with the first
    sweep = acc.add(wedge(160, 260, 3), np.eye(4), sim_time=1.04)  # overlaps 160-180 with the second
    assert sweep.shape[1] == 6
    ring = sweep[:, 4]; age = sweep[:, 5]
    assert set(ring.tolist()) == {1.0, 2.0, 3.0}
    assert np.allclose(age[ring == 1], -0.04) and np.allclose(age[ring == 2], -0.02) and np.allclose(age[ring == 3], 0.0)
    # the overlapped parts of the older wedges are gone, the rest is intact
    assert (ring == 1).sum() < 120 and (ring == 2).sum() < 120 and (ring == 3).sum() == 120
    az = np.degrees(np.arctan2(sweep[:, 1], sweep[:, 0])) % 360
    assert (az[ring == 1] < 81).all() and (az[ring == 2] < 161).all()


def test_lidar_scan_refuses_the_wrong_number_of_columns():
    import pytest
    with pytest.raises(ValueError):
        LidarScan(points=np.zeros((3, 4), np.float32))
    LidarScan(points=np.zeros((0, 4), np.float32))               # empty is fine
    LidarScan(points=np.zeros((3, 6), np.float32))
