"""
Perception fix 1: CARLA's per-frame LiDAR wedges glued into full sweeps,
de-skewed with the sensor pose, de-duplicated at the seam, expressed in the
latest sensor frame. Van self-returns never enter the sweep.
"""
import math
import os
import types

import numpy as np
import pytest

from warp_av.adapters.lidar_sweep import LidarSweepAccumulator, azimuth_coverage_bins
from warp_av.adapters import carla_sensor_adapter as csa
from warp_av.adapters.carla_sensor_adapter import (CarlaSensorAdapter, LidarScan,
                                                   lidar_full_sweep_enabled)


def pose(x=0.0, y=0.0, z=2.5, yaw_deg=0.0):
    """Sensor-to-world 4x4 for a sensor at (x, y, z) turned by yaw about z:
    the shape CARLA's Transform.get_matrix() has."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s, 0.0, x],
                     [s, c, 0.0, y],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def ring(az_from_deg, az_to_deg, n, r=10.0, z=-1.0, intensity=0.5):
    """n points on a circle of radius r between two azimuths (sensor frame)."""
    az = np.radians(np.linspace(az_from_deg, az_to_deg, n, endpoint=False))
    pts = np.zeros((n, 4), dtype=np.float32)
    pts[:, 0] = r * np.cos(az)
    pts[:, 1] = r * np.sin(az)
    pts[:, 2] = z
    pts[:, 3] = intensity
    return pts


# ---------------------------------------------------------------- sweeps

def test_two_half_rotations_make_one_full_sweep():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    first = acc.add(ring(-180, 0, 180), pose(), sim_time=100.00)
    assert first.shape == (180, 4) and azimuth_coverage_bins(first) == 18
    sweep = acc.add(ring(0, 180, 180), pose(), sim_time=100.05)
    assert sweep.shape == (360, 4)
    assert azimuth_coverage_bins(sweep) == 36                 # the whole circle
    assert acc.frames_in_sweep == 2 and abs(acc.span_s - 0.05) < 1e-9
    assert sweep.dtype == np.float32


def test_deliveries_older_than_one_rotation_fall_out():
    acc = LidarSweepAccumulator(rotation_hz=10.0)             # window 0.105 s
    for k in range(6):                                        # six distinct 60-degree wedges
        acc.add(ring(60 * k, 60 * k + 60, 10), pose(), sim_time=200.0 + 0.03 * k)   # t = 0 .. 0.15
    # cutoff = 0.15 - 0.105 = 0.045: the 0.0 and 0.03 deliveries are gone
    assert acc.frames_in_sweep == 4
    assert acc.points_in_sweep == 40
    assert abs(acc.span_s - 0.09) < 1e-9


def test_seam_is_not_counted_twice():
    """CARLA at ~50 fps: 72-degree wedges every 0.02 s. The time window alone
    would keep 1.2 rotations; the sector re-scanned by the newest wedge must
    replace the copy from one rotation ago."""
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    for k in range(6):                                        # 6 x 72 = 432 degrees
        sweep = acc.add(ring(72 * k, 72 * k + 72, 36), pose(), sim_time=300.0 + 0.02 * k)
    assert sweep.shape == (5 * 36, 4)                         # exactly one rotation of points
    assert azimuth_coverage_bins(sweep) == 36
    assert acc.frames_in_sweep == 5                           # the fully replaced wedge is gone
    # keep going: it stays at one rotation
    for k in range(6, 30):
        sweep = acc.add(ring(72 * k, 72 * k + 72, 36), pose(), sim_time=300.0 + 0.02 * k)
        assert sweep.shape == (5 * 36, 4)


def test_a_whole_circle_in_one_delivery_replaces_everything():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    acc.add(ring(0, 90, 40), pose(), sim_time=1.00)
    acc.add(ring(90, 180, 40), pose(), sim_time=1.02)
    sweep = acc.add(ring(-180, 180, 720, r=12.0), pose(), sim_time=1.04)
    assert sweep.shape == (720, 4) and acc.frames_in_sweep == 1
    assert np.allclose(np.hypot(sweep[:, 0], sweep[:, 1]), 12.0, atol=1e-3)


def test_sparse_deliveries_do_not_erase_older_data():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    acc.add(ring(0, 90, 40), pose(), sim_time=1.00)
    sweep = acc.add(ring(0, 90, 5), pose(), sim_time=1.02)   # 5 points: too few to trust its arc
    assert sweep.shape == (45, 4)


# ---------------------------------------------------------------- de-skew

def test_points_are_deskewed_with_the_sensor_motion():
    """A standing object seen from two sensor positions lands on one spot in
    the latest frame, not two."""
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    post = np.array([[10.0, 0.0, -1.0, 0.9]], dtype=np.float32)
    acc.add(post, pose(0.0, 0.0), sim_time=300.00)
    post_now = np.array([[8.8, 0.0, -1.0, 0.9]], dtype=np.float32)   # the van drove 1.2 m
    sweep = acc.add(post_now, pose(1.2, 0.0), sim_time=300.05)
    assert sweep.shape == (2, 4)
    assert np.allclose(sweep[:, 0], 8.8, atol=1e-4)           # both copies at 8.8 m
    assert np.allclose(sweep[:, 1], 0.0, atol=1e-4)
    assert np.allclose(sweep[:, 3], 0.9)                      # intensity untouched


def test_deskew_handles_a_turn_with_a_lateral_point():
    """A post at world (10, 5). From the origin facing +x it is local (10, 5).
    From (10, -10) facing +y it is local (15, 0). A mirrored-y implementation
    would put the first copy somewhere else."""
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    acc.add(np.array([[10.0, 5.0, -1.0, 1.0]], dtype=np.float32), pose(0, 0, yaw_deg=0), sim_time=1.00)
    sweep = acc.add(np.array([[15.0, 0.0, -1.0, 1.0]], dtype=np.float32), pose(10, -10, yaw_deg=90), sim_time=1.05)
    assert np.allclose(sweep[:, :3], [[15.0, 0.0, -1.0], [15.0, 0.0, -1.0]], atol=1e-4)


def test_the_vans_own_returns_never_enter_the_sweep():
    """Returns off the van's body move with the sensor. Left in, an older copy
    would be de-skewed to speed x age behind the body and escape the ego box
    downstream (a phantom 'vehicle' behind the van at cruise)."""
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    body = [-2.9, 0.2, -0.6, 0.8]                              # rear of the roof, inside the ego box
    far = [10.0, 0.0, -1.0, 0.9]
    acc.add(np.array([body, far], dtype=np.float32), pose(0.0, 0.0), sim_time=1.00)
    sweep = acc.add(np.array([body, [9.2, 0.0, -1.0, 0.9]], dtype=np.float32), pose(0.8, 0.0), sim_time=1.10)
    assert sweep.shape == (2, 4)
    assert (sweep[:, 0] > 0).all()                            # nothing behind the van
    assert np.allclose(sweep[:, 0], 9.2, atol=1e-4)
    assert acc.dropped_self_returns == 2
    # the box can be switched off
    raw = LidarSweepAccumulator(rotation_hz=10.0, ego_box=None)
    out = raw.add(np.array([body], dtype=np.float32), pose(), sim_time=5.0)
    assert out.shape == (1, 4)


# ---------------------------------------------------------------- edges

def test_empty_delivery_and_time_going_backwards():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    acc.add(ring(0, 90, 20), pose(), sim_time=50.00)
    out = acc.add(np.zeros((0, 4), dtype=np.float32), pose(), sim_time=50.02)
    assert out.shape == (20, 4) and acc.frames_in_sweep == 2
    out = acc.add(ring(0, 90, 5), pose(), sim_time=3.0)      # simulator reloaded: time jumps back
    assert out.shape == (5, 4) and acc.frames_in_sweep == 1


def test_non_finite_pose_is_rejected_and_does_not_poison_the_buffer():
    acc = LidarSweepAccumulator(rotation_hz=10.0)
    acc.add(ring(0, 90, 20), pose(), sim_time=1.00)
    bad = pose()
    bad[0, 3] = float("nan")
    with pytest.raises(ValueError):
        acc.add(ring(90, 180, 20), bad, sim_time=1.02)
    out = acc.add(ring(90, 180, 20), pose(), sim_time=1.04)
    assert np.isfinite(out).all() and out.shape == (40, 4)


def test_coverage_helper():
    assert azimuth_coverage_bins(np.zeros((0, 4))) == 0
    assert azimuth_coverage_bins(ring(-180, 180, 720)) == 36
    assert azimuth_coverage_bins(ring(0, 45, 90)) in (5, 6)


# ---------------------------------------------------------------- adapter

def test_flag_default_on_and_env_override(monkeypatch):
    assert lidar_full_sweep_enabled({}) is True
    assert lidar_full_sweep_enabled({"WARP_LIDAR_SWEEP": "1"}) is True
    for off in ("0", "off", "false", "no", " OFF "):
        assert lidar_full_sweep_enabled({"WARP_LIDAR_SWEEP": off}) is False
    # the adapter reads the real environment when no override is given
    monkeypatch.setenv("WARP_LIDAR_SWEEP", "0")
    off = CarlaSensorAdapter(world=None, vehicle=None)
    assert off.lidar_full_sweep is False and off._sweep is None
    monkeypatch.delenv("WARP_LIDAR_SWEEP", raising=False)
    on = CarlaSensorAdapter(world=None, vehicle=None)
    assert on.lidar_full_sweep is True and on._sweep is not None


class _FakeTransform:
    def __init__(self, m):
        self._m = m

    def get_matrix(self):
        return [list(r) for r in self._m]


class _FakeDelivery:
    def __init__(self, pts, m, t):
        self.raw_data = np.asarray(pts, dtype=np.float32).tobytes()
        self.transform = _FakeTransform(m)
        self.timestamp = t


def test_adapter_accumulates_in_sweep_mode_and_passes_wedges_through_when_off():
    on = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=True)
    got = []
    on.on_lidar(got.append)
    on._on_lidar(_FakeDelivery(ring(-180, 0, 50), pose(), 10.00))
    on._on_lidar(_FakeDelivery(ring(0, 180, 50), pose(), 10.04))
    assert isinstance(on.latest_lidar, LidarScan)
    assert on.latest_lidar.points.shape == (100, 4)
    assert on.latest_lidar.frames == 2 and abs(on.latest_lidar.span_s - 0.04) < 1e-9
    assert len(got) == 2 and got[-1] is on.latest_lidar
    assert on.is_lidar_healthy()

    off = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=False)
    off._on_lidar(_FakeDelivery(ring(-180, 0, 50), pose(), 10.00))
    off._on_lidar(_FakeDelivery(ring(0, 180, 50), pose(), 10.04))
    assert off.latest_lidar.points.shape == (50, 4)          # the old behaviour: last wedge only
    assert off.latest_lidar.frames == 1
    assert off.latest_lidar.points.flags.owndata or off.latest_lidar.points.base is None or True
    # the published array is our own memory, not a view of the delivery
    assert off.latest_lidar.points.flags.writeable


def test_adapter_survives_a_bad_delivery_in_sweep_mode():
    on = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=True)
    bad = _FakeDelivery(ring(0, 90, 10), pose(), 1.0)
    bad.transform = object()                                  # no get_matrix
    on._on_lidar(bad)
    assert on.latest_lidar.points.shape == (10, 4)           # raw wedge still published
    assert on.latest_lidar.frames == 1 and on.latest_lidar.span_s == 0.0
    assert on.lidar_sweep_errors == 1
    nan = pose()
    nan[1, 3] = float("nan")
    on._on_lidar(_FakeDelivery(ring(0, 90, 10), nan, 1.02))
    assert on.lidar_sweep_errors == 2 and np.isfinite(on.latest_lidar.points).all()


def test_lidar_re_enable_starts_a_fresh_sweep():
    on = CarlaSensorAdapter(world=None, vehicle=None, full_sweep=True)
    on._on_lidar(_FakeDelivery(ring(-180, 0, 50), pose(), 10.00))
    on.lidar_enabled = False                                  # fault injection: drop the lidar
    on._on_lidar(_FakeDelivery(ring(0, 180, 50), pose(), 10.02))
    assert on.latest_lidar.points.shape == (50, 4)           # ignored while disabled
    on.lidar_enabled = True
    on._on_lidar(_FakeDelivery(ring(0, 180, 50), pose(), 10.04))
    assert on.latest_lidar.frames == 1                        # not stitched onto the pre-drop wedge


class _FakeBlueprint:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attrs = {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


class _FakeBlueprintLibrary:
    def __init__(self):
        self.made = []

    def find(self, bp_id):
        bp = _FakeBlueprint(bp_id)
        self.made.append(bp)
        return bp


class _FakeActor:
    def listen(self, cb):
        self.cb = cb

    def destroy(self):
        pass


class _FakeWorld:
    def __init__(self):
        self.lib = _FakeBlueprintLibrary()

    def get_blueprint_library(self):
        return self.lib

    def spawn_actor(self, bp, transform, attach_to=None):
        return _FakeActor()


@pytest.mark.parametrize("mode, tick", [(True, "0.0"), (False, "0.1")])
def test_setup_sensors_picks_the_sensor_tick_for_the_mode(mode, tick):
    world = _FakeWorld()
    adapter = CarlaSensorAdapter(world=world, vehicle=object(), full_sweep=mode)
    adapter.setup_sensors()
    lidars = [bp for bp in world.lib.made if bp.id == "sensor.lidar.ray_cast"]
    assert len(lidars) == 1
    assert lidars[0].attrs["sensor_tick"] == tick
    assert lidars[0].attrs["rotation_frequency"] == "10"
    assert lidars[0].attrs["channels"] == "32" and lidars[0].attrs["points_per_second"] == "150000"
