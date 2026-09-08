"""
Full LiDAR sweeps from CARLA's per-frame deliveries (perception fix 1).

CARLA's ray-cast LiDAR does not hand over one 360-degree scan per rotation.
In asynchronous mode it emits, every simulator frame, only the wedge of the
rotation that frame covered (rotation_frequency x frame_time x 360 degrees,
points_per_second x frame_time points). Measured on the van, 2026-09-07:
2,000-5,000 points per delivery covering 40-290 degrees in a direction that
changes every frame; in 3 of 10 deliveries not one point was ahead of the
van. With sensor_tick 0.1 the frames in between were thrown away as well.

This module glues the deliveries back into whole sweeps:

  * returns off the van's own body (inside the ego box, in the capture
    frame) are dropped first: they move WITH the sensor, so de-skewing
    them as if they were world-fixed would smear them behind a moving van;
  * every delivery is moved into the WORLD frame using the sensor pose CARLA
    stamps on it (so a van that moved between frames does not smear a
    standing object);
  * the sector the newest delivery re-scanned is removed from the older
    deliveries (azimuth de-duplication), so the sweep holds each direction
    exactly once, whatever the frame rate;
  * deliveries older than about one rotation (window_s) are dropped;
  * the union is expressed in the LATEST sensor frame, the frame the rest of
    the stack expects (x forward, y right, z up, metres, Nx4 with intensity).

Pure numpy; no CARLA import, so it is unit-tested offline.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np

DEFAULT_ROTATION_HZ = 10.0
DEFAULT_WINDOW_FACTOR = 1.05        # a hair over one rotation: no gap at the seam
DEFAULT_MAX_FRAMES = 64             # safety cap, never reached at real frame rates
DEFAULT_EGO_BOX = (3.2, 1.3)        # half-length, half-width of the van's own body (as cluster_points)
DEDUP_MIN_POINTS = 20               # a delivery this sparse cannot tell us which sector it covered
DEDUP_FULL_CIRCLE_GAP_RAD = math.radians(5.0)   # no empty gap wider than this: the delivery is a full circle
DEDUP_MARGIN_RAD = math.radians(0.5)


class LidarSweepAccumulator:
    """Feed it every LiDAR delivery; get back the current full sweep."""

    def __init__(self, rotation_hz: float = DEFAULT_ROTATION_HZ,
                 window_factor: float = DEFAULT_WINDOW_FACTOR,
                 max_frames: int = DEFAULT_MAX_FRAMES,
                 ego_box: Optional[Tuple[float, float]] = DEFAULT_EGO_BOX):
        if rotation_hz <= 0:
            raise ValueError("rotation_hz must be positive")
        self.window_s = float(window_factor) / float(rotation_hz)
        self.max_frames = int(max_frames)
        self.ego_box = ego_box
        # each frame: [sim_time, world-frame Nx4, capture-frame azimuth (N,)]
        self._frames: Deque[List] = deque()
        self._last_time = None
        # diagnostics of the last sweep returned
        self.frames_in_sweep = 0
        self.span_s = 0.0
        self.points_in_sweep = 0
        self.dropped_self_returns = 0

    def reset(self) -> None:
        self._frames.clear()
        self._last_time = None
        self.frames_in_sweep = 0
        self.span_s = 0.0
        self.points_in_sweep = 0

    # ---- geometry -------------------------------------------------------
    @staticmethod
    def _apply(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Rigid 4x4 transform applied to the xyz of an NxK array (K >= 4);
        every column after xyz (intensity, and any extra such as a label)
        is carried along unchanged."""
        out = np.empty((pts.shape[0], pts.shape[1]), dtype=np.float32)
        if pts.shape[0] == 0:
            return out
        xyz1 = np.empty((pts.shape[0], 4), dtype=np.float64)
        xyz1[:, :3] = pts[:, :3]
        xyz1[:, 3] = 1.0
        moved = xyz1 @ matrix.T
        out[:, :3] = moved[:, :3]
        out[:, 3:] = pts[:, 3:]
        return out

    @staticmethod
    def _arc(az: np.ndarray):
        """The azimuth arc a delivery covers, as (start, end) going the positive
        way from start to end (it may wrap through +pi). None = the whole
        circle. Found as the complement of the widest empty gap."""
        a = np.sort(az)
        if a.size < 2:
            return None
        gaps = np.diff(a)
        wrap_gap = (a[0] + 2.0 * math.pi) - a[-1]
        k = int(np.argmax(gaps))
        if wrap_gap >= gaps[k]:
            if wrap_gap < DEDUP_FULL_CIRCLE_GAP_RAD:
                return None
            return float(a[0]), float(a[-1])
        if gaps[k] < DEDUP_FULL_CIRCLE_GAP_RAD:
            return None
        return float(a[k + 1]), float(a[k])

    @staticmethod
    def _in_arc(az: np.ndarray, arc) -> np.ndarray:
        s, e = arc[0] - DEDUP_MARGIN_RAD, arc[1] + DEDUP_MARGIN_RAD
        if s <= e:
            return (az >= s) & (az <= e)
        return (az >= s) | (az <= e)          # the arc wraps through +pi

    # ---- main entry -----------------------------------------------------
    def add(self, points, sensor_to_world, sim_time: float) -> np.ndarray:
        """Add one delivery. `points`: Nx4 (x, y, z, intensity) in the sensor
        frame at capture. `sensor_to_world`: the 4x4 matrix CARLA gives for
        the sensor's transform at capture (Transform.get_matrix()).
        `sim_time`: the delivery's simulation timestamp in seconds.
        Returns the sweep (Nx4, float32) in THIS delivery's sensor frame."""
        pts = np.asarray(points, dtype=np.float32)
        pts = pts.reshape(-1, 4) if pts.ndim != 2 or pts.shape[1] < 4 else pts
        M = np.asarray(sensor_to_world, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(M).all():
            raise ValueError("non-finite sensor pose")
        t = float(sim_time)
        if self._last_time is not None and t < self._last_time - 1e-6:
            self.reset()                       # simulation time went backwards: a reload
        self._last_time = t

        if self.ego_box is not None and pts.shape[0]:
            hl, hw = self.ego_box
            own = (np.abs(pts[:, 0]) < hl) & (np.abs(pts[:, 1]) < hw)
            if own.any():
                self.dropped_self_returns += int(own.sum())
                pts = pts[~own]

        az = np.arctan2(pts[:, 1], pts[:, 0]).astype(np.float32) if pts.shape[0] else np.zeros(0, np.float32)

        # 1. older than one rotation: out
        cutoff = t - self.window_s
        while self._frames and self._frames[0][0] <= cutoff:
            self._frames.popleft()
        while len(self._frames) > self.max_frames:
            self._frames.popleft()

        # 2. the sector this delivery re-scanned: out of the older frames
        if pts.shape[0] >= DEDUP_MIN_POINTS:
            arc = self._arc(az)
            if arc is None:
                self._frames.clear()           # a whole circle in one delivery
            else:
                kept: Deque[List] = deque()
                for fr in self._frames:
                    stale = self._in_arc(fr[2], arc)
                    if stale.any():
                        keep = ~stale
                        if not keep.any():
                            continue
                        fr[1] = fr[1][keep]
                        fr[2] = fr[2][keep]
                    kept.append(fr)
                self._frames = kept

        self._frames.append([t, self._apply(M, pts), az])

        world_all = (self._frames[0][1] if len(self._frames) == 1
                     else np.concatenate([f[1] for f in self._frames], axis=0))
        sweep = self._apply(np.linalg.inv(M), world_all)

        self.frames_in_sweep = len(self._frames)
        self.span_s = t - self._frames[0][0]
        self.points_in_sweep = int(sweep.shape[0])
        return sweep


def azimuth_coverage_bins(points, bin_deg: float = 10.0) -> int:
    """How many `bin_deg` sectors of the circle contain at least one point.
    36 (for 10-degree bins) means a full sweep. Diagnostics only."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return 0
    az = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    n_bins = max(1, int(round(360.0 / bin_deg)))
    idx = np.floor((az + 180.0) / bin_deg).astype(int) % n_bins     # +180 exactly wraps to bin 0
    return int(len(np.unique(idx)))
