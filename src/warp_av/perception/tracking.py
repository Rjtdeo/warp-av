"""
LiDAR clustering + multi-object tracking for camera/LiDAR perception.

Pure math on purpose: no CARLA, no OpenCV, no torch — the same code runs in
offline tests and on the vehicle. Frames:

- Cluster input: sensor-frame points (x forward, y right — CARLA LiDAR).
- Tracking runs in WORLD frame (fed the vehicle pose from localization), so
  a stationary car stays stationary while we drive past it. Velocities are
  estimated per track and smoothed; that is what car-following, cut-in
  handling, and the moving/parked distinction feed on.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------

def cluster_points(points, cell=1.0, min_points=3, max_range=55.0,
                   ego_half_len=3.2, ego_half_wid=1.3, max_clusters=60):
    """Group 2D sensor-frame points into object clusters.

    `points` is an iterable of (x, y) — pre-filtered for height by the
    caller. Grid-hash + 8-neighbour flood fill: fast, dependency-free,
    good enough for van-sized objects at 10 Hz.

    Returns clusters sorted by range: [{x, y, distance, n, extent}, ...]
    """
    cells: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for x, y in points:
        if abs(x) < ego_half_len and abs(y) < ego_half_wid:
            continue                      # our own body / mount returns
        if x * x + y * y > max_range * max_range:
            continue
        cells.setdefault((int(math.floor(x / cell)),
                          int(math.floor(y / cell))), []).append((x, y))

    seen = set()
    clusters = []
    for start in cells:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        pts: List[Tuple[float, float]] = []
        while stack:
            c = stack.pop()
            pts.extend(cells[c])
            cx, cy = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (cx + dx, cy + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(pts) < min_points:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        extent = max(math.hypot(p[0] - mx, p[1] - my) for p in pts)
        clusters.append({"x": mx, "y": my,
                         "distance": math.hypot(mx, my),
                         "n": len(pts), "extent": extent})
    clusters.sort(key=lambda c: c["distance"])
    return clusters[:max_clusters]


# ----------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------

class Track:
    __slots__ = ("tid", "wx", "wy", "vx", "vy", "cls", "confidence",
                 "last_seen", "hits")

    def __init__(self, tid, wx, wy, t):
        self.tid = tid
        self.wx = wx
        self.wy = wy
        self.vx = 0.0
        self.vy = 0.0
        self.cls = None            # 'vehicle' | 'pedestrian' | None (unknown)
        self.confidence = 0.5
        self.last_seen = t
        self.hits = 1

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


class ObjectTracker:
    """Nearest-neighbour tracker with EMA velocities in world frame.

    update(observations, t) with observations =
        [{wx, wy, cls (opt), confidence (opt)}, ...]
    returns the live tracks (confirmed = seen at least `min_hits` times).
    """

    GATE_M = 2.6            # max association distance per step
    DROP_AFTER_S = 1.2      # unseen this long -> forget
    VEL_ALPHA = 0.35        # velocity smoothing
    SPEED_DEADBAND = 0.4    # below this, report standing still
    MIN_HITS = 2            # confirmations before a track is trusted

    def __init__(self):
        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, observations, t):
        # Greedy nearest-neighbour association (fine at these densities).
        unmatched = list(range(len(observations)))
        pairs = []
        for tr in self._tracks:
            best_j, best_d = None, self.GATE_M
            px = tr.wx + tr.vx * max(0.0, t - tr.last_seen)
            py = tr.wy + tr.vy * max(0.0, t - tr.last_seen)
            for j in unmatched:
                o = observations[j]
                d = math.hypot(o["wx"] - px, o["wy"] - py)
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is not None:
                pairs.append((tr, best_j))
                unmatched.remove(best_j)

        for tr, j in pairs:
            o = observations[j]
            dt = max(1e-3, t - tr.last_seen)
            ivx = (o["wx"] - tr.wx) / dt
            ivy = (o["wy"] - tr.wy) / dt
            # Reject teleport-grade velocity (association glitch)
            if math.hypot(ivx, ivy) < 30.0:
                tr.vx = (1 - self.VEL_ALPHA) * tr.vx + self.VEL_ALPHA * ivx
                tr.vy = (1 - self.VEL_ALPHA) * tr.vy + self.VEL_ALPHA * ivy
            tr.wx, tr.wy = o["wx"], o["wy"]
            tr.last_seen = t
            tr.hits += 1
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = max(tr.confidence, o.get("confidence", 0.5))

        for j in unmatched:
            o = observations[j]
            tr = Track(self._next_id, o["wx"], o["wy"], t)
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = o.get("confidence", 0.5)
            self._next_id += 1
            self._tracks.append(tr)

        self._tracks = [tr for tr in self._tracks
                        if t - tr.last_seen <= self.DROP_AFTER_S]
        return [tr for tr in self._tracks if tr.hits >= self.MIN_HITS]

    def reported_speed(self, tr: Track) -> float:
        s = tr.speed
        return 0.0 if s < self.SPEED_DEADBAND else s
