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

FAR_RANGE_M = 15.0          # beyond this a blob may count with fewer points (perception fix 3) ...
MIN_POINTS_FAR = 2          # ... this many; the tracker then wants three sightings before it believes it
CORRIDOR_HALF_WIDTH_M = 3.0 # blobs ahead within this of the van's axis are kept before any other when capping
DEFAULT_MAX_CLUSTERS = 120  # was 60: with every point kept (fix 3) a street holds 50-100 blobs
LAST_CLUSTER_TOTAL = 0      # how many blobs the last call found before the cap (telemetry)
VEHICLE_MIN_EXTENT_M = 0.9  # the 'car-sized blob' shape rule: wider than this ...
VEHICLE_MIN_POINTS = 12     # ... with this many raw points (was 6 after 3x thinning = ~18 raw) ...
VEHICLE_MIN_HEIGHT_M = 0.5  # ... and taller than this (a planter or a bench is not a car)


def vehicle_shaped(cluster, min_points=VEHICLE_MIN_POINTS, min_extent=VEHICLE_MIN_EXTENT_M,
                   min_height=VEHICLE_MIN_HEIGHT_M):
    """Does a LiDAR blob look like a car on its own (no camera needed)?"""
    if cluster.get("extent", 0.0) < min_extent or cluster.get("n", 0) < min_points:
        return False
    h = cluster.get("height")
    return h is None or h >= min_height


def cluster_points(points, cell=1.0, min_points=3, max_range=55.0,
                   ego_half_len=3.2, ego_half_wid=1.3, max_clusters=DEFAULT_MAX_CLUSTERS, heights=None,
                   return_members=False, min_points_far=None, far_range_m=FAR_RANGE_M):
    """Group 2D sensor-frame points into object clusters.

    `points` is an iterable of (x, y) — pre-filtered for height by the
    caller. Grid-hash + 8-neighbour flood fill: fast, dependency-free,
    good enough for van-sized objects at 10 Hz.

    `heights` (optional): one value per point, the point's height above the
    local road (perception fix 2). When given, every cluster also reports
    'height' (its tallest point) and 'length' (2 x extent), which the
    road-edge rule reads. `return_members=True` adds 'members', the indices
    (into `points`) of the points in each cluster.

    `min_points_far` (perception fix 3): beyond `far_range_m` a blob needs
    only this many points (a small object 15-20 m out gives 2-3 returns).
    Such blobs are reported with 'weak': True so the tracker can ask for
    more sightings before it believes them. None = same minimum everywhere.

    Returns clusters sorted by range: [{x, y, distance, n, extent, height, length, weak}, ...]
    """
    cells: Dict[Tuple[int, int], List[Tuple[float, float, float]]] = {}
    hs = None if heights is None else list(heights)
    for i, (x, y) in enumerate(points):
        if abs(x) < ego_half_len and abs(y) < ego_half_wid:
            continue                      # our own body / mount returns
        if x * x + y * y > max_range * max_range:
            continue
        h = hs[i] if hs is not None else None
        cells.setdefault((int(math.floor(x / cell)),
                          int(math.floor(y / cell))), []).append((x, y, h, i))

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
        need = min_points
        if min_points_far is not None and len(pts) < min_points:
            # a small blob: allowed only far out, with the lower minimum
            fx = sum(p[0] for p in pts) / len(pts)
            fy = sum(p[1] for p in pts) / len(pts)
            need = min_points_far if math.hypot(fx, fy) >= far_range_m else min_points
        if len(pts) < need:
            continue
        mx = sum(p[0] for p in pts) / len(pts)
        my = sum(p[1] for p in pts) / len(pts)
        extent = max(math.hypot(p[0] - mx, p[1] - my) for p in pts)
        height = None
        if hs is not None:
            hv = [p[2] for p in pts if p[2] is not None and p[2] == p[2]]
            height = max(hv) if hv else None
        # direction of the blob's long axis, 0 = along the van's forward axis,
        # 90 = across it (from the 2x2 covariance of its points)
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
        axis_deg = abs(math.degrees(0.5 * math.atan2(2.0 * sxy, sxx - syy))) if len(pts) > 1 else 0.0
        c = {"x": mx, "y": my,
             "distance": math.hypot(mx, my),
             "n": len(pts), "extent": extent,
             "height": height, "length": 2.0 * extent,
             "axis_deg": axis_deg,
             "weak": len(pts) < min_points}
        if return_members:
            c["members"] = [p[3] for p in pts]
        clusters.append(c)
    global LAST_CLUSTER_TOTAL
    LAST_CLUSTER_TOTAL = len(clusters)
    if len(clusters) > max_clusters:
        # keep what matters first: blobs ahead in the van's corridor, then
        # strong before weak, then the nearest - so roadside clutter cannot
        # push a real obstacle 25 m ahead out of the list
        clusters.sort(key=lambda c: (not (c["x"] > 0 and abs(c["y"]) < CORRIDOR_HALF_WIDTH_M),
                                     bool(c.get("weak")), c["distance"]))
        clusters = clusters[:max_clusters]
    clusters.sort(key=lambda c: c["distance"])
    return clusters


# ----------------------------------------------------------------------
# Tracking
# ----------------------------------------------------------------------

class Track:
    __slots__ = ("tid", "wx", "wy", "vx", "vy", "cls", "confidence",
                 "last_seen", "hits", "strong_hits")

    def __init__(self, tid, wx, wy, t):
        self.tid = tid
        self.wx = wx
        self.wy = wy
        self.vx = 0.0
        self.vy = 0.0
        self.cls = None            # 'vehicle' | 'pedestrian' | None (unknown)
        self.confidence = 0.5
        self.last_seen = t
        self.hits = 1              # strong sightings count 1, weak ones WEAK_HIT
        self.strong_hits = 1       # sightings with a full-strength blob (0 for a weak-only track)

    @property
    def weak_only(self) -> bool:
        return self.strong_hits == 0

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
    WEAK_HIT = 0.67         # a weak (2-point, far) sighting counts this much: three in a row confirm

    def __init__(self):
        self._tracks: List[Track] = []
        self._next_id = 1

    def update(self, observations, t):
        # forget first, then associate: a track unseen for DROP_AFTER_S is
        # gone before this frame's sightings can revive it ("in a row" holds)
        self._tracks = [tr for tr in self._tracks if t - tr.last_seen <= self.DROP_AFTER_S]
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
            if o.get("weak"):
                tr.hits += self.WEAK_HIT
            else:
                tr.hits += 1
                tr.strong_hits += 1
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = max(tr.confidence, o.get("confidence", 0.5))

        for j in unmatched:
            o = observations[j]
            tr = Track(self._next_id, o["wx"], o["wy"], t)
            if o.get("weak"):
                tr.hits = self.WEAK_HIT
                tr.strong_hits = 0
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = o.get("confidence", 0.5)
            self._next_id += 1
            self._tracks.append(tr)

        return [tr for tr in self._tracks if tr.hits >= self.MIN_HITS - 1e-6]

    def reported_speed(self, tr: Track) -> float:
        s = tr.speed
        return 0.0 if s < self.SPEED_DEADBAND else s
