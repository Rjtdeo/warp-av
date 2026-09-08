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
        theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy) if len(pts) > 1 else 0.0
        axis_deg = abs(math.degrees(theta))
        # the blob's footprint: how long it is along that axis and how wide across it.
        # The LiDAR only ever sees the near face of a thing, so both are lower bounds.
        ct, st = math.cos(theta), math.sin(theta)
        along = [(p[0] - mx) * ct + (p[1] - my) * st for p in pts]
        across = [-(p[0] - mx) * st + (p[1] - my) * ct for p in pts]
        length_m = max(along) - min(along)
        width_m = max(across) - min(across)
        if width_m > length_m:                       # keep 'length' the longer side
            length_m, width_m = width_m, length_m
            theta += math.pi / 2
        c = {"x": mx, "y": my,
             "distance": math.hypot(mx, my),
             "n": len(pts), "extent": extent,
             "height": height, "length": 2.0 * extent,
             "axis_deg": axis_deg,
             "length_m": length_m, "width_m": width_m,
             "yaw_deg": math.degrees(math.atan2(math.sin(theta), math.cos(theta))),
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

# ---- how much the sighting of a thing wobbles, and how hard a thing can accelerate ----
# The LiDAR sees a different part of an object on every turn, so the middle of the blob
# jitters even when nothing moves; that jitter grows with range. Believing it is what made
# a parked bin look like it was doing 6.7 m/s (Perception V2 day 6).
MEAS_NOISE_BASE_M = 0.25            # wobble of a near blob's middle
MEAS_NOISE_PER_M = 0.02             # ... plus this much per metre of range
ACCEL_NOISE_MPS2 = 2.5              # how briskly a road user may change speed: normal braking.
                                    # Too small and the filter is so sure of itself that a car
                                    # which stops takes 1.5 s to be believed; too large and
                                    # parked things start to jitter again.
STILL_SPEED_MPS = 0.6               # under this, and going nowhere, it is standing still
MOVING_SPEED_MPS = 1.2              # over this it is moving again (a gap, so it cannot flicker)
STILL_WINDOW_S = 1.5                # how far back to look when asking "has it gone anywhere?"
STILL_TRAVEL_M = 0.7                # ... and how far it must have gone to count as moving
STILL_STRAIGHTNESS = 0.5            # ... and that travel must be in one direction, not a shuffle
STILL_MIN_SIGHTINGS = 4             # ... over at least this many sightings


class Track:
    """One thing the van is following, with a constant-velocity motion filter.

    The filter holds where the thing is and how fast it is going, and how sure it is of
    each. A sighting nudges the estimate rather than replacing it, by an amount that
    depends on how noisy that sighting is: a blob 30 m away barely moves the answer, a
    solid one at 8 m moves it a lot. Standing still therefore looks like standing still.
    """

    __slots__ = ("tid", "x", "P", "cls", "confidence",
                 "last_seen", "hits", "strong_hits",
                 "length_m", "width_m", "height_m", "yaw_deg",
                 "_history", "_still")

    def __init__(self, tid, wx, wy, t):
        self.tid = tid
        self.x = [float(wx), float(wy), 0.0, 0.0]          # position and velocity, world frame
        # sure about where it is, unsure how fast it is going
        self.P = [[0.5, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 4.0, 0.0],
                  [0.0, 0.0, 0.0, 4.0]]
        self.cls = None            # 'vehicle' | 'pedestrian' | None (unknown)
        self.confidence = 0.5
        self.last_seen = t
        self.hits = 1              # strong sightings count 1, weak ones WEAK_HIT
        self.strong_hits = 1       # sightings with a full-strength blob (0 for a weak-only track)
        # the footprint the van has seen so far: the biggest sighting of each side, since
        # the LiDAR only ever catches part of a thing and a bigger view is the truer one
        self.length_m = 0.0
        self.width_m = 0.0
        self.height_m = 0.0
        self.yaw_deg = 0.0         # heading of the long side, degrees, van frame at the sighting
        self._history = [(t, float(wx), float(wy))]        # where it has been lately
        self._still = True         # a thing is taken to be parked until it shows otherwise

    # ---- what the rest of the stack reads -------------------------------------------
    @property
    def wx(self):
        return self.x[0]

    @property
    def wy(self):
        return self.x[1]

    @property
    def vx(self):
        return self.x[2]

    @property
    def vy(self):
        return self.x[3]

    @property
    def weak_only(self) -> bool:
        return self.strong_hits == 0

    @property
    def speed(self):
        return math.hypot(self.x[2], self.x[3])

    @property
    def stationary(self) -> bool:
        """True while the thing is parked: slow, and it has not gone anywhere."""
        return self._still

    @property
    def travelled_m(self) -> float:
        """How far it has actually moved over the last second and a half."""
        if len(self._history) < 2:
            return 0.0
        _, x0, y0 = self._history[0]
        _, x1, y1 = self._history[-1]
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def straightness(self) -> float:
        """Of all the wandering it did, how much got it somewhere: 1 = a straight line,
        near 0 = a shuffle on the spot. A blob whose middle jumps between two parts of the
        same object shuffles; a car going past does not."""
        if len(self._history) < 3:
            return 1.0
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                   for a, b in zip(self._history, self._history[1:]))
        return self.travelled_m / path if path > 1e-6 else 1.0

    # ---- the motion filter ----------------------------------------------------------
    def predict(self, dt: float) -> None:
        """Carry the estimate forward by dt seconds, growing the uncertainty."""
        if dt <= 0.0:
            return
        x, y, vx, vy = self.x
        self.x = [x + vx * dt, y + vy * dt, vx, vy]
        p = self.P
        # P = F P F' + Q, written out for the 2 x (position, velocity) blocks
        for i, j in ((0, 2), (1, 3)):
            pii, pij, pjj = p[i][i], p[i][j], p[j][j]
            p[i][i] = pii + 2.0 * dt * pij + dt * dt * pjj
            p[i][j] = p[j][i] = pij + dt * pjj
            p[j][j] = pjj
        q = ACCEL_NOISE_MPS2 ** 2
        for i, j in ((0, 2), (1, 3)):
            p[i][i] += q * dt ** 4 / 4.0
            p[i][j] += q * dt ** 3 / 2.0
            p[j][i] = p[i][j]
            p[j][j] += q * dt * dt

    def correct(self, zx: float, zy: float, sigma_m: float, t: float) -> None:
        """Fold in one sighting, trusting it as much as `sigma_m` says."""
        r = max(0.05, sigma_m) ** 2
        if len(self._history) == 1:
            # second sighting: take the speed straight from the two positions. Waiting for
            # the filter to work it out lets its guess fall behind a fast car, and the track
            # then breaks and starts again every frame (day 6).
            dt = t - self._history[0][0]
            if dt > 1e-3:
                self.x[2] = (zx - self._history[0][1]) / dt
                self.x[3] = (zy - self._history[0][2]) / dt
                spread = 2.0 * r / (dt * dt)
                self.P[2][2] = self.P[3][3] = spread
        for axis, (i, j) in enumerate(((0, 2), (1, 3))):
            z = zx if axis == 0 else zy
            p = self.P
            innovation = z - self.x[i]
            s = p[i][i] + r
            k_pos = p[i][i] / s
            k_vel = p[j][i] / s
            self.x[i] += k_pos * innovation
            self.x[j] += k_vel * innovation
            pii, pij, pjj = p[i][i], p[i][j], p[j][j]
            p[i][i] = pii - k_pos * pii
            p[i][j] = p[j][i] = pij - k_pos * pij
            p[j][j] = pjj - k_vel * pij
        self.last_seen = t
        self._history.append((t, self.x[0], self.x[1]))
        while len(self._history) > 2 and t - self._history[0][0] > STILL_WINDOW_S:
            self._history.pop(0)
        self._update_still()

    def _update_still(self) -> None:
        """Parked or moving, decided over time and with a gap, so it cannot flicker."""
        if len(self._history) < STILL_MIN_SIGHTINGS:
            return          # too new to accuse of moving: two jittery sightings prove nothing
        speed, travelled = self.speed, self.travelled_m
        if self._still:
            if (speed > MOVING_SPEED_MPS and travelled > STILL_TRAVEL_M
                    and self.straightness > STILL_STRAIGHTNESS):
                self._still = False
        else:
            if speed < STILL_SPEED_MPS and travelled < STILL_TRAVEL_M:
                self._still = True


def measurement_noise_m(o: dict) -> float:
    """How much to distrust one sighting: mostly its range, since a far blob's middle
    jumps about as different parts of the thing come back."""
    d = o.get("distance")
    if d is None:
        d = math.hypot(o.get("wx", 0.0), o.get("wy", 0.0))
    sigma = MEAS_NOISE_BASE_M + MEAS_NOISE_PER_M * float(d)
    if o.get("weak"):
        sigma *= 2.0               # a two-point far blob is barely a measurement
    return sigma


def _grow_size(tr: Track, o: dict) -> None:
    """Keep the largest footprint seen so far, and the heading that came with it."""
    lm, wm = float(o.get("length_m", 0.0) or 0.0), float(o.get("width_m", 0.0) or 0.0)
    if lm > tr.length_m:
        tr.length_m = lm
        tr.yaw_deg = float(o.get("yaw_deg", 0.0) or 0.0)
    tr.width_m = max(tr.width_m, wm)
    tr.height_m = max(tr.height_m, float(o.get("height_m", 0.0) or 0.0))


class ObjectTracker:
    """Nearest-neighbour tracker with EMA velocities in world frame.

    update(observations, t) with observations =
        [{wx, wy, cls (opt), confidence (opt)}, ...]
    returns the live tracks (confirmed = seen at least `min_hits` times).
    """

    GATE_M = 2.0            # how far a sighting may sit from where a track was predicted...
    GATE_SPEED_MPS = 4.0    # ... plus a step's worth of travel, for at least this speed, so a
                            #     fast car is still caught on its second sighting while two
                            #     parked things 2.4 m apart never swap tracks (day 6)
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
        for o in observations:
            if "distance" not in o:
                o["distance"] = None
        # Greedy nearest-neighbour association (fine at these densities).
        unmatched = list(range(len(observations)))
        pairs = []
        for tr in self._tracks:
            dt = max(0.0, t - tr.last_seen)
            best_j, best_d = None, self.GATE_M + dt * max(self.GATE_SPEED_MPS, tr.speed)
            px = tr.wx + tr.vx * dt
            py = tr.wy + tr.vy * dt
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
            dt = max(0.0, t - tr.last_seen)
            # carry the estimate forward, then let the sighting nudge it as much as its
            # own noise deserves: no more turning blob jitter into speed (day 6)
            tr.predict(dt)
            tr.correct(o["wx"], o["wy"], measurement_noise_m(o), t)
            if o.get("weak"):
                tr.hits += self.WEAK_HIT
            else:
                tr.hits += 1
                tr.strong_hits += 1
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = max(tr.confidence, o.get("confidence", 0.5))
            _grow_size(tr, o)

        for j in unmatched:
            o = observations[j]
            tr = Track(self._next_id, o["wx"], o["wy"], t)
            if o.get("weak"):
                tr.hits = self.WEAK_HIT
                tr.strong_hits = 0
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.confidence = o.get("confidence", 0.5)
            _grow_size(tr, o)
            self._next_id += 1
            self._tracks.append(tr)

        return [tr for tr in self._tracks if tr.hits >= self.MIN_HITS - 1e-6]

    def reported_speed(self, tr: Track) -> float:
        """A thing the van has decided is parked reports exactly zero, not a wobble."""
        if getattr(tr, "stationary", False):
            return 0.0
        s = tr.speed
        return 0.0 if s < self.SPEED_DEADBAND else s
