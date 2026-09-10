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
# A car is bounded. The rule above is not: a building wall is wider than 0.9 m, has more than
# twelve points and is taller than half a metre, so it passed. Measured on the recordings,
# 99 % of the things the van called a vehicle were buildings, walls, pavements and poles.
# Since day 4 the van measures each blob's length, width and height, so the rule can simply
# ask whether the thing would fit in a car park (Perception V2, day 10 follow-up).
VEHICLE_MAX_LENGTH_M = 14.0   # a bus or an articulated lorry; a building face is longer
VEHICLE_MIN_WIDTH_M = 0.9     # a pole, a post or a fence is thinner
VEHICLE_MAX_WIDTH_M = 4.0
VEHICLE_MAX_HEIGHT_M = 3.0    # a car, a van or a small lorry. Measured on
                              # the recordings: the things wrongly called vehicles stand 3.8 to
                              # 4.0 m tall (buildings and walls) and the real car 1.5 m. A CARLA
                              # box lorry measures 2.7 m, so 2.6 turned it into an obstacle;
                              # allowing 3.0 costs four extra wrong labels and 3.4 costs
                              # seventy-three, which is where the buildings start.
WIDTH_TEST_FROM_M = 5.0       # only things this long are judged on being too thin
WIDTH_TEST_WITHIN_M = 25.0    # ... and near enough that the measurement means something
VEHICLE_MIN_LENGTH_M = 2.0    # shorter than this, near the van, is a bin or a post
VEHICLE_MIN_WIDTH_FAR_M = 0.4  # ... and thinner than this, at any distance, is a rail


def vehicle_shaped(cluster, min_points=None, min_extent=None, min_height=None,
                   max_length=None, min_width=None, max_width=None, max_height=None):
    """Does a LiDAR blob look like a car on its own, with no camera to ask?

    Being big enough is not the question; being car-shaped is. A car fits inside bounds in
    all three directions, and a building, a wall, a hedge or a pole falls outside at least
    one of them. The measured footprint is used when there is one, and a blob with no
    measurement is judged on the old test alone, since that is all there is.
    """
    # the limits are read here, not bound when this file loads, so they can be measured
    min_points = VEHICLE_MIN_POINTS if min_points is None else min_points
    min_extent = VEHICLE_MIN_EXTENT_M if min_extent is None else min_extent
    min_height = VEHICLE_MIN_HEIGHT_M if min_height is None else min_height
    max_length = VEHICLE_MAX_LENGTH_M if max_length is None else max_length
    min_width = VEHICLE_MIN_WIDTH_M if min_width is None else min_width
    max_width = VEHICLE_MAX_WIDTH_M if max_width is None else max_width
    max_height = VEHICLE_MAX_HEIGHT_M if max_height is None else max_height

    if cluster.get("extent", 0.0) < min_extent or cluster.get("n", 0) < min_points:
        return False
    h = cluster.get("height")
    if h is not None and (h < min_height or h > max_height):
        return False                    # a building, a hedge or a tree, not something driving
    length = cluster.get("length_m")
    width = cluster.get("width_m")
    if length is not None and length > max_length:
        return False                    # a wall or a building face
    if width is not None and width > max_width:
        return False                    # too broad for anything that drives
    # A car has body on both sides. A pole, a fence and a kerb do not. Close up the van can
    # see enough of a car to insist on a proper width; far off it only sees the near face,
    # which reads thinner, so the bar drops. It never drops to nothing: a rail 20 cm wide is
    # not a car at any distance (found live at 37.8 m).
    if length is not None and width is not None:
        near = cluster.get("distance", 0.0) <= WIDTH_TEST_WITHIN_M
        floor = min_width if near else VEHICLE_MIN_WIDTH_FAR_M
        if width < floor:
            return False
        if near and length < VEHICLE_MIN_LENGTH_M:
            return False
    if (length is not None and width is not None
            and length >= WIDTH_TEST_FROM_M and width < min_width):
        return False                    # long and thin: a kerb, a fence, a hedge row
    return True


# ---- putting one thing back together ------------------------------------------------
# The grid that groups points is a fixed size, and the laser's points are not: they spread
# apart with distance. So a car close to the van can come back as two blobs, its back and
# its side, with a gap between them wider than the grid. Seen live: a car 11 m ahead was
# reported twice.
#
# Two blobs are put back together when the space between them is small for their range,
# they stand at the same height, and what they make together is still a plausible size.
# Those guards matter: the whole of day 4 was about stopping a barrel merging with the
# planter beside it, and that pair differs by 0.6 m in height.
MERGE_GAP_BASE_M = 0.25
MERGE_GAP_PER_M = 0.02        # ... plus this much per metre of range
MERGE_HEIGHT_TOLERANCE_M = 0.25
MERGE_MAX_LENGTH_M = 5.5      # a car or a van; wider than this and it is two things.
                              # Measured: looser guards than these let a barrel join
                              # something beside it and be called a vehicle.


def merge_split_clusters(clusters, gap_base=None, gap_per_m=None,
                         height_tol=None, max_length=None):
    """Join blobs that are plainly two views of one thing. Returns a new list."""
    # read at call time, not bound when this file loads, so the settings can be measured
    gap_base = MERGE_GAP_BASE_M if gap_base is None else gap_base
    gap_per_m = MERGE_GAP_PER_M if gap_per_m is None else gap_per_m
    height_tol = MERGE_HEIGHT_TOLERANCE_M if height_tol is None else height_tol
    max_length = MERGE_MAX_LENGTH_M if max_length is None else max_length
    out = [dict(c) for c in clusters]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                hi, hj = a.get("height"), b.get("height")
                if hi is not None and hj is not None and abs(hi - hj) > height_tol:
                    continue
                centre = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                gap = centre - a.get("extent", 0.0) - b.get("extent", 0.0)
                allowed = gap_base + gap_per_m * min(a["distance"], b["distance"])
                if gap > allowed:
                    continue
                span = centre + a.get("extent", 0.0) + b.get("extent", 0.0)
                if span > max_length:
                    continue                       # together they would be too big to be one thing
                n = a["n"] + b["n"]
                merged = {
                    "x": (a["x"] * a["n"] + b["x"] * b["n"]) / n,
                    "y": (a["y"] * a["n"] + b["y"] * b["n"]) / n,
                    "n": n,
                    "extent": span / 2.0,
                    "height": max(h for h in (hi, hj) if h is not None) if (hi is not None or hj is not None) else None,
                    "length": span,
                    "length_m": max(a.get("length_m", 0.0), b.get("length_m", 0.0), span * 0.9),
                    "width_m": max(a.get("width_m", 0.0), b.get("width_m", 0.0)),
                    "yaw_deg": a.get("yaw_deg", 0.0) if a["n"] >= b["n"] else b.get("yaw_deg", 0.0),
                    "axis_deg": a.get("axis_deg", 0.0) if a["n"] >= b["n"] else b.get("axis_deg", 0.0),
                    "weak": bool(a.get("weak") and b.get("weak")),
                }
                merged["distance"] = math.hypot(merged["x"], merged["y"])
                if "members" in a and "members" in b:
                    merged["members"] = list(a["members"]) + list(b["members"])
                out[i] = merged
                out.pop(j)
                changed = True
                break
            if changed:
                break
    out.sort(key=lambda c: c["distance"])
    return out


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
STILL_STRAIGHTNESS = 0.92           # ... and that travel must be nearly a straight line. A car
                                    #   rounding a bend still scores about 0.96; a blob whose
                                    #   middle shuffles about scores far less.
STILL_MIN_SIGHTINGS = 4             # ... over at least this many sightings
STILL_TRAVEL_PER_M = 0.15           # far blobs wander more, so ask them to travel further
STILL_WINDOW_PER_M = 0.0            # (a longer look at far things was measured and made near
                                    #  ones worse, so the window stays fixed)
STILL_WINDOW_MAX_S = 4.0
STILL_TRAVEL_MAX_M = 2.0            # ... but never ask for more than a walking person covers in
                                    #   the window, or a pedestrian at 25 m would never count
MAX_ROAD_SPEED_MPS = 30.0           # nothing in a town does 108 km/h: above this it is a
                                    #   mis-association, not a measurement
MAX_ROAD_USER_LENGTH_M = 8.0        # longer than a bus: scenery, and scenery does not move

# ---- how big is it? (the day-7 size rule) ----------------------------------------------
# Keeping the largest view a track ever had is right for a car, which reveals more of itself
# as you approach, and wrong the moment one frame glues an object to a wall: the inflated
# size then sticks for the life of the track. A person read 8.6 m long that way. So keep the
# middle of the recent sightings instead: one bad frame is outvoted, while a size several
# frames agree on still comes through, which is how a person pushing a pram gets their room.
SIZE_HISTORY = 12                   # sightings kept per track
SIZE_MIN_FOR_MEDIAN = 3             # below this, take the largest, since there is nothing to vote
# Generous bounds, used ONLY to throw away an impossible sighting, never to rewrite a real one.
# A person can be 2 m tall with a bag and a bike; a person cannot be 8 m long.
CLASS_SIZE_LIMITS = {
    # long side, short side, tall. Roomy on purpose: a person wheeling a bicycle is about
    # 2 m long, one pushing a pram about 1.5 m, and a group walking abreast is wider than
    # one person. Only a merge with a wall gets past these.
    "pedestrian": (2.5, 1.8, 2.4),
    "cyclist": (2.8, 1.8, 2.4),            # rider plus bicycle, or a motorbike
    "vehicle": (14.0, 4.0, 4.5),           # a bus or a lorry is still a vehicle
}
# When do the sightings disagree enough to admit the van does not know the size? Not as a
# share of the size: any natural swing is a big share of a person, so a ratio test calls
# every pedestrian unsure and the warning becomes noise. It is an absolute swing, and it
# grows with range because the LiDAR's points spread out. Measured on the four recordings:
# a walker at 6 m swings 0.05-0.17 m, a barrel at 9 m 0.15 m, a bin at 12 m 0.27 m, a car at
# 22 m 0.32 m, while a blob that merges with its neighbour swings 1.1-2.7 m.
SIZE_SPREAD_BASE_M = 0.15
SIZE_SPREAD_PER_M = 0.02
# What the van should leave room for, whatever it measured. People change direction without
# warning, so their margin is generous however small they look.
MIN_CLEARANCE_M = {"pedestrian": 0.6, "cyclist": 0.8, "vehicle": 1.2}
DEFAULT_MIN_CLEARANCE_M = 0.4


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
                 "_history", "_still", "range_m", "_sizes", "size_uncertain",
                 "cls_source", "_unnamed")

    def __init__(self, tid, wx, wy, t):
        self.tid = tid
        self.x = [float(wx), float(wy), 0.0, 0.0]          # position and velocity, world frame
        # sure about where it is, unsure how fast it is going
        self.P = [[0.5, 0.0, 0.0, 0.0],
                  [0.0, 0.5, 0.0, 0.0],
                  [0.0, 0.0, 4.0, 0.0],
                  [0.0, 0.0, 0.0, 4.0]]
        self.cls = None            # 'vehicle' | 'pedestrian' | 'cyclist' | None (unknown)
        # where that name came from, and how many sightings in a row have offered none.
        # A name used to be set once and kept for ever, so a blob that looked car-shaped in
        # a single frame stayed a "vehicle" for the life of the track, however tall it grew
        # (found live, Perception V2 day 10 follow-up).
        self.cls_source = None
        self._unnamed = 0
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
        self.range_m = 0.0         # how far away it was last seen, for judging its wobble
        self._sizes = []           # the recent size sightings, to take a middle value from
        self.size_uncertain = False   # True when those sightings disagree badly

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
    def window_s(self) -> float:
        """How far back to look. Further away means a longer look."""
        return min(STILL_WINDOW_MAX_S, STILL_WINDOW_S + STILL_WINDOW_PER_M * self.range_m)

    @property
    def history_span_s(self) -> float:
        """How long the kept history actually covers."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1][0] - self._history[0][0]

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
                vx = (zx - self._history[0][1]) / dt
                vy = (zy - self._history[0][2]) / dt
                if math.hypot(vx, vy) <= MAX_ROAD_SPEED_MPS:
                    self.x[2], self.x[3] = vx, vy
                    self.P[2][2] = self.P[3][3] = 2.0 * r / (dt * dt)
                # a jump faster than any road user is two different things being confused,
                # not a measurement: start from standing still instead
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
        speed = math.hypot(self.x[2], self.x[3])
        if speed > MAX_ROAD_SPEED_MPS:            # keep the estimate inside the possible
            scale = MAX_ROAD_SPEED_MPS / speed
            self.x[2] *= scale
            self.x[3] *= scale
        self.last_seen = t
        self._history.append((t, self.x[0], self.x[1]))
        while len(self._history) > 2 and t - self._history[0][0] > self.window_s:
            self._history.pop(0)
        self._update_still()

    def _update_still(self) -> None:
        """Parked or moving, decided over time and with a gap, so it cannot flicker."""
        if len(self._history) < STILL_MIN_SIGHTINGS:
            return          # too new to accuse of moving: two jittery sightings prove nothing
        speed, travelled = self.speed, self.travelled_m
        # how far this thing must travel before the van believes it: further away, and the
        # middle of its blob wanders more, so ask for more
        need = min(STILL_TRAVEL_MAX_M, max(STILL_TRAVEL_M, STILL_TRAVEL_PER_M * self.range_m))
        if self._still:
            if self.length_m > MAX_ROAD_USER_LENGTH_M:
                return                     # a 20 m blob is a wall or a hedge, not a road user
            if (speed > MOVING_SPEED_MPS and travelled > need
                    and self.straightness > STILL_STRAIGHTNESS):
                self._still = False
        else:
            if speed < STILL_SPEED_MPS and travelled < need:
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


#: how many sightings in a row may offer no name before an old one is dropped. A name that
#: came from the camera is kept longer, because a detector missing one frame is ordinary; a
#: name that came from the blob's shape is dropped quickly, because the shape is measured
#: fresh every time and its silence means the thing does not look like that any more.
FORGET_NAME_AFTER = {"shape": 3, "camera": 15}


def _forget_name(tr: "Track") -> None:
    if tr.cls is None:
        return
    tr._unnamed += 1
    if tr._unnamed >= FORGET_NAME_AFTER.get(tr.cls_source or "shape", 3):
        tr.cls = None
        tr.cls_source = None
        tr.confidence = 0.5
        tr._unnamed = 0


def _median(values):
    vs = sorted(values)
    n = len(vs)
    if not n:
        return 0.0
    return vs[n // 2] if n % 2 else 0.5 * (vs[n // 2 - 1] + vs[n // 2])


def plausible_size(cls, length_m: float, width_m: float, height_m: float) -> bool:
    """Could a thing of this kind really be this big? Generous on purpose: the limits are
    here to throw away a blob merged with a wall, not to decide what size a person is."""
    limit = CLASS_SIZE_LIMITS.get(cls)
    if limit is None:
        return True
    return length_m <= limit[0] and width_m <= limit[1] and height_m <= limit[2]


def _note_size(tr: Track, o: dict) -> None:
    """Record this sighting's footprint and report the middle of the recent ones."""
    lm = float(o.get("length_m", 0.0) or 0.0)
    wm = float(o.get("width_m", 0.0) or 0.0)
    hm = float(o.get("height_m", 0.0) or 0.0)
    if lm <= 0.0 and wm <= 0.0 and hm <= 0.0:
        return                                  # nothing measured this time
    if not plausible_size(tr.cls, lm, wm, hm):
        # a person is not 8 m long: that frame is a merge with something else. Throw the
        # measurement away, but admit that the van is now unsure rather than saying nothing.
        tr.size_uncertain = True
        return
    tr._sizes.append((lm, wm, hm, float(o.get("yaw_deg", 0.0) or 0.0)))
    if len(tr._sizes) > SIZE_HISTORY:
        tr._sizes.pop(0)
    if len(tr._sizes) < SIZE_MIN_FOR_MEDIAN:
        # too few to vote: take the biggest so far, as the van did before
        tr.length_m = max(tr.length_m, lm)
        tr.width_m = max(tr.width_m, wm)
        tr.height_m = max(tr.height_m, hm)
        if lm >= tr.length_m:
            tr.yaw_deg = float(o.get("yaw_deg", 0.0) or 0.0)
        return
    lengths = [s[0] for s in tr._sizes]
    tr.length_m = _median(lengths)
    tr.width_m = _median([s[1] for s in tr._sizes])
    tr.height_m = _median([s[2] for s in tr._sizes])
    # the heading of the sighting nearest that middle length, so it matches the shape reported
    tr.yaw_deg = min(tr._sizes, key=lambda s: abs(s[0] - tr.length_m))[3]
    spread = max(lengths) - min(lengths)
    tr.size_uncertain = bool(spread > expected_size_spread_m(tr.range_m))


def expected_size_spread_m(range_m: float) -> float:
    """How much a thing's measured length is expected to wobble at that range, from the
    LiDAR's points spreading out. Beyond this, the sightings genuinely disagree."""
    return SIZE_SPREAD_BASE_M + SIZE_SPREAD_PER_M * max(0.0, float(range_m))


def _recheck_name(tr: "Track") -> None:
    """A name given on one frame, checked against everything the track has learned since.

    The shape rule looks at a single sighting. The track keeps the middle of the last
    twelve, which is a better description of the thing. Seen live: something 3.7 m tall and
    something 0 cm wide were both still labelled vehicles, because each had one frame that
    happened to look car-shaped. If what the track has settled on could not be a vehicle,
    the name goes.

    A name from the camera used to be left alone, on the grounds that the camera saw the
    thing itself and not just its outline. That held on Town03's quiet streets. In a dense
    town it does not: at 34-48 m the camera called poles and signs "vehicle", and one of
    them measured 3.1 m tall and another 0.6 m long (measured live in Town10HD, 2026-09-09).
    So a camera name is now checked too, but ONLY against what is flatly impossible: too
    tall, or too long. A minimum-length test was tried and backed out -- it stripped the
    name off a real car at 22 m, because a car seen end-on measures under 2 m long. Width
    is excluded for the same reason: at range it is the least reliable number there is, and
    the camera really did see something. The thin poles the camera mislabels at 34-48 m
    therefore still get through; the honest fix for those is trusting the camera less with
    distance, not a size gate the laser cannot support. A track that loses its name keeps its place, its
    size and its room to leave; it becomes an obstacle, and the van still stops for it.
    """
    if tr.cls != "vehicle" or tr.cls_source not in ("shape", "camera"):
        return
    if len(tr._sizes) < SIZE_MIN_FOR_MEDIAN:
        return
    if tr.height_m > VEHICLE_MAX_HEIGHT_M or tr.length_m > VEHICLE_MAX_LENGTH_M:
        tr.cls = None
        tr.cls_source = None
        tr.confidence = 0.5
        return
    if tr.cls_source == "shape" and tr.width_m > 0.0 and tr.width_m < VEHICLE_MIN_WIDTH_FAR_M:
        tr.cls = None
        tr.cls_source = None
        tr.confidence = 0.5


def clearance_radius_m(cls, length_m: float, width_m: float) -> float:
    """How much room to leave around a thing: what was measured, but never less than the
    kind of thing deserves. A person may be small and still step sideways without warning."""
    measured = 0.5 * math.hypot(max(0.0, length_m), max(0.0, width_m))
    return max(measured, MIN_CLEARANCE_M.get(cls, DEFAULT_MIN_CLEARANCE_M))


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
    # ...but bounded, three ways, because the line above ran away with itself.
    #
    # A track's OWN speed widened its OWN search radius, with nothing above it, multiplied by
    # a gap of up to DROP_AFTER_S. That is a loop with the gain on the wrong side: one wrong
    # match gives a track a speed it never had, the speed widens the radius, the wider radius
    # finds another wrong match further away, and so on. Beside a kerb, where the LiDAR
    # returns dozens of near-identical slivers, it runs away completely.
    #
    # Measured live in Town10HD on 2026-09-10, van crawling under 1.2 m/s on an empty road:
    # tracks 0.1 m wide reported a MEDIAN speed of 19 to 27 m/s, pinned at the 30 m/s clamp,
    # and jumped 19, 22 and 31 m between one frame and the next. At 30 m/s with a 1.2 s gap
    # the old radius was 38 m. Those phantoms fired the crossing-prediction on 12% of readings
    # and stopped the van on 18%, on a street with nothing in it.
    #
    # What each bound is for, and why it cannot cost a real sighting:
    #   * a brand new track has speed 0, so its gate is GATE_M + dt * GATE_SPEED_MPS and none
    #     of these touch it -- the day-6 "catch a fast car on its second sighting" case is
    #     exactly as it was;
    #   * an established track already carries its velocity in the PREDICTED position, so the
    #     gate only has to cover error, not travel. At 9 Hz a car doing 14 m/s moves 1.6 m a
    #     frame and is predicted there.
    GATE_SPEED_CAP_MPS = 14.0   # a track may widen its own gate, but only up to a fast car
    GATE_MAX_DT_S = 0.35        # a long gap does not licence a huge gate: about three frames
    GATE_MAX_M = 4.0            # and nothing is ever the same object 4 m from where it should be
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
            reach = min(self.GATE_MAX_DT_S, dt) * min(self.GATE_SPEED_CAP_MPS,
                                                      max(self.GATE_SPEED_MPS, tr.speed))
            best_j, best_d = None, min(self.GATE_MAX_M, self.GATE_M + reach)
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
            if o.get("distance") is not None:
                tr.range_m = float(o["distance"])
            tr.correct(o["wx"], o["wy"], measurement_noise_m(o), t)
            if o.get("weak"):
                tr.hits += self.WEAK_HIT
            else:
                tr.hits += 1
                tr.strong_hits += 1
            if o.get("cls"):
                # A rider stays a rider. The camera sees the bicycle only in some frames,
                # and in the others it reports the person alone; without this, every such
                # frame demoted a cyclist back to a pedestrian and the name flickered.
                demotion = tr.cls == "cyclist" and o["cls"] == "pedestrian"
                if not demotion:
                    tr.cls = o["cls"]
                    tr.cls_source = o.get("cls_source", "shape")
                    tr.confidence = max(tr.confidence, o.get("confidence", 0.5))
                tr._unnamed = 0
            else:
                _forget_name(tr)
            _note_size(tr, o)
            _recheck_name(tr)

        for j in unmatched:
            o = observations[j]
            tr = Track(self._next_id, o["wx"], o["wy"], t)
            if o.get("distance") is not None:
                tr.range_m = float(o["distance"])
            if o.get("weak"):
                tr.hits = self.WEAK_HIT
                tr.strong_hits = 0
            if o.get("cls"):
                tr.cls = o["cls"]
                tr.cls_source = o.get("cls_source", "shape")
                tr.confidence = o.get("confidence", 0.5)
            _note_size(tr, o)
            self._next_id += 1
            self._tracks.append(tr)

        return [tr for tr in self._tracks if tr.hits >= self.MIN_HITS - 1e-6]

    def reported_speed(self, tr: Track) -> float:
        """A thing the van has decided is parked reports exactly zero, not a wobble.

        Capping this by how far the thing has actually travelled was tried and taken back
        out. It sounds right -- the filter's velocity chases jitter, displacement is a fact --
        but measured on 4702 live readings of tracks the van called moving, the claim and the
        truth differ by a median of 1.1x, and a cap at 1.6x silences 3% of them. These blobs
        are not jittering in place. They really do slide along the kerb at 2 to 8 m/s, because
        the association hops from one sliver to the next. The fault is in what the sliding is
        taken to MEAN, which is why the fix lives in the crossing prediction.
        """
        if getattr(tr, "stationary", False):
            return 0.0
        s = tr.speed
        return 0.0 if s < self.SPEED_DEADBAND else s

    def reported_velocity(self, tr: Track):
        """The velocity that AGREES with reported_speed.

        These two had drifted apart. Speed was zeroed for anything the van had decided was
        parked -- the day-6 decision, which weighs how far the thing has actually travelled
        and how straight it went, not just one speed reading. The velocity was not: it stayed
        the raw filter estimate, wobble and all. So the same object on the same sheet said
        "parked, 0.0 m/s" in one field and 1.2 m/s in another, and the crossing-prediction
        reads the second. Measured live on an EMPTY street, 2026-09-09: prediction fired 9
        times and 5 of them were on objects the van itself called parked. Each one cost the
        van several metres per second of speed for something that never moved.
        """
        if self.reported_speed(tr) == 0.0:
            return 0.0, 0.0
        return tr.vx, tr.vy
