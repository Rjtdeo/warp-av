"""
Find a parking bay from the lidar, not the map.

Town10's parking lanes are unmarked strips along the kerb, and a real test
course has no map at all, so a bay has to be SEEN: the kerb is a line of low
points to the van's right; parked cars are clumps of higher points along it;
a long enough empty stretch between them, beside the kerb, is a bay.

Input is one lidar sweep in the SENSOR frame as CARLA delivers it: rows of
(x forward, y right, z up, intensity), metres, with the sensor mounted at
(0, 0, LIDAR_HEIGHT_M) on the van. Output slots are in the VEHICLE frame the
rest of this codebase uses for vehicle-relative objects: x forward, y LEFT.

Pure numpy: no CARLA, so it is tested on synthetic scenes and only then
pointed at the simulator (tools/probe_bay_finder.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

LIDAR_HEIGHT_M = 2.5          # where the sensor sits above the road
SLOT_LEN_M = 7.0
SLOT_WID_M = 2.5
KERB_LOW_M = 0.04             # kerb candidates: this far above the road ...
KERB_HIGH_M = 0.35            # ... up to this (kerbs are 10-25 cm; bumpers start ~0.3)
BODY_LOW_M = 0.35             # parked-car candidates: from here ...
BODY_HIGH_M = 2.2             # ... to here above the road
SEARCH_RIGHT_MIN_M = 0.8      # look for the kerb this far to the right of the van ...
SEARCH_RIGHT_MAX_M = 11.0     # ... out to here (a set-back bay's kerb sits 7-9 m out)
SEARCH_BACK_M = 12.0          # and this far behind ...
SEARCH_AHEAD_M = 30.0         # ... to this far ahead (beyond ~25 m a car is 2-5 points)
BIN_M = 0.5                   # occupancy bins along the kerb
MAX_CURVE = 0.012             # |c| in y = a + b x + c x^2: a gentle bend (radius > ~40 m)
ROAD_FIT_HALF_WIDTH_M = 3.5   # road-surface plane is fitted from points this close beside the van
OCCUPIED_POINTS = 2           # a bin with this many body points is taken (near) ...
OCCUPIED_FAR_M = 15.0         # ... and beyond this range a single point is enough:
                              # a parked car 25 m away returns 2-5 points in total
END_MARGIN_M = 0.5            # a bay needs this much clear beyond each end
KERB_GAP_M = 0.3              # a parked van sits this far off the kerb face


@dataclass
class Bay:
    x: float            # vehicle frame, metres ahead of the sensor
    y: float            # vehicle frame, metres LEFT (negative = to the right)
    yaw: float          # radians, relative to the van's heading
    length: float       # how long the free stretch is
    along_start: float  # free stretch, in along-kerb metres from the sensor
    along_end: float
    slot_start: float   # the 7 m slot itself, centred in the free stretch
    slot_end: float


@dataclass
class Kerb:
    a: float            # kerb edge, sensor frame: y_right = a + b * x + c * x^2
    b: float
    c: float
    n_points: int

    def y_at(self, x):
        return self.a + self.b * x + self.c * x * x

    def yaw_at(self, x):
        return math.atan(self.b + 2.0 * self.c * x)

    @property
    def yaw(self) -> float:
        return self.yaw_at(0.0)

    def right_of(self, x, y_right):
        """Signed distance from the kerb edge; positive = road side (van side)."""
        slope = self.b + 2.0 * self.c * x
        return (self.y_at(x) - y_right) / np.sqrt(1.0 + slope * slope)


def fit_road_plane(points: np.ndarray):
    """(a, b, c) of the road surface z = a + b x + c y, fitted from the lowest
    points beside and ahead of the van. Heights are measured from THIS, not
    from an assumed flat road: on a cambered street the pavement 8 m out can
    sit below a fixed 'kerb height' band and vanish."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    m = (np.abs(y) < ROAD_FIT_HALF_WIDTH_M) & (x > -6.0) & (x < 20.0) & (z < -LIDAR_HEIGHT_M + 0.6)
    if m.sum() < 50:
        return (-LIDAR_HEIGHT_M, 0.0, 0.0)
    xs, ys, zs = x[m], y[m], z[m]
    keep = zs <= np.percentile(zs, 60)          # the lowest 60 % are the road
    A = np.stack([np.ones(keep.sum()), xs[keep], ys[keep]], axis=1)
    try:
        (a, b, c), *_ = np.linalg.lstsq(A, zs[keep], rcond=None)
    except Exception:
        return (-LIDAR_HEIGHT_M, 0.0, 0.0)
    if abs(b) > 0.15 or abs(c) > 0.15:          # nonsense slope: fall back to flat
        return (-LIDAR_HEIGHT_M, 0.0, 0.0)
    return (float(a), float(b), float(c))


def heights_above_road(points: np.ndarray, plane=None):
    if plane is None:
        plane = fit_road_plane(points)
    a, b, c = plane
    return points[:, 2] - (a + b * points[:, 0] + c * points[:, 1])


EDGE_CELL_M = 0.5             # lateral cell used to look for the edge cluster
EDGE_MIN_POINTS = 2           # a cell needs this many raised points and the next cell
                              # at least one (a pavement), OR EDGE_DENSE points on its
                              # own (a bare kerb with nothing behind it): a kerb is a
                              # dense cluster, a stray road-surface point is not
EDGE_DENSE = 3


def edge_per_strip(xs, ys):
    """For each half-metre strip along the van, the lateral position of the
    nearest CLUSTER of raised points - the pavement's near edge. A single
    raised point nearer in (road noise, a stone) does not count: with the kerb
    9 m out, taking the nearest single point put the edge at 2-7 m."""
    strip = np.floor((xs + SEARCH_BACK_M) / BIN_M).astype(int)
    cell = np.floor((ys - SEARCH_RIGHT_MIN_M) / EDGE_CELL_M).astype(int)
    n_cells = int((SEARCH_RIGHT_MAX_M - SEARCH_RIGHT_MIN_M) / EDGE_CELL_M) + 2
    strips = np.unique(strip)
    ex, ey = [], []
    for st in strips:
        sel = strip == st
        counts = np.bincount(np.clip(cell[sel], 0, n_cells - 1), minlength=n_cells)
        for k in range(n_cells - 1):
            if counts[k] >= EDGE_DENSE or (counts[k] >= EDGE_MIN_POINTS and counts[k + 1] >= 1):
                inside = sel & (cell == k)
                ex.append(float(xs[inside].mean()))
                ey.append(float(ys[inside].min()))
                break
    return np.array(ex), np.array(ey)


def fit_kerb(points: np.ndarray, trim_m: float = 0.25, rounds: int = 3) -> Optional[Kerb]:
    """Kerb edge as a gentle curve through the nearest raised points in each
    half-metre strip to the van's right (the pavement's near edge), with
    heights taken from a fitted road plane. A straight line is the special
    case c = 0; a bend tighter than MAX_CURVE is refused."""
    if points is None or len(points) == 0:
        return None
    x, y = points[:, 0], points[:, 1]
    h = heights_above_road(points)
    m = ((y > SEARCH_RIGHT_MIN_M) & (y < SEARCH_RIGHT_MAX_M) & (x > -SEARCH_BACK_M)
         & (x < SEARCH_AHEAD_M) & (h > KERB_LOW_M) & (h < KERB_HIGH_M))
    xs, ys = x[m], y[m]
    if len(xs) < 20:
        return None
    ex, ey = edge_per_strip(xs, ys)
    if len(ex) < 6:
        return None
    keep = np.ones(len(ex), dtype=bool)
    a = b = c = 0.0
    for _ in range(rounds):
        if keep.sum() < 5:
            return None
        A = np.stack([np.ones(keep.sum()), ex[keep], ex[keep] ** 2], axis=1)
        (a, b, c), *_ = np.linalg.lstsq(A, ey[keep], rcond=None)
        if abs(c) > MAX_CURVE:                     # too bent: straight line instead
            A = np.stack([np.ones(keep.sum()), ex[keep]], axis=1)
            (a, b), *_ = np.linalg.lstsq(A, ey[keep], rcond=None)
            c = 0.0
        resid = np.abs(ey - (a + b * ex + c * ex ** 2))
        keep = resid < trim_m
    if keep.sum() < 5 or abs(b) > math.tan(math.radians(35)):
        return None
    return Kerb(float(a), float(b), float(c), int(keep.sum()))


def find_bays(points: np.ndarray, min_len_m: float = SLOT_LEN_M) -> List[Bay]:
    """All free stretches at least min_len_m long beside the kerb, nearest first
    (measured from the van), each with the slot pose a van should aim for."""
    kerb = fit_kerb(points)
    if kerb is None:
        return []
    x, y = points[:, 0], points[:, 1]
    z = heights_above_road(points)
    d = kerb.right_of(x, y)                       # + towards the van
    body = ((z > BODY_LOW_M) & (z < BODY_HIGH_M) & (d > -0.3) & (d < SLOT_WID_M + 1.0)
            & (x > -SEARCH_BACK_M) & (x < SEARCH_AHEAD_M))
    # along-kerb coordinate: for a gentle curve, x along the tangent at the
    # sensor is within a few centimetres of arc length over 30 m
    c, s = math.cos(kerb.yaw), math.sin(kerb.yaw)
    along = x * c + y * s
    edges = np.arange(-SEARCH_BACK_M, SEARCH_AHEAD_M + BIN_M, BIN_M)
    counts, _ = np.histogram(along[body], bins=edges)
    centres = edges[:-1] + BIN_M / 2.0
    need = np.where(np.abs(centres) > OCCUPIED_FAR_M, 1, OCCUPIED_POINTS)
    occupied = counts >= need
    bays: List[Bay] = []
    i = 0
    while i < len(occupied):
        if occupied[i]:
            i += 1
            continue
        j = i
        while j < len(occupied) and not occupied[j]:
            j += 1
        free_start, free_end = edges[i], edges[j]
        usable = free_end - free_start - 2 * END_MARGIN_M
        if usable >= min_len_m:
            # centre the slot in the free stretch, one slot back from the kerb
            mid = 0.5 * (free_start + free_end)
            off = KERB_GAP_M + SLOT_WID_M / 2.0          # from the kerb face, road side
            kx = mid * c                                    # a point on the kerb edge
            ky = kerb.y_at(kx)
            yaw_here = kerb.yaw_at(kx)                      # the edge's direction there
            cc, ss = math.cos(yaw_here), math.sin(yaw_here)
            nx, ny = -ss, cc                                # unit normal (sensor frame, y right)
            if ny > 0:                                      # towards the van = decreasing y_right
                nx, ny = -nx, -ny
            sx, sy = kx + nx * off, ky + ny * off
            bays.append(Bay(x=float(sx), y=float(-sy), yaw=float(yaw_here),
                            length=float(free_end - free_start),
                            along_start=float(free_start), along_end=float(free_end),
                            slot_start=float(mid - SLOT_LEN_M / 2.0),
                            slot_end=float(mid + SLOT_LEN_M / 2.0)))
        i = j
    bays.sort(key=lambda bay: abs(bay.x))
    return bays
