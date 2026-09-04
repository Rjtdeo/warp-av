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
SEARCH_RIGHT_MAX_M = 8.0      # ... out to here
SEARCH_BACK_M = 12.0          # and this far behind ...
SEARCH_AHEAD_M = 40.0         # ... to this far ahead
BIN_M = 0.5                   # occupancy bins along the kerb
OCCUPIED_POINTS = 3           # a bin with this many body points is taken
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


@dataclass
class Kerb:
    a: float            # kerb line, sensor frame: y_right = a + b * x
    b: float
    n_points: int

    @property
    def yaw(self) -> float:
        return math.atan(self.b)

    def right_of(self, x, y_right):
        """Signed distance from the kerb line; positive = road side (van side)."""
        return (self.a + self.b * x - y_right) / math.sqrt(1.0 + self.b * self.b)


def fit_kerb(points: np.ndarray, trim_m: float = 0.25, rounds: int = 3) -> Optional[Kerb]:
    """Straight line along the kerb FACE: the road-side edge of the pavement.

    The lidar sees the whole pavement surface at kerb height, so a plain fit
    through every low point runs down the middle of the pavement, a metre or
    more beyond the kerb (first probe: +1.2 m, at all 40 spots). So: in each
    half-metre strip along the van, keep only the nearest low point to the van;
    those are the edge; fit the line through them, trimming stragglers."""
    if points is None or len(points) == 0:
        return None
    x, y, z = points[:, 0], points[:, 1], points[:, 2] + LIDAR_HEIGHT_M
    m = ((y > SEARCH_RIGHT_MIN_M) & (y < SEARCH_RIGHT_MAX_M) & (x > -SEARCH_BACK_M)
         & (x < SEARCH_AHEAD_M) & (z > KERB_LOW_M) & (z < KERB_HIGH_M))
    xs, ys = x[m], y[m]
    if len(xs) < 20:
        return None
    # nearest low point per along-strip = the pavement's near edge
    strip = np.floor((xs + SEARCH_BACK_M) / BIN_M).astype(int)
    order = np.lexsort((ys, strip))
    strip_sorted, first = np.unique(strip[order], return_index=True)
    ex, ey = xs[order][first], ys[order][first]
    if len(ex) < 8:
        return None
    keep = np.ones(len(ex), dtype=bool)
    a = b = 0.0
    for _ in range(rounds):
        if keep.sum() < 6:
            return None
        A = np.stack([np.ones(keep.sum()), ex[keep]], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, ey[keep], rcond=None)
        resid = np.abs(ey - (a + b * ex))
        keep = resid < trim_m
    if keep.sum() < 6 or abs(b) > math.tan(math.radians(35)):
        return None
    return Kerb(float(a), float(b), int(keep.sum()))


def find_bays(points: np.ndarray, min_len_m: float = SLOT_LEN_M) -> List[Bay]:
    """All free stretches at least min_len_m long beside the kerb, nearest first
    (measured from the van), each with the slot pose a van should aim for."""
    kerb = fit_kerb(points)
    if kerb is None:
        return []
    x, y, z = points[:, 0], points[:, 1], points[:, 2] + LIDAR_HEIGHT_M
    d = kerb.right_of(x, y)                       # + towards the van
    body = ((z > BODY_LOW_M) & (z < BODY_HIGH_M) & (d > -0.3) & (d < SLOT_WID_M + 1.0)
            & (x > -SEARCH_BACK_M) & (x < SEARCH_AHEAD_M))
    # along-kerb coordinate: project onto the kerb direction
    c, s = math.cos(kerb.yaw), math.sin(kerb.yaw)
    along = x * c + y * s
    edges = np.arange(-SEARCH_BACK_M, SEARCH_AHEAD_M + BIN_M, BIN_M)
    counts, _ = np.histogram(along[body], bins=edges)
    occupied = counts >= OCCUPIED_POINTS
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
            kx = mid * c                                    # a point on the kerb line
            ky = kerb.a + kerb.b * kx
            # step off the kerb towards the van (perpendicular, road side)
            nx, ny = -s, c                                  # unit normal in sensor frame (y right)
            # choose the normal that points towards the van (decreasing y_right)
            if ny > 0:
                nx, ny = -nx, -ny
            sx, sy = kx + nx * off, ky + ny * off
            bays.append(Bay(x=float(sx), y=float(-sy), yaw=float(kerb.yaw),
                            length=float(free_end - free_start),
                            along_start=float(free_start), along_end=float(free_end)))
        i = j
    bays.sort(key=lambda bay: abs(bay.x))
    return bays
