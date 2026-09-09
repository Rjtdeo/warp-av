"""
A map of the space around the van: what is free, what is blocked, what it
cannot see. (Perception V2, day 9.)

Until now the van kept a list of things. A list is the right shape for "stop
for that", and the wrong shape for two other questions it has to answer:

  * **where can I go?** A list of obstacles cannot tell you where the gap is.
    Steering around anything needs free space, not objects.
  * **what is that long thing?** A wall or a hedge is one continuous surface,
    and chopping it into blobs invents objects that were never there. On the
    recorded scenes the van reported about thirteen of them per update as
    moving, which is a wall's blob wandering, not traffic.

The grid answers both. It sits on the van, x forward and y to the right, and
every square is one of three things:

    FREE      a laser beam went through it and carried on
    OCCUPIED  a beam stopped there
    UNKNOWN   no beam has been through it, so nothing is claimed

Unknown is a real answer and is kept separate from free on purpose. Behind a
parked van is unknown, not empty, and a planner that treats the two the same
will drive into whatever is hiding there.

How it is filled in: every laser return is turned into a bearing and a range.
For each narrow slice of bearing, the nearest return is where the world stops.
Everything nearer along that slice was passed through, so it is free; the
return itself is occupied; everything beyond it is hidden, so it stays
unknown. That is the classic conversion from one turn of a scanner into free
space, and it is fast because it is one pass over the points.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

# A second, thinner layer over the same squares: which of them the laser actually saw
# ROAD SURFACE in, rather than merely passed a beam through. Free is not drivable: a beam
# travels along a pavement perfectly well (Perception V2 day 10).
NOT_ROAD = 0
ROAD = 1

DEFAULT_RANGE_M = 30.0        # how far the grid reaches, in every direction
DEFAULT_CELL_M = 0.25         # how big one square is
DEFAULT_BEARING_STEP_DEG = 0.25   # at 30 m two neighbouring slices are then 13 cm apart,
                                  # half a square, so the free space between rays joins up
EGO_NOSE_M = 3.0                  # the grid sits on the LiDAR, at the middle of the van;
                                  # its nose is about three metres ahead of that
DEFAULT_LANE_HALF_M = 1.75        # the van's own corridor, as the planner draws it
DEFAULT_BLIND_REACH_M = 1.5       # unseen space this close to the lane edge is a pocket that
                                  # matters: whatever steps out of it is beside us at once
DEFAULT_BLIND_LOOK_M = 15.0       # further ahead than this and there is time to see it first

DEFAULT_SPREAD_SLICES = 4         # how far an object's shadow reaches sideways, in slices.
                                  # Measured on the recordings: it takes the share of real
                                  # things wrongly called free from 6.8 % to 2.7 %, and costs
                                  # under one point of the road being called free.


@dataclass
class GridSummary:
    """The few numbers a planner or an operator actually reads."""

    free_ahead_m: float           # how far the van can see clear straight ahead, in its own lane
    free_left_m: float            # ... and in the lane to each side
    free_right_m: float
    free_cells: int
    occupied_cells: int
    unknown_cells: int
    cell_m: float
    range_m: float
    road_left_m: Optional[float] = None    # where the road surface reaches, 8 m ahead
    road_right_m: Optional[float] = None
    road_width_m: Optional[float] = None

    def as_dict(self) -> dict:
        total = max(1, self.free_cells + self.occupied_cells + self.unknown_cells)
        return {"free_ahead_m": round(self.free_ahead_m, 1),
                "free_left_m": round(self.free_left_m, 1),
                "free_right_m": round(self.free_right_m, 1),
                "cells": {"free": self.free_cells, "occupied": self.occupied_cells,
                          "unknown": self.unknown_cells},
                "seen_share": round((self.free_cells + self.occupied_cells) / total, 3),
                "road": {"left_m": None if self.road_left_m is None else round(self.road_left_m, 2),
                         "right_m": None if self.road_right_m is None else round(self.road_right_m, 2),
                         "width_m": None if self.road_width_m is None else round(self.road_width_m, 2)},
                "cell_m": self.cell_m, "range_m": self.range_m}


class OccupancyGrid:
    """Free, blocked and unseen space around the van, from one turn of the laser."""

    def __init__(self, range_m: float = DEFAULT_RANGE_M, cell_m: float = DEFAULT_CELL_M,
                 bearing_step_deg: float = DEFAULT_BEARING_STEP_DEG,
                 spread_slices: int = DEFAULT_SPREAD_SLICES):
        self.range_m = float(range_m)
        self.cell_m = float(cell_m)
        self.bearing_step_deg = float(bearing_step_deg)
        self.spread_slices = int(spread_slices)
        self.n = int(round(2.0 * self.range_m / self.cell_m))
        self.cells = np.full((self.n, self.n), UNKNOWN, dtype=np.uint8)
        #: squares the laser hit road surface in. Sparse near the van, sparser far away.
        self.road = np.full((self.n, self.n), NOT_ROAD, dtype=np.uint8)
        self.bearings = int(round(360.0 / self.bearing_step_deg))
        #: nearest thing along each bearing, metres. inf = nothing seen that way
        self.wall_range = np.full(self.bearings, np.inf, dtype=np.float32)
        self.updated = False

    # ---- geometry ---------------------------------------------------------------
    def to_cell(self, x, y):
        """Van-frame metres -> (row, column). Row grows forward, column grows right."""
        r = np.asarray((np.asarray(x) + self.range_m) / self.cell_m, dtype=np.int32)
        c = np.asarray((np.asarray(y) + self.range_m) / self.cell_m, dtype=np.int32)
        return r, c

    def to_metres(self, row, col):
        return ((row + 0.5) * self.cell_m - self.range_m,
                (col + 0.5) * self.cell_m - self.range_m)

    def inside(self, row, col) -> bool:
        return 0 <= row < self.n and 0 <= col < self.n

    def at(self, x: float, y: float) -> int:
        """What the grid says about one spot, in metres."""
        r, c = self.to_cell(x, y)
        r, c = int(r), int(c)
        return int(self.cells[r, c]) if self.inside(r, c) else UNKNOWN

    # ---- filling it in ----------------------------------------------------------
    def update(self, points_xy: np.ndarray, occupied: np.ndarray) -> "OccupancyGrid":
        """One turn of the laser: `points_xy` is Nx2 in the van's frame, `occupied` says
        which of those returns came off something solid rather than the road."""
        self.cells[:] = UNKNOWN
        self.road[:] = NOT_ROAD
        self.wall_range[:] = np.inf
        self.updated = True
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return self
        occ = np.asarray(occupied, dtype=bool)

        rng = np.hypot(pts[:, 0], pts[:, 1])
        bearing = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360.0
        slot = np.minimum((bearing / self.bearing_step_deg).astype(np.int32), self.bearings - 1)

        # where the world stops along each bearing: the nearest solid return
        solid = occ & (rng > 0.1) & (rng <= self.range_m)
        if solid.any():
            np.minimum.at(self.wall_range, slot[solid], rng[solid])


        # free: every square a beam passed through on its way to that stop. Walking out
        # along each bearing in half-cell steps covers every square the ray crosses.
        step = self.cell_m * 0.5
        reach = np.where(np.isinf(self.wall_range), 0.0, self.wall_range - self.cell_m)
        # bearings with no solid return still saw free space out to the furthest road point
        road = (~occ) & (rng <= self.range_m)
        if road.any():
            road_reach = np.zeros(self.bearings, dtype=np.float32)
            np.maximum.at(road_reach, slot[road], rng[road])
            reach = np.where(np.isinf(self.wall_range), road_reach, reach)
        # A thin thing, a post or a cone, can fall between two slices, and the slice beside it
        # would then be swept free straight past it. So every slice is also held back by the
        # nearest wall its neighbours found. Two mistakes are avoided here. Taking the
        # neighbours' *reach* wipes the map out, because with slices this narrow most of them
        # hold no points at all and a reach of zero spreads everywhere. Taking the neighbours'
        # wall on its own lets a slice that saw nothing inherit a far wall and claim more
        # space than it ever saw. The smaller of the two is the only version that can lose
        # free space and never invent it.
        if self.spread_slices:
            w = self.spread_slices
            neigh = np.stack([np.roll(self.wall_range, k) for k in range(-w, w + 1)]).min(axis=0)
            reach = np.minimum(reach, np.where(np.isinf(neigh), np.inf, neigh - self.cell_m))

        live = reach > step
        if live.any():
            angles = np.radians((np.arange(self.bearings, dtype=np.float32) + 0.5)
                                * self.bearing_step_deg)
            steps = np.arange(1, int(self.range_m / step) + 1, dtype=np.float32) * step
            d = steps[None, :] * live[:, None]                      # (bearing, step)
            keep = (d > 0) & (d <= reach[:, None])
            if keep.any():
                bi, si = np.nonzero(keep)
                dist = steps[si]
                xs = dist * np.cos(angles[bi])
                ys = dist * np.sin(angles[bi])
                r, c = self.to_cell(xs, ys)
                ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
                self.cells[r[ok], c[ok]] = FREE

        # Flat ground: where a beam actually landed on a surface the van could roll on,
        # as opposed to free space it merely passed through. Marking only the squares the
        # returns land in draws thin arcs, because the laser's rings touch the ground in
        # rings; so each slice is filled out to its furthest ground return, exactly the way
        # free space is filled, and never past the nearest solid thing.
        ground = (~occ) & (rng <= self.range_m)
        if ground.any():
            surface = np.zeros(self.bearings, dtype=np.float32)
            np.maximum.at(surface, slot[ground], rng[ground])
            surface = np.minimum(surface, np.where(np.isinf(self.wall_range),
                                                   self.range_m, self.wall_range))
            live_s = surface > step
            if live_s.any():
                angles_s = np.radians((np.arange(self.bearings, dtype=np.float32) + 0.5)
                                      * self.bearing_step_deg)
                steps_s = np.arange(1, int(self.range_m / step) + 1, dtype=np.float32) * step
                keep_s = (steps_s[None, :] <= surface[:, None]) & live_s[:, None]
                if keep_s.any():
                    bi, si = np.nonzero(keep_s)
                    dist = steps_s[si]
                    r, c = self.to_cell(dist * np.cos(angles_s[bi]), dist * np.sin(angles_s[bi]))
                    ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
                    self.road[r[ok], c[ok]] = ROAD

        # occupied: where the beams actually stopped. Written last so it always wins.
        if solid.any():
            r, c = self.to_cell(pts[solid, 0], pts[solid, 1])
            ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
            self.cells[r[ok], c[ok]] = OCCUPIED
        return self

    # ---- reading it -------------------------------------------------------------
    def free_distance(self, bearing_deg: float = 0.0, half_width_m: float = 1.5,
                      max_m: Optional[float] = None, start_m: float = EGO_NOSE_M) -> float:
        """How far the van can go along a bearing before the grid stops being free.

        A strip is used rather than a single line, because the van is wider than one square
        and a gap it cannot fit through is not a gap. The walk starts at the nose: the
        squares under the van itself can never be seen, since its own returns are thrown
        away, and asking about them makes every answer zero.
        """
        max_m = self.range_m if max_m is None else min(max_m, self.range_m)
        a = math.radians(bearing_deg)
        ca, sa = math.cos(a), math.sin(a)
        offsets = np.arange(-half_width_m, half_width_m + 1e-9, self.cell_m, dtype=np.float32)
        d = max(self.cell_m * 0.5, float(start_m))
        while d <= max_m:
            xs = d * ca - offsets * sa
            ys = d * sa + offsets * ca
            r, c = self.to_cell(xs, ys)
            ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
            if not ok.all():
                return d
            band = self.cells[r[ok], c[ok]]
            if np.any(band != FREE):
                return d
            d += self.cell_m
        return max_m

    def blind_spot_ahead(self, lane_half_m: float = DEFAULT_LANE_HALF_M,
                         reach_m: float = DEFAULT_BLIND_REACH_M,
                         max_m: float = DEFAULT_BLIND_LOOK_M,
                         start_m: float = EGO_NOSE_M) -> Optional[float]:
        """How far ahead is the nearest place beside our lane that the van CANNOT SEE INTO?

        Unknown is kept apart from free for exactly this reason. Behind a parked lorry is
        not empty; it is a pocket, and a person can walk out of it. This walks up the lane
        and returns the first distance at which an unseen square sits within `reach_m` of
        the lane's own edge -- close enough that whatever is hiding there is beside the van
        before it can be seen. None means nothing unseen that near, all the way out.

        Measured on Town10HD, 2026-09-09, out to 15 m: an ordinary street answers None,
        because the van can see the road and the kerb and the unseen part starts beyond
        them. Park a lorry 3.0 m to the right and the answer is 5.5 m -- its shadow reaches
        1.50 m across, which is inside our own lane. A car at 2.6 m answers 8.0 m. So the
        question is quiet when there is nothing to worry about, which is what makes it
        worth asking every tick.
        """
        limit = lane_half_m + reach_m
        ys = np.arange(-limit, limit + 1e-9, self.cell_m, dtype=np.float32)
        d = max(self.cell_m * 0.5, float(start_m))
        max_m = min(max_m, self.range_m)
        while d <= max_m:
            r, c = self.to_cell(np.full(ys.shape, d, dtype=np.float32), ys)
            ok = (r >= 0) & (r < self.n) & (c >= 0) & (c < self.n)
            if ok.any() and np.any(self.cells[r[ok], c[ok]] == UNKNOWN):
                return float(d)
            d += self.cell_m
        return None

    def road_edge(self, x_m: float, side: str, max_m: float = 12.0,
                  gap_m: float = 1.5) -> Optional[float]:
        """How far the road surface reaches sideways at a given distance ahead.

        Walk out from the van's own line until the road runs out and stays out for
        `gap_m`. A short break is stepped over, because the laser's rings leave gaps
        between them and a single empty square is not the edge of the road.
        Returns None when the road was never found there at all.
        """
        sign = -1.0 if side == "left" else 1.0
        step = self.cell_m
        last_road = None
        y = 0.0
        while abs(y) <= max_m:
            r, c = self.to_cell(x_m, y)
            r, c = int(r), int(c)
            if not self.inside(r, c):
                break
            if self.road[r, c] == ROAD:
                last_road = y
            elif last_road is not None and abs(y - last_road) > gap_m:
                break
            y += sign * step
        return last_road

    def road_width(self, x_m: float = 8.0) -> Optional[float]:
        left = self.road_edge(x_m, "left")
        right = self.road_edge(x_m, "right")
        if left is None or right is None:
            return None
        return abs(right - left)

    def drivable_at(self, x_m: float, y_m: float) -> bool:
        """Free AND road: somewhere the van could actually put a wheel."""
        r, c = self.to_cell(x_m, y_m)
        r, c = int(r), int(c)
        if not self.inside(r, c):
            return False
        return bool(self.cells[r, c] == FREE and self.road[r, c] == ROAD)

    def summary(self, lane_m: float = 3.5) -> GridSummary:
        free = int(np.count_nonzero(self.cells == FREE))
        occupied = int(np.count_nonzero(self.cells == OCCUPIED))
        unknown = int(self.cells.size - free - occupied)
        side = math.degrees(math.atan2(lane_m, 10.0))     # a lane over, ten metres ahead
        left = self.road_edge(8.0, "left")
        right = self.road_edge(8.0, "right")
        return GridSummary(
            free_ahead_m=self.free_distance(0.0),
            free_left_m=self.free_distance(-side),
            free_right_m=self.free_distance(side),
            free_cells=free, occupied_cells=occupied, unknown_cells=unknown,
            cell_m=self.cell_m, range_m=self.range_m,
            road_left_m=left, road_right_m=right,
            road_width_m=None if (left is None or right is None) else abs(right - left))

    def as_text(self, span_m: float = 12.0, step_cells: int = 2) -> str:
        """A small picture for a terminal: the van at the bottom, ahead going up."""
        rows = []
        half = int(span_m / self.cell_m)
        mid = self.n // 2
        for r in range(mid + half, mid - 1, -step_cells):
            line = []
            for c in range(mid - half, mid + half + 1, step_cells):
                if not self.inside(r, c):
                    line.append(" ")
                    continue
                v = self.cells[r, c]
                line.append("." if v == FREE else "#" if v == OCCUPIED else " ")
            rows.append("".join(line))
        return "\n".join(rows)
