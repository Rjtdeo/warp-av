"""
Ground (road) removal for the LiDAR: local patches (perception fix 2).

Today's rule is one flat line, 35 cm above the road under the van
(`minimum_lidar_z = -2.15` for a 2.5 m mount): everything below it is thrown
away. That deletes every low object (planters, kerbs, the feet of people,
the bottom half of a barrel) and keeps the road itself as soon as it climbs
a hill.

This module replaces the line with local patches:

  * the ground around the van is split into square tiles (1.5 m);
  * in each tile the lowest point is "the road here", if it is a believable
    road height for that distance (a band that widens with range, so a hill
    far away is still road);
  * a tile whose lowest point is a car, not road (much higher than the
    typical road height of its neighbours, their median), borrows that
    typical height; a tile with no road point at all takes its neighbours'
    median, and failing that a road PLANE fitted through every believable
    tile (so the far field of a hill is still a hill, not "assume flat");
    nothing ever borrows more than 3 m down (a bridge deck must not borrow
    the ground under the bridge);
  * every point is measured against the HIGHEST ground of the four tile
    centres nearest to it. A pavement strip sharing a tile with the road,
    a shallow ditch beside the road, or the uphill sliver of a tile on a
    grade are then ground, not objects;
  * a point is kept when it is more than `keep_above_m` (12 cm) above that
    ground, and below a ceiling (4 m above the road: bridges, trees, wires).

Pure numpy, sensor frame in (x forward, y right, z up, metres), no CARLA.
Also here: the road-edge rule used after clustering. A long, low blob beside
the lane is a kerb or a road edge, not an obstacle; a planter is low but
short and stays an obstacle; a long, low thing IN the lane stays an obstacle.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_LIDAR_HEIGHT_M = 2.5
DEFAULT_TILE_M = 1.5
DEFAULT_KEEP_ABOVE_M = 0.12
DEFAULT_BORROW_STEP_M = 0.25        # own lowest point this much above the reference: it is a car, borrow
DEFAULT_MAX_BORROW_M = 3.0          # reference lower than this: a different level (bridge), do not borrow
DEFAULT_CEILING_ABOVE_ROAD_M = 4.0  # today's maximum_lidar_z (1.5) for a 2.5 m mount
DEFAULT_RANGE_M = 50.0
ROAD_BAND_BASE_M = 0.5              # a tile's lowest point can be this far above/below the road under the van ...
ROAD_BAND_PER_M = 0.12              # ... plus 12 % of the tile's distance (a 12 % grade), and still be road
PLANE_MIN_TILES = 24                # fewer believable tiles than this: no plane, assume flat
PLANE_ENVELOPE_M = 0.50             # refit the plane on tiles within this of the first fit (either side)
MIN_NEIGHBOURS_FOR_REF = 3          # fewer road-like neighbours than this: the plane is the reference
RING_MIN_MATES = 2                  # a sparse far tile with this many same-height mates in a row is a road ring
RING_MATE_TOL_M = 0.30
RING_MATE_REACH = 3                 # tiles (4.5 m) along the row / column
REF_SLACK_BASE_M = 0.5              # a reference road height may sit this far above the fitted plane ...
REF_SLACK_PER_M = 0.04              # ... plus 4 % of the distance (hills bend away from a plane); higher = not road
NEIGHBOUR_STEP_M = 0.5              # a nearest tile more than this above a point's own tile is another level, ignored

ROAD_EDGE_MAX_HEIGHT_M = 0.30       # a blob lower than this ...
ROAD_EDGE_MIN_EXTENT_M = 1.5        # ... and longer than 2 x this ...
ROAD_EDGE_MIN_LATERAL_M = 1.2       # ... and centred this far off the van's axis is a kerb / road edge.
                                    # A low, long thing IN our lane (a traffic island, a slab) stays an obstacle.

MODES = ("patches", "flat")


def ground_filter_mode_from_env(env=None) -> str:
    """WARP_GROUND_FILTER=patches (default) | flat (the old 35 cm line)."""
    env = os.environ if env is None else env
    v = str(env.get("WARP_GROUND_FILTER", "patches")).strip().lower()
    return v if v in MODES else "patches"


@dataclass
class GroundResult:
    keep: np.ndarray            # bool (N,): point survives the road removal
    above: np.ndarray           # float32 (N,): height above the local road (NaN when out of range)
    ground_tiles: int           # tiles whose own lowest point was believed to be road
    borrowed_tiles: int         # tiles whose lowest point was a car and took the neighbours' road height
    ms: float


class GroundFilter:
    def __init__(self, lidar_height_m: float = DEFAULT_LIDAR_HEIGHT_M, tile_m: float = DEFAULT_TILE_M,
                 keep_above_m: float = DEFAULT_KEEP_ABOVE_M, borrow_step_m: float = DEFAULT_BORROW_STEP_M,
                 max_borrow_m: float = DEFAULT_MAX_BORROW_M,
                 ceiling_above_road_m: float = DEFAULT_CEILING_ABOVE_ROAD_M, range_m: float = DEFAULT_RANGE_M):
        self.lidar_height_m = float(lidar_height_m)
        self.tile_m = float(tile_m)
        self.keep_above_m = float(keep_above_m)
        self.borrow_step_m = float(borrow_step_m)
        self.max_borrow_m = float(max_borrow_m)
        self.ceiling_above_road_m = float(ceiling_above_road_m)
        self.range_m = float(range_m)
        self.n = int(np.ceil(2.0 * self.range_m / self.tile_m))
        # tile centres, their distance from the van, the plausible road band,
        # and the design matrix of the road plane, all computed once
        c = (np.arange(self.n) + 0.5) * self.tile_m - self.range_m
        cx, cy = np.meshgrid(c, c, indexing="ij")
        self._tile_cx = cx.reshape(-1)
        self._tile_cy = cy.reshape(-1)
        self._tile_range = np.hypot(self._tile_cx, self._tile_cy)
        self._band = ROAD_BAND_BASE_M + ROAD_BAND_PER_M * self._tile_range
        self._slack = (REF_SLACK_BASE_M + REF_SLACK_PER_M * self._tile_range).reshape(self.n, self.n)
        self._plane_A = np.c_[self._tile_cx, self._tile_cy, np.ones(self.n * self.n)]
        self.last: Optional[GroundResult] = None
        self.last_plane = None

    # the typical (median) road height of the neighbouring tiles, ignoring
    # empty ones; radius 1 = the 8 tiles around, 2 = the 24 around.
    # A median, not a minimum: a road on an embankment has a field 2 m
    # lower on ONE side, and must not borrow the field as its road.
    @staticmethod
    def _neighbour_median(grid: np.ndarray, radius: int, with_count: bool = False):
        n = grid.shape[0]
        layers = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                layer = np.full_like(grid, np.nan)
                src = grid[max(0, di):n + min(0, di), max(0, dj):n + min(0, dj)]
                layer[max(0, -di):n + min(0, -di), max(0, -dj):n + min(0, -dj)] = src
                layers.append(layer)
        stack = np.stack(layers, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slices -> NaN, wanted
            med = np.nanmedian(stack, axis=0)
        if with_count:
            return med, np.isfinite(stack).sum(axis=0)
        return med

    @staticmethod
    def _same_height_mates(own: np.ndarray) -> np.ndarray:
        """For every tile: how many tiles within 3 along the same row (x) or
        the same column (y) have a lowest point within RING_MATE_TOL_M of
        its own. The larger of the two directions."""
        n = own.shape[0]
        along_x = np.zeros_like(own, dtype=np.int64)
        along_y = np.zeros_like(own, dtype=np.int64)
        for d in range(1, RING_MATE_REACH + 1):
            for sign in (-1, 1):
                sh = np.full_like(own, np.nan)
                if sign > 0:
                    sh[:, d:] = own[:, :n - d]
                else:
                    sh[:, :n - d] = own[:, d:]
                along_y += (np.abs(sh - own) <= RING_MATE_TOL_M).astype(np.int64)
                sh = np.full_like(own, np.nan)
                if sign > 0:
                    sh[d:, :] = own[:n - d, :]
                else:
                    sh[:n - d, :] = own[d:, :]
                along_x += (np.abs(sh - own) <= RING_MATE_TOL_M).astype(np.int64)
        return np.maximum(along_x, along_y)

    def _road_plane(self, own_flat: np.ndarray, road_under_van: float) -> np.ndarray:
        """A plane through the believable tiles: the fallback where a tile
        has no road of its own and no road next door. Refitted once on its
        lower side so cars and walls do not lift it. Flat if too few tiles."""
        idx = np.flatnonzero(np.isfinite(own_flat))
        fallback = np.full(self.n * self.n, road_under_van)
        self.last_plane = None
        if idx.size < PLANE_MIN_TILES:
            return fallback
        A = self._plane_A[idx]
        zt = own_flat[idx]
        coef, *_ = np.linalg.lstsq(A, zt, rcond=None)
        near = np.abs(zt - A @ coef) < PLANE_ENVELOPE_M      # drop cars/walls above AND lower levels below
        if near.sum() >= PLANE_MIN_TILES:
            coef, *_ = np.linalg.lstsq(A[near], zt[near], rcond=None)
        plane = self._plane_A @ coef
        self.last_plane = coef
        # the plane may not leave the plausible band either
        return np.clip(plane, road_under_van - self._band, road_under_van + self._band)

    def apply(self, points) -> GroundResult:
        t0 = time.perf_counter()
        pts = np.asarray(points, dtype=np.float32)
        N = pts.shape[0]
        keep = np.zeros(N, dtype=bool)
        above = np.full(N, np.nan, dtype=np.float32)
        if N == 0:
            self.last = GroundResult(keep, above, 0, 0, 0.0)
            return self.last
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        xs = np.where(finite, x, 0.0).astype(np.float64)
        ys = np.where(finite, y, 0.0).astype(np.float64)
        zs = np.where(finite, z, 0.0).astype(np.float64)
        ix = np.floor((xs + self.range_m) / self.tile_m).astype(np.int64)
        iy = np.floor((ys + self.range_m) / self.tile_m).astype(np.int64)
        inr = finite & (ix >= 0) & (ix < self.n) & (iy >= 0) & (iy < self.n)
        flat = np.clip(ix * self.n + iy, 0, self.n * self.n - 1)

        # 1. the lowest point of every tile, believed only at a road-like height
        tile_min = np.full(self.n * self.n, np.inf, dtype=np.float64)
        np.minimum.at(tile_min, flat[inr], zs[inr])
        road_under_van = -self.lidar_height_m
        plausible = np.abs(tile_min - road_under_van) <= self._band
        own_flat = np.where(plausible, tile_min, np.nan)
        own = own_flat.reshape(self.n, self.n)

        # 2. the reference road height next door, and the plane behind it all
        fallback = self._road_plane(own_flat, road_under_van).reshape(self.n, self.n)
        nb1, n_nb = self._neighbour_median(own, 1, with_count=True)
        # the reference road height: the neighbours' typical height when
        # there are enough road-like neighbours, else the fitted plane.
        # Neighbours far above the plane are wall faces or car roofs seen
        # without their road (sparse far rings): capped to the plane + slack.
        from_nb = np.isfinite(nb1) & (n_nb >= MIN_NEIGHBOURS_FOR_REF)
        ref = np.where(from_nb, np.minimum(nb1, fallback + self._slack), fallback)
        # Far out the road returns are sparse rings 8-15 m apart, so a tile
        # often has no road-like neighbour at all and the plane is the only
        # reference. A road ring on a hill sits well above the plane too, so
        # height alone cannot tell it from a car's lowest ring. Shape can: a
        # ring runs across the road, so its tile has mates of the same height
        # along a row; a car does not. Tiles with such mates are never
        # borrowed against the plane.
        mates = self._same_height_mates(own)
        ring_like = ~from_nb & (mates >= RING_MIN_MATES)
        step = own - ref                                   # NaN only where own is missing
        borrow = np.isfinite(step) & (step > self.borrow_step_m) & (step <= self.max_borrow_m) & ~ring_like
        ground = np.where(borrow, ref, own)
        # tiles with no road of their own take their neighbours' typical
        # CORRECTED ground (car tiles already replaced), so an empty tile next
        # to a car does not inherit the car's height
        missing = np.isnan(ground)
        if missing.any():
            nbg1 = self._neighbour_median(ground, 1)
            ground = np.where(missing & np.isfinite(nbg1), nbg1, ground)
            still = np.isnan(ground)
            if still.any():
                nbg2 = self._neighbour_median(ground, 2)
                ground = np.where(still & np.isfinite(nbg2), nbg2, ground)
            ground = np.where(np.isnan(ground), fallback, ground)

        # 3. every point against the HIGHEST ground of its four nearest tile
        #    centres: strips, ditches and hill slivers inside a tile are ground.
        #    A nearest tile a whole step above the point's own tile is another
        #    level (a wall base, a car roof far away) and does not count.
        fx = (xs + self.range_m) / self.tile_m - 0.5
        fy = (ys + self.range_m) / self.tile_m - 0.5
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, self.n - 1)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, self.n - 1)
        i1 = np.minimum(i0 + 1, self.n - 1)
        j1 = np.minimum(j0 + 1, self.n - 1)
        g_own = ground[np.clip(ix, 0, self.n - 1), np.clip(iy, 0, self.n - 1)]
        g = g_own.copy()
        for gi, gj in ((i0, j0), (i0, j1), (i1, j0), (i1, j1)):
            gk = ground[gi, gj]
            g = np.maximum(g, np.where(gk - g_own <= NEIGHBOUR_STEP_M, gk, g_own))
        h = (zs - g).astype(np.float32)
        above[inr] = h[inr]
        keep = inr & (h > self.keep_above_m) & (h < self.ceiling_above_road_m)
        self.last = GroundResult(keep, above, int((np.isfinite(own) & ~borrow).sum()), int(borrow.sum()),
                                 (time.perf_counter() - t0) * 1000.0)
        return self.last


def flat_cut(points, lidar_height_m: float = DEFAULT_LIDAR_HEIGHT_M, min_above_m: float = 0.35,
             ceiling_above_road_m: float = DEFAULT_CEILING_ABOVE_ROAD_M) -> GroundResult:
    """The old rule, kept for A/B comparison: one line `min_above_m` above
    the road under the van."""
    t0 = time.perf_counter()
    pts = np.asarray(points, dtype=np.float32)
    above = (pts[:, 2] + lidar_height_m).astype(np.float32) if pts.shape[0] else np.zeros(0, np.float32)
    keep = (above > min_above_m) & (above < ceiling_above_road_m)
    return GroundResult(keep, above, 0, 0, (time.perf_counter() - t0) * 1000.0)


def is_road_edge(cluster: dict, max_height_m: float = ROAD_EDGE_MAX_HEIGHT_M,
                 min_extent_m: float = ROAD_EDGE_MIN_EXTENT_M,
                 min_lateral_m: float = ROAD_EDGE_MIN_LATERAL_M) -> bool:
    """A long, low blob beside the lane is a kerb or a road edge, not
    something to stop for. A long, low blob in the lane is kept.
    `cluster` is a cluster_points() dict with 'extent', 'y' (sensor frame,
    metres off the van's axis) and (optionally) 'height'."""
    height = cluster.get("height")
    if height is None:
        return False
    return (float(height) < max_height_m and float(cluster.get("extent", 0.0)) > min_extent_m
            and abs(float(cluster.get("y", 0.0))) > min_lateral_m)


def remove_road_edge_points(sel: np.ndarray, heights: np.ndarray, cluster_fn=None):
    """Pass 1 of the clustering (fix 2): cluster the LOW points on their own;
    a long, low blob clear of the lane is a kerb / road edge and its POINTS
    (those clear of the lane) are removed before the main clustering, so a
    kerb strip can never glue itself to a lamp post, a bin or a pedestrian
    on the pavement and drag their centroid. Returns
    (sel, heights, road_edges_dropped) where the count is of blobs that
    actually lost points."""
    if cluster_fn is None:
        from .tracking import cluster_points as cluster_fn      # local import: tracking imports nothing from here
    sel = np.asarray(sel)
    heights = np.asarray(heights)
    if sel.shape[0] == 0:
        return sel, heights, 0
    low = heights < ROAD_EDGE_MAX_HEIGHT_M
    if not low.any():
        return sel, heights, 0
    low_idx = np.flatnonzero(low)
    low_clusters = cluster_fn(sel[low_idx, :2].tolist(), heights=heights[low_idx].tolist(), return_members=True)
    edges = [c for c in low_clusters if is_road_edge(c)]
    if not edges:
        return sel, heights, 0
    keep_pt = np.ones(sel.shape[0], dtype=bool)
    dropped_blobs = 0
    for c in edges:
        members = np.asarray(c["members"], dtype=np.int64)
        clear = np.abs(sel[low_idx[members], 1]) > ROAD_EDGE_MIN_LATERAL_M    # only the part beside the lane
        if clear.any():
            keep_pt[low_idx[members[clear]]] = False
            dropped_blobs += 1
    return sel[keep_pt], heights[keep_pt], dropped_blobs
