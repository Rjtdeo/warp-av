"""
Is a parking spot free? Read off the van's own LiDAR (2026-09-11).

It used to be CARLA's list of every car in the world, plus the map's decorative parked cars
-- knowledge no real van has. Now it is the free-space map the LiDAR builds every sweep
(perception/occupancy.py): a square is FREE where a laser beam passed through it, OCCUPIED
where one stopped, and UNKNOWN where none has been. The spot is read a little in from its
edges, so the kerb running along its far side is not counted as being in it.

Three answers, and the third is not the first: UNSEEN never counts as free. A spot behind a
parked van, or out of the map's 30 m reach, is unseen -- the van does not turn into it.
"""
from __future__ import annotations

import math

import numpy as np

TAKEN, FREE, UNSEEN = "taken", "free", "unseen"

SPOT_INSET_M = 0.3        # judged this far in from its edges: the kerb beside it is not IN it
SPOT_TAKEN_CELLS = 4      # this many blocked 25 cm squares inside it, and it is taken
SPOT_FREE_SHARE = 0.75    # it is free only once the laser has seen this much of it empty
SPOT_DEFAULT_LEN_M = 7.0
SPOT_DEFAULT_WID_M = 2.2

_OCCUPIED, _FREE = 2, 1   # perception/occupancy.py's values


def spot_counts(grid, spot: dict, pose_x: float, pose_y: float, pose_yaw: float):
    """(free, blocked, unseen) squares inside `spot` ({"x", "y", "yaw", "length", "width"},
    world frame, yaw in radians), read off `grid` (an OccupancyGrid centred on the van at the
    pose, x forward, y right); None when there is no map or the spot is too small to read."""
    if grid is None or not getattr(grid, "updated", False):
        return None
    half_l = (spot.get("length") or SPOT_DEFAULT_LEN_M) / 2.0 - SPOT_INSET_M
    half_w = (spot.get("width") or SPOT_DEFAULT_WID_M) / 2.0 - SPOT_INSET_M
    if half_l <= 0 or half_w <= 0:
        return None
    step = grid.cell_m
    a, b = np.meshgrid(np.arange(-half_l, half_l + 1e-6, step),
                       np.arange(-half_w, half_w + 1e-6, step))
    sc, ss = math.cos(spot["yaw"]), math.sin(spot["yaw"])
    wx = spot["x"] + a * sc - b * ss                 # spot frame -> world (CARLA: y to the right)
    wy = spot["y"] + a * ss + b * sc
    dx, dy = wx - pose_x, wy - pose_y
    c, s = math.cos(pose_yaw), math.sin(pose_yaw)
    r, col = grid.to_cell(dx * c + dy * s, -dx * s + dy * c)      # world -> the van's map
    inside = (r >= 0) & (r < grid.n) & (col >= 0) & (col < grid.n)
    cells = np.zeros(r.shape, dtype=np.uint8)
    cells[inside] = grid.cells[r[inside], col[inside]]
    free = int((cells == _FREE).sum())
    blocked = int((cells == _OCCUPIED).sum())
    return free, blocked, int(cells.size) - free - blocked


def spot_view(grid, spot: dict, pose_x: float, pose_y: float, pose_yaw: float) -> str:
    """TAKEN, FREE or UNSEEN for `spot` -- see spot_counts."""
    counts = spot_counts(grid, spot, pose_x, pose_y, pose_yaw)
    if counts is None:
        return UNSEEN
    free, blocked, unseen = counts
    if blocked >= SPOT_TAKEN_CELLS:
        return TAKEN
    if free / float(free + blocked + unseen) >= SPOT_FREE_SHARE:
        return FREE
    return UNSEEN
