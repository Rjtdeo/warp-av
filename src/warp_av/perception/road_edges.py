"""
Where the road stops. (Perception V2, day 10.)

Yesterday's map says where there is space. It does not say where the van is
allowed to put its wheels. A laser beam travels along a pavement perfectly
well, so a pavement comes back as free space, and free is not the same as
drivable. Until now the only thing that knew where the lane was, was the
simulator's map, handed to the van for nothing.

A kerb has a shape the laser can find on its own: a step of ten or fifteen
centimetres that runs alongside the road for many metres, roughly parallel to
it. That is quite unlike a bin, a person or a car, which are tall and short.
So:

  1. take the points that sit in the kerb band, a little above the road but
     well below knee height;
  2. split them into the ones on the van's left and the ones on its right;
  3. fit a straight line through each side, throwing out the points that do
     not agree, because a parked car's wheels and a shallow verge both leave
     points in the same band;
  4. report how far the road runs each way, and how sure the van is.

The result is deliberately modest: two lines and two distances. It says
nothing about lane markings, junctions, or roads with no kerb at all, and it
says so by returning nothing rather than guessing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

KERB_MIN_M = 0.05           # a kerb sits at least this far above the road...
KERB_MAX_M = 0.35           # ... and no higher than this; above it is an object
AHEAD_M = 25.0              # how far forward to look
BEHIND_M = 5.0
MAX_LATERAL_M = 12.0        # a kerb further off than this is another street
MIN_POINTS = 12             # fewer than this is not a road edge
MIN_LENGTH_M = 4.0          # ... and neither is a short stub
MAX_TILT_DEG = 25.0         # a kerb runs alongside the road, not across it
FIT_TOLERANCE_M = 0.30      # how far off the line a point may sit and still count
MIN_INLIER_SHARE = 0.55     # and how many must agree before the line is believed


@dataclass
class RoadEdge:
    """One side of the road, as a straight line in the van's frame."""

    side: str                  # 'left' or 'right'
    offset_m: float            # how far the edge sits from the van, at the van
    heading_deg: float         # which way it runs, 0 = straight ahead
    length_m: float            # how much of it was actually seen
    points: int
    inlier_share: float

    @property
    def confident(self) -> bool:
        return (self.points >= MIN_POINTS and self.length_m >= MIN_LENGTH_M
                and self.inlier_share >= MIN_INLIER_SHARE
                and abs(self.heading_deg) <= MAX_TILT_DEG)

    def lateral_at(self, x_m: float) -> float:
        """Where the edge sits, sideways, at a given distance ahead."""
        return self.offset_m + math.tan(math.radians(self.heading_deg)) * float(x_m)

    def as_dict(self) -> dict:
        return {"side": self.side, "offset_m": round(self.offset_m, 2),
                "heading_deg": round(self.heading_deg, 1),
                "length_m": round(self.length_m, 1), "points": self.points,
                "inlier_share": round(self.inlier_share, 2), "confident": self.confident}


@dataclass
class RoadEdges:
    left: Optional[RoadEdge] = None
    right: Optional[RoadEdge] = None

    @property
    def width_m(self) -> Optional[float]:
        """How wide the drivable strip is, when both sides were found."""
        if self.left is None or self.right is None:
            return None
        if not (self.left.confident and self.right.confident):
            return None
        return abs(self.right.offset_m - self.left.offset_m)

    def drivable(self, x_m: float, y_m: float, margin_m: float = 0.0) -> Optional[bool]:
        """Is that spot inside the road? None when the van does not know."""
        checks = []
        if self.left is not None and self.left.confident:
            checks.append(y_m >= self.left.lateral_at(x_m) + margin_m)
        if self.right is not None and self.right.confident:
            checks.append(y_m <= self.right.lateral_at(x_m) - margin_m)
        if not checks:
            return None
        return all(checks)

    def as_dict(self) -> dict:
        return {"left": self.left.as_dict() if self.left else None,
                "right": self.right.as_dict() if self.right else None,
                "width_m": None if self.width_m is None else round(self.width_m, 2)}


def _fit_side(pts: np.ndarray, side: str) -> Optional[RoadEdge]:
    """Fit y = a + b*x through one side's kerb points, ignoring the ones that disagree.

    A plain least-squares fit is dragged off by a parked car's wheels or a driveway,
    so the fit is repeated on the points that agreed with it, twice, which is enough
    to shake off a handful of strays without needing a full random-sample search.
    """
    if pts.shape[0] < MIN_POINTS:
        return None
    x, y = pts[:, 0].astype(np.float64), pts[:, 1].astype(np.float64)
    keep = np.ones(len(x), dtype=bool)
    a = b = 0.0
    for _ in range(3):
        if keep.sum() < MIN_POINTS:
            return None
        b, a = np.polyfit(x[keep], y[keep], 1)
        residual = np.abs(y - (a + b * x))
        keep = residual <= FIT_TOLERANCE_M
    if keep.sum() < MIN_POINTS:
        return None
    used = pts[keep]
    length = float(used[:, 0].max() - used[:, 0].min())
    return RoadEdge(side=side, offset_m=float(a), heading_deg=float(math.degrees(math.atan(b))),
                    length_m=length, points=int(keep.sum()),
                    inlier_share=float(keep.sum() / len(x)))


def find_road_edges(points_xy: np.ndarray, heights_m: np.ndarray,
                    ahead_m: float = AHEAD_M, behind_m: float = BEHIND_M) -> RoadEdges:
    """Find the kerb each side of the van from one sweep.

    `points_xy` is Nx2 in the van's frame, `heights_m` how far each point sits above the
    road beneath it, which the ground filter already works out.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    h = np.asarray(heights_m, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or h.shape[0] != pts.shape[0]:
        return RoadEdges()
    band = ((h >= KERB_MIN_M) & (h <= KERB_MAX_M)
            & (pts[:, 0] >= -behind_m) & (pts[:, 0] <= ahead_m)
            & (np.abs(pts[:, 1]) <= MAX_LATERAL_M) & (np.abs(pts[:, 1]) >= 0.8))
    if not band.any():
        return RoadEdges()
    kerb = pts[band]
    left = kerb[kerb[:, 1] < 0]
    right = kerb[kerb[:, 1] > 0]
    return RoadEdges(left=_fit_side(left, "left"), right=_fit_side(right, "right"))
