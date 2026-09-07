"""
Vehicle footprint and swept-path geometry.

Today the planner treats the van as a line (the route) and every obstacle as
a point, then compares the point's distance from the line with hand-tuned
bands (1.40 / 1.75 / 2.20 m). Those bands stand in for "half a van plus half
a car plus a margin" and were tuned after specific scrapes. In a bend the
van's outer corner swings wide of the route line, and the bands do not know.

This module puts the van's real rectangle on the route instead:

    * VehicleFootprint  - the van's half-length, half-width and a safety
                          margin, all in metres.
    * sweep_conflict    - slide that rectangle along the route ahead, one
                          station at a time, turned to face along the route
                          at each station, and ask whether an obstacle disc
                          touches it. Returns where the first touch happens.

Pure geometry: no CARLA, no perception types. Route points may be anything
with .x and .y attributes (planner Waypoints) or (x, y) pairs.

Conventions: world frame, metres. The van's centre is assumed to follow the
route line (a rear-axle model would put the centre slightly inside a bend;
this is the simpler, slightly more cautious choice).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class VehicleFootprint:
    """The van as a rectangle, plus the room we insist on around it."""
    half_length: float = 2.96          # CARLA Sprinter bounding box, measured 2026-09-05
    half_width: float = 0.99
    safety_margin: float = 0.30        # extra clearance on every side

    @property
    def swept_half_length(self) -> float:
        return self.half_length + self.safety_margin

    @property
    def swept_half_width(self) -> float:
        return self.half_width + self.safety_margin


@dataclass(frozen=True)
class SweepHit:
    """Where the swept body first touches the obstacle."""
    along_m: float          # distance ahead of the van's centre, along the route
    station_x: float        # van centre at the moment of contact
    station_y: float
    heading: float          # route heading there (rad)
    lateral_m: float        # obstacle's sideways offset from the route line (+ left, - right)


def _xy(p) -> Tuple[float, float]:
    if hasattr(p, "x"):
        return float(p.x), float(p.y)
    return float(p[0]), float(p[1])


def _polyline(route_pts: Iterable) -> List[Tuple[float, float]]:
    pts = [_xy(p) for p in route_pts]
    # drop repeated points: they carry no direction
    out: List[Tuple[float, float]] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


def _project(px: float, py: float, pts: Sequence[Tuple[float, float]]):
    """Nearest point on the polyline: (arc length there, signed lateral offset,
    segment index). Lateral is + to the LEFT of the route direction."""
    best_d2, best_arc, best_i, best_lat = float("inf"), 0.0, 0, 0.0
    arc = 0.0
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        seg = math.sqrt(L2)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            # sign: cross product of direction and (point - foot); + means left
            cross = dx * (py - cy) - dy * (px - cx)
            best_d2, best_arc, best_i = d2, arc + t * seg, i
            best_lat = math.copysign(math.sqrt(d2), cross) if d2 > 0 else 0.0
        arc += seg
    return best_arc, best_lat, best_i


def _point_at_arc(s: float, pts: Sequence[Tuple[float, float]]):
    """(x, y, heading) of the point `s` metres along the polyline."""
    arc = 0.0
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i + 1]
        seg = math.hypot(bx - ax, by - ay)
        if s <= arc + seg or i == len(pts) - 2:
            t = 0.0 if seg == 0 else max(0.0, min(1.0, (s - arc) / seg))
            return ax + t * (bx - ax), ay + t * (by - ay), math.atan2(by - ay, bx - ax)
        arc += seg
    ax, ay = pts[-2]
    bx, by = pts[-1]
    return bx, by, math.atan2(by - ay, bx - ax)


def _disc_touches_box(ox: float, oy: float, radius: float,
                      cx: float, cy: float, heading: float,
                      half_len: float, half_wid: float) -> bool:
    """Does a disc (centre ox, oy) touch a rectangle centred at (cx, cy) and
    turned by `heading`? Exact: nearest point of the box to the disc centre."""
    dx, dy = ox - cx, oy - cy
    c, s = math.cos(-heading), math.sin(-heading)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    qx = max(-half_len, min(half_len, lx))
    qy = max(-half_wid, min(half_wid, ly))
    return math.hypot(lx - qx, ly - qy) <= radius + 1e-9


def sweep_conflict(route_pts: Iterable, ego_xy, footprint: VehicleFootprint,
                   obstacle_xy, obstacle_radius: float = 0.0,
                   horizon_m: float = 20.0, step_m: float = 0.5) -> Optional[SweepHit]:
    """Slide the van's rectangle (inflated by the safety margin) along the
    route from the van's current position for `horizon_m` metres, facing
    along the route at every station. Return the first station whose body
    touches the obstacle disc, or None.

    route_pts       the planned route ahead (and possibly behind) the van
    ego_xy          the van's centre now
    obstacle_xy     the obstacle's centre in the same frame
    obstacle_radius how big the obstacle is (0 for a bare point)
    Obstacles whose route position is behind the van's centre are ignored:
    the sweep only looks where the van is going.
    """
    pts = _polyline(route_pts)
    if len(pts) < 2 or horizon_m <= 0 or step_m <= 0:
        return None
    ex, ey = _xy(ego_xy)
    ox, oy = _xy(obstacle_xy)
    ego_arc, _, _ = _project(ex, ey, pts)
    obs_arc, obs_lat, _ = _project(ox, oy, pts)
    if obs_arc < ego_arc:
        return None                         # behind us along the route
    total = 0.0
    for i in range(len(pts) - 1):
        total += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    end = min(ego_arc + horizon_m, total)
    # quick reject: the obstacle is further ahead than the sweep can reach,
    # even allowing for the van's nose and the disc
    if obs_arc - end > footprint.swept_half_length + obstacle_radius:
        return None
    hl, hw = footprint.swept_half_length, footprint.swept_half_width
    s = ego_arc
    while s <= end + 1e-9:
        cx, cy, heading = _point_at_arc(s, pts)
        if _disc_touches_box(ox, oy, obstacle_radius, cx, cy, heading, hl, hw):
            return SweepHit(along_m=round(s - ego_arc, 3), station_x=cx, station_y=cy,
                            heading=heading, lateral_m=round(obs_lat, 3))
        s += step_m
    return None
