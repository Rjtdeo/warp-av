"""
Debug visualisation of the vehicle footprint and swept path (Planning V2,
phase 1C-VIS). Drawing only: nothing here decides anything.

Two layers, kept apart on purpose:

  * build_frame(...)  - pure geometry. From the van's pose, its footprint,
                        the route and the perceived objects it produces a
                        DebugFrame: the van's body rectangle, the protected
                        rectangle (body + safety margin), the swept-path
                        rectangles every ~1 m, the stationary obstacles the
                        blocking rule would look at, the first sweep contact
                        (computed with the SAME pure sweep_conflict and the
                        SAME constants the planner uses - for display), and the
                        text labels. No CARLA.
  * FootprintDebugDrawer - turns a DebugFrame into world.debug.draw_* calls
                        with short lifetimes. Every call is wrapped: a drawing
                        failure is counted and logged (rate-limited) and never
                        raised, so autonomy is never touched by a bad frame.

The planner's verdict (path_blocked, closest distance) is shown alongside the
sweep result, so the picture can be compared with what the van actually did,
with footprint blocking ON or OFF.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .footprint import VehicleFootprint, sweep_conflict, _polyline, _project, _point_at_arc
from .planner import DEFAULT_OBSTACLE_RADIUS_M, FOOTPRINT_STATIONARY_REACH_M

STATIONARY_MPS = 0.5                       # same threshold as the planner's stationary test
DEFAULT_STEP_M = 1.0                       # coarser than the 0.5 m collision sampling: it is a picture
DEFAULT_LIFETIME_S = 0.3                   # drawings vanish on their own; nothing accumulates

Point = Tuple[float, float]


@dataclass
class DebugFrame:
    body: List[Point] = field(default_factory=list)            # 4 corners, physical van
    protected: List[Point] = field(default_factory=list)       # 4 corners, van + safety margin
    stations: List[List[Point]] = field(default_factory=list)  # protected rectangles along the route
    obstacles: List[Tuple[float, float, float, str]] = field(default_factory=list)  # x, y, radius, tag
    conflict: Optional[Tuple[float, float, float]] = None      # x, y, along_m of the first contact
    texts: List[Tuple[str, str]] = field(default_factory=list) # (kind, text): mode / sweep / planner
    blocking_enabled: bool = False


def rectangle_corners(cx: float, cy: float, yaw: float, half_len: float, half_wid: float) -> List[Point]:
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for fx, fy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        lx, ly = fx * half_len, fy * half_wid
        out.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return out


def _radius_for(obj) -> float:
    r = getattr(obj, "radius", None)
    if r is not None:
        return float(r)
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    return DEFAULT_OBSTACLE_RADIUS_M.get(kind, 0.5)


def build_frame(pose_xy_yaw, footprint: VehicleFootprint, route_pts: Sequence, objects: Sequence,
                blocking_enabled: bool, planner_blocked: Optional[bool] = None,
                planner_distance: Optional[float] = None,
                horizon_m: float = FOOTPRINT_STATIONARY_REACH_M, step_m: float = DEFAULT_STEP_M) -> DebugFrame:
    """Everything to draw this tick. Pure; safe to call with an empty route."""
    ex, ey, eyaw = pose_xy_yaw
    frame = DebugFrame(blocking_enabled=bool(blocking_enabled))
    frame.body = rectangle_corners(ex, ey, eyaw, footprint.half_length, footprint.half_width)
    frame.protected = rectangle_corners(ex, ey, eyaw, footprint.swept_half_length, footprint.swept_half_width)
    frame.texts.append(("mode", f"FOOTPRINT BLOCKING: {'ON' if blocking_enabled else 'OFF'}"))

    pts = _polyline(route_pts or [])
    if len(pts) < 2:
        frame.texts.append(("sweep", "SWEPT PATH: no route"))
        return frame

    ego_arc, _, _ = _project(ex, ey, pts)
    total = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
    s = ego_arc
    while s <= min(ego_arc + horizon_m, total) + 1e-9:
        cx, cy, heading = _point_at_arc(s, pts)
        frame.stations.append(rectangle_corners(cx, cy, heading, footprint.swept_half_length,
                                                footprint.swept_half_width))
        s += step_m

    # stationary objects the blocking rule would consider, mirrored from the planner
    cos_y, sin_y = math.cos(eyaw), math.sin(eyaw)
    lat_limit = footprint.swept_half_length + 1.0
    best = None
    for obj in objects or []:
        if getattr(obj, "speed", 0.0) >= STATIONARY_MPS:
            continue
        wx = ex + cos_y * obj.x - sin_y * obj.y
        wy = ey + sin_y * obj.x + cos_y * obj.y
        oarc, lat, seg = _project(wx, wy, pts)
        along = oarc - ego_arc
        if along < -1.0 or along > horizon_m or abs(lat) > lat_limit:
            continue
        radius = _radius_for(obj)
        wp = route_pts[seg] if seg < len(route_pts) else None
        wp_next = route_pts[min(seg + 1, len(route_pts) - 1)] if route_pts else None
        near_junction = bool(getattr(wp, "is_junction", False) or getattr(wp_next, "is_junction", False))
        if near_junction:
            frame.obstacles.append((wx, wy, radius, "junction: old rules"))
            continue
        hit = sweep_conflict(pts, (ex, ey), footprint, (wx, wy), obstacle_radius=radius,
                             horizon_m=horizon_m + footprint.swept_half_length)
        frame.obstacles.append((wx, wy, radius, "hit" if hit is not None else "checked"))
        if hit is not None and (best is None or hit.along_m < best.along_m):
            best = hit
    if best is not None:
        frame.conflict = (best.station_x, best.station_y, best.along_m)
        frame.texts.append(("sweep", f"SWEPT PATH CONFLICT @ {best.along_m:.1f} m"))
    else:
        frame.texts.append(("sweep", "SWEPT PATH CLEAR"))
    if planner_blocked is not None:
        if planner_blocked:
            d = f" ({planner_distance:.1f} m)" if planner_distance is not None and planner_distance < 900 else ""
            frame.texts.append(("planner", f"PLANNER: BLOCKED{d}"))
        else:
            frame.texts.append(("planner", "PLANNER: CLEAR"))
    return frame


class FootprintDebugConfig:
    """The visualisation switch. Independent of the blocking flag. DEFAULT OFF."""

    def __init__(self, enabled: bool = False, lifetime_s: float = DEFAULT_LIFETIME_S):
        self.enabled = bool(enabled)
        self.lifetime_s = float(lifetime_s)

    def set(self, enabled=None):
        if enabled is not None:
            if isinstance(enabled, str):
                v = enabled.strip().lower()
                if v in ("true", "1", "on", "yes"):
                    enabled = True
                elif v in ("false", "0", "off", "no"):
                    enabled = False
                else:
                    return {"success": False, "reason": "debug_enabled must be true or false", **self.state()}
            self.enabled = bool(enabled)
        return {"success": True, **self.state()}

    def state(self) -> dict:
        return {"footprint_debug_enabled": self.enabled}


class FootprintDebugDrawer:
    """Draws a DebugFrame with CARLA's debug helper. Never raises."""

    LOG_EVERY_S = 10.0

    def __init__(self, world, lifetime_s: float = DEFAULT_LIFETIME_S):
        self.world = world
        self.lifetime_s = float(lifetime_s)
        self.failures = 0
        self.frames_drawn = 0
        self._last_log = 0.0

    # -- colours: (r, g, b); converted lazily so this module never imports carla at load
    BODY = (40, 200, 255)          # cyan: the physical van
    PROTECTED = (255, 210, 40)     # yellow: body + safety margin
    SWEEP = (120, 120, 255)        # blue-violet: where the protected body will be
    OBSTACLE = (255, 140, 0)       # orange: stationary object being checked
    JUNCTION = (150, 150, 150)     # grey: near a junction, judged by the old rules
    HIT = (255, 30, 30)            # red: contact
    TEXT = (255, 255, 255)

    def note_failure(self, error) -> None:
        self.failures += 1
        now = time.time()
        if now - self._last_log >= self.LOG_EVERY_S:
            self._last_log = now
            print(f"[FootprintDebug] drawing failed ({self.failures} so far): {error}")

    def draw(self, frame: DebugFrame, z: float = 0.1) -> bool:
        """Draw one frame. Returns True if every call succeeded."""
        try:
            import carla  # local import: the stack has it; offline tests stub it
            dbg = self.world.debug
            color_cls = getattr(carla, "Color", None)

            def col(rgb):
                return color_cls(*rgb) if color_cls is not None else None

            def loc(x, y, dz=0.0):
                return carla.Location(x=float(x), y=float(y), z=float(z + dz))

            def poly(corners, rgb, thickness):
                for i in range(len(corners)):
                    a, b = corners[i], corners[(i + 1) % len(corners)]
                    kw = dict(thickness=thickness, life_time=self.lifetime_s)
                    c = col(rgb)
                    if c is not None:
                        kw["color"] = c
                    dbg.draw_line(loc(*a), loc(*b), **kw)

            poly(frame.body, self.BODY, 0.06)
            poly(frame.protected, self.PROTECTED, 0.04)
            for rect in frame.stations:
                poly(rect, self.SWEEP, 0.02)
            for (ox, oy, radius, tag) in frame.obstacles:
                rgb = self.HIT if tag == "hit" else self.JUNCTION if tag.startswith("junction") else self.OBSTACLE
                kw = dict(size=0.15, life_time=self.lifetime_s)
                c = col(rgb)
                if c is not None:
                    kw["color"] = c
                dbg.draw_point(loc(ox, oy, 0.2), **kw)
                if radius > 0.05:
                    # an approximate circle: 8 short lines
                    ring = [(ox + radius * math.cos(a), oy + radius * math.sin(a))
                            for a in [k * math.pi / 4 for k in range(8)]]
                    poly(ring, rgb, 0.02)
            if frame.conflict is not None:
                x, y, along = frame.conflict
                kw = dict(size=0.25, life_time=self.lifetime_s)
                c = col(self.HIT)
                if c is not None:
                    kw["color"] = c
                dbg.draw_point(loc(x, y, 0.3), **kw)
            # labels stacked above the van's centre
            if frame.body:
                cx = sum(p[0] for p in frame.body) / 4.0
                cy = sum(p[1] for p in frame.body) / 4.0
                for i, (kind, text) in enumerate(frame.texts):
                    rgb = self.HIT if ("CONFLICT" in text or "BLOCKED" in text) else self.TEXT
                    kw = dict(draw_shadow=False, life_time=self.lifetime_s)
                    c = col(rgb)
                    if c is not None:
                        kw["color"] = c
                    dbg.draw_string(loc(cx, cy, 2.6 - 0.35 * i), text, **kw)
            self.frames_drawn += 1
            return True
        except Exception as e:      # drawing must never touch autonomy
            self.note_failure(e)
            return False
