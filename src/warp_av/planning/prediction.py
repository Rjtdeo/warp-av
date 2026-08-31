"""
Motion prediction: guess where every MOVING object will be over the next few
seconds, and warn when a guessed position lands on OUR path before we have
cleared it.

Design rules (anti-phantom-braking):
- Only objects moving faster than 0.8 m/s are predicted (parked cars and
  tracker jitter never trigger anything).
- Only objects currently OUTSIDE our corridor are considered: a lead car
  driving ahead of us lives inside the corridor and is the car-following
  logic's job. Prediction exists for CROSSERS and CUT-INS — things that are
  not a problem yet but will be.
- A conflict needs both sides of the appointment to match: the object's
  guessed position must be on our path AND we must be near that spot at
  around that time.

Pure math, no CARLA — the same code runs in offline tests.
"""
from __future__ import annotations

import math

HORIZON_S = 3.5          # how far into the future we guess
STEP_S = 0.5             # guessing resolution
MIN_SPEED = 0.8          # slower than this = treated as standing
CORRIDOR_M = 2.2         # same body-width band the live corridor uses
MEET_WINDOW_M = 7.0      # how close our arrival must be to count as a meeting


def _project(px, py, wps, n):
    """(arc position along the route polyline, lateral distance)."""
    best_d2, best_arc = float("inf"), 0.0
    arc = 0.0
    for i in range(n - 1):
        ax, ay = wps[i].x, wps[i].y
        bx, by = wps[i + 1].x, wps[i + 1].y
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 > 1e-9:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d2:
                best_d2, best_arc = d2, arc + t * math.sqrt(L2)
        arc += math.sqrt(L2)
    return best_arc, math.sqrt(best_d2)


def predict_route_conflict(objects, route_wps, ego_x, ego_y, ego_yaw, ego_speed):
    """Earliest predicted conflict, or None.

    `objects` carry ego-frame x/y plus world-frame vx_world/vy_world.
    Returns {"t": s, "along_m": m ahead of us, "type": str, "id": int}.
    """
    n = len(route_wps)
    if n < 2 or not objects:
        return None
    ego_arc, _ = _project(ego_x, ego_y, route_wps, n)
    cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)

    best = None
    for obj in objects:
        vx = getattr(obj, "vx_world", 0.0)
        vy = getattr(obj, "vy_world", 0.0)
        if math.hypot(vx, vy) < MIN_SPEED:
            continue
        # ego frame -> world (same transform the corridor uses)
        wx = ego_x + cos_y * obj.x - sin_y * obj.y
        wy = ego_y + sin_y * obj.x + cos_y * obj.y
        arc_now, lat_now = _project(wx, wy, route_wps, n)
        if lat_now <= CORRIDOR_M and arc_now - ego_arc > -1.0:
            continue          # already in our corridor: following logic's job
        t = STEP_S
        while t <= HORIZON_S:
            px, py = wx + vx * t, wy + vy * t
            arc_p, lat_p = _project(px, py, route_wps, n)
            along = arc_p - ego_arc
            if lat_p <= CORRIDOR_M and 0.0 < along <= 35.0:
                # will WE be near that spot around then?
                ours = ego_speed * t
                if abs(along - ours) <= MEET_WINDOW_M + 0.3 * ego_speed:
                    cand = {"t": round(t, 1), "along_m": round(along, 1),
                            "type": getattr(obj.object_type, "value", str(obj.object_type)),
                            "id": getattr(obj, "id", 0)}
                    if best is None or cand["t"] < best["t"]:
                        best = cand
                    break
            t += STEP_S
    return best
