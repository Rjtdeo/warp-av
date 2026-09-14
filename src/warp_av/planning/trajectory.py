"""
The next few seconds, written down: a minimal trajectory (Planning V2 task 3).

Until now the van had "aim there, go this fast": one target point and one speed, decided
afresh every tick, and nothing that said where the van would be in two seconds. Nothing
could be checked against that. The off-route body sweep (task 1) had to assume "straight
ahead", which is wrong exactly when the van is steering back to its lane.

This is the smallest thing that answers the question. Each tick, from what the van already
decides: the path the controller would drive -- pure pursuit, aiming at the route the
distance ahead it aims now, stepped along until the horizon -- and along it the speed the
behaviour asked for, held to the bend cap and to comfortable acceleration and braking, and
the time each point is reached. Pure geometry, no CARLA.

It predicts, it does not command: the controller still steers at the aim point and the
behaviour still sets the speed. It is for anything that must ask "what happens if the van
keeps doing what it is doing" -- the off-route body sweep reads it now; the ground under a
pass path and time-to-collision (tasks 4 and 5) can read it next.

Frame: CARLA's. yaw turns +x towards +y, so positive curvature turns the van to its RIGHT.
The geometry always runs to the horizon; a stop only sends the speed to zero along it, so a
van told to stop still knows what lies on the path it would take when it moves again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .footprint import _polyline, _project, _point_at_arc

#: Comfortable pull-away: the controller lets the speed target rise 0.15 m/s per 0.1 s tick
#: (VehicleController.SPEED_SLEW_UP), which is this.
ACCEL_MPS2 = 1.5
#: The Sprinter's steering lock in CARLA: 70 deg on a 3.66 m wheelbase (measured 2026-09-14
#: from vehicle.get_physics_control), a kinematic turning radius of 1.33 m. Far tighter than
#: any real van; the arc practically never meets it. Kept so an aim point behind the van
#: cannot ask for a spiral.
MAX_CURVATURE = 0.75
HORIZON_M = 25.0
STEP_M = 0.5
#: Time is counted at no less than this speed: a standing van has no future at 0 m/s.
V_FLOOR_MPS = 0.2


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class TrajectoryPoint:
    s: float            # metres along the path from the van's centre
    x: float
    y: float
    yaw: float          # radians, CARLA frame
    curvature: float    # 1/m; positive turns the van to its right
    speed: float        # m/s the van intends to be doing here
    t: float            # seconds from now


@dataclass
class Trajectory:
    points: List[TrajectoryPoint]
    aim_xy: Optional[Tuple[float, float]] = None
    #: where the speed reaches zero along the path, when it does
    stops_at_m: Optional[float] = None

    @property
    def length_m(self) -> float:
        return self.points[-1].s if self.points else 0.0

    @property
    def duration_s(self) -> float:
        return self.points[-1].t if self.points else 0.0

    def xy(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self.points]

    def at_time(self, t: float) -> Optional[TrajectoryPoint]:
        """The first point reached at or after `t` seconds; the last one when `t` is beyond."""
        for p in self.points:
            if p.t >= t:
                return p
        return self.points[-1] if self.points else None

    def as_dict(self, every_m: float = 2.0) -> dict:
        """For the API: one point every `every_m` metres plus the last, rounded."""
        rows, last_bin = [], None
        for i, p in enumerate(self.points):
            b = int(p.s // every_m)
            if b != last_bin or i == len(self.points) - 1:
                last_bin = b
                rows.append({"s": round(p.s, 1), "x": round(p.x, 2), "y": round(p.y, 2),
                             "yaw_deg": round(math.degrees(p.yaw), 1),
                             "curvature": round(p.curvature, 3),
                             "speed": round(p.speed, 2), "t": round(p.t, 2)})
        return {"points": rows, "length_m": round(self.length_m, 1),
                "duration_s": round(self.duration_s, 1),
                "stops_at_m": None if self.stops_at_m is None else round(self.stops_at_m, 1),
                "aim": None if self.aim_xy is None else [round(self.aim_xy[0], 2), round(self.aim_xy[1], 2)]}


#: The controller's aim point is never closer than this (main.py: lookahead = max(5.0, ...)).
AIM_FLOOR_M = 5.0


def _geometry(x, y, yaw, route_pts, aim_xy, max_curvature, horizon_m, step_m):
    """(s, x, y, yaw, curvature) samples of the path the controller would drive.

    Pure pursuit, step by step, the way the controller does it: at every step aim at the
    point on the route a fixed distance ahead of where we are (the distance the controller
    is using now: from the van to its aim point), turn on the circle through that point,
    advance one step, aim again. A van off its line rejoins it smoothly; a van on it
    follows it. Beyond the route's end, or without a route: straight on."""
    out = [(0.0, x, y, yaw, 0.0)]
    poly = _polyline(route_pts or [])
    s, px, py, ph = 0.0, x, y, yaw
    if len(poly) >= 2:
        L = AIM_FLOOR_M
        if aim_xy is not None:
            L = max(2.5, math.hypot(float(aim_xy[0]) - x, float(aim_xy[1]) - y))
        # only the piece of route we can reach: from just behind us to horizon + aim ahead
        _, _, seg0 = _project(x, y, poly)
        lo = max(0, seg0 - 1)
        hi, reach = lo + 1, 0.0
        while hi < len(poly) - 1 and reach < horizon_m + L + 10.0:
            reach += math.hypot(poly[hi + 1][0] - poly[hi][0], poly[hi + 1][1] - poly[hi][1])
            hi += 1
        poly = poly[lo:hi + 1]
        total = sum(math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
                    for i in range(len(poly) - 1))
        while s < horizon_m - 1e-6:
            a_here, _, _ = _project(px, py, poly)
            if a_here >= total - 1e-6:
                break                              # the route has run out: straight on below
            ax, ay, _ = _point_at_arc(min(total, a_here + L), poly)
            d = math.hypot(ax - px, ay - py)
            if d < 0.5:
                kappa = 0.0
            else:
                alpha = _wrap(math.atan2(ay - py, ax - px) - ph)
                kappa = max(-max_curvature, min(max_curvature, 2.0 * math.sin(alpha) / d))
            step = min(step_m, horizon_m - s)
            if abs(kappa) < 1e-9:
                px, py = px + step * math.cos(ph), py + step * math.sin(ph)
            else:
                ph_new = ph + kappa * step
                px += (math.sin(ph_new) - math.sin(ph)) / kappa
                py -= (math.cos(ph_new) - math.cos(ph)) / kappa
                ph = _wrap(ph_new)
            s += step
            out.append((s, px, py, ph, kappa))
    while s < horizon_m - 1e-6:                    # no route (left): straight on
        step = min(step_m, horizon_m - s)
        s += step
        px, py = px + step * math.cos(ph), py + step * math.sin(ph)
        out.append((s, px, py, ph, 0.0))
    return out


def _speeds(geom, v0, target, accel, decel, lat_accel, v_turn_min):
    """The speed at every sample, and where it reaches zero (None when it does not)."""
    n = len(geom)
    v0 = max(0.0, float(v0))
    if target <= 0.05:
        # asked to stop: brake comfortably from where we are
        v = [math.sqrt(max(0.0, v0 * v0 - 2.0 * decel * g[0])) for g in geom]
        stop = 0.0 if v0 <= 0.0 else v0 * v0 / (2.0 * decel)
        return v, stop
    cap = []
    for g in geom:
        k = abs(g[4])
        c = target
        if k > 1e-3:
            c = min(c, max(v_turn_min, math.sqrt(lat_accel / k)))
        cap.append(c)
    v = [0.0] * n
    v[0] = v0
    for i in range(1, n):                          # pull away no harder than accel
        ds = geom[i][0] - geom[i - 1][0]
        v[i] = min(cap[i], math.sqrt(v[i - 1] ** 2 + 2.0 * accel * ds))
    for i in range(n - 2, -1, -1):                 # ...and slow in time for what is ahead
        ds = geom[i + 1][0] - geom[i][0]
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * decel * ds))
    v[0] = v0                                      # now is a fact, not an intention
    return v, None


def plan_trajectory(pose_xy_yaw, speed_now: float, route_pts: Sequence, aim_xy, desired_speed: float,
                    *, accel: float = ACCEL_MPS2, decel: Optional[float] = None,
                    lat_accel: Optional[float] = None, v_turn_min: Optional[float] = None,
                    max_curvature: float = MAX_CURVATURE, horizon_m: float = HORIZON_M,
                    step_m: float = STEP_M) -> Trajectory:
    """The van's intended path and speed over the next `horizon_m` metres.

    pose_xy_yaw    where the van's centre is and which way it faces (CARLA frame, radians)
    speed_now      m/s now
    route_pts      the route ahead (waypoints or (x, y) pairs); may be empty
    aim_xy         the controller's aim point on it, or None. Its distance from the van is the
                   aim distance the path is driven with (AIM_FLOOR_M without one); without a
                   route the path is straight on
    desired_speed  what the behaviour asked for this tick; 0 or less means stop
    decel, lat_accel, v_turn_min default to the planner's own bend-cap numbers
    (RoutePlanner.A_DECEL, A_LAT_MAX, V_TURN_MIN), so a bend slows this the way it slows the van.
    """
    if decel is None or lat_accel is None or v_turn_min is None:
        from .planner import RoutePlanner           # here, not at the top: that module wants carla
        decel = RoutePlanner.A_DECEL if decel is None else decel
        lat_accel = RoutePlanner.A_LAT_MAX if lat_accel is None else lat_accel
        v_turn_min = RoutePlanner.V_TURN_MIN if v_turn_min is None else v_turn_min
    x, y, yaw = float(pose_xy_yaw[0]), float(pose_xy_yaw[1]), float(pose_xy_yaw[2])
    geom = _geometry(x, y, yaw, route_pts, aim_xy, max_curvature, horizon_m, step_m)
    v, stop = _speeds(geom, speed_now, float(desired_speed), accel, decel, lat_accel, v_turn_min)
    pts, t = [], 0.0
    for i, (s, px, py, ph, k) in enumerate(geom):
        if i > 0:
            ds = s - geom[i - 1][0]
            t += ds / max(0.5 * (v[i - 1] + v[i]), V_FLOOR_MPS)
        pts.append(TrajectoryPoint(s=s, x=px, y=py, yaw=ph, curvature=k, speed=v[i], t=t))
    return Trajectory(points=pts, aim_xy=(None if aim_xy is None else (float(aim_xy[0]), float(aim_xy[1]))),
                      stops_at_m=stop)
