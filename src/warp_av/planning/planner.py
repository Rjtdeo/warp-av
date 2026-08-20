"""
Route Planner

YOUR ROVER equivalent:
    Your rover has no route planning — it just goes forward and avoids obstacles.

THIS VERSION:
    Uses CARLA's road network to plan a route from A to B.
    Outputs a list of waypoints the vehicle should follow.
    The controller steers toward the next waypoint.
"""

import carla
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0       # desired heading at this point
    speed: float = 8.0     # desired speed at this point
    is_junction: bool = False   # inside an intersection (from the CARLA map)


@dataclass
class Route:
    waypoints: List[Waypoint]
    total_distance: float = 0.0
    timestamp: float = field(default_factory=time.time)


class RoutePlanner:
    """
    Plans a route from current position to destination.
    Uses CARLA's GlobalRoutePlanner for road-aware routing.
    """

    def __init__(self, carla_map, sampling_resolution=2.0):
        self.carla_map = carla_map

        # Use CARLA's built-in route planner
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        self._grp = GlobalRoutePlanner(self.carla_map, sampling_resolution)

        self._enabled = True

    def plan_route(self, start_x, start_y, end_x, end_y) -> Optional[Route]:
        """
        Plan a route between two points.
        Returns a list of waypoints the vehicle should follow.
        """
        if not self._enabled:
            print("[Planner] DISABLED — cannot plan route")
            return None

        try:
            start_loc = carla.Location(x=start_x, y=start_y, z=0)
            end_loc = carla.Location(x=end_x, y=end_y, z=0)

            # Get route from CARLA's planner
            route = self._grp.trace_route(start_loc, end_loc)

            if not route:
                print("[Planner] No route found!")
                return None

            waypoints = []
            total_dist = 0.0
            prev = None

            for wp, road_option in route:
                point = Waypoint(
                    x=wp.transform.location.x,
                    y=wp.transform.location.y,
                    z=wp.transform.location.z,
                    yaw=math.radians(wp.transform.rotation.yaw),
                    is_junction=bool(getattr(wp, "is_junction", False))
                )
                waypoints.append(point)

                if prev:
                    dx = point.x - prev.x
                    dy = point.y - prev.y
                    total_dist += math.sqrt(dx**2 + dy**2)
                prev = point

            print(f"[Planner] Route planned: {len(waypoints)} waypoints, {total_dist:.0f}m")
            return Route(waypoints=waypoints, total_distance=total_dist)

        except Exception as e:
            print(f"[Planner] Route planning failed: {e}")
            return None

    # --- Curve-aware speed (Troy #2/#3: left & right turns) ---
    A_LAT_MAX = 1.3     # m/s^2 comfortable lateral accel for a cargo van (higher clipped kerbs)
    A_DECEL = 1.2       # m/s^2 gentle pre-corner deceleration (earlier slowdown)
    V_TURN_MIN = 2.5    # m/s never asked to go slower than this for a bend
    CURVE_HORIZON_M = 30.0

    def curve_speed_cap(self, route: Route, current_x, current_y, cruise=8.0) -> float:
        """
        How fast may we go RIGHT NOW given the bends in the next 30 m?

        For each upcoming waypoint: local curvature (heading change / distance)
        gives a comfortable in-turn speed v_turn = sqrt(a_lat / curvature); a
        bend d metres away allows sqrt(v_turn^2 + 2*a_decel*d) now — so the cap
        tightens gradually as the corner approaches instead of braking late.
        """
        if not route or len(route.waypoints) < 3:
            return cruise

        # locate ourselves on the route (same scan as get_next_waypoint)
        ci, cd = 0, float("inf")
        for i, wp in enumerate(route.waypoints):
            d = math.hypot(wp.x - current_x, wp.y - current_y)
            if d < cd:
                cd, ci = d, i

        cap = cruise
        dist = 0.0
        prev = route.waypoints[ci]
        for i in range(ci + 1, len(route.waypoints) - 1):
            a, b, c = route.waypoints[i - 1], route.waypoints[i], route.waypoints[i + 1]
            seg = math.hypot(b.x - a.x, b.y - a.y)
            dist += seg
            if dist > self.CURVE_HORIZON_M:
                break
            h1 = math.atan2(b.y - a.y, b.x - a.x)
            h2 = math.atan2(c.y - b.y, c.x - b.x)
            dh = abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
            step = max(0.5, math.hypot(c.x - b.x, c.y - b.y))
            curvature = dh / step
            if curvature < 1e-3:        # straight enough
                continue
            v_turn = max(self.V_TURN_MIN, math.sqrt(self.A_LAT_MAX / curvature))
            # subtract the controller's coast band so actual speed (which rides
            # ~0.8 m/s above a falling target) meets v_turn AT the bend
            allowed_now = max(v_turn, math.sqrt(v_turn ** 2 + 2.0 * self.A_DECEL * max(0.0, dist)) - 0.8)
            cap = min(cap, allowed_now)
        return max(self.V_TURN_MIN, min(cruise, cap))

    TURN_ANGLE_RAD = 0.35   # ~20 deg heading change across a junction = a turn

    def upcoming_turn(self, route: Route, current_x, current_y, horizon_m=20.0):
        """
        Is there a TURN at a junction within `horizon_m` along the route?
        Returns {"distance_m": d, "direction": "left"|"right"} or None.
        Direction uses CARLA's yaw convention (positive yaw change = right).
        """
        if not route or len(route.waypoints) < 3:
            return None
        wps = route.waypoints
        ci, cd = 0, float("inf")
        for i, wp in enumerate(wps):
            d = math.hypot(wp.x - current_x, wp.y - current_y)
            if d < cd:
                cd, ci = d, i

        dist = 0.0
        j_start = None
        for i in range(ci + 1, len(wps)):
            dist += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
            if dist > horizon_m and j_start is None:
                return None
            if wps[i].is_junction and j_start is None:
                j_start = i
                j_dist = dist
            if j_start is not None and not wps[i].is_junction:
                # heading change across the junction span
                dyaw = (wps[i].yaw - wps[max(0, j_start - 1)].yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(dyaw) < self.TURN_ANGLE_RAD:
                    j_start = None      # straight through — keep scanning
                    continue
                return {"distance_m": round(j_dist, 1),
                        "direction": "right" if dyaw > 0 else "left"}
        return None

    def distance_to_next_junction(self, route: Route, current_x, current_y, horizon_m=45.0):
        """Distance along the route to the first junction waypoint (turning or
        straight-through), or None. Used as the stop-line fallback at lights."""
        if not route or len(route.waypoints) < 2:
            return None
        wps = route.waypoints
        ci, cd = 0, float("inf")
        for i, wp in enumerate(wps):
            d = math.hypot(wp.x - current_x, wp.y - current_y)
            if d < cd:
                cd, ci = d, i
        if wps[ci].is_junction:
            return 0.0
        dist = 0.0
        for i in range(ci + 1, len(wps)):
            dist += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
            if dist > horizon_m:
                return None
            if wps[i].is_junction:
                return round(dist, 1)
        return None

    def signed_cross_track(self, route: Route, current_x, current_y) -> float:
        """
        Signed lateral offset of the vehicle from the route polyline.
        Positive = left of the path direction (right-handed frame, consistent
        with the atan2/yaw math used everywhere else). Used by the controller's
        centreline-correction term.
        """
        if not route or len(route.waypoints) < 2:
            return 0.0
        best_d2 = float("inf")
        best_sign = 0.0
        wps = route.waypoints
        for i in range(len(wps) - 1):
            ax, ay = wps[i].x, wps[i].y
            bx, by = wps[i + 1].x, wps[i + 1].y
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            if L2 < 1e-9:
                continue
            t = max(0.0, min(1.0, ((current_x - ax) * dx + (current_y - ay) * dy) / L2))
            cx, cy = ax + t * dx, ay + t * dy
            d2 = (current_x - cx) ** 2 + (current_y - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                cross = dx * (current_y - ay) - dy * (current_x - ax)
                best_sign = 1.0 if cross > 0 else -1.0
        return best_sign * math.sqrt(best_d2)

    def get_next_waypoint(
        self,
        route: Route,
        current_x,
        current_y,
        lookahead=5.0
    ) -> Optional[Waypoint]:
        """
        Find a waypoint ahead of the vehicle.

        First find the route point closest to the vehicle.
        Then only search FORWARD from that point.
        This prevents steering back toward old waypoints.
        """

        if not route or not route.waypoints:
            return None

        # Find where we currently are on the route.
        closest_index = 0
        closest_distance = float("inf")

        for i, wp in enumerate(route.waypoints):
            dx = wp.x - current_x
            dy = wp.y - current_y
            distance = math.sqrt(dx ** 2 + dy ** 2)

            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        # Walk FORWARD along the route accumulating arc length, and return the
        # point exactly `lookahead` metres along the road (interpolated).
        # The old version returned the first waypoint at a straight-line
        # distance >= lookahead — in a bend that point is much further around
        # the corner, so the van aimed across the inside and clipped kerbs.
        acc = 0.0
        for i in range(closest_index + 1, len(route.waypoints)):
            a = route.waypoints[i - 1]
            b = route.waypoints[i]
            seg = math.hypot(b.x - a.x, b.y - a.y)
            if seg <= 1e-6:
                continue
            if acc + seg >= lookahead:
                t = (lookahead - acc) / seg
                return Waypoint(
                    x=a.x + (b.x - a.x) * t,
                    y=a.y + (b.y - a.y) * t,
                    z=a.z + (b.z - a.z) * t,
                    yaw=b.yaw,
                )
            acc += seg

        # Near destination, use final waypoint.
        return route.waypoints[-1]

    def distance_to_destination(self, route: Route, current_x, current_y) -> float:
        """How far to the end of the route."""
        if not route or not route.waypoints:
            return 999.0
        last = route.waypoints[-1]
        dx = last.x - current_x
        dy = last.y - current_y
        return math.sqrt(dx**2 + dy**2)

    def disable(self):
        self._enabled = False
        print("[Planner] DISABLED")

    def enable(self):
        self._enabled = True
        print("[Planner] Re-enabled")
