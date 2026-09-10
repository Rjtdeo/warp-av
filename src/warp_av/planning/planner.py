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

from .footprint import VehicleFootprint, sweep_conflict
from .instrumentation import (PlannerDecision, debug_planning_enabled,
                              CLEAR, NO_ROUTE, BLOCKED_TRACKED_OBJECT,
                              BLOCKED_SWEPT_PATH, BLOCKED_SCRAPE, BLOCKED_VRU)

# Planning V2 (phase 1B): perception objects carry no size, so when the swept
# body is checked against a STATIONARY object it is given a radius by type.
# Half a car for vehicles; a bin/pole/planter for other things.
DEFAULT_OBSTACLE_RADIUS_M = {"vehicle": 0.9, "pedestrian": 0.4, "obstacle": 0.5, "unknown": 0.5}
FOOTPRINT_STATIONARY_REACH_M = 12.0   # sweep decides hard-blocks for stationary objects this far ahead

VAN_HALF_WIDTH_M = 0.99               # the CARLA Sprinter, measured (planning/footprint_config.py)
VAN_HALF_LENGTH_M = 2.96              # ... and its nose, that far ahead of the point it steers about
MAX_BLOCK_HALFWIDTH_M = 2.20          # nothing beyond this is looked at at all, so no band may exceed it


def obstacle_radius_m(obj) -> float:
    """How much room to leave around one object.

    Perception measures this per object and puts it on every DetectedObject as
    `clearance_radius_m` -- its own footprint, but never less than its kind needs. The
    planner used to look for an attribute called `radius`, which nothing sets, and so
    always fell back to the table below. That table is also meaner than perception:
    it allowed a pedestrian 0.4 m where perception asks for 0.6 m.
    """
    measured = getattr(obj, "clearance_radius_m", None)
    if measured is None:
        measured = getattr(obj, "radius", None)
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    fallback = DEFAULT_OBSTACLE_RADIUS_M.get(kind, 0.5)
    try:
        measured = float(measured)
    except (TypeError, ValueError):
        return fallback
    if measured != measured or measured <= 0.0:      # NaN or nonsense
        return fallback
    return max(measured, fallback)


#: whose room is allowed to widen the band that stops the van. Only people.
WIDENS_THE_BAND = ("pedestrian", "cyclist")

# How wide a thing's BODY is, for deciding whether the van would scrape it. This is not the
# same question as how much room to leave around it, and using the clearance figure for both
# is what made the van stop for kerbs.
#
# obstacle_radius_m answers "how much room does this deserve", and floors an unnamed lump at
# 0.4 m for safety. That floor is right for keeping a polite distance and wrong for asking
# "would I hit it": a kerb sliver measured 0.3 x 0.1 m does not become 0.8 m wide because the
# van is being careful. Measured live in Town10HD on 2026-09-10 with the road empty, the van
# stopped on 18% of frames for stationary things 1.36 to 2.20 m off its line -- railings,
# posts and kerb, none of them on any road, all of them things it passes every day.
#
# So the scrape test uses what was MEASURED, with a floor only where under-measurement is both
# likely and expensive. The LiDAR sees one face of a car and can report it 1.8 x 0.5 m, so a
# vehicle keeps a floor near its real half-width; an unnamed lump is taken at its measured size.
SCRAPE_HALF_WIDTH_FLOOR_M = {"vehicle": 0.90, "pedestrian": 0.30, "cyclist": 0.30}
DEFAULT_SCRAPE_HALF_WIDTH_M = 0.15    # an unnamed lump is measured, not assumed to be car-sized
SCRAPE_MARGIN_M = 0.25                # how close its body may come to ours before we stop


def scrape_half_width_m(obj):
    """Half the body of a thing, for asking whether the van would touch it -- or None.

    The half-diagonal, so it is an over-estimate whichever way the thing is turned: a car
    measured 4.5 x 1.8 m comes out at 2.42 m and blocks readily, which is the point.

    None means NOTHING WAS MEASURED, and the caller must then do what the van always did and
    stop. Not knowing how big something is has never been a reason to drive at it.
    """
    length = getattr(obj, "length_m", None) or 0.0
    width = getattr(obj, "width_m", None) or 0.0
    try:
        length, width = float(length), float(width)
    except (TypeError, ValueError):
        return None
    if length != length or width != width or (length <= 0.0 and width <= 0.0):
        return None                                 # never measured: assume the worst
    measured = 0.5 * math.hypot(length, width)
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    return max(measured, SCRAPE_HALF_WIDTH_FLOOR_M.get(kind, DEFAULT_SCRAPE_HALF_WIDTH_M))


#: How far a thing whose size was never measured is assumed to reach. Half a car's length:
#: enough that an unmeasured lump beside the driver's door still stops the van, and not so
#: much that one well behind the tail does. Not knowing how big something is has never been
#: a reason to drive at it.
UNMEASURED_REACH_M = 2.5


def reach_toward_us_m(obj) -> float:
    """How far this thing's BODY might extend back toward our bumper.

    The half-diagonal of what was measured, so it over-estimates whichever way the thing is
    turned -- which is the safe direction. This is a geometric question and takes the
    measured body, not obstacle_radius_m: that one answers "how much room does it deserve",
    floors an unnamed lump at 0.4 m for politeness, and using it here read a nine-metre
    lorry as half a metre long.
    """
    half = scrape_half_width_m(obj)
    return UNMEASURED_REACH_M if half is None else max(half, 0.0)


def would_scrape(obj, lat_m: float) -> bool:
    """Would this thing's body reach the van's, at this distance off the line?"""
    half = scrape_half_width_m(obj)
    if half is None:
        return True                                 # size unknown -> treat it as in the way
    return (lat_m - half) < (VAN_HALF_WIDTH_M + SCRAPE_MARGIN_M)


def block_band_m(obj, floor_m: float) -> float:
    """How far off the line an object can sit and still stop us.

    For a person or someone on a bicycle: the van's own half-width plus the room
    perception says they need. That is 1.59 m for a walker and 1.79 m for a rider, against
    the single 1.40 m everything used to get -- and perception had been asking for 0.6 m
    and 0.8 m of room all along while the planner allowed 0.4 m.

    For everything else the band is left exactly as it was, on purpose. The 1.40 m figure
    for vehicles is not an oversight: a car waiting at a cross-street stop line just round
    the corner sits about 1.6 m off our line, and widening the band freezes the mission
    every time one does. A stationary vehicle out there is already caught by the wide-body
    rule below, which is what that rule is for. Tried it the other way first; two tests
    that exist precisely to catch this said no.

    The band can only ever grow, and never past the range the corridor looks at.
    """
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    if kind not in WIDENS_THE_BAND:
        return floor_m
    return min(MAX_BLOCK_HALFWIDTH_M, max(floor_m, VAN_HALF_WIDTH_M + obstacle_radius_m(obj)))


def _int_or_none(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Waypoint:
    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0       # desired heading at this point
    speed: float = 8.0     # desired speed at this point
    is_junction: bool = False   # inside an intersection (from the CARLA map)
    # Which piece of road and which lane this point sits on. CARLA gives both and this code
    # used to throw them away, which is why a traffic light could not be tied to OUR lane --
    # the van had to wait until it was 2 m from the stop line for the simulator to admit a
    # light governed it. None when the route was not built from a CARLA map.
    road_id: Optional[int] = None
    lane_id: Optional[int] = None


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
        # Planning V2 phase 0: what the last path judgement decided and on what evidence.
        # Read by main.py for the telemetry. Set on every call to filter_to_route_corridor,
        # including the early returns -- a decision that was never made must not silently
        # read as the previous tick's.
        self.last_decision = PlannerDecision()
        self.carla_map = carla_map

        # Use CARLA's built-in route planner
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        self._grp = GlobalRoutePlanner(self.carla_map, sampling_resolution)

        self._enabled = True
        # Planning V2 feature flag. OFF: filter_to_route_corridor behaves exactly
        # as before. ON: pass self.footprint to it and stationary obstacles are
        # hard-blocked by the swept van body instead of the 1.40/2.20 m bands.
        # Nothing reads this yet (phase 1B); main.py still calls without a footprint.
        self.use_footprint_blocking = False
        self.footprint = VehicleFootprint()

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
                    is_junction=bool(getattr(wp, "is_junction", False)),
                    # kept so a traffic light can be matched to this exact lane
                    road_id=_int_or_none(getattr(wp, "road_id", None)),
                    lane_id=_int_or_none(getattr(wp, "lane_id", None)),
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
        wps = route.waypoints           # snapshot: writers swap, never mutate
        if not route or len(wps) < 3:
            return cruise

        # locate ourselves on the route (same scan as get_next_waypoint)
        ci, cd = 0, float("inf")
        for i, wp in enumerate(wps):
            d = math.hypot(wp.x - current_x, wp.y - current_y)
            if d < cd:
                cd, ci = d, i

        cap = cruise
        dist = 0.0
        prev = wps[ci]
        for i in range(ci + 1, len(wps) - 1):
            a, b, c = wps[i - 1], wps[i], wps[i + 1]
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

    # --- Parking / pull-over (Troy #7) ---
    PARK_BLEND_M = 15.0        # length of the pull-over ramp before the spot
    PARK_CURB_MARGIN_M = 1.2   # keep the van's centre this far off the lane edge
    PARK_MAX_PULLBACK_M = 40.0 # may park up to this far BEFORE a pin that sits in a bend/junction

    def _straight_run_before(self, wps, idx, need_m):
        """Is there >= need_m of straight, non-junction road ending at wps[idx]?"""
        if wps[idx].is_junction:
            return False
        # the anchor itself must be locally straight (not the first point of a bend)
        if idx > 0 and abs((wps[idx - 1].yaw - wps[idx].yaw + math.pi) % (2 * math.pi) - math.pi) > math.radians(5):
            return False
        run = 0.0
        for i in range(idx, 0, -1):
            dyaw = abs((wps[i - 1].yaw - wps[idx].yaw + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > math.radians(20) or wps[i - 1].is_junction:
                return run >= need_m
            run += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
            if run >= need_m:
                return True
        return run >= need_m

    def _right_bay(self, x, y, z):
        """Is there a Parking/Shoulder bay to the right of the driving lane at
        this point? Returns (bx, by, byaw_rad, width) or None. CARLA-only."""
        wp = self.carla_map.get_waypoint(
            carla.Location(x=x, y=y, z=z), project_to_road=True,
            lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        for _ in range(4):      # slide to the rightmost same-direction driving lane
            r = wp.get_right_lane()
            if (r is not None and r.lane_type == carla.LaneType.Driving
                    and abs((r.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180) < 60):
                wp = r
            else:
                break
        bay = wp.get_right_lane()
        if (bay is not None
                and bay.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder)
                and bay.lane_width >= 1.8):
            t = bay.transform
            return (t.location.x, t.location.y, math.radians(t.rotation.yaw), bay.lane_width)
        return None

    def _find_bay_anchor(self, wps):
        """Scan the final PARK_MAX_PULLBACK_M of the route (nearest-to-pin
        first) for a point with a real stopping bay to the right AND enough
        straight road behind it to blend in. Returns (index, target) or None."""
        arc = 0.0
        for i in range(len(wps) - 1, 2, -1):
            if i < len(wps) - 1:
                arc += math.hypot(wps[i + 1].x - wps[i].x, wps[i + 1].y - wps[i].y)
            if arc > self.PARK_MAX_PULLBACK_M:
                return None
            if not self._straight_run_before(wps, i, 6.0):
                continue
            bay = self._right_bay(wps[i].x, wps[i].y, wps[i].z)
            if bay is not None:
                bx, by, byaw, bw = bay
                off = math.hypot(bx - wps[i].x, by - wps[i].y)
                return i, (bx, by, byaw, off)
        return None

    def apply_pullover(self, route: Route, side="right"):
        """
        Bend the end of the route so the mission finishes at the kerb on the
        right-hand side (or a real Parking/Shoulder lane if the map has one)
        instead of dead-centre on the road.

        If the pin itself sits in a bend or junction, park like a driver would:
        at the kerb on the nearest STRAIGHT stretch before it (up to 40 m back,
        the route is truncated there). Returns {"x","y","yaw","offset_m",
        "moved_back_m"} or None when no safe spot exists within the pullback.
        """
        if not route or len(route.waypoints) < 4:
            return None
        wps = route.waypoints

        # Step 0: PREFER a real stopping bay (parking/shoulder strip beyond the
        # lane line) anywhere in the last 40 m — park fully OFF the driving lane.
        bay_target = None
        kind = "kerb"
        try:
            found = self._find_bay_anchor(wps)
        except Exception:
            found = None
        if found is not None:
            a, bay_target = found
            kind = "bay"
            moved_back = 0.0
            for i in range(a + 1, len(wps)):
                moved_back += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
        else:
            # Step 1: kerb-hug fallback — last waypoint with >=6 m of straight
            # road behind it, at most PARK_MAX_PULLBACK_M before the pin.
            moved_back = 0.0
            a = len(wps) - 1
            while a > 2 and moved_back <= self.PARK_MAX_PULLBACK_M:
                if self._straight_run_before(wps, a, 6.0):
                    break
                moved_back += math.hypot(wps[a].x - wps[a - 1].x, wps[a].y - wps[a - 1].y)
                a -= 1
            else:
                return None
            if a <= 2 or moved_back > self.PARK_MAX_PULLBACK_M:
                return None
        if a < len(wps) - 1:
            route.waypoints = wps = wps[:a + 1]   # mission now ends at the bay / before the bend
        last = wps[-1]

        # Step 2: how much straight tail do we have to blend over?
        usable = 0.0
        i0 = len(wps) - 1
        for i in range(len(wps) - 1, 0, -1):
            dyaw = abs((wps[i - 1].yaw - last.yaw + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > math.radians(20) or wps[i - 1].is_junction:
                break
            usable += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
            i0 = i - 1
            if usable >= self.PARK_BLEND_M:
                break
        if usable < 6.0:
            return None

        if bay_target is not None:
            tx, ty, tyaw, off = bay_target
        else:
            tx, ty, tyaw, off = self._pullover_target(last)
        if off <= 0.1:
            return None      # nowhere to pull over (very narrow lane)

        self._blend_tail_to(route, i0, tx, ty, tyaw, usable)
        return {"x": round(tx, 2), "y": round(ty, 2), "yaw": round(tyaw, 3),
                "offset_m": round(off, 2), "moved_back_m": round(moved_back, 1), "kind": kind}

    def _blend_tail_to(self, route: Route, i0: int, tx, ty, tyaw, usable):
        """Replace the route tail after index i0 with a smooth ramp to the
        target, finishing with a straight-in section so the vehicle arrives
        parallel (shared by kerbside pull-over and slot parking)."""
        wps = list(route.waypoints)     # snapshot; writer swaps atomically at the end
        last = wps[-1]
        p0 = wps[i0]
        h = last.yaw
        fwd = (math.cos(h), math.sin(h))
        right = (-math.sin(h), math.cos(h))     # CARLA frame: right of the heading
        dx, dy = tx - p0.x, ty - p0.y
        along = dx * fwd[0] + dy * fwd[1]
        lat = dx * right[0] + dy * right[1]
        straight_in = min(9.5, max(0.0, along - 6.0))
        cut = max(0.5, along - straight_in)
        K = max(6, int(usable / 2.0))
        new_tail = []
        for k in range(1, K + 1):
            a = k / K
            t = min(1.0, (a * along) / cut)
            smooth = t * t * (3 - 2 * t)        # smoothstep: no lateral jerk
            new_tail.append(Waypoint(
                x=p0.x + fwd[0] * (a * along) + right[0] * (smooth * lat),
                y=p0.y + fwd[1] * (a * along) + right[1] * (smooth * lat),
                z=last.z, yaw=tyaw))
        # atomic swap: the 10 Hz tick thread may be iterating the old list
        route.waypoints = wps[:i0 + 1] + new_tail

    # ---------------- FIND PARKING: explicit van-sized slots ----------------
    SLOT_LEN_M = 7.0

    def find_parking_slots(self, route: Route, search_back_m=70.0):
        """Slice the parking bays along the final stretch of the route into
        van-sized slot rectangles. Returns a list ordered far -> near the
        destination: {x, y, yaw, length, width, corners: [[x,y]*4]}."""
        if not route or len(route.waypoints) < 4:
            return []
        wps = route.waypoints
        # collect bay centreline points alongside the route tail (route order)
        arc_from_end = [0.0] * len(wps)
        for i in range(len(wps) - 2, -1, -1):
            arc_from_end[i] = arc_from_end[i + 1] + math.hypot(
                wps[i + 1].x - wps[i].x, wps[i + 1].y - wps[i].y)
        # bay points near junctions are unusable (crossings, building access,
        # curved corner sections) — mask them out so no slot can exist there
        near_junction = set()
        for i, wp in enumerate(wps):
            if wp.is_junction:
                for j in range(max(0, i - 3), min(len(wps), i + 4)):
                    near_junction.add(j)
        bay_pts = []
        for i, wp in enumerate(wps):
            if arc_from_end[i] > search_back_m:
                continue
            bay = None
            if i not in near_junction:
                try:
                    bay = self._right_bay(wp.x, wp.y, wp.z)
                except Exception:
                    bay = None
            bay_pts.append(bay)      # None marks gaps

        slots = []
        run = []
        for b in bay_pts + [None]:
            if b is not None:
                run.append(b)
                continue
            if len(run) >= 2:
                slots.extend(self._slice_run_into_slots(run))
            run = []
        return slots

    SLOT_MAX_CURVE_RAD = 0.14   # ~8 deg heading spread across a slot = too curved

    def _slice_run_into_slots(self, run):
        """run = consecutive (x, y, yaw, width) bay points along the road."""
        arcs = [0.0]
        for a, b in zip(run, run[1:]):
            arcs.append(arcs[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        seg_yaws = [math.atan2(b[1] - a[1], b[0] - a[0]) for a, b in zip(run, run[1:])]
        total = arcs[-1]
        n = int(total // self.SLOT_LEN_M)
        out = []
        for k in range(n):
            mid = (k + 0.5) * self.SLOT_LEN_M
            # a slot must sit on a STRAIGHT piece of bay: parking tilted on a
            # curved corner section is exactly what a driver would never do
            lo, hi = mid - self.SLOT_LEN_M / 2.0, mid + self.SLOT_LEN_M / 2.0
            span = [y_ for y_, a0, a1 in zip(seg_yaws, arcs, arcs[1:])
                    if a1 >= lo and a0 <= hi]
            if span:
                ref = span[0]
                spread = max(abs((y_ - ref + math.pi) % (2 * math.pi) - math.pi) for y_ in span)
                if spread > self.SLOT_MAX_CURVE_RAD:
                    continue
            # interpolate centre + heading at arc position `mid`
            for i in range(len(arcs) - 1):
                if arcs[i + 1] >= mid:
                    seg = max(1e-6, arcs[i + 1] - arcs[i])
                    t = (mid - arcs[i]) / seg
                    x = run[i][0] + (run[i + 1][0] - run[i][0]) * t
                    y = run[i][1] + (run[i + 1][1] - run[i][1]) * t
                    yaw = math.atan2(run[i + 1][1] - run[i][1], run[i + 1][0] - run[i][0])
                    width = run[i][3]
                    break
            else:
                continue
            fwd = (math.cos(yaw), math.sin(yaw))
            right = (-math.sin(yaw), math.cos(yaw))
            hl, hw = self.SLOT_LEN_M / 2.0, width / 2.0
            corners = [[round(x + sx * fwd[0] * hl + sy * right[0] * hw, 2),
                        round(y + sx * fwd[1] * hl + sy * right[1] * hw, 2)]
                       for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
            out.append({"x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 3),
                        "length": self.SLOT_LEN_M, "width": round(width, 2),
                        "corners": corners})
        return out

    @staticmethod
    def point_in_slot(px, py, slot, inflate=0.0):
        dx, dy = px - slot["x"], py - slot["y"]
        c, s_ = math.cos(-slot["yaw"]), math.sin(-slot["yaw"])
        lx = dx * c - dy * s_
        ly = dx * s_ + dy * c
        return (abs(lx) <= slot["length"] / 2.0 + inflate
                and abs(ly) <= slot["width"] / 2.0 + inflate)

    @staticmethod
    def van_in_slot(vx, vy, vyaw, half_len, half_wid, slot):
        """(inside, margin_along_m, margin_side_m) for the van's rectangle."""
        c, s_ = math.cos(vyaw), math.sin(vyaw)
        worst_lx = worst_ly = 0.0
        for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            px = vx + sx * c * half_len - sy * s_ * half_wid
            py = vy + sx * s_ * half_len + sy * c * half_wid
            dx, dy = px - slot["x"], py - slot["y"]
            cs, ss = math.cos(-slot["yaw"]), math.sin(-slot["yaw"])
            lx = abs(dx * cs - dy * ss)
            ly = abs(dx * ss + dy * cs)
            worst_lx = max(worst_lx, lx)
            worst_ly = max(worst_ly, ly)
        m_along = slot["length"] / 2.0 - worst_lx
        m_side = slot["width"] / 2.0 - worst_ly
        return (m_along >= 0 and m_side >= 0, round(m_along, 2), round(m_side, 2))

    @staticmethod
    def choose_free_slot(slots):
        """Best slot = nearest the destination that is FREE and whose
        PREDECESSOR is also free (the approach ramp sweeps through it).
        Fallback: any free slot. Returns an index or None."""
        for i in range(len(slots) - 1, -1, -1):
            if not slots[i]["occupied"] and (i == 0 or not slots[i - 1]["occupied"]):
                return i
        for i in range(len(slots) - 1, -1, -1):
            if not slots[i]["occupied"]:
                return i
        return None

    def retarget_to_slot(self, route: Route, slot):
        """Trim the route beside the chosen slot and blend into it."""
        wps = route.waypoints
        if len(wps) < 6:
            return False
        ci = min(range(len(wps)),
                 key=lambda i: math.hypot(wps[i].x - slot["x"], wps[i].y - slot["y"]))
        if ci < 4:
            return False
        trimmed = wps[:ci + 1]
        trimmed[-1] = Waypoint(x=trimmed[-1].x, y=trimmed[-1].y,
                               z=trimmed[-1].z, yaw=slot["yaw"])
        route.waypoints = trimmed          # atomic swap
        i0 = max(0, len(trimmed) - 8)
        self._blend_tail_to(route, i0, slot["x"], slot["y"], slot["yaw"], usable=14.0)
        return True

    def _pullover_target(self, last: Waypoint):
        """Kerb-side point for the final stop. Uses the CARLA map when
        available (rightmost driving lane edge, or a Parking/Shoulder lane);
        falls back to pure geometry 1.2 m right of the final waypoint."""
        try:
            wp = self.carla_map.get_waypoint(
                carla.Location(x=last.x, y=last.y, z=last.z),
                project_to_road=True, lane_type=carla.LaneType.Driving)
            # walk to the rightmost same-direction driving lane
            for _ in range(4):
                r = wp.get_right_lane()
                if (r is not None and r.lane_type == carla.LaneType.Driving
                        and abs((r.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180) < 60):
                    wp = r
                else:
                    break
            park = wp.get_right_lane()
            if (park is not None and park.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder)
                    and park.lane_width > 2.2):
                t = park.transform
                return (t.location.x, t.location.y, math.radians(t.rotation.yaw), park.lane_width / 2.0)
            t = wp.transform
            rv = t.get_right_vector()
            edge = max(0.0, wp.lane_width / 2.0 - self.PARK_CURB_MARGIN_M)
            return (t.location.x + rv.x * edge, t.location.y + rv.y * edge,
                    math.radians(t.rotation.yaw), edge)
        except Exception:
            edge = 1.2
            return (last.x - math.sin(last.yaw) * edge,
                    last.y + math.cos(last.yaw) * edge, last.yaw, edge)

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

    def filter_to_route_corridor(self, perception, route: Route, ego_x, ego_y, ego_yaw,
                                 corridor_halfwidth_m=1.75, block_halfwidth_m=1.40,
                                 danger_m=8.0, max_ahead_m=50.0, footprint=None):
        """
        Recompute perception's "in my path" verdict against the ROUTE CORRIDOR
        instead of a straight box along the vehicle's nose.

        Mid-turn the nose points across neighbouring lanes, so the ego-frame box
        flags vehicles that are not on our path (false stop) and misses
        obstacles around the bend (late stop). Here an object counts only if it
        lies within corridor_halfwidth of the route polyline AND ahead of us
        along the route. Mutates and returns the PerceptionOutput.

        footprint (Planning V2, optional): a VehicleFootprint. When given, a
        STATIONARY object on a non-junction stretch hard-blocks only if the
        van's swept body (footprint plus safety margin, slid along the route)
        touches it within FOOTPRINT_STATIONARY_REACH_M, instead of the
        1.40 m / 2.20 m centre-line bands. Everything else - moving objects,
        pedestrians, junction segments, the slow zone, what counts as
        "closest" - is unchanged. None (the default) = exactly the old rules.
        """
        if not route or len(route.waypoints) < 2:
            self.last_decision = PlannerDecision(reason=NO_ROUTE)
            return perception
        if not getattr(perception, "objects", None):
            self.last_decision = PlannerDecision(reason=CLEAR,
                                                 route_points_used=len(route.waypoints))
            return perception

        wps = route.waypoints
        n = len(wps)

        def arc_pos(px, py):
            """(arc-length along route of nearest point, lateral distance,
            nearest segment index)."""
            best_d2, best_arc, best_i = float("inf"), 0.0, 0
            arc = 0.0
            for i in range(n - 1):
                ax, ay, bx, by = wps[i].x, wps[i].y, wps[i + 1].x, wps[i + 1].y
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                seg = math.sqrt(L2) if L2 > 1e-9 else 0.0
                if seg > 0:
                    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                    cx, cy = ax + t * dx, ay + t * dy
                    d2 = (px - cx) ** 2 + (py - cy) ** 2
                    if d2 < best_d2:
                        best_d2, best_arc, best_i = d2, arc + t * seg, i
                arc += seg
            return best_arc, math.sqrt(best_d2), best_i

        ego_arc, _, _ = arc_pos(ego_x, ego_y)
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
        # how far ahead of the steering point the van's nose is: its real body when the
        # caller gave us one, otherwise the measured Sprinter
        front_bumper_m = (footprint.half_length if footprint is not None else VAN_HALF_LENGTH_M)

        closest = 999.0
        closest_type = perception.closest_obstacle_type
        closest_speed = 0.0
        closest_lat = None
        blocked = False
        found = False
        # --- phase 0 bookkeeping. Records what was decided; decides nothing. -------------
        want_detail = debug_planning_enabled()
        seen = 0            # objects the corridor actually looked at
        in_corridor = 0     # ... of those, how many sat inside the slow band
        why = None          # which rule said "blocked", for the first blocker found
        blocker = None      # and the object it said it about
        detail = []

        def _note_block(rule, obj_, along_, lat_):
            """Remember the NEAREST blocker and the rule that caught it."""
            nonlocal why, blocker
            if blocker is None or max(0.0, along_) < max(0.0, blocker[1]):
                why, blocker = rule, (obj_, along_, lat_)

        for obj in perception.objects:
            seen += 1
            # ego frame (x fwd, y left) -> world
            wx = ego_x + cos_y * obj.x - sin_y * obj.y
            wy = ego_y + sin_y * obj.x + cos_y * obj.y
            oarc, lat, oseg = arc_pos(wx, wy)
            along = oarc - ego_arc
            # Wide-body check needs objects slightly beyond the slow corridor
            # too (centre at 1.75-2.20 m still overlaps the van's swept width —
            # 2.20 covers SUV-class half-widths; run 77 clipped a parked
            # Patrol at ~2.1 m while sweeping through a bend).
            if along < -1.0 or along > max_ahead_m:
                continue
            near_junction = (wps[oseg].is_junction
                             or wps[min(oseg + 1, n - 1)].is_junction)
            stationary = getattr(obj, "speed", 0.0) < 0.5
            # Is any part of it at or ahead of the FRONT BUMPER? Driving forward cannot
            # reach anything behind that line.
            #
            # `along` is measured from the point the van steers about, which sits 2.96 m
            # behind its nose, and the loop keeps everything from 1 m behind that point.
            # So an object level with the driver's door counted as an obstacle "ahead", and
            # one beside the rear wheels was reported at 0.0 m -- max(0.0, along) clamped
            # it -- and hard-stopped the van. Seen live on 2026-09-10: eleven ticks in a row
            # blocked by object 3845 "at 0.0 m", with nothing in front of the van.
            #
            # The object's own size counts, so a lorry whose middle is level with our door
            # but whose nose reaches past our bumper still blocks.
            nose_gap = (along - front_bumper_m) + reach_toward_us_m(obj)
            reaches_our_nose = nose_gap > 0.0
            # Planning V2: does the swept body decide this object's hard-block?
            sweep_decides = (footprint is not None and stationary and not near_junction)
            # Old rules never look beyond 2.20 m from the line. The swept body
            # can reach further in a bend (the outer corner swings wide), so in
            # footprint mode a stationary object is kept for the sweep up to
            # the van's own reach; the sweep itself decides precisely.
            lat_limit = (footprint.swept_half_length + 1.0) if sweep_decides else 2.20
            if lat > lat_limit:
                continue
            if want_detail:
                detail.append({"id": int(getattr(obj, "id", 0) or 0),
                               "kind": getattr(getattr(obj, "object_type", None), "value", None),
                               "along_m": round(along, 1), "lateral_m": round(lat, 2),
                               "stationary": bool(stationary),
                               "near_junction": bool(near_junction)})
            if lat <= corridor_halfwidth_m:
                in_corridor += 1
                found = True
                dist = max(0.0, along)
                if dist < closest:
                    closest = dist
                    closest_type = obj.object_type
                    closest_speed = obj.speed
                    closest_lat = round(lat, 2)
                # Two-tier: only an object near the path CENTRE can stop us (a
                # real lead vehicle sits at 0-0.8 m). The 1.4-1.75 m band —
                # e.g. a car waiting at the cross-street stop line just around
                # the corner — slows us but must not freeze the mission.
                # How wide a band stops us is the object's own, not one figure for
                # everything (Perception V2 day 11). Perception already measures how much
                # room each thing needs -- 1.2 m for a car, 0.8 m for a cyclist, 0.6 m for
                # a person, 0.4 m for a bin -- and until now nothing read it. The band can
                # only ever GROW: a bin keeps today's 1.40 m, a person gains 0.19 m, and
                # nothing anywhere gets a narrower band than it had before.
                if (dist < danger_m and lat <= block_band_m(obj, block_halfwidth_m)
                        and not sweep_decides and reaches_our_nose):
                    blocked = True
                    _note_block(BLOCKED_TRACKED_OBJECT, obj, along, lat)
            # Physical-width conflict: centre-line thresholds ignore that the
            # van (~2.0 m) plus a parked car (~1.8 m) cannot share 2×1.75 m.
            # A STATIONARY body whose centre sits in the 1.40-2.05 m band
            # close ahead on a straight would be scraped — stop instead of
            # squeezing (a parked mini in a narrow bay was hit at 2.8 m/s in
            # sweep run 62). Moving traffic and the central corridor keep the
            # existing rules (following/lead logic), and junction-adjacent
            # objects stay exempt: cross-street geometry is the give-way
            # logic's job (re-blocking it was the original false-stop bug).
            if (lat > block_halfwidth_m and lat <= 2.20
                    and stationary
                    and max(0.0, along) < 12.0 and not near_junction):
                # SEEING it is unconditional -- the slow-zone bookkeeping wants everything
                # out here, whatever its size.
                found = True
                dist = max(0.0, along)
                if dist < closest:
                    closest = dist
                    closest_type = obj.object_type
                    closest_speed = obj.speed
                    closest_lat = round(lat, 2)
                # STOPPING for it needs its body to actually reach ours. The old rule stopped
                # for anything in this band, which on a normal street means stopping for the
                # kerb: measured live, 18% of frames on an empty road, for railings and posts
                # 1.36 to 2.20 m off the line and not on any road at all. See would_scrape.
                if not sweep_decides and would_scrape(obj, lat) and reaches_our_nose:
                    blocked = True
                    _note_block(BLOCKED_SCRAPE, obj, along, lat)
            # Planning V2: the van's real body, slid along the route, decides
            # whether a stationary object is in the way. A parked car 1.6 m off
            # the line still blocks (half a car reaches into our margin); a
            # planter at 1.9 m no longer does; a body 2.4 m off the line on the
            # outside of a bend is caught when the corner sweeps over it.
            if sweep_decides and max(0.0, along) < FOOTPRINT_STATIONARY_REACH_M:
                radius = obstacle_radius_m(obj)
                hit = sweep_conflict(wps, (ego_x, ego_y), footprint, (wx, wy),
                                     obstacle_radius=radius,
                                     horizon_m=FOOTPRINT_STATIONARY_REACH_M + footprint.swept_half_length)
                if hit is not None:
                    found = True
                    dist = max(0.0, along)
                    if dist < closest:
                        closest = dist
                        closest_type = obj.object_type
                        closest_speed = obj.speed
                        closest_lat = round(lat, 2)
                    blocked = True
                    _note_block(BLOCKED_SWEPT_PATH, obj, along, lat)

        perception.closest_obstacle_distance = closest
        perception.closest_obstacle_speed = closest_speed
        perception.closest_obstacle_lateral_m = closest_lat
        perception.path_blocked = blocked
        if found:
            perception.closest_obstacle_type = closest_type

        # --- phase 0: say what was decided and on what evidence. Reads the same state the
        # lines above just wrote; it cannot and must not change any of it.
        decision = PlannerDecision(reason=CLEAR,
                                   objects_considered=seen,
                                   objects_in_corridor=in_corridor,
                                   route_points_used=n,
                                   candidates=detail)
        if blocked and blocker is not None:
            obj_, along_, lat_ = blocker
            kind = getattr(getattr(obj_, "object_type", None), "value",
                           str(getattr(obj_, "object_type", "")) or None)
            # A person or someone riding is worth its own reason: "the van stopped" and
            # "the van stopped FOR A PERSON" are different lines in a report.
            reason = BLOCKED_VRU if kind in ("pedestrian", "cyclist") else why
            decision.reason = reason
            decision.blocker_id = int(getattr(obj_, "id", 0) or 0) or None
            decision.blocker_kind = kind
            decision.blocker_distance_m = max(0.0, along_)
            decision.blocker_lateral_m = lat_
            decision.used_footprint = (why == BLOCKED_SWEPT_PATH)
        elif blocked:
            # blocked with nothing recorded should be impossible; say so rather than
            # reporting a clear path.
            decision.reason = BLOCKED_TRACKED_OBJECT
        self.last_decision = decision
        return perception

    def blend_departure(self, route: Route, ego_x, ego_y):
        """Ease OUT of a parking bay at mission start: the route begins on
        the lane centre while the van sits metres off in its bay, and pure
        pursuit answers with near-full lock and an overshoot across the
        lane (operator: "it drives between lanes at the start"; max steer
        0.996 recorded). Decay the initial lateral offset smoothly over the
        first stretch instead. Returns True when a blend was applied."""
        wps = route.waypoints
        if len(wps) < 8:
            return False
        d0x, d0y = wps[1].x - wps[0].x, wps[1].y - wps[0].y
        seg = math.hypot(d0x, d0y)
        if seg < 1e-6:
            return False
        d0x, d0y = d0x / seg, d0y / seg
        ex, ey = ego_x - wps[0].x, ego_y - wps[0].y
        # +ve = ego sits to the ROUTE's right (CARLA left-handed frame)
        lat0 = d0x * ey - d0y * ex
        if abs(lat0) < 1.5 or abs(lat0) > 8.0:
            return False                     # already in lane / implausible
        blend_len = max(8.0, min(14.0, abs(lat0) * 2.2))
        new_head = []
        arc = 0.0
        i = 0
        for i in range(len(wps) - 1):
            wp = wps[i]
            if arc >= blend_len:
                break
            t = arc / blend_len
            s = t * t * (3 - 2 * t)          # smoothstep
            off = lat0 * (1.0 - s)
            right = (-math.sin(wp.yaw), math.cos(wp.yaw))
            new_head.append(Waypoint(
                x=wp.x + right[0] * off, y=wp.y + right[1] * off,
                z=wp.z, yaw=wp.yaw, speed=wp.speed, is_junction=wp.is_junction))
            arc += math.hypot(wps[i + 1].x - wp.x, wps[i + 1].y - wp.y)
        # atomic swap: the 10 Hz tick thread may be iterating the old list
        route.waypoints = new_head + wps[i:]
        return True

    OVERTAKE_SHIFT_M = 3.6       # one lane to the LEFT around the dead car
    OVERTAKE_PASS_M = 8.0        # stay shifted this far beyond the obstacle
    OVERTAKE_REJOIN_M = 16.0     # fully back in lane this far beyond it

    def plan_overtake(self, route: Route, ego_x, ego_y, obstacle_along_m,
                      lane_ok=None):
        """Rewrite the route to swing one lane LEFT around a dead vehicle
        ahead and rejoin beyond it (straights only: refuses near junctions
        or in bends). `lane_ok(x, y)` must confirm the shifted position is
        on a real driving lane. Returns the rejoin point (Waypoint) or None
        when the geometry does not allow a safe pass."""
        wps = route.waypoints
        n = len(wps)
        if n < 10 or obstacle_along_m is None:
            return None
        ci = min(range(n), key=lambda i: math.hypot(wps[i].x - ego_x,
                                                    wps[i].y - ego_y))
        # cumulative arc from the ego's nearest waypoint
        arcs = [0.0]
        for i in range(ci, n - 1):
            arcs.append(arcs[-1] + math.hypot(wps[i + 1].x - wps[i].x,
                                              wps[i + 1].y - wps[i].y))
        # Full offset only PAST the car's centre: the whole approach is ramp,
        # putting ~2.7 m of clearance at its rear corner already (v1 clipped
        # the corner by demanding a full lane change inside 6 m).
        shift_done = obstacle_along_m + 1.0
        pass_end = obstacle_along_m + self.OVERTAKE_PASS_M
        rejoin = obstacle_along_m + self.OVERTAKE_REJOIN_M
        if arcs[-1] < rejoin + 3.0:
            return None                       # destination too close — hold
        # straight-and-open guard over the whole detour region
        for k, i in enumerate(range(ci, min(ci + len(arcs), n))):
            if arcs[k] > rejoin + 2.0:
                break
            if wps[i].is_junction:
                return None
            dyaw = abs((wps[i].yaw - wps[ci].yaw + math.pi) % (2 * math.pi) - math.pi)
            if dyaw > math.radians(14):
                return None                   # bend — sight lines too poor
        new_tail = []
        rejoin_wp = None
        for k, i in enumerate(range(ci, n)):
            a = arcs[k] if k < len(arcs) else arcs[-1]
            if a <= shift_done:
                t = a / shift_done
            elif a <= pass_end:
                t = 1.0
            elif a <= rejoin:
                t = (rejoin - a) / max(0.5, rejoin - pass_end)
            else:
                t = 0.0
            s = t * t * (3 - 2 * t)           # smoothstep, no lateral jerk
            wp = wps[i]
            right = (-math.sin(wp.yaw), math.cos(wp.yaw))
            off = -self.OVERTAKE_SHIFT_M * s  # minus right-vector = LEFT
            nx, ny = wp.x + right[0] * off, wp.y + right[1] * off
            if s > 0.5 and lane_ok is not None and k % 4 == 0:
                if not lane_ok(nx, ny):
                    return None               # no drivable lane to borrow
            new_tail.append(Waypoint(x=nx, y=ny, z=wp.z, yaw=wp.yaw,
                                     speed=wp.speed, is_junction=wp.is_junction))
            if rejoin_wp is None and a >= rejoin:
                rejoin_wp = new_tail[-1]
        if rejoin_wp is None:
            rejoin_wp = new_tail[-1]
        # atomic swap: the 10 Hz tick thread may be iterating the old list
        route.waypoints = wps[:ci] + new_tail
        return rejoin_wp

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

        wps = route.waypoints           # snapshot: writers swap, never mutate
        if not route or not wps:
            return None

        # Find where we currently are on the route.
        closest_index = 0
        closest_distance = float("inf")

        for i, wp in enumerate(wps):
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
        for i in range(closest_index + 1, len(wps)):
            a = wps[i - 1]
            b = wps[i]
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

        # Near the destination the route runs out before the lookahead: aim at
        # a virtual point extended past the end along the final heading, so the
        # vehicle ALIGNS with the parking direction instead of beelining
        # diagonally at the endpoint.
        last = wps[-1]
        ext = max(0.0, lookahead - acc)
        if ext > 0.1:
            # direction from the last real segment (yaw fields can be unset/stale)
            hd = last.yaw
            for j in range(len(wps) - 2, -1, -1):
                pv = wps[j]
                if math.hypot(last.x - pv.x, last.y - pv.y) > 0.3:
                    hd = math.atan2(last.y - pv.y, last.x - pv.x)
                    break
            return Waypoint(x=last.x + math.cos(hd) * ext,
                            y=last.y + math.sin(hd) * ext,
                            z=last.z, yaw=hd)
        return last

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
