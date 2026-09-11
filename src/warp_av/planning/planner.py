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
from dataclasses import dataclass, field, replace
from typing import List, Optional

from .footprint import VehicleFootprint, ObstacleBox, sweep_conflict
from .instrumentation import (PlannerDecision, debug_planning_enabled,
                              CLEAR, NO_ROUTE, BLOCKED_TRACKED_OBJECT,
                              BLOCKED_SWEPT_PATH, BLOCKED_SCRAPE, BLOCKED_VRU)

# Planning V2 (phase 1B): perception objects carry no size, so when the swept
# body is checked against a STATIONARY object it is given a radius by type.
# Half a car for vehicles; a bin/pole/planter for other things.
DEFAULT_OBSTACLE_RADIUS_M = {"vehicle": 0.9, "pedestrian": 0.4, "obstacle": 0.5, "unknown": 0.5}
FOOTPRINT_STATIONARY_REACH_M = 12.0   # sweep decides hard-blocks for stationary objects this far ahead
# ...and never further off the line than this. The swept body exists to catch what the
# centre-line bands miss: "a parked car 1.6 m off the line still blocks, a planter at 1.9 m
# no longer does, a body 2.4 m off the line on the outside of a bend is caught". So 2.4 m is
# the case it was written for, and this is that plus a little.
#
# Without a bound it reaches much further, because the box is 3.26 m half-LENGTH: its CORNER
# sits 3.51 m from its centre, and on a curving path that corner sweeps an arc wider still.
# Measured, with the bound off:
#
#     bend radius   how far off the line a "hit" could be
#        20 m                    2.70 m
#        12 m                    3.60 m
#         8 m                    5.90 m
#
# Live, that meant blocked_swept_path on 97% of ticks, the van never above 0.35 m/s, and the
# things blocking it 4.2 m off the centreline -- kerb and street furniture it drives past
# every day. A first attempt bounded this by JUNCTIONS instead, on the theory that turns are
# where the arc tightens. It did not help: the road bends without being flagged a junction,
# so the sweep was still active through the turn. The honest bound is on the answer itself.
SWEEP_MAX_LATERAL_M = 2.6

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


# How much to trust a measured heading. The tracker keeps the yaw from the sighting whose
# length was nearest the median, in the van's frame AT THAT SIGHTING, so if the van has turned
# since, the object's heading is off by that much. Rather than pretend the heading is exact,
# the box is widened by what an error this size would swing its ends through. For a 3.5 m
# kerb strip that is 0.24 m -- plenty, and still a tenth of the 1.75 m the circle claimed.
YAW_TOLERANCE_DEG = 8.0
#: ... for a rectangle perception FITTED (tracking.fit_rectangle) and kept on the map frame,
#: the heading is far better: measured live on a parked car passed at 0.55 m, 268 readings,
#: edge direction off by 0.5 deg at the median and 2.7 deg at the 90th percentile (the
#: spread-of-points heading: 16.6 and 21.0). 5 deg covers ~94% of single readings, and the
#: track takes its heading from its median-length sighting, not from a stray one.
YAW_TOLERANCE_FITTED_DEG = 5.0
#: Every side of an obstacle box is grown by this much. The LiDAR sees a thing's near face
#: only, so what it measures is a lower bound on the thing -- never let the box be smaller.
OBSTACLE_BOX_PAD_M = 0.10


# A kerb only needs TYRE clearance (fix 3, 2026-09-10). Something no taller than a kerb
# cannot touch the van's body -- the body passes over it -- only a tyre can, and the tyres
# sit inboard of the body's side. So a kerb-height thing is judged against the van's body
# plus KERB_CLEARANCE_M, not the full safety margin, and at its measured width.
#
# Why it matters: pulling into a parking slot, the kerb sits beside where the van will
# stand. Measured in 4 stalled scenarios: 0.5-0.6 m from the parked van's side to the kerb,
# against 0.55 m of padding the check piled onto it (0.30 m margin, 0.10 m pad, the 0.15 m
# width floor, 0.09 m for the heading) -- so the van waited at the kerb for ever.
#
# Why it is safe: kerb-like means no taller than KERB_LIKE_MAX_HEIGHT_M, standing still, and
# never named a person, cyclist or vehicle. In both static-truth recordings (18,000 blobs),
# nothing within 15 m of the van that measured 0.30 m or less was ever a person or vehicle
# -- the LiDAR sees them whole at that range. Anything taller keeps the full margin, and a
# kerb-height thing IN the path still blocks: tyre clearance is not zero.
KERB_LIKE_MAX_HEIGHT_M = 0.20
KERB_CLEARANCE_M = 0.10


def kerb_like(obj) -> bool:
    """Low enough that only a tyre could touch it, standing still, and not a road user."""
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    if kind in ("pedestrian", "cyclist", "vehicle"):
        return False
    try:
        height = float(getattr(obj, "height_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return False
    if not (0.0 < height <= KERB_LIKE_MAX_HEIGHT_M):
        return False
    return bool(getattr(obj, "stationary", False))


# Pass a parked vehicle with care instead of waiting behind it for ever (fix 2, 2026-09-10).
# A car parked on the shoulder, its middle 2.5-2.6 m off the van's line, held the van for
# minutes in two live runs. The real gap was about 0.7 m; the check added 0.30 m for a
# heading the car might have, 0.10 m because the LiDAR sees only its near side, and the
# van's own 0.30 m margin -- 0.70 m -- and called it a touch. The padding stays: it is
# right to be unsure of a car's heading. What changes is the answer when only the MARGIN
# is in the way: the van then passes with PASS_CLEARANCE_M instead of 0.30 m, slowly (it is
# reported as the nearest obstacle, so the behaviour's 3 m/s slow zone applies).
#
# Vehicles only, standing still. A person or anything unnamed keeps the full margin --
# unknown is not free. A static post keeps it too: it stands at the height of the van's
# mirrors, which reach past the body the footprint describes; a car's roof is below them.
PASS_CLEARANCE_M = 0.15


def can_pass_with_care(obj) -> bool:
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    return kind == "vehicle" and bool(getattr(obj, "stationary", False))


class WaitingIsPointless:
    """Blocked on the way into the parking spot by something that will not move: say when to
    stop waiting and finish there (fix 3).

    Seen 2026-09-10: 4 of 5 pedestrian scenarios ended with the van in front of a kerb piece
    for the rest of the run, "replan or operator action required". Waiting only makes sense
    for what can move. So: near the spot, blocked, by something static or kerb-height, for
    AFTER_S -> finish. A person or vehicle is never "will not move" -- the van keeps waiting,
    and the slot re-check deals with a car that took the spot.
    """
    WITHIN_M = 30.0     # only on the way into the spot
    AFTER_S = 6.0       # the behaviour already waits 3 s before it calls the route blocked
    # An unnamed thing that is not labelled static but has sat in the way this long is not
    # going anywhere either (a post inside the parking lane is never "static": the lane is
    # road, and road is where parked cars are). People and vehicles never count.
    STILL_UNNAMED_AFTER_S = 20.0
    # A moment of "clear" is the kerb's measured box flickering, not the way opening: seen
    # live, blocked 4.2 s, clear 1.1 s while the van crept 1.2 m, blocked again. Only a
    # longer clear restarts the clock.
    CLEAR_GRACE_S = 2.0

    def __init__(self):
        self.since = None
        self.clear_since = None

    @staticmethod
    def what_is_it(obj) -> str:
        if kerb_like(obj):
            return "kerb"
        return {"pole": "post", "structure": "wall", "low": "kerb"}.get(
            getattr(obj, "static_rule", "") or "", "fixed object")

    @staticmethod
    def will_not_move(obj) -> bool:
        return obj is not None and (getattr(obj, "motion_class", "dynamic") == "static" or kerb_like(obj))

    @staticmethod
    def can_move(obj) -> bool:
        """A person, a cyclist or a vehicle: always worth waiting for."""
        kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
        return kind in ("pedestrian", "cyclist", "vehicle")

    def update(self, blocked: bool, distance_to_spot, blocker, now: float):
        """What is in the way, once it is time to stop waiting -- else None."""
        near = distance_to_spot is not None and distance_to_spot <= self.WITHIN_M
        if near and blocked and blocker is not None and self.can_move(blocker):
            self.since = self.clear_since = None           # wait for it, however long
            return None
        if not (near and blocked and blocker is not None):
            if self.since is not None:
                if self.clear_since is None:
                    self.clear_since = now
                if now - self.clear_since >= self.CLEAR_GRACE_S or not near:
                    self.since = self.clear_since = None
            return None
        self.clear_since = None
        if self.since is None:
            self.since = now
            return None
        waited = now - self.since
        if self.will_not_move(blocker):
            return self.what_is_it(blocker) if waited >= self.AFTER_S else None
        if bool(getattr(blocker, "stationary", False)) and waited >= self.STILL_UNNAMED_AFTER_S:
            return "fixed object"
        return None


def obstacle_box_for(obj, ego_yaw: float):
    """The obstacle as the rectangle perception measured, in the world frame -- or None.

    None means the size was never measured, and the caller must fall back to the circle,
    which is the cautious answer: not knowing a thing's shape is not a reason to assume it
    is thin.

    Widths get the same floors as the scrape test, for the same reason. A car the laser sees
    end-on comes out 1.8 x 0.5 m, so a vehicle never gets a half-width under 0.9 m however
    thin it measured; a person never under 0.3 m.
    """
    # the smallest rectangle round its points when perception fitted one (fix 2): the spread
    # of a car's points seen from its corner runs diagonally and gets the heading wrong
    fitted = float(getattr(obj, "box_length_m", 0.0) or 0.0) > 0.0
    length = (getattr(obj, "box_length_m", None) if fitted else getattr(obj, "length_m", None)) or 0.0
    width = (getattr(obj, "box_width_m", None) if fitted else getattr(obj, "width_m", None)) or 0.0
    try:
        length, width = float(length), float(width)
    except (TypeError, ValueError):
        return None
    if length != length or width != width or (length <= 0.0 and width <= 0.0):
        return None
    kind = getattr(getattr(obj, "object_type", None), "value", "unknown")
    # a kerb-height thing is taken at its measured width: the floor is there because the
    # LiDAR can see a car end-on, and nothing kerb-height hides a car behind it
    floor = 0.0 if kerb_like(obj) else SCRAPE_HALF_WIDTH_FLOOR_M.get(kind, DEFAULT_SCRAPE_HALF_WIDTH_M)
    half_w = max(0.5 * width, floor)
    half_l = max(0.5 * length, half_w)
    # cover a heading error by what it would swing the ends through
    half_w += half_l * math.sin(math.radians(YAW_TOLERANCE_FITTED_DEG if fitted else YAW_TOLERANCE_DEG))
    yaw_obj = math.radians(float((getattr(obj, "box_yaw_deg", 0.0) if fitted
                                  else getattr(obj, "yaw_deg", 0.0)) or 0.0))
    return ObstacleBox(half_length=half_l + OBSTACLE_BOX_PAD_M,
                       half_width=half_w + OBSTACLE_BOX_PAD_M,
                       heading=ego_yaw + yaw_obj)


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
    PARK_CURB_MARGIN_M = 1.2   # keep the van's centre this far off the lane edge
    PARK_MAX_PULLBACK_M = 40.0 # may park up to this far BEFORE a pin that sits in a bend/junction

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

    # --- The way into a parking spot (2026-09-11) ---
    # Measured against CARLA on 2026-09-11: the ramp into a parking strip was squeezed into
    # whatever straight road was left. CARLA's route changes lane just before the pin, the
    # blend window reached back across that change, and one ramp had to move the van 6.5 m
    # sideways in about 6 m of road -- up to 57 degrees. The van cut in, its path met the
    # pavement kerb and it finished across the lane. Now the ramp has a fixed gentlest angle,
    # lies entirely in ONE lane before the spot, and when the road before the pin is too short
    # the spot moves along the strip, up to PARK_PAST_PIN_M past the pin, instead.
    PULL_IN_MAX_DEG = 15.0      # the steepest point of the planned ramp
    PULL_IN_STRAIGHT_M = 5.0    # then dead straight for this long, so it arrives parallel
    PULL_IN_MIN_RAMP_M = 6.0    # never shorter than this, however small the move
    LANE_TO_SPOT_MAX_M = 4.6    # the spot must be beside the lane the van is in: a bay next to a
                                # 3.5 m lane is ~3.05 m from its centre, the kerb of the lane
                                # beyond ~4.05 m; a bay two lanes over (6.55 m, 2026-09-11) is not
    SAME_LANE_TOL_M = 0.75      # a route point this far off the lane line is in another lane
    PULL_IN_SETTLE_M = 10.0     # before the ramp, this long in the lane after a lane change, bend
                                # or junction. Starting it right at CARLA's 2 m lane change stacked
                                # the two turns: 27 degrees in the closed-loop test, 12 without
    PARK_PAST_PIN_M = 80.0      # how far on past the pin a spot may be -- across a junction if need
                                # be: a pin on a stop line has none before it (2026-09-11, light 21)
    PARK_KERB_PENALTY_M = 15.0  # choosing: a kerb stop in the lane counts as this much further off
    PARK_LANE_PENALTY_M = 40.0  # ...and a plain stop in the lane as this much (bays are for parking)
    LANE_STOP_STRAIGHT_M = 6.0  # a stop in the lane needs this much straight lane before it...
    LANE_STOP_CLEAR_M = 15.0    # ...and no junction this close ahead: never on a stop line

    @classmethod
    def pull_in_ramp_m(cls, lateral_m: float) -> float:
        """How long a ramp moving the van `lateral_m` sideways must be, so its steepest point
        is PULL_IN_MAX_DEG. The ramp is a smoothstep: steepest in the middle, at 1.5 times the
        average slope."""
        return max(cls.PULL_IN_MIN_RAMP_M,
                   1.5 * abs(lateral_m) / math.tan(math.radians(cls.PULL_IN_MAX_DEG)))

    @classmethod
    def bay_needed_behind_m(cls, lateral_m: float) -> float:
        """How much bay must run back beside the ramp. The van's front corner leaves its lane a
        little under a third of the way down the ramp; from there on the bay must be there."""
        return 0.7 * cls.pull_in_ramp_m(lateral_m) + cls.PULL_IN_STRAIGHT_M

    def _same_lane_start(self, wps, i, need_m, settle_m=None):
        """Walk back from wps[i] while the route stays in the same lane -- on the line through
        wps[i] along its heading -- straight, and out of junctions. The index need_m back, if the
        lane also carries on settle_m (PULL_IN_SETTLE_M) further back, or the route begins in it;
        else None -- a bend, a junction, or a lane change comes first."""
        settle_m = self.PULL_IN_SETTLE_M if settle_m is None else settle_m
        a = wps[i]
        c, s = math.cos(a.yaw), math.sin(a.yaw)
        run, start = 0.0, None
        for j in range(i, 0, -1):
            p = wps[j - 1]
            if p.is_junction and start is None:
                return None            # the ramp itself never starts inside a junction...
            if abs(-(p.x - a.x) * s + (p.y - a.y) * c) > self.SAME_LANE_TOL_M:
                return None            # ...but straight on THROUGH one, the van is settled
            if abs((p.yaw - a.yaw + math.pi) % (2 * math.pi) - math.pi) > math.radians(10):
                return None
            run += math.hypot(wps[j].x - p.x, wps[j].y - p.y)
            if start is None and run >= need_m:
                start = j - 1
            if start is not None and run >= need_m + settle_m:
                return start
        return start          # the route itself begins in this lane: nothing to settle from

    def _pull_in_plan(self, wps, k, tx, ty, tyaw, needs_bay):
        """Can the van pull in to (tx, ty) beside route point k, gently, from its own lane?
        (k, ramp start index, tx, ty, tyaw, sideways m, ramp m) or None."""
        a = wps[k]
        c, s = math.cos(a.yaw), math.sin(a.yaw)
        lat = -(tx - a.x) * s + (ty - a.y) * c        # CARLA frame: + is to the right
        along = (tx - a.x) * c + (ty - a.y) * s
        if not (0.1 < lat <= self.LANE_TO_SPOT_MAX_M) or abs(along) > 2.0:
            return None
        if abs((tyaw - a.yaw + math.pi) % (2 * math.pi) - math.pi) > math.radians(10):
            return None
        ramp = self.pull_in_ramp_m(lat)
        i0 = self._same_lane_start(wps, k, ramp + self.PULL_IN_STRAIGHT_M)
        if i0 is None:
            return None
        if needs_bay and not self._bay_runs_back(wps, k, self.bay_needed_behind_m(lat)):
            return None
        return k, i0, tx, ty, tyaw, lat, ramp

    def extend_past_pin(self, route: Route, metres: float) -> int:
        """Carry the route on past its last point for up to `metres`, straight on -- through a
        junction too, on the branch that keeps the heading the pin had -- so a parking spot can
        be chosen past the pin when there is none before it. A pin on a junction's stop line
        (2026-09-11, light 21) has its nearest spot across the junction. Stops where the road
        turns away (over 30 degrees from the pin's heading) or ends. The lights on the way are
        found as for any route (the points keep their road and lane). CARLA map only; returns
        how many points it added."""
        cmap = getattr(self, "carla_map", None)
        if cmap is None or not route or not route.waypoints or metres <= 0:
            return 0
        last = route.waypoints[-1]
        wp = cmap.get_waypoint(carla.Location(x=last.x, y=last.y, z=last.z),
                               project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0

        def gap(a, b):
            return abs((a - b + 180) % 360 - 180)
        heading = wp.transform.rotation.yaw
        added, d = [], 0.0
        while d < metres:
            options = wp.next(2.0)
            if not options:
                break
            nxt = min(options, key=lambda n: gap(n.transform.rotation.yaw, heading))
            if (gap(nxt.transform.rotation.yaw, wp.transform.rotation.yaw) > 12
                    or gap(nxt.transform.rotation.yaw, heading) > 30):
                break
            wp, d = nxt, d + 2.0
            loc = wp.transform.location
            added.append(Waypoint(x=loc.x, y=loc.y, z=loc.z,
                                  yaw=math.radians(wp.transform.rotation.yaw), speed=last.speed,
                                  is_junction=bool(wp.is_junction),
                                  road_id=wp.road_id, lane_id=wp.lane_id))
        if added:
            route.waypoints = route.waypoints + added
        return len(added)

    def _bay_runs_back(self, wps, idx, need_m):
        """Does the bay beside the route carry on unbroken for need_m behind wps[idx]? The
        turn-in into a bay happens alongside it (APPROACH_BAY_BEHIND_M), so a short bay --
        one with kerb just behind the spot -- cannot be driven into forwards."""
        run = 0.0
        for i in range(idx, 0, -1):
            run += math.hypot(wps[i].x - wps[i - 1].x, wps[i].y - wps[i - 1].y)
            if wps[i - 1].is_junction:
                return False
            try:
                if self._right_bay(wps[i - 1].x, wps[i - 1].y, wps[i - 1].z) is None:
                    return False
            except Exception:
                return False
            if run >= need_m:
                return True
        return False

    def apply_pullover(self, route: Route, side="right", pin_index=None, avoid=(), ahead_of=None):
        """
        Choose where the mission ends, and bend the route into it. In order of preference:
          * "bay"  -- a real Parking/Shoulder strip beside the lane, pulled into gently
          * "kerb" -- the kerb edge of the lane, when there is no strip to use
          * "lane" -- straight in the lane, where a straight stretch allows and no junction is
                      just ahead (never on a stop line)
        Every option is scored by how far it is from the pin in a straight line -- how far to
        walk -- plus PARK_KERB_PENALTY_M / PARK_LANE_PENALTY_M, and the best one taken. Looked
        for from PARK_MAX_PULLBACK_M before the pin to PARK_PAST_PIN_M after it (the route must
        have been carried on past it: extend_past_pin). Every option ends STRAIGHT: the van
        never finishes part-way through a lane change (2026-09-11: 22 degrees across the line,
        called arrived).

        When nothing works, the route is cut back to the pin and None returned.

        `avoid`: spots already turned down (x, y) -- nothing within 6 m of one is chosen again.
        `ahead_of`: the van's (x, y) when choosing again on the way; the pull-in must start
        ahead of it, since it cannot back up to start one it has already driven past.
        """
        if not route or len(route.waypoints) < 4:
            return None
        wps = route.waypoints
        pin = len(wps) - 1 if pin_index is None else max(0, min(int(pin_index), len(wps) - 1))
        px, py = wps[pin].x, wps[pin].y
        arc = [0.0] * len(wps)
        for k in range(1, len(wps)):
            arc[k] = arc[k - 1] + math.hypot(wps[k].x - wps[k - 1].x, wps[k].y - wps[k - 1].y)
        window = [k for k in range(3, len(wps))
                  if -self.PARK_MAX_PULLBACK_M <= arc[k] - arc[pin] <= self.PARK_PAST_PIN_M
                  and not wps[k].is_junction]
        first_i0 = 1
        if ahead_of is not None:
            here = min(range(len(wps)), key=lambda k: (wps[k].x - ahead_of[0]) ** 2
                       + (wps[k].y - ahead_of[1]) ** 2)
            first_i0 = here + 2

        best = None                              # (score, plan, kind, strip width)

        def consider(plan, kind, penalty, width=None):
            nonlocal best
            if plan is None or plan[1] < first_i0:
                return
            if any(math.hypot(plan[2] - ax, plan[3] - ay) < 6.0 for ax, ay in avoid):
                return
            score = math.hypot(plan[2] - px, plan[3] - py) + penalty
            if best is None or score < best[0]:
                best = (score, plan, kind, width)

        for k in window:
            try:
                bay = self._right_bay(wps[k].x, wps[k].y, wps[k].z)
            except Exception:
                bay = None
            if bay is not None:
                consider(self._pull_in_plan(wps, k, bay[0], bay[1], bay[2], needs_bay=True),
                         "bay", 0.0, width=bay[3])
            tx, ty, tyaw, off = self._pullover_target(wps[k])
            if off > 0.1:
                consider(self._pull_in_plan(wps, k, tx, ty, tyaw, needs_bay=False),
                         "kerb", self.PARK_KERB_PENALTY_M)
            consider(self._lane_stop_plan(wps, k, arc), "lane", self.PARK_LANE_PENALTY_M)
        if best is None:
            if pin < len(wps) - 1:
                route.waypoints = wps[:pin + 1]  # nothing past the pin is wanted after all
            return None
        _, (k, i0, tx, ty, tyaw, lat, ramp), kind, width = best
        route.waypoints = wps[:k + 1]
        if kind != "lane":
            self._blend_tail_to(route, i0, tx, ty, tyaw, ramp_m=ramp)
        approach = ramp + self.PULL_IN_STRAIGHT_M if kind != "lane" else 0.0
        return {"x": round(tx, 2), "y": round(ty, 2), "yaw": round(tyaw, 3),
                "offset_m": round(lat, 2), "moved_back_m": round(max(0.0, arc[pin] - arc[k]), 1),
                "past_pin_m": round(max(0.0, arc[k] - arc[pin]), 1),
                "from_pin_m": round(math.hypot(tx - px, ty - py), 1), "ramp_m": round(ramp, 1),
                "approach_m": round(approach, 1),
                "ramp_start": [round(wps[i0].x, 2), round(wps[i0].y, 2)], "kind": kind,
                "width": None if width is None else round(width, 2)}

    def _lane_stop_plan(self, wps, k, arc):
        """Stopping straight in the lane at route point k: a straight run of lane before it
        (settled after any lane change), and no junction within LANE_STOP_CLEAR_M ahead -- a
        van stopped on a stop line blocks the junction. The route must run on that far to tell."""
        if self._same_lane_start(wps, k, self.LANE_STOP_STRAIGHT_M) is None:
            return None
        if arc[-1] - arc[k] < self.LANE_STOP_CLEAR_M:
            return None
        for j in range(k + 1, len(wps)):
            if arc[j] - arc[k] > self.LANE_STOP_CLEAR_M:
                break
            if wps[j].is_junction:
                return None
        a = wps[k]
        return k, max(0, k - 2), a.x, a.y, a.yaw, 0.0, 0.0

    def _blend_tail_to(self, route: Route, i0: int, tx, ty, tyaw, ramp_m: float):
        """Replace the route tail after index i0 with a smooth ramp `ramp_m` long to the
        target's side offset, then straight in to the target, so the van arrives parallel
        (shared by kerbside pull-over and slot parking)."""
        wps = list(route.waypoints)     # snapshot; writer swaps atomically at the end
        last = wps[-1]
        p0 = wps[i0]
        h = last.yaw
        fwd = (math.cos(h), math.sin(h))
        right = (-math.sin(h), math.cos(h))     # CARLA frame: right of the heading
        dx, dy = tx - p0.x, ty - p0.y
        along = dx * fwd[0] + dy * fwd[1]
        lat = dx * right[0] + dy * right[1]
        cut = max(0.5, min(along, ramp_m))
        K = max(6, int(along / 2.0))
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
        bay_id = 0
        for b in bay_pts + [None]:
            if b is not None:
                run.append(b)
                continue
            if len(run) >= 2:
                slots.extend(self._slice_run_into_slots(run, bay_id))
                bay_id += 1
            run = []
        return slots

    SLOT_MAX_CURVE_RAD = 0.14   # ~8 deg heading spread across a slot = too curved

    def _slice_run_into_slots(self, run, bay_id=0):
        """run = consecutive (x, y, yaw, width) bay points along the road. Every slot says
        which bay it is in, its place in it (k, counted along the road), and how much bay
        lies behind its centre -- what decides whether the van can drive into it."""
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
                        "corners": corners, "bay": bay_id, "k": k,
                        "bay_behind_m": round(mid, 2)})
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

    # How much BAY the van needs behind a slot's centre to drive into it forwards. The van has
    # no reverse gear, so it turns in from the lane, and from where its body leaves the lane
    # the turn-in must run alongside bay, not kerb (fix 3: 3 of the 4 parking stalls on
    # 2026-09-10 were bays one slot long). For the gentle ramp into a bay beside a 3.5 m lane
    # (bay_needed_behind_m(3.05)): 17 m -- the third slot of a strip is the first it can use.
    APPROACH_BAY_BEHIND_M = 17.0

    @classmethod
    def slot_reachable(cls, slots, i):
        """Can the van drive into slot i forwards? It must be free, have enough bay behind
        it, and the slots its turn-in sweeps through must be free too. Slots that do not
        say which bay they are in (from the LiDAR) keep the old test: the one before is free."""
        sl = slots[i]
        if sl.get("occupied"):
            return False
        if "bay" not in sl or "k" not in sl:
            return i == 0 or not slots[i - 1].get("occupied")
        if sl.get("bay_behind_m", 0.0) < cls.APPROACH_BAY_BEHIND_M:
            return False
        # every slot the turn-in sweeps through must be there -- straight bay -- and free. A
        # slot missing from the list is a piece of bay that BENDS, and a bay bends round
        # something: a kerb build-out, a tree pit. Seen live 2026-09-10: two turn-ins across
        # such bends, one ending at a kerb and one behind a 3.9 m tree for a minute.
        length = sl.get("length", cls.SLOT_LEN_M)
        swept = int(math.ceil((cls.APPROACH_BAY_BEHIND_M - length / 2.0) / length))
        behind = {other.get("k"): other for other in slots if other.get("bay") == sl["bay"]}
        for k in range(sl["k"] - swept, sl["k"]):
            if k not in behind or behind[k].get("occupied"):
                return False
        return True

    @classmethod
    def free_slots_by_distance(cls, slots, near):
        """Indices of the slots the van can drive into forwards, nearest `near` (x, y) first."""
        ok = [i for i in range(len(slots)) if cls.slot_reachable(slots, i)]
        return sorted(ok, key=lambda i: math.hypot(slots[i]["x"] - near[0], slots[i]["y"] - near[1]))

    @classmethod
    def choose_free_slot(cls, slots):
        """Best slot = the one nearest the destination that the van can drive into
        forwards (slot_reachable). For map slots there is no fallback: None, and the mission
        parks at the kerb in the lane rather than aim at a slot it would have to scrape into.
        Slots that do not say which bay they are in keep their old last resort, any free one."""
        for i in range(len(slots) - 1, -1, -1):
            if cls.slot_reachable(slots, i):
                return i
        for i in range(len(slots) - 1, -1, -1):
            if "bay" not in slots[i] and not slots[i].get("occupied"):
                return i
        return None

    def retarget_to_slot(self, route: Route, slot):
        """End the route in `slot`, pulling in gently from our lane (_pull_in_plan). A point in
        the lane itself (a hold-short point) is simply driven to. Returns {"approach_m": how far
        before the slot the pull-in starts, "ramp_m"}, or None when there is no gentle way in."""
        wps = route.waypoints
        if len(wps) < 6:
            return None
        ci = min(range(len(wps)),
                 key=lambda i: math.hypot(wps[i].x - slot["x"], wps[i].y - slot["y"]))
        if ci < 4:
            return None
        a = wps[ci]
        lat = -(slot["x"] - a.x) * math.sin(a.yaw) + (slot["y"] - a.y) * math.cos(a.yaw)
        if abs(lat) < 0.5:
            route.waypoints = wps[:ci] + [Waypoint(x=slot["x"], y=slot["y"], z=a.z, yaw=slot["yaw"],
                                                   road_id=a.road_id, lane_id=a.lane_id)]
            return {"approach_m": 0.0, "ramp_m": 0.0}
        plan = self._pull_in_plan(wps, ci, slot["x"], slot["y"], slot["yaw"], needs_bay=False)
        if plan is None:
            return None                    # no gentle way in from our lane: another slot
        _, i0, tx, ty, tyaw, _, ramp = plan
        route.waypoints = wps[:ci + 1]     # atomic swap
        self._blend_tail_to(route, i0, tx, ty, tyaw, ramp_m=ramp)
        return {"approach_m": round(ramp + self.PULL_IN_STRAIGHT_M, 1), "ramp_m": round(ramp, 1)}

    def _pullover_target(self, last: Waypoint):
        """Kerb-side point for a stop in the lane: the right edge of the rightmost same-way
        driving lane, the van's centre PARK_CURB_MARGIN_M in from it. Falls back to pure
        geometry 1.2 m right of the waypoint. A Parking/Shoulder strip is NOT this function's
        answer any more: a strip is only pulled into as a bay, where it is checked to run back
        far enough (_pull_in_plan) -- otherwise a strip too short to enter could be chosen."""
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

    def pull_in_blocker(self, perception, route: Route, ego_x, ego_y, ego_yaw, footprint):
        """The nearest thing standing still that the van's body would touch on the REST of the
        route -- the whole pull-in, not the FOOTPRINT_STATIONARY_REACH_M the running check looks
        ahead -- judged as filter_to_route_corridor judges it (the measured box when there is
        one, a kerb at tyre clearance, a parked vehicle passable with care). (object, metres
        away, the SweepHit, its centre (x, y), its ObstacleBox or None) or None. Asked before turning into a parking spot, while choosing another is still
        possible: on 2026-09-11 the van turned in first and was stopped 12 degrees across the
        lane by a bus shelter beside the spot."""
        if footprint is None or not route or len(route.waypoints) < 2:
            return None
        wps = route.waypoints
        cos_y, sin_y = math.cos(ego_yaw), math.sin(ego_yaw)
        best = None
        for obj in getattr(perception, "objects", None) or []:
            if getattr(obj, "speed", 0.0) >= 0.5:
                continue                     # moving things are the running check's business
            wx = ego_x + cos_y * obj.x - sin_y * obj.y
            wy = ego_y + sin_y * obj.x + cos_y * obj.y
            box = obstacle_box_for(obj, ego_yaw)
            where = (wx, wy)
            if box is not None:
                bdx = float(getattr(obj, "box_dx", 0.0) or 0.0)
                bdy = float(getattr(obj, "box_dy", 0.0) or 0.0)
                where = (wx + cos_y * bdx - sin_y * bdy, wy + sin_y * bdx + cos_y * bdy)
            body = (replace(footprint, safety_margin=min(footprint.safety_margin, KERB_CLEARANCE_M))
                    if kerb_like(obj) else footprint)
            radius = obstacle_radius_m(obj)
            hit = sweep_conflict(wps, (ego_x, ego_y), body, where, obstacle_radius=radius,
                                 horizon_m=1e4, obstacle_box=box)
            if hit is None:
                continue
            if can_pass_with_care(obj):
                tight = replace(footprint, safety_margin=min(footprint.safety_margin, PASS_CLEARANCE_M))
                if sweep_conflict(wps, (ego_x, ego_y), tight, where, obstacle_radius=radius,
                                  horizon_m=1e4, obstacle_box=box) is None:
                    continue
            d = math.hypot(wx - ego_x, wy - ego_y)
            if best is None or d < best[1]:
                best = (obj, d, hit, where, box)
        return best

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
        # How far ahead the path stops being a plain road. The swept-body check is only
        # trustworthy on a straight-ish stretch, so this is where it must hand back.
        #
        # A junction turn is a tight arc -- 8 m radius is an ordinary town corner -- and the
        # van is 5.92 m long, so its body genuinely sweeps a wide band round it. Measured:
        #
        #     bend radius   how far off the line the swept body reaches
        #        20 m                    2.70 m
        #        12 m                    3.60 m
        #         8 m                    5.90 m
        #
        # At a real junction there is always something within six metres -- kerb, poles,
        # parked cars in the next lane, the corner of a building -- so sweeping the turn
        # blocks on all of it and the van can never turn anywhere in a town. Measured live
        # on 2026-09-10: with the sweep on, the planner said blocked_swept_path on 98% of
        # ticks and the van never exceeded 0.54 m/s, blocked by objects 3.98 m off the line.
        #
        # The rule already tried to allow for this and asked the wrong question: it exempted
        # objects STANDING IN a junction, not the swept PATH going through one. An object on
        # a straight stretch, with the route bending into a junction five metres further on,
        # was still swept through the turn.
        junction_arc = None
        _a = 0.0
        for _i in range(n):
            if wps[_i].is_junction and _a >= ego_arc:
                junction_arc = _a
                break
            if _i + 1 < n:
                _a += math.hypot(wps[_i + 1].x - wps[_i].x, wps[_i + 1].y - wps[_i].y)
        # how far ahead the plain road runs out; None means "no junction on the route"
        plain_road_m = None if junction_arc is None else max(0.0, junction_arc - ego_arc)
        # how far ahead the van's nose is: its real body when we were given one
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
        passing_obj = None  # a parked vehicle being passed with care: (obj, along, lat)
        detail = []

        def _note_block(rule, obj_, along_, lat_):
            """Remember the NEAREST blocker and the rule that caught it."""
            nonlocal why, blocker
            if blocker is None or max(0.0, along_) < max(0.0, blocker[1]):
                why, blocker = rule, (obj_, along_, lat_)

        for obj in perception.objects:
            seen += 1
            # the van's frame (x forward, y to the right, as camera_lidar_perception reports
            # objects -- measured live 2026-09-11) -> world
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
            footprint_reach = (footprint.swept_half_length if footprint is not None else 0.0)
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
            # Only where the path it would sweep is a plain road: the body must reach the
            # object before the road turns, or the swept envelope is not something to judge
            # by. See plain_road_m above for what a turn does to it.
            sweep_on_plain_road = (plain_road_m is None
                                   or along + footprint_reach <= plain_road_m)
            sweep_decides = (footprint is not None and stationary
                             and not near_junction and sweep_on_plain_road
                             and lat <= SWEEP_MAX_LATERAL_M)
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
                # The measured rectangle when there is one; the circle only when there is not.
                # See ObstacleBox for what the circle did to a kerb strip.
                box = obstacle_box_for(obj, ego_yaw)
                # the rectangle's own centre, not the average of its points (fix 2): the two
                # are 0.9 m apart for a car seen from its corner
                where = (wx, wy)
                if box is not None:
                    bdx = float(getattr(obj, "box_dx", 0.0) or 0.0)
                    bdy = float(getattr(obj, "box_dy", 0.0) or 0.0)
                    where = (wx + cos_y * bdx - sin_y * bdy, wy + sin_y * bdx + cos_y * bdy)
                # a kerb needs tyre clearance, not the full safety margin (KERB_CLEARANCE_M)
                body = (replace(footprint, safety_margin=min(footprint.safety_margin, KERB_CLEARANCE_M))
                        if kerb_like(obj) else footprint)
                hit = sweep_conflict(wps, (ego_x, ego_y), body, where,
                                     obstacle_radius=radius,
                                     horizon_m=FOOTPRINT_STATIONARY_REACH_M + footprint.swept_half_length,
                                     obstacle_box=box)
                passing = False
                if hit is not None and can_pass_with_care(obj):
                    # only the margin in the way? then pass it slowly (PASS_CLEARANCE_M)
                    tight = replace(footprint, safety_margin=min(footprint.safety_margin, PASS_CLEARANCE_M))
                    passing = sweep_conflict(wps, (ego_x, ego_y), tight, where,
                                             obstacle_radius=radius,
                                             horizon_m=FOOTPRINT_STATIONARY_REACH_M + footprint.swept_half_length,
                                             obstacle_box=box) is None
                if passing:
                    # seen, and the nearest thing ahead, so the slow zone applies -- not a stop
                    found = True
                    dist = max(0.0, along)
                    if dist < closest:
                        closest = dist
                        closest_type = obj.object_type
                        closest_speed = obj.speed
                        closest_lat = round(lat, 2)
                    if passing_obj is None or dist < passing_obj[1]:
                        passing_obj = (obj, dist, lat)
                elif hit is not None:
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
        if passing_obj is not None and not blocked:
            decision.passing_id = int(getattr(passing_obj[0], "id", 0) or 0) or None
            decision.passing_lateral_m = passing_obj[2]
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

    def route_left_m(self, route: Route, current_x, current_y) -> float:
        """How far there is still to DRIVE along the route to its end, from the nearest point of
        the route (found the way get_next_waypoint finds it)."""
        wps = route.waypoints if route else None
        if not wps:
            return 999.0
        if len(wps) < 2:
            return math.hypot(wps[0].x - current_x, wps[0].y - current_y)
        best_j, best_t, best_d2 = 0, 0.0, float("inf")
        for j in range(len(wps) - 1):
            ax, ay, bx, by = wps[j].x, wps[j].y, wps[j + 1].x, wps[j + 1].y
            dx, dy = bx - ax, by - ay
            seg2 = dx * dx + dy * dy
            tt = 0.0 if seg2 <= 1e-12 else max(0.0, min(1.0, ((current_x - ax) * dx + (current_y - ay) * dy) / seg2))
            d2 = (ax + tt * dx - current_x) ** 2 + (ay + tt * dy - current_y) ** 2
            if d2 < best_d2:
                best_j, best_t, best_d2 = j, tt, d2
        left = (1.0 - best_t) * math.hypot(wps[best_j + 1].x - wps[best_j].x, wps[best_j + 1].y - wps[best_j].y)
        for k in range(best_j + 1, len(wps) - 1):
            left += math.hypot(wps[k + 1].x - wps[k].x, wps[k + 1].y - wps[k].y)
        return left

    def distance_to_destination(self, route: Route, current_x, current_y) -> float:
        """How far to the end of the route: by road, or in a straight line, whichever is more.

        It used to be the straight line alone. A route that loops round a block passes close to
        its own end long before it gets there: measured live on 2026-09-11, 19.5 m from it with
        about 200 m still to drive, and 22.7 m from the parking spot on the wrong street. The van
        slowed for its destination there, and the parking check stopped it and looked for a spot
        hidden behind buildings -- then gave the spot up. At the end the two agree; past the end
        (an overshoot) the straight line is the one that keeps growing."""
        if not route or not route.waypoints:
            return 999.0
        last = route.waypoints[-1]
        straight = math.hypot(last.x - current_x, last.y - current_y)
        return max(straight, self.route_left_m(route, current_x, current_y))

    def disable(self):
        self._enabled = False
        print("[Planner] DISABLED")

    def enable(self):
        self._enabled = True
        print("[Planner] Re-enabled")
