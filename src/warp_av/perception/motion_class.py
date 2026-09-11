"""
Static or dynamic: can this thing move? (Planning V2)

Every tracked thing starts DYNAMIC -- "not detected as moving" is not the same as "safe".
The van calls a thing STATIC only when it has earned it, three ways at once:

  1. SHAPE AND PLACE. One of three rules, learned from CARLA's answer key
     (tools/record_static_truth.py records what the van's LiDAR sees next to what each
     point really hit; tools/learn_static_rules.py learns the thresholds):

       POLE       at least 2.3 m tall, at most 0.6 m across, at least 30 % of its points
                  above 2 m, and at least 0.25 m clear of every lane a vehicle can use
       STRUCTURE  at least 70 % of its points above 2 m, and 0.25 m clear of every lane
       LOW        at most 0.3 m tall, and no more than 0.25 m into a lane: a kerb

     "Every lane a vehicle can use" means driving, parking, shoulder, two-way and bike
     lanes, from the map -- Town10HD draws its parking strips as shoulders.
  2. AGAIN AND AGAIN. The rule has held on 3 sightings in a row, over at least 1 s.
  3. NOBODY NAMED IT. No camera -- and no shape rule -- has ever called this track a
     person, a cyclist or a vehicle. One such name and it stays dynamic for good.

It goes back to DYNAMIC at once when a name arrives, when the tracker stops calling it
stationary, or when the rule fails 3 sightings in a row.

What STATIC changes: the crossing prediction no longer asks whether it might step or
pull into our path. What it does NOT change: a static thing IN the path still stops the
van exactly as before -- a cone in the lane is still a cone in the lane.

Why the rules are lopsided. Calling a pole "dynamic" costs a needless caution; calling a
person "static" is the one mistake that is not allowed. So each threshold is the loosest
that let NO person or vehicle through on the roads it learned from, with a margin: one
step looser must still let none through. Measured on Town10HD, 2026-09-10:

  * learned on half the roads, tested on the other half, 5 different splits: 0 of 2,684
    people and vehicles called static -- counting the ones glued to a pole or a shop front
    inside a blob that is mostly wall -- and 24 % of the static things called static;
  * the final rules on a fresh recording they were never tuned on: 0 of 1,211, and 24 %
    (poles 47 %, kerbs 88 %, walls, fences and trees 20-30 %, cones and bins about 1 %).

Cones and bins stay dynamic on purpose. To the LiDAR a bin on the pavement is the same
shape as a parked scooter or a crouching child -- a "short thing off the road" rule was
tried and called parked bicycles static. Telling those apart is the camera's job.

Nothing further than 12 m from any lane is labelled at all: two lorries parked in a yard
31-41 m from the road look exactly like a building corner to the laser, and nothing that
far off can reach the road in the next few seconds (the same 12 m the crossing
prediction uses).

CARLA validation only: the lane distances come from CARLA's map.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

# ---- the learned rules (tools/learn_static_rules.py on static_truth, 2026-09-10) -------
POLE_MIN_HEIGHT_M = 2.3          # never lower, whatever the data allows: the tallest person
                                 # recorded measured 1.92 m, and real people come taller
POLE_MAX_ACROSS_M = 0.6
POLE_MIN_HIGH_SHARE = 0.30       # a pole has points all the way up it; 4 points of a person
                                 # plus ONE point of the pole behind them is not a pole
STRUCTURE_MIN_HIGH_SHARE = 0.70
CLEAR_OF_LANES_M = 0.25          # POLE and STRUCTURE: the whole blob this far off every lane
LOW_MAX_HEIGHT_M = 0.30
LOW_MAX_INTO_LANE_M = 0.25       # a kerb sits on the lane edge; a car half-hidden behind
                                 # another shows as 3-5 points 0.3-0.4 m up, INSIDE the lane
HIGH_POINT_M = 2.0               # "high up" means above this, for the two share rules
MIN_POINTS = 3                   # a 2-point blob is never judged
IN_REACH_OF_ROAD_M = 12.0        # further off any lane than this: not labelled at all
GAP_SAMPLE_POINTS = 24           # the gap is measured at up to this many of a blob's points

# ---- a camera's "vehicle" that cannot be one (2026-09-11) ---------------------------------
# A bus shelter on the pavement was named a VEHICLE by the camera, so it could never be
# static, got a vehicle's width and was waited for as a road user: parking beside it, the
# van stopped 12 degrees into its turn and waited for ever. Measured against CARLA's answer
# key on both static_truth recordings (17,828 blobs): of every blob at least this tall, this
# long and this far clear of every lane a vehicle can use, NOT ONE is a real vehicle -- nor
# with any threshold one step looser (2.2 m / 2.8 m / 0.27 m). The vehicles that do stand off
# the road are parked bicycles and motorbikes, 0.7-1.2 m tall, and bikes or a car glued to a
# pole, 1.3-1.9 m long; a car glued to a pole and a sign sat 0.26 m clear.
# Not one of those blobs is MOSTLY a vehicle, but 23 and 33 of them hold a few road-user
# points (a person against a shop front, a motorbike in a hedge; learn_static_rules' own test,
# 3 points). That is why only a VEHICLE name is ever dropped -- a person or cyclist name is
# never questioned here -- and why dropping a name never makes a thing static: the rules
# above still have to hold, and they let no such blob through.
NOT_A_VEHICLE_MIN_HEIGHT_M = 2.3
NOT_A_VEHICLE_MIN_LENGTH_M = 3.0
NOT_A_VEHICLE_CLEAR_M = 0.3


def vehicle_name_implausible(height_m: Optional[float], longest_m: Optional[float],
                             gap_m: Optional[float]) -> bool:
    """Would calling this blob a vehicle be implausible? Tall, long and wholly off the road:
    street furniture -- a shelter, a kiosk -- not a vehicle. Unknown is never enough."""
    if height_m is None or longest_m is None or gap_m is None or not math.isfinite(gap_m):
        return False
    return (height_m >= NOT_A_VEHICLE_MIN_HEIGHT_M and longest_m >= NOT_A_VEHICLE_MIN_LENGTH_M
            and NOT_A_VEHICLE_CLEAR_M <= gap_m <= IN_REACH_OF_ROAD_M)


# ---- earning it, and losing it ----------------------------------------------------------
STATIC_AFTER_SIGHTINGS = 3
STATIC_AFTER_S = 1.0
DYNAMIC_AFTER_MISSES = 3
# the gap to the road is looked up once per track and reused (a lookup is 25 us, and a
# static thing does not move), unless the track has moved or the answer has gone stale
GAP_REFRESH_S = 2.0
GAP_REFRESH_MOVE_M = 0.5

STATIC = "static"
DYNAMIC = "dynamic"
POLE, STRUCTURE, LOW = "pole", "structure", "low"


def high_share(heights: Sequence[float]) -> float:
    """What share of a blob's points sit more than HIGH_POINT_M above the road."""
    h = np.asarray(heights, dtype=float)
    h = h[np.isfinite(h)]
    return float(np.mean(h > HIGH_POINT_M)) if h.size else 0.0


def shape_rules(height_m: float, across_m: float, high: float, n: int) -> Tuple[str, ...]:
    """The rules this blob's SHAPE fits, before asking where it is. Cheap, so it can run on
    every blob; only these candidates pay for a lookup on the map."""
    if n < MIN_POINTS or height_m is None or not height_m > 0.0:
        return ()
    out = []
    if high >= STRUCTURE_MIN_HIGH_SHARE:
        out.append(STRUCTURE)
    if height_m >= POLE_MIN_HEIGHT_M and across_m <= POLE_MAX_ACROSS_M and high >= POLE_MIN_HIGH_SHARE:
        out.append(POLE)
    if height_m <= LOW_MAX_HEIGHT_M:
        out.append(LOW)
    return tuple(out)


def rule_holds(rule: str, gap_m: Optional[float]) -> bool:
    """Does the PLACE part of a rule hold, given how far the blob is from the nearest lane
    (+ = clear of it by that much, - = that far into it)? Unknown is never good enough."""
    if gap_m is None or not math.isfinite(gap_m) or gap_m > IN_REACH_OF_ROAD_M:
        return False
    if rule in (POLE, STRUCTURE):
        return gap_m >= CLEAR_OF_LANES_M
    if rule == LOW:
        return gap_m >= -LOW_MAX_INTO_LANE_M
    return False


def sample_for_gap(points_xy: np.ndarray) -> np.ndarray:
    """Up to GAP_SAMPLE_POINTS of a blob's points, evenly spread: what the recorder used."""
    pts = np.asarray(points_xy, dtype=float)
    if len(pts) <= GAP_SAMPLE_POINTS:
        return pts
    return pts[np.linspace(0, len(pts) - 1, GAP_SAMPLE_POINTS).astype(int)]


class RoadGap:
    """How far a blob is from the nearest lane a vehicle can use.

    `lookup(x, y, z)` returns the nearest such lane's centre and width as (cx, cy, width),
    or None. The gap of one point is its distance to that centre less half the width; the
    gap of a blob is the smallest over its points, so the part nearest the road decides.
    """

    def __init__(self, lookup: Callable[[float, float, float], Optional[Tuple[float, float, float]]]):
        self._lookup = lookup
        self.lookups = 0

    def gap_m(self, world_xy: Iterable[Tuple[float, float]], z: float = 0.0) -> Optional[float]:
        best = None
        for x, y in world_xy:
            self.lookups += 1
            hit = self._lookup(float(x), float(y), float(z))
            if hit is None:
                continue
            cx, cy, width = hit
            g = math.hypot(x - cx, y - cy) - width / 2.0
            best = g if best is None else min(best, g)
        return best


def carla_road_gap(carla_map) -> RoadGap:
    """A RoadGap reading CARLA's map (imported here so the rest runs without CARLA)."""
    import carla
    lanes = (carla.LaneType.Driving | carla.LaneType.Parking | carla.LaneType.Bidirectional
             | carla.LaneType.Shoulder | carla.LaneType.Biking)

    def lookup(x, y, z):
        wp = carla_map.get_waypoint(carla.Location(x=x, y=y, z=z), project_to_road=True, lane_type=lanes)
        if wp is None:
            return None
        loc = wp.transform.location
        return loc.x, loc.y, wp.lane_width

    return RoadGap(lookup)


class MotionMemory:
    """One track's answer to "can it move?", and the evidence behind it."""

    __slots__ = ("state", "rule", "streak", "streak_since", "misses", "ever_named",
                 "gap_m", "gap_at", "gap_xy", "why")

    def __init__(self):
        self.state = DYNAMIC
        self.rule = None             # which rule made it static
        self.streak = 0              # sightings in a row that fitted a rule
        self.streak_since = None
        self.misses = 0              # sightings in a row that did not, once static
        self.ever_named = False      # a person, cyclist or vehicle name, ever
        self.gap_m = None
        self.gap_at = None
        self.gap_xy = None
        self.why = "not judged yet"  # the reason for the current answer, in words, for the console

    @property
    def is_static(self) -> bool:
        return self.state == STATIC

    def _drop(self):
        self.state = DYNAMIC
        self.rule = None
        self.streak = 0
        self.streak_since = None
        self.misses = 0

    def _gap(self, gap_fn, wx, wy, t) -> Optional[float]:
        fresh = (self.gap_at is not None and t - self.gap_at < GAP_REFRESH_S
                 and self.gap_xy is not None
                 and math.hypot(wx - self.gap_xy[0], wy - self.gap_xy[1]) < GAP_REFRESH_MOVE_M)
        if not fresh:
            self.gap_m = gap_fn() if gap_fn is not None else None
            self.gap_at, self.gap_xy = t, (wx, wy)
        return self.gap_m

    def note(self, shapes: Sequence[str], gap_fn: Optional[Callable[[], Optional[float]]],
             named: bool, stationary: bool, wx: float, wy: float, t: float) -> str:
        """One sighting. `shapes` is what shape_rules() said about it, `gap_fn` looks its
        distance from the road up (only called when a shape fits), `named` is True when a
        camera or the shape rule has called this track a person, cyclist or vehicle."""
        if named:
            self.ever_named = True
        if self.ever_named or not stationary:
            self._drop()
            self.why = "named a road user" if self.ever_named else "moving"
            return self.state
        rule, gap = None, None
        if shapes:
            gap = self._gap(gap_fn, wx, wy, t)
            rule = next((r for r in shapes if rule_holds(r, gap)), None)
        if rule is None:
            if self.state == STATIC:
                self.misses += 1
                if self.misses >= DYNAMIC_AFTER_MISSES:
                    self._drop()
            else:
                self.streak, self.streak_since = 0, None
            if self.state != STATIC:
                self.why = ("shape fits no rule" if not shapes
                            else "no lane found near it" if gap is None
                            else "too far from any road to matter" if gap > IN_REACH_OF_ROAD_M
                            else "too close to a lane")
            return self.state
        self.misses = 0
        if self.streak == 0:
            self.streak_since = t
        self.streak += 1
        if self.state != STATIC:
            self.rule = rule
            if self.streak >= STATIC_AFTER_SIGHTINGS and t - self.streak_since >= STATIC_AFTER_S - 1e-9:
                self.state = STATIC
            else:
                self.why = f"fits {rule}, still checking"
        if self.state == STATIC:
            self.why = self.rule
        return self.state


def static_dynamic_wanted(env=None) -> bool:
    """WARP_STATIC_DYNAMIC=0 turns the labelling off (every thing dynamic), for A/B runs."""
    import os
    e = os.environ if env is None else env
    return str(e.get("WARP_STATIC_DYNAMIC", "1")).strip().lower() not in ("0", "false", "off", "no")
