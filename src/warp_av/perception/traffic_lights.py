"""Knowing about a traffic light BEFORE you are underneath it.

The problem this replaces
------------------------
The van used to ask CARLA "is a light affecting me right now?" every tick. That is a
proximity switch, not a lookahead: CARLA answers only once the van is inside a small
trigger box at the stop line. Measured on Town10HD, 2026-09-09, backing away from a red
light: nothing at 30 m, 20 m, 10 m, 6 m or 4 m, and RED first reported at **2 m**. At
8 m/s the van needs about 14 m to stop. So it could not stop for a red light at all, and
the red-light demo failed for that reason on every commit going back.

It was also the most expensive thing in the tick after perception -- `get_traffic_light()`
alone costs about 65 ms, worst case 121 ms, because it is a search on the simulator's side.

How this works instead
----------------------
A traffic light's geometry never moves, so it is looked up ONCE and remembered. Every
light reports the lanes it governs (`get_stop_waypoints()`), and every lane has a road id
and a lane id. The route is built from CARLA waypoints carrying the same ids. So a light
can be tied to OUR lane on OUR route exactly, rather than guessed at by distance.

Measured on Town10HD: all 15 lights report their stop lanes; the whole database builds in
234 ms once; on a 226-point route four lights matched, correctly ordered at 29, 254, 379
and 444 m along it, with each stop line landing within 0.3 m of a route waypoint.

Why lane matching and not nearest-distance: at the first junction on that route, another
light sits **12.8 m** from ours and governs the cross traffic. Picking the nearest light
would stop the van for traffic crossing in front of it. Lane matching excludes it.

Three separate things
---------------------
Kept apart on purpose, so the last one can be swapped later without touching the others:

  * WHERE the signals are and WHICH one applies to us -- `SignalMap`, `RouteSignals`
  * WHAT COLOUR it is -- a `state_source` function, `light id -> colour`
  * WHAT THE VAN DOES about it -- unchanged, in the behaviour layer

Today the colour comes from the CARLA actor. When the front camera can read a light, only
`carla_state_source` gets replaced; none of the geometry, matching or timing changes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------------------
# How far ahead we care, and how often we ask. Named, not scattered through the code.
# ---------------------------------------------------------------------------------------

#: Beyond this, a signal is known about but its colour is never asked for. There is nothing
#: to do about a red light 60 m away that is not already handled by simply driving.
LOOKAHEAD_M = 50.0

#: Inside this, the colour is asked for as often as the van thinks. This is the band where
#: the answer changes what the van does.
NEAR_M = 25.0

#: How often the colour is asked in the far band (50 m .. 25 m).
FAR_CHECK_HZ = 4.0

#: ... and in the near band (25 m .. the stop line).
NEAR_CHECK_HZ = 10.0

#: Once the van is this far PAST a stop line, that signal is finished with and the next one
#: is selected. A little slack, because the van's own length straddles the line.
PASSED_BY_M = 4.0

#: A colour older than this is not trusted. Better to say "unknown" -- which the van treats
#: as a reason to stop -- than to act on a stale green.
STATE_STALE_S = 1.0

RED, YELLOW, GREEN, UNKNOWN, NONE = "red", "yellow", "green", "unknown", "none"

#: Colours that mean "do not proceed". UNKNOWN is deliberately in here: at a stop line we
#: know about, not being able to read the light is not permission to carry on.
STOP_COLOURS = (RED, YELLOW, UNKNOWN)


# ---------------------------------------------------------------------------------------
# A. WHERE the signals are
# ---------------------------------------------------------------------------------------


@dataclass
class SignalGeometry:
    """One traffic light, as the map describes it. Never changes while the map is loaded."""

    light_id: int
    stop_points: List[Tuple[float, float]] = field(default_factory=list)   # world x, y
    lanes: Set[Tuple[int, int]] = field(default_factory=set)               # (road_id, lane_id)
    actor: object = None                                                   # the CARLA actor

    def nearest_stop_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if not self.stop_points:
            return None
        return min(self.stop_points, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)


class SignalMap:
    """Every traffic light on the map, and the lanes each one governs. Built ONCE."""

    def __init__(self, signals: Optional[Dict[int, SignalGeometry]] = None):
        self.signals: Dict[int, SignalGeometry] = dict(signals or {})
        self.by_lane: Dict[Tuple[int, int], List[int]] = {}
        self.build_ms = 0.0
        self._index()

    def _index(self) -> None:
        self.by_lane = {}
        for sig in self.signals.values():
            for lane in sig.lanes:
                self.by_lane.setdefault(lane, []).append(sig.light_id)

    def __len__(self) -> int:
        return len(self.signals)

    @classmethod
    def from_world(cls, world) -> "SignalMap":
        """Read the whole map's traffic lights once. Measured at 234 ms on Town10HD."""
        t0 = time.perf_counter()
        found: Dict[int, SignalGeometry] = {}
        try:
            lights = list(world.get_actors().filter("traffic.traffic_light*"))
        except Exception:
            lights = []
        for tl in lights:
            pts: List[Tuple[float, float]] = []
            lanes: Set[Tuple[int, int]] = set()
            try:
                for wp in tl.get_stop_waypoints():
                    loc = wp.transform.location
                    pts.append((float(loc.x), float(loc.y)))
                    lanes.add((int(wp.road_id), int(wp.lane_id)))
            except Exception:
                continue
            if pts:
                found[int(tl.id)] = SignalGeometry(light_id=int(tl.id), stop_points=pts,
                                                   lanes=lanes, actor=tl)
        out = cls(found)
        out.build_ms = (time.perf_counter() - t0) * 1000.0
        return out


# ---------------------------------------------------------------------------------------
# B. WHAT COLOUR it is -- the one piece meant to be replaced later
# ---------------------------------------------------------------------------------------

#: A state source answers: light id -> "red" | "yellow" | "green" | "unknown".
StateSource = Callable[[int], str]

_CARLA_COLOURS = {"Red": RED, "Yellow": YELLOW, "Green": GREEN}


def carla_state_source(signal_map: SignalMap) -> StateSource:
    """Colour from the CARLA actor. THIS is the piece the camera replaces later.

    Reading the state off an actor we already hold is cheap; it is FINDING the actor that
    cost 65 ms, and that is now done once at map load.
    """
    def read(light_id: int) -> str:
        sig = signal_map.signals.get(light_id)
        if sig is None or sig.actor is None:
            return UNKNOWN
        try:
            return _CARLA_COLOURS.get(str(sig.actor.get_state()).split(".")[-1], UNKNOWN)
        except Exception:
            return UNKNOWN
    return read


# ---------------------------------------------------------------------------------------
# WHICH signal applies to us, and how far away it is
# ---------------------------------------------------------------------------------------


@dataclass
class RouteSignal:
    """A signal that our route actually passes, and where along it."""

    light_id: int
    along_m: float                     # distance along the route to its stop line
    stop_point: Tuple[float, float]
    lane: Tuple[int, int]


def route_signals(signal_map: SignalMap, waypoints: Sequence) -> List[RouteSignal]:
    """Which signals this route passes, in the order we will meet them.

    A signal counts only if it governs a (road, lane) the route actually travels. That is
    what keeps cross traffic out: at the first junction of the measured route another light
    sat 12.8 m away, and it governs the crossing road, so it is not on this list.
    """
    if not waypoints:
        return []
    # distance along the route to each waypoint, worked out once
    along = [0.0] * len(waypoints)
    for i in range(1, len(waypoints)):
        a, b = waypoints[i - 1], waypoints[i]
        along[i] = along[i - 1] + math.hypot(b.x - a.x, b.y - a.y)

    # which lanes does this route actually travel, and which waypoints are on each
    lane_points: Dict[Tuple[int, int], List[int]] = {}
    for i, wp in enumerate(waypoints):
        road, lane = getattr(wp, "road_id", None), getattr(wp, "lane_id", None)
        if road is None or lane is None:
            continue
        lane_points.setdefault((int(road), int(lane)), []).append(i)
    if not lane_points:
        return []

    out: List[RouteSignal] = []
    for lane, idxs in lane_points.items():
        for light_id in signal_map.by_lane.get(lane, ()):
            sig = signal_map.signals[light_id]
            # the route waypoint ON THAT LANE closest to this light's stop line. Searching
            # only that lane's points is what stops a light at the far end of the route
            # matching a waypoint at the near end.
            best_i, best_d = None, float("inf")
            for i in idxs:
                wp = waypoints[i]
                for px, py in sig.stop_points:
                    d = math.hypot(px - wp.x, py - wp.y)
                    if d < best_d:
                        best_i, best_d = i, d
            if best_i is None:
                continue
            pt = sig.nearest_stop_point(waypoints[best_i].x, waypoints[best_i].y)
            out.append(RouteSignal(light_id=light_id, along_m=along[best_i],
                                   stop_point=pt, lane=lane))
    # one entry per light, the earliest place we meet it
    first: Dict[int, RouteSignal] = {}
    for s in out:
        was = first.get(s.light_id)
        if was is None or s.along_m < was.along_m:
            first[s.light_id] = s
    return sorted(first.values(), key=lambda s: s.along_m)


# ---------------------------------------------------------------------------------------
# What the rest of the van is told
# ---------------------------------------------------------------------------------------


@dataclass
class SignalAhead:
    """The next signal on our route, or nothing."""

    state: str = NONE                  # red | yellow | green | unknown | none
    distance_m: Optional[float] = None  # along the route, to the stop line
    light_id: Optional[int] = None
    state_age_s: Optional[float] = None
    watching: bool = False             # inside the band where the colour is asked for

    @property
    def valid(self) -> bool:
        return self.state in (RED, YELLOW, GREEN)

    @property
    def must_not_proceed(self) -> bool:
        return self.state in STOP_COLOURS

    def as_dict(self) -> dict:
        return {"state": self.state,
                "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
                "light_id": self.light_id,
                "state_age_s": None if self.state_age_s is None else round(self.state_age_s, 2),
                "watching": self.watching, "valid": self.valid}


class TrafficLightLookahead:
    """Which signal is next, how far off, and what colour -- without asking every tick.

    The van's own position gives the distance; the colour is asked for only inside the
    activation band, and more often the closer it gets.
    """

    def __init__(self, signal_map: SignalMap, state_source: Optional[StateSource] = None,
                 lookahead_m: float = LOOKAHEAD_M, near_m: float = NEAR_M,
                 far_hz: float = FAR_CHECK_HZ, near_hz: float = NEAR_CHECK_HZ):
        self.signal_map = signal_map
        self.state_source = state_source or carla_state_source(signal_map)
        self.lookahead_m = float(lookahead_m)
        self.near_m = float(near_m)
        self.far_period = 1.0 / max(0.1, far_hz)
        self.near_period = 1.0 / max(0.1, near_hz)
        self._signals: List[RouteSignal] = []
        self._route_stamp = None
        self._current: Optional[RouteSignal] = None
        self._state = NONE
        self._state_at: Optional[float] = None
        self._asked_at: Optional[float] = None
        # what it cost, for the telemetry
        self.queries = 0
        self.last_geometry_ms = 0.0
        self.last_query_ms = 0.0
        self.last_ego_along_m = 0.0
        self.last_route_len = 0

    # ---- route ---------------------------------------------------------------------
    def set_route(self, route) -> None:
        """Called when a route is made or changed. Stale associations are dropped."""
        waypoints = getattr(route, "waypoints", None) or []
        self._signals = route_signals(self.signal_map, waypoints)
        self._route_stamp = getattr(route, "timestamp", None)
        self._current = None
        self._state, self._state_at, self._asked_at = NONE, None, None

    def _route_changed(self, route) -> bool:
        return getattr(route, "timestamp", None) != self._route_stamp

    # ---- every tick ----------------------------------------------------------------
    def update(self, route, ego_x: float, ego_y: float, now: Optional[float] = None) -> SignalAhead:
        now = time.time() if now is None else float(now)
        t0 = time.perf_counter()
        if route is None or not getattr(route, "waypoints", None):
            self._current = None
            self.last_geometry_ms = (time.perf_counter() - t0) * 1000.0
            return SignalAhead()
        if self._route_changed(route):
            self.set_route(route)
        if not self._signals:
            self.last_geometry_ms = (time.perf_counter() - t0) * 1000.0
            return SignalAhead()

        ego_along = _along_route(route.waypoints, ego_x, ego_y)
        self.last_ego_along_m = ego_along
        self.last_route_len = len(route.waypoints)
        nxt = None
        for sig in self._signals:
            if sig.along_m - ego_along > -PASSED_BY_M:
                nxt = sig
                break
        if nxt is None or (self._current is not None and nxt.light_id != self._current.light_id):
            # either finished with them all, or moved on to the next: forget the old colour
            self._state, self._state_at, self._asked_at = NONE, None, None
        self._current = nxt
        self.last_geometry_ms = (time.perf_counter() - t0) * 1000.0
        if nxt is None:
            return SignalAhead()

        distance = nxt.along_m - ego_along
        if distance > self.lookahead_m:
            # known about, but there is nothing to do about it yet: no colour asked for
            return SignalAhead(distance_m=distance, light_id=nxt.light_id, watching=False)

        period = self.near_period if distance <= self.near_m else self.far_period
        if self._asked_at is None or now - self._asked_at >= period:
            self._asked_at = now
            q0 = time.perf_counter()
            got = None
            try:
                got = self.state_source(nxt.light_id)
            except Exception:
                got = None
            self.last_query_ms = (time.perf_counter() - q0) * 1000.0
            self.queries += 1
            if got in (RED, YELLOW, GREEN):
                self._state, self._state_at = got, now
            else:
                # Whatever tells us the colour has failed, or said it does not know. The old
                # answer is LEFT AS IT WAS and its clock is NOT restarted, so it goes stale
                # and turns into UNKNOWN below -- which the van treats as a reason to stop.
                # Restarting the clock here was a bug of mine: it kept an old green looking
                # fresh forever, which is the exact failure requirement 5 exists to prevent.
                if self._state not in (RED, YELLOW, GREEN):
                    self._state = UNKNOWN
        age = None if self._state_at is None else now - self._state_at
        state = self._state
        if state in (RED, YELLOW, GREEN) and age is not None and age > STATE_STALE_S:
            state = UNKNOWN        # a stale green is not permission
        return SignalAhead(state=state, distance_m=distance, light_id=nxt.light_id,
                           state_age_s=age, watching=True)

    # ---- for the telemetry ----------------------------------------------------------
    def as_dict(self) -> dict:
        return {"signals_on_route": len(self._signals),
                "signals_on_map": len(self.signal_map),
                "map_build_ms": round(self.signal_map.build_ms, 1),
                "geometry_ms": round(self.last_geometry_ms, 3),
                "query_ms": round(self.last_query_ms, 3),
                "queries": self.queries,
                "ego_along_m": round(getattr(self, "last_ego_along_m", 0.0), 1),
                "route_points": getattr(self, "last_route_len", 0),
                "signal_along_m": (round(self._current.along_m, 1)
                                   if self._current is not None else None)}


def _along_route(waypoints: Sequence, x: float, y: float) -> float:
    """How far along the route we are. Straight-line to the nearest waypoint, then its own
    distance along -- the route is sampled every 2 m, so this is good to about a metre."""
    best_i, best_d2 = 0, float("inf")
    for i, wp in enumerate(waypoints):
        d2 = (wp.x - x) ** 2 + (wp.y - y) ** 2
        if d2 < best_d2:
            best_i, best_d2 = i, d2
    total = 0.0
    for i in range(1, best_i + 1):
        total += math.hypot(waypoints[i].x - waypoints[i - 1].x,
                            waypoints[i].y - waypoints[i - 1].y)
    return total
