"""Stop and give-way signs, read from the map.

The same place the traffic-light stop lines come from (perception/traffic_lights.py): the
map's own OpenDRIVE signals. A stop sign is landmark type 206, a give-way 205, and each one
says which road, how far along it, and -- through its lane validities -- exactly which lanes
it governs. Town10HD carries 9 stop signs and 1 give-way.

Nothing is read off a camera here, and nothing is asked of the simulator that a map could not
answer: which lanes a sign governs is a rule of the road, like a stop line.

What the van does about a sign is the behaviour's business (behavior._rule_road_sign): come
to a full stop at a stop line and then give way; slow and give way at a give-way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

STOP, GIVE_WAY = "stop", "give_way"
#: OpenDRIVE sign types. 206 is a stop sign, 205 a give-way ("yield").
SIGN_TYPES = {"206": STOP, "205": GIVE_WAY}
#: Lanes narrower than this are kerbs, medians and pavements that a sign's validity range
#: sweeps up along with the driving lanes: a 0.6 m "lane" is not a lane the van can be in.
MIN_LANE_WIDTH_M = 2.0


@dataclass(frozen=True)
class RoadSign:
    """One sign, and the one lane it governs. A sign over three lanes is three of these."""
    kind: str                    # STOP or GIVE_WAY
    road_id: int
    lane_id: int
    x: float
    y: float
    yaw_deg: float               # the direction of travel in that lane, where the sign is
    name: str = ""

    @property
    def lane(self) -> Tuple[int, int]:
        return (self.road_id, self.lane_id)


def read_signs(carla_map) -> List[RoadSign]:
    """Every stop and give-way sign in the map, one entry per lane it governs."""
    out: List[RoadSign] = []
    for code, kind in SIGN_TYPES.items():
        try:
            marks = list(carla_map.get_all_landmarks_of_type(code))
        except Exception:
            continue
        for lm in marks:
            try:
                validities = list(lm.get_lane_validities() or [])
            except Exception:
                validities = []
            for first, last in validities:
                for lane_id in range(min(first, last), max(first, last) + 1):
                    if lane_id == 0:
                        continue                     # the reference line is not a lane
                    try:
                        wp = carla_map.get_waypoint_xodr(lm.road_id, lane_id, lm.s)
                    except Exception:
                        wp = None
                    if wp is None or float(getattr(wp, "lane_width", 0.0)) < MIN_LANE_WIDTH_M:
                        continue
                    t = wp.transform
                    out.append(RoadSign(kind=kind, road_id=int(lm.road_id), lane_id=int(lane_id),
                                        x=float(t.location.x), y=float(t.location.y),
                                        yaw_deg=float(t.rotation.yaw), name=str(lm.name)))
    return out


def _along_route(waypoints: Sequence) -> List[float]:
    along = [0.0] * len(waypoints)
    for i in range(1, len(waypoints)):
        a, b = waypoints[i - 1], waypoints[i]
        along[i] = along[i - 1] + math.hypot(b.x - a.x, b.y - a.y)
    return along


def signs_on_route(signs: Sequence[RoadSign], waypoints: Sequence,
                   max_off_m: float = 4.0) -> List[Tuple[float, RoadSign]]:
    """(metres along the route, sign) for every sign this route actually meets, in order.

    A sign counts only if it governs a (road, lane) the route travels AND sits near a point of
    the route -- which is what keeps the stop sign on the crossing road out of our list, the
    same rule route_signals uses for traffic lights.
    """
    if not waypoints or not signs:
        return []
    along = _along_route(waypoints)
    lanes = {(getattr(w, "road_id", None), getattr(w, "lane_id", None)) for w in waypoints}
    found: List[Tuple[float, RoadSign]] = []
    for sign in signs:
        if sign.lane not in lanes:
            continue
        best, best_at = None, None
        for i, w in enumerate(waypoints):
            if (getattr(w, "road_id", None), getattr(w, "lane_id", None)) != sign.lane:
                continue
            d = math.hypot(w.x - sign.x, w.y - sign.y)
            if best is None or d < best:
                best, best_at = d, along[i]
        if best is not None and best <= max_off_m:
            found.append((best_at, sign))
    found.sort(key=lambda pair: pair[0])
    return found


def next_sign(on_route: Sequence[Tuple[float, RoadSign]], along_now_m: float,
              within_m: float = 60.0) -> Optional[Tuple[float, RoadSign]]:
    """The next sign ahead of us on the route: (metres to it, sign), or None."""
    for at, sign in on_route:
        gap = at - along_now_m
        if -1.0 < gap <= within_m:
            return (gap, sign)
    return None
