"""Where the van stops for a red light: the lane's stop line from the map, not the zebra.

Measured against CARLA on 2026-09-11 (route 78,139 -> -67.3,28.0, eleven lights): the van
crossed the stop line on red at three lights out of four. It held at the ZEBRA crossing, which
on Town10HD starts 2.2 to 3.9 m past the painted stop bar at every light measured but one (at
light 22 it starts level with the bar) -- and held the middle of the van 2.6 m short of it,
which put the bumper 0.35 m past even that.

Overhead pictures of the paint at all 15 lights gave the rule below; the table in
test_every_measured_lane_stops_at_or_before_its_paint is those measurements.
"""
import math

import pytest

from warp_av.perception.traffic_lights import (
    stop_line_for_lane, route_signals, SignalGeometry, SignalMap, TrafficLightLookahead,
    _along_route, STOP_LINE_STEP_M, RED, GREEN, UNKNOWN, NONE)
from warp_av.planning.planner import Waypoint


# ---------------------------------------------------------------- a lane that walks like CARLA's

class _Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = x, y, 0.0


class _Rot:
    def __init__(self, yaw):
        self.yaw = yaw


class _Tf:
    def __init__(self, x, y, yaw):
        self.location, self.rotation = _Loc(x, y), _Rot(yaw)


class LaneWP:
    """A point on a straight lane running along `yaw_deg` from (x0, y0); the junction starts
    `junction_at` metres along. `sideways` shifts it across the lane (another lane)."""

    def __init__(self, s, junction_at, road=5, lane=-1, yaw_deg=0.0, x0=0.0, y0=0.0,
                 sideways=0.0, branch=None):
        self.s, self.junction_at = s, junction_at
        self.road_id, self.lane_id = road, lane
        self.yaw_deg, self.x0, self.y0, self.sideways = yaw_deg, x0, y0, sideways
        h = math.radians(yaw_deg)
        x = x0 + s * math.cos(h) - sideways * math.sin(h)
        y = y0 + s * math.sin(h) + sideways * math.cos(h)
        self.transform = _Tf(x, y, yaw_deg)
        self.is_junction = s >= junction_at
        self._branch = branch

    def next(self, d):
        straight = LaneWP(self.s + d, self.junction_at, self.road_id, self.lane_id,
                          self.yaw_deg, self.x0, self.y0, self.sideways)
        if self._branch is not None and self.s + d >= self._branch:
            turning = LaneWP(self.s + d, self.junction_at, 99, 1, self.yaw_deg + 90.0,
                             self.x0, self.y0)
            return [turning, straight]          # the wrong branch comes first, as CARLA may
        return [straight]


def a_lane(junction_at, signal_at=None, zebra_at=None, signal_sideways=0.0, **kw):
    stop = LaneWP(0.0, junction_at, **kw)
    affected = []
    if signal_at is not None:
        affected.append(LaneWP(signal_at, junction_at, sideways=signal_sideways,
                               lane=kw.get("lane", -1) if signal_sideways == 0.0 else -2,
                               **{k: v for k, v in kw.items() if k != "lane"}))
    on_zebra = None
    if zebra_at is not None:
        on_zebra = lambda x, y: zebra_at <= x <= zebra_at + 3.0     # lane runs along +x
    return stop, affected, on_zebra


# ---------------------------------------------------------------- which one is the line

def test_the_signal_is_the_line_when_the_junction_starts_later():
    stop, aff, zebra = a_lane(junction_at=7.0, signal_at=6.26, zebra_at=10.5)
    x, y, past, why = stop_line_for_lane(stop, aff, zebra)
    assert past == pytest.approx(6.26, abs=0.01) and "map" in why
    assert (x, y) == pytest.approx((6.26, 0.0), abs=0.01)


def test_the_junction_entry_is_the_line_when_the_signal_is_past_it():
    """Light 18, lane 2: its map position is 7.77 m in, the junction starts at 5.75 m, and
    the paint is at 7.47 m. The map position alone would put the bumper over the paint."""
    stop, aff, zebra = a_lane(junction_at=5.75, signal_at=7.77, zebra_at=11.25)
    _, _, past, why = stop_line_for_lane(stop, aff, zebra)
    assert past == pytest.approx(5.75 - STOP_LINE_STEP_M, abs=0.01) and "junction" in why
    assert past < 7.47


def test_the_zebra_is_never_the_line_when_it_comes_after():
    stop, aff, zebra = a_lane(junction_at=4.5, signal_at=4.78, zebra_at=10.0)
    _, _, past, why = stop_line_for_lane(stop, aff, zebra)
    assert "zebra" not in why and past < 10.0


def test_a_zebra_before_everything_else_is_where_it_stops():
    """Not seen on Town10HD, but the rule holds either way: the earliest wins."""
    stop, aff, zebra = a_lane(junction_at=9.0, signal_at=8.0, zebra_at=3.0)
    _, _, past, why = stop_line_for_lane(stop, aff, zebra)
    assert "zebra" in why and past == pytest.approx(3.0 - STOP_LINE_STEP_M, abs=0.3)


def test_a_signal_listed_on_the_other_lane_still_counts_along_ours():
    """Lights 11 and 13 list their map position on only one of their two lanes."""
    stop, aff, zebra = a_lane(junction_at=7.0, signal_at=6.26, signal_sideways=3.5, zebra_at=10.5)
    x, y, past, why = stop_line_for_lane(stop, aff, zebra)
    assert past == pytest.approx(6.26, abs=0.01)
    assert y == pytest.approx(0.0, abs=0.01), "the line must be on OUR lane, not the other one"


def test_nothing_known_falls_back_to_carlas_point_which_is_always_early():
    stop, aff, _ = a_lane(junction_at=99.0)
    x, y, past, why = stop_line_for_lane(stop, aff, None)
    assert past == 0.0 and (x, y) == (0.0, 0.0)


def test_the_walk_never_follows_a_branch_onto_another_road():
    """wp.next() returns a LIST at junctions; taking [0] moved a walk to another street."""
    stop = LaneWP(0.0, junction_at=6.0, branch=3.0)
    _, y, past, _ = stop_line_for_lane(stop, [], None)
    assert past == pytest.approx(6.0 - STOP_LINE_STEP_M, abs=0.01)
    assert y == pytest.approx(0.0, abs=0.01)


# The measurements (metres along the lane past CARLA's stop point): the light's map position,
# the first junction point, the first zebra point, and the painted bar found in the picture.
MEASURED = [
    # light lane  signal  junction  zebra   bar
    (9, 1, 2.16, 2.25, 4.50, 2.19), (9, 2, 2.20, 2.25, 4.75, 2.22),
    (11, -1, 6.26, 6.50, 10.50, 7.62), (11, -2, 6.24, 6.25, 10.25, 7.62),
    (12, 1, 4.78, 4.50, 10.00, 6.81), (12, 2, 4.76, 4.50, 9.75, 6.78),
    (13, -1, 5.73, 5.75, 10.50, 7.47), (13, -2, 5.74, 5.75, 10.50, 7.47),
    (14, 5, 4.85, 5.00, None, 6.93),
    (15, 1, 6.03, 6.00, 9.50, 7.08), (15, 2, 6.08, 6.00, 9.50, 7.02),
    (16, -1, 6.21, 6.25, 10.25, 7.68), (16, -2, 6.91, 6.25, 10.50, 7.71),
    (18, 1, 7.20, 5.75, 11.50, 7.59), (18, 2, 7.77, 5.75, 11.25, 7.47),
    (19, 1, 6.88, 7.00, 9.50, 6.99), (19, 2, 7.40, 7.00, 9.25, 7.05),
    (20, -1, 6.79, 6.75, None, 6.99), (20, -2, 6.78, 6.75, None, 6.99),
    (22, 4, 5.32, 5.50, 6.75, 6.81), (22, 5, 5.32, 5.50, 6.75, 6.78),
]


@pytest.mark.parametrize("light,lane,signal,junction,zebra,bar", MEASURED)
def test_every_measured_lane_stops_at_or_before_its_paint(light, lane, signal, junction, zebra, bar):
    stop, aff, on_zebra = a_lane(junction_at=junction, signal_at=signal, zebra_at=zebra)
    _, _, past, _ = stop_line_for_lane(stop, aff, on_zebra)
    assert past <= bar, f"light {light} lane {lane}: the line is past the paint"
    assert bar - past <= 3.0, f"light {light} lane {lane}: stops {bar - past:.1f} m short of the paint"


def test_the_zebra_was_never_before_the_paint():
    """Why the old rule failed: the zebra starts at the paint or past it -- 2.2 to 3.9 m past
    it everywhere but light 22, where the two are level (within 6 cm). Holding the middle of
    the van 2.6 m short of the zebra put the bumper over the stop line at every light."""
    for light, lane, _, _, zebra, bar in MEASURED:
        if zebra is not None:
            assert zebra >= bar - 0.1, f"light {light} lane {lane}"
            assert zebra - 2.6 + 2.95 > bar, f"light {light} lane {lane}: old rule stopped short"


# ---------------------------------------------------------------- the distance to it

LANE = (5, -1)


def _route(n=60, step=2.0):
    return [Waypoint(x=i * step, y=0.0, road_id=LANE[0], lane_id=LANE[1]) for i in range(n)]


def test_the_route_distance_is_to_the_line_not_to_carlas_point():
    sig = SignalGeometry(light_id=11, stop_points=[(40.0, 0.0)], lanes={LANE},
                         lines={LANE: (46.2, 0.0)})
    got = route_signals(SignalMap({11: sig}), _route())
    assert len(got) == 1
    assert got[0].along_m == pytest.approx(46.2, abs=0.01)
    assert got[0].stop_point == (46.2, 0.0)


def test_without_a_line_it_is_carlas_point_as_before():
    sig = SignalGeometry(light_id=11, stop_points=[(40.0, 0.0)], lanes={LANE})
    got = route_signals(SignalMap({11: sig}), _route())
    assert got[0].along_m == pytest.approx(40.0, abs=0.01)


def test_where_we_are_is_not_rounded_to_a_waypoint():
    """Waypoints are 2 m apart; the old answer was good to about a metre."""
    assert _along_route(_route(), 41.3, 0.4) == pytest.approx(41.3, abs=0.01)
    assert _along_route(_route(), 0.0, 0.0) == 0.0


# ---------------------------------------------------------------- no light on the route

def _lookahead(colour, asked=None):
    sig = SignalGeometry(light_id=21, stop_points=[(20.0, 0.0)], lanes={LANE},
                         lines={LANE: (23.5, 0.0)})

    def read(light_id):
        if asked is not None:
            asked.append(light_id)
        return colour
    return TrafficLightLookahead(SignalMap({21: sig}), state_source=read)


def test_the_map_finds_the_light_on_our_lane_and_the_colour_source_reads_it():
    asked = []
    out = _lookahead(RED, asked).on_lane(LANE, 10.0, 0.0, 0.0)
    assert out.light_id == 21 and out.state == RED
    assert out.distance_m == pytest.approx(13.5, abs=0.01)
    assert asked == [21], "the colour must come from the given source (the camera)"


def test_a_colour_that_cannot_be_read_is_unknown_not_nothing():
    out = _lookahead(None).on_lane(LANE, 10.0, 0.0, 0.0)
    assert out.state == UNKNOWN


def test_a_light_on_another_lane_behind_or_beside_us_is_not_ours():
    look = _lookahead(RED)
    assert look.on_lane((9, 1), 10.0, 0.0, 0.0).light_id is None         # another lane
    assert look.on_lane(LANE, 40.0, 0.0, 0.0).light_id is None           # well past it
    assert look.on_lane(LANE, 10.0, 0.0, math.pi).light_id is None      # facing away
    assert look.on_lane(LANE, -30.0, 0.0, 0.0).light_id is None          # too far off
    assert look.on_lane(None, 10.0, 0.0, 0.0).light_id is None


def test_the_van_never_asks_the_simulator_for_a_lights_colour():
    """CARLA is for the map: where lights and lines are. What colour one shows comes from the
    camera. The fallback for a light missing from the route asked the simulator."""
    import pathlib
    main = (pathlib.Path(__file__).parent.parent / "src/warp_av/main.py").read_text()
    assert "current_light_state" not in main
