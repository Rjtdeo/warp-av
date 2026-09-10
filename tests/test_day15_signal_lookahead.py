"""Knowing about a traffic light before you are underneath it.

The fault this replaces, measured on Town10HD 2026-09-09 by backing the van away from a
red light: nothing reported at 30, 20, 10, 6 or 4 m, and RED first seen at 2 m. At 8 m/s
the van needs about 14 m to stop, so it could not stop for a red light at all.
"""
import math

import pytest

from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior, LIGHT_MEANS_STOP
from warp_av.localization.localization import Pose
from warp_av.perception.perception import PerceptionOutput, ObjectType
from warp_av.planning.planner import Route, Waypoint
from warp_av.perception.traffic_lights import (
    SignalMap, SignalGeometry, TrafficLightLookahead, route_signals, SignalAhead,
    LOOKAHEAD_M, NEAR_M, FAR_CHECK_HZ, NEAR_CHECK_HZ, PASSED_BY_M, STATE_STALE_S,
    RED, YELLOW, GREEN, UNKNOWN, NONE)


# ---------------------------------------------------------------- a little world

OUR_LANE = (7, -1)
CROSS_LANE = (99, 1)


def straight_route(length_m=200.0, step=2.0, lane=OUR_LANE):
    """A straight road east, every 2 m, all on one lane -- like CARLA's own routes."""
    n = int(length_m / step) + 1
    return Route(waypoints=[Waypoint(x=i * step, y=0.0, road_id=lane[0], lane_id=lane[1])
                            for i in range(n)])


def signal_at(light_id, x, y=0.0, lane=OUR_LANE):
    return SignalGeometry(light_id=light_id, stop_points=[(x, y)], lanes={lane})


def a_map(*signals):
    return SignalMap({s.light_id: s for s in signals})


def _explodes(_light_id):
    raise RuntimeError("the simulator went away")


def fixed_colour(colour, counter=None):
    def read(light_id):
        if counter is not None:
            counter.append(light_id)
        return colour
    return read


# ---------------------------------------------------------------- 1-2: nothing to do

def test_no_route_returns_at_once_and_asks_nothing():
    asked = []
    look = TrafficLightLookahead(a_map(signal_at(1, 60.0)), fixed_colour(RED, asked))
    out = look.update(None, 0.0, 0.0, now=100.0)
    assert out.state == NONE and out.light_id is None and out.watching is False
    assert asked == [], "asked the simulator with no route at all"


def test_a_route_with_no_signals_on_it_asks_nothing():
    asked = []
    look = TrafficLightLookahead(a_map(signal_at(1, 60.0, lane=CROSS_LANE)), fixed_colour(RED, asked))
    out = look.update(straight_route(), 0.0, 0.0, now=100.0)
    assert out.state == NONE and out.light_id is None
    assert asked == []


# ---------------------------------------------------------------- 3-5: the RIGHT signal

def test_the_signal_on_our_lane_is_chosen_from_several():
    """Three lights, one of them ours. The others are nearer in a straight line."""
    ours = signal_at(10, 80.0, y=0.0, lane=OUR_LANE)
    cross_near = signal_at(11, 78.0, y=6.0, lane=CROSS_LANE)
    cross_nearer = signal_at(12, 76.0, y=-4.0, lane=(55, 3))
    look = TrafficLightLookahead(a_map(ours, cross_near, cross_nearer), fixed_colour(GREEN))
    out = look.update(straight_route(), 40.0, 0.0, now=100.0)
    assert out.light_id == 10, "picked a light that does not govern our lane"


def test_cross_traffic_is_never_chosen():
    """Measured live: at the first junction another light sat 12.8 m from ours and governs
    the crossing road. Nearest-distance would have stopped the van for cross traffic."""
    cross = signal_at(11, 62.8, y=0.0, lane=CROSS_LANE)
    look = TrafficLightLookahead(a_map(cross), fixed_colour(RED))
    out = look.update(straight_route(), 50.0, 0.0, now=100.0)
    assert out.light_id is None and out.state == NONE


def test_a_signal_behind_us_is_not_chosen():
    behind = signal_at(10, 20.0)
    ahead = signal_at(20, 120.0)
    look = TrafficLightLookahead(a_map(behind, ahead), fixed_colour(GREEN))
    out = look.update(straight_route(), 60.0, 0.0, now=100.0)
    assert out.light_id == 20


# ---------------------------------------------------------------- 6-9: when it looks

def test_beyond_the_lookahead_it_is_known_about_but_not_asked_about():
    asked = []
    look = TrafficLightLookahead(a_map(signal_at(10, 150.0)), fixed_colour(RED, asked))
    out = look.update(straight_route(), 0.0, 0.0, now=100.0)
    assert out.light_id == 10, "should still know the signal is there"
    assert out.distance_m == pytest.approx(150.0, abs=2.0)
    assert out.watching is False and out.state == NONE
    assert asked == [], "asked for a colour 150 m out"


def test_crossing_the_lookahead_line_switches_it_on():
    asked = []
    look = TrafficLightLookahead(a_map(signal_at(10, 100.0)), fixed_colour(RED, asked))
    look.update(straight_route(), 100.0 - LOOKAHEAD_M - 5.0, 0.0, now=100.0)
    assert asked == []
    out = look.update(straight_route(), 100.0 - LOOKAHEAD_M + 2.0, 0.0, now=100.1)
    assert out.watching is True and out.state == RED
    assert len(asked) == 1


def measure_rate(look, ego_x, seconds=3.0, ticks_hz=100.0):
    """Ask it as fast as the van thinks, and count how often it really queries."""
    asked = []
    look.state_source = fixed_colour(RED, asked)
    route = straight_route()
    now = 100.0
    step = 1.0 / ticks_hz
    while now < 100.0 + seconds:
        look.update(route, ego_x, 0.0, now=now)
        now += step
    return len(asked) / seconds


def test_in_the_far_band_it_asks_at_about_the_far_rate():
    look = TrafficLightLookahead(a_map(signal_at(10, 100.0)))
    hz = measure_rate(look, ego_x=100.0 - (LOOKAHEAD_M + NEAR_M) / 2.0)
    assert FAR_CHECK_HZ - 1.0 <= hz <= FAR_CHECK_HZ + 1.0, f"asked at {hz:.1f} Hz"


def test_in_the_near_band_it_asks_at_about_the_near_rate():
    look = TrafficLightLookahead(a_map(signal_at(10, 100.0)))
    hz = measure_rate(look, ego_x=100.0 - NEAR_M / 2.0)
    assert NEAR_CHECK_HZ - 1.5 <= hz <= NEAR_CHECK_HZ + 1.5, f"asked at {hz:.1f} Hz"


def test_it_asks_far_more_often_close_up_than_far_out():
    far = measure_rate(TrafficLightLookahead(a_map(signal_at(10, 100.0))),
                       ego_x=100.0 - (LOOKAHEAD_M + NEAR_M) / 2.0)
    near = measure_rate(TrafficLightLookahead(a_map(signal_at(10, 100.0))),
                        ego_x=100.0 - NEAR_M / 2.0)
    assert near > far * 1.8


# ---------------------------------------------------------------- 10-13: what it means

def light_decision(state, distance):
    """Driven the same way tests/test_traffic_lights.py drives it."""
    b = BehaviorSystem()
    b.set_mission()
    return b.update(PerceptionOutput(traffic_light=state, traffic_light_distance_m=distance),
                    Pose(healthy=True), 500, True)


def test_red_seen_far_enough_out_reaches_the_driving_decision():
    out = light_decision(RED, 30.0)
    assert "RED" in out.reason.upper()
    assert out.desired_speed_mps < 8.0


def test_green_does_not_stop_the_van():
    out = light_decision(GREEN, 30.0)
    assert out.behavior is not DrivingBehavior.STOPPED_RED_LIGHT
    assert "GREEN" not in out.reason.upper()


def test_yellow_behaves_as_it_always_did():
    assert "yellow" in LIGHT_MEANS_STOP
    out = light_decision(YELLOW, 30.0)
    assert "YELLOW" in out.reason.upper()


def test_an_unreadable_light_is_never_taken_as_green():
    """The whole point of requirement 5."""
    assert UNKNOWN in LIGHT_MEANS_STOP
    out = light_decision(UNKNOWN, 12.0)
    assert out.should_stop is True or out.desired_speed_mps < 8.0
    assert "UNREADABLE" in out.reason.upper()
    assert "GREEN" not in out.reason.upper()


def test_a_stale_colour_becomes_unknown_rather_than_a_stale_green():
    look = TrafficLightLookahead(a_map(signal_at(10, 100.0)), fixed_colour(GREEN))
    route = straight_route()
    out = look.update(route, 90.0, 0.0, now=100.0)
    assert out.state == GREEN
    look.state_source = _explodes
    # nothing new can be read, and time passes
    out = look.update(route, 90.0, 0.0, now=100.0 + STATE_STALE_S + 0.5)
    assert out.state == UNKNOWN, "an old green must not keep counting as green"


# ---------------------------------------------------------------- 14-15: moving on

def test_once_past_the_line_it_moves_to_the_next_signal():
    look = TrafficLightLookahead(a_map(signal_at(10, 60.0), signal_at(20, 160.0)),
                                 fixed_colour(RED))
    route = straight_route()
    assert look.update(route, 50.0, 0.0, now=100.0).light_id == 10
    after = look.update(route, 60.0 + PASSED_BY_M + 2.0, 0.0, now=100.5)
    assert after.light_id == 20, "still watching the signal it has driven past"


def test_the_old_colour_is_dropped_when_the_signal_changes():
    look = TrafficLightLookahead(a_map(signal_at(10, 60.0), signal_at(20, 160.0)),
                                 fixed_colour(RED))
    route = straight_route()
    look.update(route, 50.0, 0.0, now=100.0)
    out = look.update(route, 60.0 + PASSED_BY_M + 2.0, 0.0, now=100.05)
    # too far for the new one, so no colour has been read for it yet
    assert out.light_id == 20 and out.state in (NONE, RED)
    assert out.state != YELLOW


def test_a_new_route_forgets_the_old_association():
    look = TrafficLightLookahead(a_map(signal_at(10, 60.0)), fixed_colour(RED))
    old = straight_route()
    assert look.update(old, 40.0, 0.0, now=100.0).light_id == 10
    fresh = straight_route(lane=CROSS_LANE)      # a different road entirely
    fresh.timestamp = old.timestamp + 10.0
    out = look.update(fresh, 40.0, 0.0, now=101.0)
    assert out.light_id is None, "kept a signal from the old route"


# ---------------------------------------------------------------- the distance itself

def test_distance_is_measured_along_the_route_not_as_the_crow_flies():
    """A signal can be near in a straight line and far along the road. An L-shaped route:
    the signal sits 20 m away across the corner but 120 m along the road."""
    wps = [Waypoint(x=float(i) * 2, y=0.0, road_id=7, lane_id=-1) for i in range(31)]     # east 60 m
    wps += [Waypoint(x=60.0, y=float(i) * 2, road_id=7, lane_id=-1) for i in range(1, 31)]  # north 60 m
    route = Route(waypoints=wps)
    sig = signal_at(10, 60.0, y=60.0)
    look = TrafficLightLookahead(a_map(sig), fixed_colour(RED))
    out = look.update(route, 0.0, 0.0, now=100.0)
    crow = math.hypot(60.0, 60.0)
    assert crow < 90.0
    assert out.distance_m == pytest.approx(120.0, abs=4.0), "used the straight-line distance"


def test_signals_come_back_in_the_order_we_will_meet_them():
    sigs = route_signals(a_map(signal_at(30, 160.0), signal_at(10, 40.0), signal_at(20, 100.0)),
                         straight_route().waypoints)
    assert [s.light_id for s in sigs] == [10, 20, 30]


def test_waypoints_without_road_ids_are_simply_ignored():
    """A route not built from a CARLA map must not crash anything."""
    route = Route(waypoints=[Waypoint(x=float(i) * 2, y=0.0) for i in range(50)])
    look = TrafficLightLookahead(a_map(signal_at(10, 60.0)), fixed_colour(RED))
    assert look.update(route, 20.0, 0.0, now=100.0).light_id is None
