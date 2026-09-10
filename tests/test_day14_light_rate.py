"""Asking the simulator what colour the light is, a few times a second instead of ten.

Measured on the van 2026-09-09: one `get_traffic_light()` costs about 65 ms, worst case
121 ms -- it is a search on the simulator's side, not a lookup -- and it was 29 % of the
whole tick. Watched for 30 seconds at 10 a second, the answer changed 0 times in 190
readings.
"""
import types

import pytest

from warp_av.perception.perception import (PerceptionSystem, LIGHT_NEAR_REFRESH_S,
                                           LIGHT_FAR_REFRESH_S)


class FakeLight:
    def __init__(self, state="Red"):
        self.id = 1
        self._state = state

    def get_state(self):
        return f"TrafficLightState.{self._state}"

    def get_stop_waypoints(self):
        loc = types.SimpleNamespace(x=10.0, y=0.0)
        return [types.SimpleNamespace(transform=types.SimpleNamespace(location=loc))]


class FakeVehicle:
    def __init__(self, light=None):
        self.light = light
        self.asks = 0

    def get_traffic_light(self):
        self.asks += 1
        return self.light

    def get_location(self):
        return types.SimpleNamespace(x=0.0, y=0.0)


def system(light=None):
    p = object.__new__(PerceptionSystem)
    p.vehicle = FakeVehicle(light)
    p._tl_stop_cache = {}
    p._tl_asked_at = None
    p._tl_last = ("none", None)
    return p


def test_the_first_question_is_always_asked():
    p = system(FakeLight("Red"))
    assert p.current_light_state(now=100.0)[0] == "red"
    assert p.vehicle.asks == 1


def test_it_is_not_asked_again_straight_away():
    p = system(FakeLight("Red"))
    p.current_light_state(now=100.0)
    for k in range(8):
        p.current_light_state(now=100.0 + 0.01 * k)
    assert p.vehicle.asks == 1, "asked the simulator every tick again"


def test_with_a_light_there_it_is_asked_often():
    p = system(FakeLight("Red"))
    p.current_light_state(now=100.0)
    p.current_light_state(now=100.0 + LIGHT_NEAR_REFRESH_S + 0.01)
    assert p.vehicle.asks == 2


def test_with_no_light_at_all_it_is_asked_rarely():
    """Mid-road there is nothing to be late about."""
    p = system(None)
    p.current_light_state(now=100.0)
    p.current_light_state(now=100.0 + LIGHT_NEAR_REFRESH_S + 0.01)
    assert p.vehicle.asks == 1, "asking often when there is no light is wasted time"
    p.current_light_state(now=100.0 + LIGHT_FAR_REFRESH_S + 0.01)
    assert p.vehicle.asks == 2


def test_a_light_turning_red_is_noticed_within_the_refresh():
    p = system(FakeLight("Green"))
    assert p.current_light_state(now=100.0)[0] == "green"
    p.vehicle.light._state = "Red"
    # still the old answer, briefly
    assert p.current_light_state(now=100.0 + LIGHT_NEAR_REFRESH_S / 2)[0] == "green"
    # and the new one, once it is asked again
    assert p.current_light_state(now=100.0 + LIGHT_NEAR_REFRESH_S + 0.01)[0] == "red"


def test_being_late_costs_a_metre_or_so_not_a_stopping_distance():
    """The whole safety argument, written down: at 8 m/s the worst delay is the near
    refresh, and the van's measured stopping margin is about 7.5 m."""
    worst_late_m = 8.0 * LIGHT_NEAR_REFRESH_S
    assert worst_late_m < 2.0
    assert worst_late_m < 7.5 / 3.0


def test_the_stop_line_distance_still_comes_back():
    p = system(FakeLight("Red"))
    state, dist = p.current_light_state(now=100.0)
    assert state == "red"
    assert dist == pytest.approx(10.0)
