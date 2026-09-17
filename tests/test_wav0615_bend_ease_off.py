"""This test exists because WAV-0615 hit a parked car after easing a safety-relevant slowdown.

V1.5, 2026-09-17. Run v1_full_A, scenario WAV-0615, north-east bend of Town10HD, two cars parked
in the shoulder. Reproduced in run v15_before (cruise 8 m/s: 2 collisions in 4 crossings).

The numbers below are the trace at T1 (t+35.8 s):
  * the van was doing 6.65 m/s; its previous speed target was 7.6 m/s;
  * the planner's nearest object was a stationary vehicle 8.7 m ahead, 2.54 m to the side,
    on the swept path but released by the free-space second opinion -> level SLOW;
  * rule 18 (object_ahead) asked for 2.0 m/s -- the van needed 9.4 m to get there under its
    own comfort model and had 8.7 m (4.0 m by CARLA truth);
  * the comfort ease-off raised the request to 7.25 m/s (7.6 - 0.35), the controller kept
    throttle 0.45, the first brake came 0.6 s later from the occupancy stop, 0.2 s before
    the impact at 6.6 m/s.

Pre-fix code fails here: BehaviorOutput had no idea whether a request was required or a
comfort choice, and the ease-off treated every slow-zone request as comfort.
"""
from types import SimpleNamespace

import pytest

from warp_av.behavior.behavior import (BehaviorSystem, EASE_OFF_MPS, distance_to_slow,
                                       stopping_speed_for)
from warp_av.behavior import transitions as T
from warp_av.control.controller import VehicleController
from warp_av.localization.localization import Pose
from warp_av.main import WarpAV
from warp_av.perception.perception import ObjectType, PerceptionOutput
from warp_av.planning.instrumentation import (PlannerDecision, BLOCKED_SWEPT_PATH, GROUND_RELEASED,
                                              PATH_SLOW)

SPEED = 6.65          # m/s, the van at T1
PREVIOUS_TARGET = 7.6  # m/s, what the ease-off would ease from
CAR_AHEAD_M = 8.7      # the planner's distance to the parked car
SLOW_SPEED = 2.0       # rule 18's request in camera mode


def van():
    b = BehaviorSystem()
    b.set_mission()
    b.slow_speed = SLOW_SPEED
    return b


def the_bend_at_t1():
    """The path record exactly as WAV-0615 had it: a swept-path block released by the ground."""
    return PlannerDecision(reason=BLOCKED_SWEPT_PATH, level=PATH_SLOW, closest_distance_m=CAR_AHEAD_M,
                           closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0, closest_lateral_m=2.54,
                           second_opinion=GROUND_RELEASED)


def decide(b, path, speed, **kw):
    return b.update(perception=PerceptionOutput(), pose=Pose(healthy=True, speed=speed),
                    destination_distance=115.0, safety_ok=True, path=path, **kw)


def ease(out, previous_target):
    """The ease-off exactly as main.py runs it, on a stand-in for the system."""
    stub = SimpleNamespace(_eased_speed=previous_target)
    WarpAV._ease_off(stub, out)
    return stub


def command(desired, speed=SPEED):
    c = VehicleController()
    return c.compute_command(current_x=0.0, current_y=0.0, current_yaw=0.0, current_speed=speed,
                             target_x=10.0, target_y=0.0, desired_speed=desired, should_stop=False)


def test_the_distance_the_van_needs_is_more_than_it_had():
    assert distance_to_slow(SPEED, SLOW_SPEED) == pytest.approx(9.36, abs=0.05)
    assert distance_to_slow(SPEED, SLOW_SPEED) > CAR_AHEAD_M
    assert distance_to_slow(1.0, 2.0) == 0.0           # already slower: nothing to do


def test_wav0615_the_slow_request_is_required_and_is_not_eased():
    out = decide(van(), the_bend_at_t1(), SPEED)
    assert out.why == T.OBJECT_AHEAD_SLOW and out.desired_speed_mps == SLOW_SPEED and not out.should_stop
    assert out.safety_required is True
    stub = ease(out, PREVIOUS_TARGET)
    # pre-fix: max(2.0, 7.6 - 0.35) = 7.25 and "| easing off"
    assert out.desired_speed_mps == SLOW_SPEED
    assert out.desired_speed_mps != max(SLOW_SPEED, PREVIOUS_TARGET - EASE_OFF_MPS)
    assert "easing off" not in out.reason
    assert stub._speed_request == {"raw": 2.0, "eased": 2.0, "why": T.OBJECT_AHEAD_SLOW, "safety": True, "stop": False}


def test_wav0615_the_controller_then_brakes_instead_of_pulling():
    cmd = command(SLOW_SPEED)
    assert cmd.brake > 0.0 and cmd.throttle == 0.0
    # ...which is what the pre-fix target did NOT get: 7.25 m/s at 6.65 m/s is a pull
    eased_as_before = max(SLOW_SPEED, PREVIOUS_TARGET - EASE_OFF_MPS)
    old = command(eased_as_before)
    assert old.throttle > 0.0 and old.brake == 0.0


def test_a_slow_zone_object_with_room_still_eases_for_comfort():
    # 19 m: inside the 20 m slow zone, but 13.2 m is all the van needs to reach 2.0 m/s from 8 m/s
    far = PlannerDecision(reason=BLOCKED_SWEPT_PATH, level=PATH_SLOW, closest_distance_m=19.0,
                          closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0, second_opinion=GROUND_RELEASED)
    out = decide(van(), far, 8.0)
    assert out.why == T.OBJECT_AHEAD_SLOW and out.safety_required is False
    assert distance_to_slow(8.0, SLOW_SPEED) == pytest.approx(13.2) and distance_to_slow(8.0, SLOW_SPEED) < 19.0
    ease(out, 8.0)
    assert out.desired_speed_mps == pytest.approx(8.0 - EASE_OFF_MPS) and "easing off" in out.reason


def test_the_cannot_see_past_cap_is_required_too():
    out = decide(van(), PlannerDecision(reason="clear"), 8.0, seen_ahead_m=10.8)
    assert out.desired_speed_mps == pytest.approx(stopping_speed_for(10.8))
    assert out.safety_required is True and "only seen clear" in out.reason
    ease(out, 8.0)
    assert out.desired_speed_mps == pytest.approx(stopping_speed_for(10.8))


def test_a_stop_and_plain_cruising_are_untouched():
    out = decide(van(), PlannerDecision(reason="clear"), 5.0)
    assert out.why == T.ROUTE_CLEAR and out.safety_required is False
    stub = ease(out, 5.0)
    assert stub._eased_speed == out.desired_speed_mps
    out = decide(van(), PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=6.0,
                                        closest_kind=ObjectType.VEHICLE), 5.0)
    assert out.should_stop
    stub = ease(out, 5.0)
    assert stub._eased_speed == 0.0 and stub._speed_request["stop"] is True
