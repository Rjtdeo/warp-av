"""P-B05 (2026-09-17): the early lane-edge safety slow is a tier of its own.

Live (V3B, six bend approaches at 5-7 m/s) the parked Mustang's own track produced the early slow
in one approach; the first required slow otherwise came at 1.2-1.7 m from a hard block on the next
car's box. Replayed from the record: the V1.7 hold asked for the block margin (0.30 m) along the
intended path, and the correctly measured car cleared the swept body by 0.3-1.0 m (the true car by
0.9-1.0 m), so the "early" slow could only fire when the block would; and it was switched off
inside the 2.6 m route-sweep band, where a collision course along the intended path was answered
CLEAR. Now the hold asks for EDGE_SLOW_CLEARANCE_M (1.0 m) and looks at every stationary vehicle in
reach. The hard block is untouched.

Frame as the planner's: route along +x at y = 0; objects handed over in the van's frame (x ahead,
y to the right) exactly as the planner converts them. Case A is a recorded live tick.
"""
import math
import time
import pytest

from warp_av.behavior.behavior import BehaviorSystem
from warp_av.behavior import transitions as T
from warp_av.localization.localization import Pose
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PATH_SLOW, PATH_CLEAR, PATH_BLOCKED
from warp_av.planning import planner as planner_mod
from warp_av.planning.planner import RoutePlanner, Route, Waypoint, EDGE_SLOW_CLEARANCE_M
from test_v2a_end_to_end import van  # noqa: F401  (the end-to-end harness fixture)

FOOT = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)
CAR_LEN, CAR_WID = 4.7, 1.9


def planner():
    return RoutePlanner.__new__(RoutePlanner)          # no CARLA: the corridor filter needs no map


def straight():
    return Route(waypoints=[Waypoint(x=i * 2.0, y=0.0) for i in range(60)])


def in_van_frame(ego, wx, wy):
    ex, ey, yaw = ego
    dx, dy = wx - ex, wy - ey
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def parked_car(ego, wx, wy, length=CAR_LEN, width=CAR_WID, kind=ObjectType.VEHICLE, speed=0.0, oid=7):
    x, y = in_van_frame(ego, wx, wy)
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed, id=oid,
                          stationary=speed < 0.5, length_m=length, width_m=width, height_m=1.4,
                          box_length_m=length, box_width_m=width, box_yaw_deg=-math.degrees(ego[2]),
                          clearance_radius_m=0.5 * math.hypot(length, width), motion_class="static")


def bin_at(ego, wx, wy):
    x, y = in_van_frame(ego, wx, wy)
    return DetectedObject(object_type=ObjectType.OBSTACLE, x=x, y=y, distance=math.hypot(x, y), speed=0.0, id=9,
                          stationary=True, length_m=0.6, width_m=0.6, height_m=1.0, box_length_m=0.6, box_width_m=0.6,
                          clearance_radius_m=0.5, motion_class="static")


def decide(pl, ego, objects, route=None, intended=None):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = False
    p.closest_obstacle_distance = 999.0
    return pl.filter_to_route_corridor(p, route or straight(), *ego, danger_m=8.0, footprint=FOOT,
                                       intended_path=intended)


def behaviour_on(path, speed=6.5):
    b = BehaviorSystem()
    b.set_mission()
    b.slow_speed = 2.0
    return b.update(perception=PerceptionOutput(), pose=Pose(healthy=True, speed=speed), destination_distance=115.0,
                    safety_ok=True, path=path)


# ---------------------------------------------------------------- Case A: recorded geometry that should slow
# v3b_after/WAV-0888__2 tick 701 (t+53.8): the van 6.3 m (true gap) from the parked Mustang at 7.0 m/s, its track
# reported 2.29 x 1.79 m with its centre 11.9 m ahead (the first tick inside the hold's 12 m reach); the live
# planner said clear and nothing was required until a block at 1.7 m. Route = the rebuilt WAV-0888 route around
# the north-east bend; intended path = the trajectory the van had planned the tick before (the record's every-2 m
# sample of it).
A_POSE = (96.8, 126.57, math.radians(-43.1))
A_INTENDED = [(95.88, 127.46), (97.34, 126.09), (98.68, 124.61), (99.91, 123.03), (101.03, 121.38), (102.07, 119.67),
              (103.02, 117.91), (103.89, 116.11), (104.69, 114.27), (105.43, 112.41), (106.1, 110.53),
              (106.7, 108.62), (107.24, 106.7), (107.49, 105.73)]
A_ROUTE = [(59.33, 141.18, 0.006, False), (61.33, 141.19, 0.006, False), (63.33, 141.21, 0.006, False),
           (65.43, 141.21, -0.013, False), (67.72, 141.12, -0.068, False), (69.99, 140.9, -0.123, False),
           (72.26, 140.55, -0.178, False), (74.5, 140.09, -0.234, False), (76.71, 139.49, -0.289, False),
           (78.88, 138.78, -0.344, False), (81.02, 137.95, -0.399, False), (83.1, 137.0, -0.454, False),
           (85.13, 135.94, -0.509, False), (87.1, 134.77, -0.564, False), (89.0, 133.5, -0.619, False),
           (90.82, 132.12, -0.675, False), (92.57, 130.64, -0.73, False), (94.23, 129.07, -0.785, False),
           (95.81, 127.4, -0.84, False), (97.28, 125.67, -0.892, False), (98.64, 123.92, -0.933, False),
           (99.92, 122.11, -0.974, False), (101.13, 120.26, -1.015, False), (102.25, 118.35, -1.056, False),
           (103.31, 116.4, -1.097, False), (104.28, 114.41, -1.138, False), (105.16, 112.38, -1.179, False),
           (105.96, 110.31, -1.221, False), (106.68, 108.22, -1.262, False), (107.31, 106.09, -1.303, False),
           (107.86, 103.94, -1.344, False), (108.31, 101.77, -1.385, False), (108.67, 99.59, -1.426, False),
           (108.95, 97.39, -1.467, False), (109.13, 95.18, -1.508, False), (109.23, 92.97, -1.549, False),
           (109.25, 90.89, -1.564, False), (109.26, 88.89, -1.564, False), (109.27, 86.89, -1.564, False),
           (109.3, 82.96, -7.847, True), (109.31, 80.96, -7.847, True), (109.33, 78.96, -7.847, True),
           (109.34, 76.96, -7.847, True), (109.35, 74.96, -7.847, True), (109.37, 72.96, -7.847, True),
           (109.38, 70.96, -7.847, True), (109.4, 68.96, -7.847, True), (109.41, 66.96, -7.847, True)]
A_CAR = {"x": 105.47, "y": 118.45, "distance": 11.9, "length_m": 2.64, "width_m": 1.68, "height_m": 1.2, "yaw_deg": -54.3,
         "box_dx": 0.36, "box_dy": 0.19, "box_length_m": 2.29, "box_width_m": 1.79, "box_yaw_deg": 163.6,
         "clearance_radius_m": 1.57, "confidence": 0.82, "id": 3618, "motion_class": "dynamic",
         "motion_why": "named a road user", "size_uncertain": True, "speed": 0.0, "stationary": True}


def recorded_bend_route():
    return Route(waypoints=[Waypoint(x=x, y=y, yaw=yaw, is_junction=j) for x, y, yaw, j in A_ROUTE])


def recorded_mustang():
    x, y = in_van_frame(A_POSE, A_CAR["x"], A_CAR["y"])
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y,
                          **{k: v for k, v in A_CAR.items() if k not in ("x", "y")})


def test_case_a_the_recorded_bend_approach_is_a_required_slow_from_12m():
    out = decide(planner(), A_POSE, [recorded_mustang()], route=recorded_bend_route(), intended=A_INTENDED)
    assert out.level == PATH_SLOW and out.edge_hold is True and out.blocked is False, out.one_line()
    assert 10.5 <= out.closest_distance_m <= 12.5
    o = behaviour_on(out, speed=6.98)
    assert o.why == T.OBJECT_AHEAD_SLOW and o.desired_speed_mps == 2.0 and o.safety_required is True


def test_case_a_before_the_change_the_same_tick_was_clear(monkeypatch):
    # the V1.7 rule: the block margin along the intended path -- the live answer at that tick
    monkeypatch.setattr(planner_mod, "EDGE_SLOW_CLEARANCE_M", FOOT.safety_margin)
    out = decide(planner(), A_POSE, [recorded_mustang()], route=recorded_bend_route(), intended=A_INTENDED)
    assert out.level == PATH_CLEAR and out.edge_hold is False and out.closest_distance_m == 999.0, out.one_line()


# ---------------------------------------------------------------- Case B: a safely separated shoulder car stays clear
def test_case_b_a_safely_separated_shoulder_car_stays_clear_and_a_close_one_does_not():
    on_line = (10.0, 0.0, 0.0)
    # centre 3.9 m off the line: about 2 m body to body -> nothing
    out = decide(planner(), on_line, [parked_car(on_line, 21.0, 3.9)])
    assert out.level == PATH_CLEAR and out.edge_hold is False and out.closest_distance_m == 999.0
    # centre 2.9 m off the line: 0.96 m body to body -> the pass speed, required, never a block
    out = decide(planner(), on_line, [parked_car(on_line, 21.0, 2.9)])
    assert out.level == PATH_SLOW and out.edge_hold is True and out.blocked is False
    assert behaviour_on(out).safety_required is True
    # a moving car in the same place is left to the traffic rules
    out = decide(planner(), on_line, [parked_car(on_line, 21.0, 2.9, speed=5.0)])
    assert out.edge_hold is False and out.level == PATH_CLEAR


# ---------------------------------------------------------------- Case C: crossing the 2.6 m band leaves no CLEAR hole
@pytest.mark.parametrize("lat", [2.9, 2.7, 2.62, 2.58, 2.5, 2.4])
def test_case_c_a_car_on_the_intended_path_is_never_clear_whichever_side_of_the_band(lat):
    # the van 0.6 m off its line toward the car (inside the 1.75 m corridor), intended path parallel from there
    ego = (10.0, 0.6, 0.0)
    intended = [(10.0 + 0.5 * i, 0.6) for i in range(40)]
    out = decide(planner(), ego, [parked_car(ego, 22.0, lat)], intended=intended)
    assert out.level in (PATH_SLOW, PATH_BLOCKED), (lat, out.one_line())
    if out.level == PATH_SLOW:
        assert out.edge_hold is True and behaviour_on(out).safety_required is True
    # and from the line itself the same car is at least a required slow
    on_line = (10.0, 0.0, 0.0)
    out = decide(planner(), on_line, [parked_car(on_line, 22.0, lat)])
    assert out.level in (PATH_SLOW, PATH_BLOCKED) and (out.blocked or out.edge_hold), (lat, out.one_line())


def test_case_c_before_the_change_the_band_answered_a_collision_course_clear(monkeypatch):
    # what Step 4 found: inside the band the hold was off and the route sweep judged from the line
    monkeypatch.setattr(planner_mod, "EDGE_SLOW_CLEARANCE_M", FOOT.safety_margin)
    ego = (10.0, 0.6, 0.0)
    intended = [(10.0 + 0.5 * i, 0.6) for i in range(40)]
    # the fix keeps the hold on inside the band: with the old margin it still fires here only because
    # the body along the intended path overlaps the box (0.27 m); the OLD code answered clear
    out = decide(planner(), ego, [parked_car(ego, 22.0, 2.4)], intended=intended)
    assert out.level == PATH_SLOW and out.edge_hold is True


# ---------------------------------------------------------------- Case E: a real hard blocker still blocks
def test_case_e_a_car_in_the_lane_is_still_a_hard_block_and_a_narrow_bay_car_too():
    ego = (10.0, 0.0, 0.0)
    out = decide(planner(), ego, [parked_car(ego, 16.0, 0.6)])
    assert out.level == PATH_BLOCKED and out.blocked is True and out.edge_hold is False
    out = decide(planner(), ego, [parked_car(ego, 16.0, 1.6)])
    assert out.level == PATH_BLOCKED and out.blocked is True and out.edge_hold is False
    o = behaviour_on(out)
    assert o.should_stop is True


# ---------------------------------------------------------------- Case F: a small roadside object outside the body path
def test_case_f_a_bin_beside_the_lane_and_a_far_car_are_judged_exactly_as_before(monkeypatch):
    ego = (10.0, 0.0, 0.0)
    # a bin 1.9 m off the line, 6 m ahead: not a vehicle, so the tier never looks at it. What it gets is
    # what it always got -- the slow-zone bookkeeping of the 1.4-2.2 m band (seen at 6.0 m, a slow that the
    # behaviour makes required inside the 8 m danger distance) -- identical with the tier and without it.
    with_tier = decide(planner(), ego, [bin_at(ego, 16.0, 1.9)])
    assert with_tier.blocked is False and with_tier.edge_hold is False
    monkeypatch.setattr(planner_mod, "EDGE_SLOW_CLEARANCE_M", FOOT.safety_margin)
    without = decide(planner(), ego, [bin_at(ego, 16.0, 1.9)])
    assert (with_tier.level, with_tier.reason, with_tier.closest_distance_m, with_tier.edge_hold) == \
           (without.level, without.reason, without.closest_distance_m, without.edge_hold)
    assert behaviour_on(with_tier).safety_required == behaviour_on(without).safety_required
    monkeypatch.setattr(planner_mod, "EDGE_SLOW_CLEARANCE_M", EDGE_SLOW_CLEARANCE_M)
    out = decide(planner(), ego, [parked_car(ego, 24.0, 5.0)])
    assert out.level == PATH_CLEAR and out.edge_hold is False


# ---------------------------------------------------------------- Case G: stable and hysteretic
def test_case_g_a_box_flickering_about_the_tier_edge_keeps_one_steady_hold(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(planner_mod.time, "time", lambda: now[0])
    pl = planner()
    ego = (10.0, 0.0, 0.0)
    held = []
    for k in range(8):
        # the reported centre wobbles 2.9 / 3.35 m tick to tick: inside the tier, then 0.1 m outside it
        lat = 2.9 if k % 2 == 0 else 3.35
        out = decide(pl, ego, [parked_car(ego, 21.0, lat)])
        held.append(out.edge_hold)
        now[0] += 0.25
    assert all(held), held
    # clear of the body by the slow clearance plus the block margin for a full second: released
    for _ in range(6):
        out = decide(pl, ego, [parked_car(ego, 21.0, 4.6)])
        now[0] += 0.25
    assert out.edge_hold is False and out.level == PATH_CLEAR


# ---------------------------------------------------------------- Step 11: the slow becomes braking, end to end
def test_step11_the_early_slow_reaches_the_brake_pedal_through_the_whole_chain(van):
    from test_v2a_end_to_end import start_mission, place, see, chain
    start_mission(van)
    place(van, x=0.0, y=0.0, yaw=0.0, speed=6.5)
    see(van)
    van.tick()                                       # a cruise tick: the intended path from this pose
    car = parked_car((0.0, 0.0, 0.0), 11.0, 2.9)     # the Case B car: 0.96 m body to body, 11 m ahead
    place(van, x=0.0, y=0.0, yaw=0.0, speed=6.5)
    see(van, [car])
    van.tick()
    c = chain(van)
    print("Step 11 chain:", {k: c[k] for k in ("planner_level", "why", "safety_required", "raw_mps", "eased_mps", "brake", "throttle")})
    assert c["planner_level"] == PATH_SLOW and getattr(van._path, "edge_hold", False) is True, c
    assert c["why"] == T.OBJECT_AHEAD_SLOW, c
    assert c["safety_required"] is True, c
    assert c["raw_mps"] == van.behavior.slow_speed < 6.5, c      # the pass speed, from 6.5 m/s
    assert c["brake"] > 0.0 and c["throttle"] == 0.0, c


def test_step11_without_the_tier_the_same_car_was_cruised_past(van, monkeypatch):
    from test_v2a_end_to_end import start_mission, place, see, chain
    monkeypatch.setattr(planner_mod, "EDGE_SLOW_CLEARANCE_M", FOOT.safety_margin)
    start_mission(van)
    place(van, x=0.0, y=0.0, yaw=0.0, speed=6.5)
    see(van)
    van.tick()
    place(van, x=0.0, y=0.0, yaw=0.0, speed=6.5)
    see(van, [parked_car((0.0, 0.0, 0.0), 11.0, 2.9)])
    van.tick()
    c = chain(van)
    assert c["planner_level"] == PATH_CLEAR and c["safety_required"] is False and c["brake"] == 0.0, c
