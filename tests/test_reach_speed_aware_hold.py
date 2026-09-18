"""P-B05 follow-up (2026-09-18): the early lane-edge slow's reach grows with speed.

The hold looked 12 m ahead of the van's centre (FOOTPRINT_STATIONARY_REACH_M, the BLOCK sweep's reach) whatever the
speed, and the hold exists to reach 2 m/s before the body is within 1.0 m of the car -- which takes 0.4 s + v^2 / 6 of
road in the stack's own model (behavior.distance_to_slow). Recorded at 6.5-7 m/s the gate opened with 5-7 m of free run
where 8.5-10.7 m were needed; the brake stepped to 0.6 and the van stopped dead. Now the tick hands the planner
distance_to_slow(v, slow) + the van's length as the reach, floored at 12 m. Cases A-H of the brief, the Step 6 replay
of two recorded approaches (tests/fixtures/reach), and the Step 12 chain in the real stack."""
import json
import math
import os

import pytest

from test_pb05_early_lane_edge_slow import planner, straight, parked_car, decide, FOOT, in_van_frame
from test_v2a_end_to_end import van, place, see, start_mission, chain, straight_road  # noqa: F401
from warp_av.behavior.behavior import distance_to_slow
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PATH_SLOW, PATH_CLEAR, PATH_BLOCKED
from warp_av.planning.planner import Route, Waypoint, FOOTPRINT_STATIONARY_REACH_M

HERE = os.path.dirname(__file__)
FIX = json.load(open(os.path.join(HERE, "fixtures", "reach", "bend_approach_ticks.json"), encoding="utf-8"))
ROUTES = json.load(open(os.path.join(HERE, "fixtures", "endpoint", "recorded_routes.json"), encoding="utf-8"))
SLOW = 2.0


def reach_for(speed, fp=FOOT):
    """What main.py hands the planner: the road a comfortable required slow needs plus the van's length."""
    return distance_to_slow(speed, SLOW) + 2.0 * fp.half_length


def route_of(name):
    return Route(waypoints=[Waypoint(x=p["x"], y=p["y"], yaw=p["yaw"], is_junction=bool(p["is_junction"]),
                                     road_id=p.get("road_id"), lane_id=p.get("lane_id")) for p in ROUTES[name]["route"]])


def published_object(o, pose):
    """A track as /api/state published it (world x, y), back in the van's frame as the planner takes it."""
    x, y = in_van_frame(pose, o["x"], o["y"])
    return DetectedObject(object_type=ObjectType(o["type"]), x=x, y=y, distance=o["distance"], speed=o.get("speed", 0.0),
                          confidence=o.get("confidence", 1.0), id=o["id"], length_m=o.get("length_m", 0.0), width_m=o.get("width_m", 0.0),
                          height_m=o.get("height_m", 0.0), yaw_deg=o.get("yaw_deg", 0.0), stationary=o.get("stationary", True),
                          size_uncertain=o.get("size_uncertain", False), clearance_radius_m=o.get("clearance_radius_m", 0.4),
                          motion_class=o.get("motion_class", "dynamic"), static_rule=o.get("static_rule", ""), motion_why=o.get("motion_why", ""),
                          box_dx=o.get("box_dx", 0.0), box_dy=o.get("box_dy", 0.0), box_length_m=o.get("box_length_m", 0.0),
                          box_width_m=o.get("box_width_m", 0.0), box_yaw_deg=o.get("box_yaw_deg", 0.0))


def replay(key, speed_aware):
    """The recorded approach through the real corridor filter, tick after tick, with the reach as recorded (12 m) or
    as the tick now hands it in. Returns one record per tick."""
    fx = FIX[key]; fp = VehicleFootprint(*fx["footprint"]); route = route_of(fx["route_fixture"].split(":")[1])
    rp = planner(); rp._edge_hold = None
    out = []
    for tk in fx["ticks"]:
        pose = (tk["pose"][0], tk["pose"][1], math.radians(tk["pose"][2]))
        objs = [published_object(tk["mustang"], pose)] if tk["mustang"] else []
        p = PerceptionOutput(objects=objs); p.path_blocked = False; p.closest_obstacle_distance = 999.0
        dec = rp.filter_to_route_corridor(p, route, *pose, danger_m=8.0, footprint=fp, intended_path=tk["intended"],
                                          edge_reach_m=(reach_for(tk["speed"], fp) if speed_aware else None))
        out.append(dict(t=tk["t"], gap=tk["gap_truth_m"], speed=tk["speed"], cx=(objs[0].x if objs else None),
                        hold=bool(dec.edge_hold), level=dec.level, reach=dec.edge_reach_m, recorded=tk["recorded"]["edge_hold"],
                        need=distance_to_slow(tk["speed"], SLOW)))
    return out


# ---------------------------------------------------------------- Step 6: the recorded holds, exactly
def test_step6_the_recorded_high_speed_hold_is_reproduced_tick_for_tick_at_12_m():
    seq = replay("high_speed_WAV-0888", speed_aware=False)
    assert [r["hold"] for r in seq] == [r["recorded"] for r in seq], [(r["t"], r["hold"], r["recorded"]) for r in seq]
    first = next(r for r in seq if r["hold"])
    assert first["t"] == 45.4 and abs(first["gap"] - 5.35) < 0.05 and 10.5 < first["cx"] <= 12.0, first
    assert all(r["reach"] == 12.0 for r in seq)
    # ...and the tick before it was held back by the gate alone: the centroid 12.2 m out, the box already 3.2 m long
    before = seq[seq.index(first) - 1]
    assert not before["hold"] and 12.0 < before["cx"] < 12.6 and before["speed"] > 6.9, before


def test_step6_the_recorded_five_mps_hold_is_reproduced_at_12_m():
    seq = replay("five_mps_BEND5", speed_aware=False)
    assert [r["hold"] for r in seq] == [r["recorded"] for r in seq], [(r["t"], r["hold"], r["recorded"]) for r in seq]
    first = next(r for r in seq if r["hold"])
    assert abs(first["gap"] - 5.08) < 0.05 and first["speed"] < 4.5, first


# ---------------------------------------------------------------- Case B: high speed
def test_case_b_at_seven_mps_the_hold_opens_where_the_slow_needs_it_not_at_12_m():
    seq = replay("high_speed_WAV-0888", speed_aware=True)
    first = next(r for r in seq if r["hold"])
    old = next(r for r in replay("high_speed_WAV-0888", speed_aware=False) if r["hold"])
    assert first["t"] == 44.8 and abs(first["gap"] - 9.42) < 0.05 and 15.5 < first["cx"] < 16.0, first
    assert first["reach"] >= 16.0 and first["reach"] == round(max(12.0, reach_for(first["speed"])), 1)
    assert first["gap"] - old["gap"] >= 4.0                                   # four metres of road more, at 7 m/s
    free_run = first["gap"] - 1.0                                              # to the 1.0 m slow clearance
    assert free_run >= 0.8 * first["need"], (free_run, first["need"])         # about what the slow needs (10.4 m)
    assert all(r["hold"] for r in seq if r["t"] >= first["t"])                # and it stays held


# ---------------------------------------------------------------- Case C: low speed unchanged
def test_case_c_at_five_mps_nothing_changes_the_reach_is_the_12_m_floor():
    with_rule = replay("five_mps_BEND5", speed_aware=True)
    without = replay("five_mps_BEND5", speed_aware=False)
    assert [r["hold"] for r in with_rule] == [r["hold"] for r in without]
    assert all(r["reach"] == 12.0 for r in with_rule), [r["reach"] for r in with_rule]
    assert all(r["speed"] < 5.0 and r["need"] < 5.0 for r in with_rule)


# ---------------------------------------------------------------- Case A: a long parked car, its centre beyond 12 m
def car_beside(ego, x, y, length=4.7, width=1.9):
    return parked_car(ego, x, y, length=length, width=width)


def test_case_a_a_long_parked_car_is_eligible_by_its_near_edge_when_the_speed_says_so():
    ego = (0.0, 0.0, 0.0)
    car = car_beside(ego, 14.5, 2.3)                                            # near edge 12.15 m, inner side 1.35 m off the line
    old = decide(planner(), ego, [car])
    new = decide(planner(), ego, [car], intended=None) if False else None
    pl = planner(); p = PerceptionOutput(objects=[car]); p.path_blocked = False; p.closest_obstacle_distance = 999.0
    new = pl.filter_to_route_corridor(p, straight(), *ego, danger_m=8.0, footprint=FOOT, edge_reach_m=reach_for(7.0))
    assert not old.edge_hold and old.level == PATH_CLEAR, (old.level, old.reason)   # 12 m: the centroid is 14.5 m out
    assert new.edge_hold and new.level == PATH_SLOW and new.edge_reach_m >= 16.0, (new.level, new.reason)


# ---------------------------------------------------------------- Case D / G: safe shoulder, the 1.0 m clearance intact
@pytest.mark.parametrize("y_centre, held", [(2.85, True), (3.3, False), (3.9, False)])
def test_case_d_and_g_the_one_metre_clearance_decides_at_the_new_reach_too(y_centre, held):
    """The slow body reaches 1.994 m from the line (half-width 0.994 + EDGE_SLOW_CLEARANCE_M): a car whose inner side
    is 1.9 m off holds; 2.35 m off (the box carries a 0.1 m pad) and further does not -- at 7 m/s and 16 m out just
    as at 12 m (P-B05: 2.9 m car centre required the slow, 3.9 m clear)."""
    ego = (0.0, 0.0, 0.0)
    car = car_beside(ego, 14.5, y_centre)
    pl = planner(); p = PerceptionOutput(objects=[car]); p.path_blocked = False; p.closest_obstacle_distance = 999.0
    dec = pl.filter_to_route_corridor(p, straight(), *ego, danger_m=8.0, footprint=FOOT, edge_reach_m=reach_for(7.0))
    assert bool(dec.edge_hold) is held, (y_centre, dec.level, dec.reason)
    if not held:
        assert dec.level == PATH_CLEAR


# ---------------------------------------------------------------- Case E: far but irrelevant
def test_case_e_a_car_beyond_the_reach_is_not_looked_at():
    ego = (0.0, 0.0, 0.0)
    car = car_beside(ego, 30.0, 2.3)
    pl = planner(); p = PerceptionOutput(objects=[car]); p.path_blocked = False; p.closest_obstacle_distance = 999.0
    dec = pl.filter_to_route_corridor(p, straight(), *ego, danger_m=8.0, footprint=FOOT, edge_reach_m=reach_for(7.0))
    assert not dec.edge_hold and dec.level == PATH_CLEAR and dec.edge_reach_m < 20.0


# ---------------------------------------------------------------- Case F: a real blocker still blocks
def test_case_f_a_car_in_the_lane_is_blocked_with_either_reach():
    ego = (0.0, 0.0, 0.0)
    car = car_beside(ego, 9.0, 0.0)
    for reach in (None, reach_for(7.0)):
        pl = planner(); p = PerceptionOutput(objects=[car]); p.path_blocked = False; p.closest_obstacle_distance = 999.0
        dec = pl.filter_to_route_corridor(p, straight(), *ego, danger_m=8.0, footprint=FOOT, edge_reach_m=reach)
        assert dec.level == PATH_BLOCKED, (reach, dec.level, dec.reason)


# ---------------------------------------------------------------- Case H: the reach itself
def test_case_h_the_reach_is_the_slow_distance_plus_the_van_and_never_below_12_m():
    assert distance_to_slow(0.0, SLOW) == 0.0 and reach_for(0.0) == 2.0 * FOOT.half_length
    r = [max(12.0, reach_for(v)) for v in (0.0, 2.0, 4.0, 5.0, 5.28, 6.0, 6.5, 7.0, 8.0)]
    assert r[:4] == [12.0] * 4 and abs(r[4] - 12.0) < 0.1                          # the floor to about 5.3 m/s
    assert 15.9 < r[7] < 16.7 and 18.5 < r[8] < 19.5, r                          # 16.3 m at 7 m/s, 19.1 at 8
    assert all(b >= a for a, b in zip(r, r[1:]))
    # a junk reach falls back to 12 m (the record carries the reach once something was looked at)
    pl = planner(); p = PerceptionOutput(objects=[car_beside((0.0, 0.0, 0.0), 30.0, 2.3)]); p.path_blocked = False; p.closest_obstacle_distance = 999.0
    assert pl.filter_to_route_corridor(p, straight(), 0.0, 0.0, 0.0, danger_m=8.0, footprint=FOOT, edge_reach_m="x").edge_reach_m == 12.0


# ---------------------------------------------------------------- Step 12: the chain, in the real stack
def test_step12_earlier_eligibility_is_an_earlier_required_slow_and_an_earlier_brake(van, monkeypatch):
    """The van at 7 m/s on a straight, a parked car with its centroid 14.5 m ahead 2.3 m off the line: the planner
    says slow with the hold, the behaviour makes it required, and the controller brakes on the same tick. With the
    reach held at 12 m (the slow distance zeroed) the same tick is clear and the throttle stays on."""
    import warp_av.main as main
    start_mission(van, route=straight_road(), dest=(220.0, 0.0))
    van.behavior.slow_speed = SLOW                       # as the live stack runs (main sets 2.0 in truth/lidar mode)
    ego = (0.0, 0.0, 0.0)
    for old_reach in (True, False):
        if old_reach:
            monkeypatch.setattr(main, "distance_to_slow", lambda v, to: 0.0)
        else:
            monkeypatch.setattr(main, "distance_to_slow", distance_to_slow)
        van.planner._edge_hold = None
        for _ in range(2):
            place(van, x=0.0, y=0.0, yaw=0.0, speed=7.0)
            see(van, [car_beside(ego, 14.5, 2.3)])
            van.tick()
            van.clock.advance(0.1)
        c = chain(van); st = van._current_state["planner"]
        if old_reach:
            assert not st["edge_hold"] and st["edge_reach_m"] == 12.0 and c["brake"] == 0.0, (st, c)
        else:
            assert st["edge_hold"] and st["edge_reach_m"] >= 16.0, st
            assert c["planner_level"] == "slow" and c["safety_required"] and c["raw_mps"] <= 2.5, c
            assert c["brake"] > 0.3 and c["throttle"] == 0.0, c
