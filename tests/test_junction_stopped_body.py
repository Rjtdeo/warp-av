"""Junction stopped-body (2026-09-18): a stopped vehicle lying across the van's straight path through a
junction, judged by its CENTRE and never by its body.

WAV-0386 three times on stack 8b4fa9d (scratch/runs/sweep_4, sweep_6, sweep_7): a truck 5.2 x 2.6 m standing
across Town10 junction 675 with its centre 2.5-3.3 m off the route line and its end 0.1 m from it. The route
through the junction is straight. Perception had it right (fitted 5.0 x 2.1-2.5 m, heading within 3 deg,
centre within 0.5 m at the last 4 m; fit_is_believable). The corridor filter never examined it: the body
sweep -- the one rule that reads a shape -- is switched off for anything standing in a junction or past the
"plain road" (proxies for "the route polyline sweep is not trusted where the route turns", 604837b) and for
any centre over 2.6 m from the line; the centre bands then skipped it. Only the laser's ground map stopped
the van, at 5.9-6.7 m, and its dropouts (a 0.1 m intrusion is 0-1 cells) gave the throttle back at 1.5-2.4 m:
0.32 m, 1.47 m, then a collision at 3.2 m/s.

The fix: where the route sweep is not trusted, a stationary body is swept along the path the van is actually
steering (the intended path, a real arc) when there is one, and a body whose own reach brings it within the
sweep bound is a candidate; on a plain road nothing changes. And the ground's "seen free" release may not lift
a block the van's BARE body would run into (decision.blocker_touches_body).
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from warp_av.perception.occupancy import OccupancyGrid
from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import (PATH_BLOCKED, PATH_CLEAR, PATH_SLOW, BLOCKED_SWEPT_PATH,
                                              GROUND_RELEASED, PlannerDecision)
from warp_av.planning.planner import Route, RoutePlanner, Waypoint, SWEEP_MAX_LATERAL_M
from test_v2a_end_to_end import van  # noqa: F401  (the end-to-end harness fixture, for the ground release)

FIX = json.load(open(Path(__file__).parent / "fixtures" / "junction" / "wav0386_ticks.json", encoding="utf-8"))
FOOT = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)


def planner():
    return RoutePlanner.__new__(RoutePlanner)       # no CARLA map: the corridor filter needs none


def recorded_route():
    return Route(waypoints=[Waypoint(x=w["x"], y=w["y"], yaw=float(w.get("yaw") or 0.0), is_junction=bool(w["is_junction"]),
                                     road_id=w.get("road_id"), lane_id=w.get("lane_id")) for w in FIX["route"]])


def to_ego(o, pose):
    """A published object (world x, y; the box in the van's frame of that tick) as perception handed it over."""
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = o["x"] - px, o["y"] - py
    return DetectedObject(object_type=ObjectType(o["type"]), x=c * dx + s * dy, y=-s * dx + c * dy, distance=o["distance"],
                          speed=o.get("speed", 0.0), confidence=o.get("confidence", 1.0), id=o["id"],
                          length_m=o.get("length_m", 0.0), width_m=o.get("width_m", 0.0), height_m=o.get("height_m", 0.0),
                          yaw_deg=o.get("yaw_deg", 0.0), stationary=o.get("stationary", True), size_uncertain=o.get("size_uncertain", False),
                          clearance_radius_m=o.get("clearance_radius_m", 0.4), motion_class=o.get("motion_class", "dynamic"),
                          static_rule=o.get("static_rule", ""), motion_why=o.get("motion_why", ""),
                          box_dx=o.get("box_dx", 0.0), box_dy=o.get("box_dy", 0.0), box_length_m=o.get("box_length_m", 0.0),
                          box_width_m=o.get("box_width_m", 0.0), box_yaw_deg=o.get("box_yaw_deg", 0.0))


def tick(i):
    return FIX["ticks"][i]


def replay(tk, objects=None, intended=True):
    """The corridor decision for a recorded tick, exactly as main.py asks it (the previous tick's trajectory
    as the intended path)."""
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    objs = objects if objects is not None else [to_ego(o, pose) for o in tk["objects"]]
    closest = min(objs, key=lambda o: o.distance) if objs else None
    p = PerceptionOutput(objects=objs, closest_obstacle_distance=closest.distance if closest else 999.0,
                         closest_obstacle_type=closest.object_type if closest else ObjectType.UNKNOWN, timestamp=tk["timestamp"])
    fp = VehicleFootprint(**tk["footprint"])
    path = [tuple(p_) for p_ in tk["intended_path"]] if (intended and tk["intended_path"]) else None
    return planner().filter_to_route_corridor(p, recorded_route(), *pose, danger_m=8.0, footprint=fp, intended_path=path)


def truck_id(tk):
    """The recorded truck: the tracked vehicle nearest the truth box (the truth is for scoring only)."""
    tb = tk["truth_truck_box"]
    return min((o for o in tk["objects"] if o["type"] == "vehicle"), key=lambda o: math.hypot(o["x"] - tb[0], o["y"] - tb[1]))["id"]


# ---- CASE A: the recorded ticks -------------------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(FIX["ticks"])))
def test_case_a_every_recorded_tick_now_blocks_on_the_truck_itself(i):
    tk = tick(i)
    assert tk["live"]["level"] in ("clear", "slow"), "the record: the corridor never blocked on the truck"   # what changes
    dec = replay(tk)
    assert dec.level == PATH_BLOCKED and dec.reason == BLOCKED_SWEPT_PATH, (tk["note"], dec.level, dec.reason)
    assert dec.blocker_id == truck_id(tk), (dec.blocker_id, truck_id(tk))
    assert dec.blocker_touches_body is True, "the bare body runs into the measured rectangle: not a margin brushed"
    assert dec.used_footprint


def test_case_a_the_first_block_comes_at_the_sweep_reach_with_the_van_at_speed():
    """Run 3, t+44.2: 12 m centre distance, 4.9 m/s, truth gap 7.4 m — the record says clear and throttle."""
    tk = tick(0)
    assert tk["live"]["throttle"] > 0.0 and tk["pose"]["speed"] > 4.0
    dec = replay(tk)
    assert dec.level == PATH_BLOCKED and 11.0 <= dec.blocker_distance_m <= 12.0, dec.blocker_distance_m


def test_case_a_the_contact_tick_is_a_block_not_a_throttle():
    """Run 3, t+47.2: the record is clear/clear at 3.2 m/s with the truth gap at 0.00 m."""
    tk = tick(2)
    assert tk["live"]["level"] == "clear" and tk["pose"]["speed"] > 2.0
    dec = replay(tk)
    assert dec.level == PATH_BLOCKED and dec.blocker_id == truck_id(tk)


def test_case_a_with_no_intended_path_the_old_verdict_stands():
    """Through a junction the sweep needs the path the van is steering; a bare heading line is the tilted-van
    false stop of 2026-08 and is not used. Live the trajectory exists from the second tick on."""
    dec = replay(tick(0), intended=False)
    assert dec.level != PATH_BLOCKED and dec.blocker_id is None


# ---- CASE B: the ground's dropout and its release cannot expose the truck ------------------------------

def test_case_b_the_block_stands_on_the_tracked_body_alone():
    """The corridor filter has no ground map input: its verdict cannot drop out with the grid's."""
    for i in range(len(FIX["ticks"])):
        dec = replay(tick(i))
        assert dec.level == PATH_BLOCKED and dec.second_opinion is None


def swept_grid(reach=25.0):
    """A laser map around a van at the origin facing +x: every bearing seen empty to `reach` metres."""
    ang = np.radians(np.arange(0.0, 360.0, 0.25))
    ground = np.stack([reach * np.cos(ang), reach * np.sin(ang)], axis=1)
    return OccupancyGrid().update(ground, np.zeros(len(ground), dtype=bool))


def _release(stack, touches):
    """main._drive_on_if_the_ground_is_seen_free on a corridor BLOCK 6 m ahead over ground seen empty."""
    from warp_av.localization.localization import Pose
    path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, level=PATH_BLOCKED, closest_distance_m=6.0,
                           closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0, blocker_id=7, blocker_kind="vehicle",
                           blocker_distance_m=6.0, blocker_touches_body=touches)
    stack._drive_on_if_the_ground_is_seen_free(path, Pose(healthy=True), swept_grid(), 2.95)
    return path


def test_case_b_the_ground_may_not_release_a_block_the_bare_body_would_run_into(van):
    path = _release(van, touches=True)
    assert path.level == PATH_BLOCKED and path.second_opinion is None
    assert van._ground_seen_free is None


def test_case_b_but_a_margin_only_touch_is_still_released_as_before(van):
    path = _release(van, touches=False)
    assert path.level == PATH_SLOW and path.second_opinion == GROUND_RELEASED


# ---- CASE C / D / I: a junction body beside the path, cross traffic, a real full blocker ---------------

def beside_path(tk, along_m, right_m):
    """(world x, y, path heading deg) `along_m` down the recorded intended path and `right_m` to its right."""
    pts = [tuple(p) for p in tk["intended_path"]]
    s = 0.0
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if s + seg >= along_m:
            t = (along_m - s) / seg
            hx, hy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg          # along; right = (hy, -hx) in this map frame
            px, py = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
            return px - hy * right_m * -1.0, py + hx * right_m * -1.0, math.degrees(math.atan2(hy, hx))
        s += seg
    raise ValueError("beyond the path")


def car_at_world(tk, wx, wy, world_yaw_deg, length=4.5, width=1.9, speed=0.0, oid=901, kind="vehicle"):
    """A stationary vehicle placed in the world, published as perception would (box in the van's frame)."""
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    o = dict(type=kind, x=wx, y=wy, distance=math.hypot(wx - pose[0], wy - pose[1]), speed=speed, confidence=0.9, id=oid,
             length_m=length, width_m=width, height_m=1.6, yaw_deg=world_yaw_deg - tk["pose"]["yaw_deg"], stationary=speed < 0.5,
             size_uncertain=False, clearance_radius_m=1.2, motion_class="static" if speed < 0.5 else "dynamic", static_rule="", motion_why="",
             box_dx=0.0, box_dy=0.0, box_length_m=length, box_width_m=width, box_yaw_deg=world_yaw_deg - tk["pose"]["yaw_deg"])
    return to_ego(o, pose)


def test_case_c_a_stopped_car_beside_the_straight_path_in_the_junction_stays_clear():
    """Its centre 3.3 m off the path, parked along the road: the body is 0.7 m outside the swept band."""
    tk = tick(0)
    x, y, h = beside_path(tk, 10.0, 3.3)
    car = car_at_world(tk, x, y, h)
    assert abs(car.y) > SWEEP_MAX_LATERAL_M                    # the centre bound alone would never look at it
    dec = replay(tk, objects=[car])
    assert dec.level == PATH_CLEAR and dec.blocker_id is None, (dec.level, dec.reason)


def test_case_c_a_car_waiting_at_the_cross_street_nose_short_of_the_line_stays_clear():
    """Across the road like the truck, but its nose 1.7 m from the path (a hand outside the body's 1.29 m
    swept half-width, pad included): the body stops short of ours."""
    tk = tick(0)
    x, y, h = beside_path(tk, 10.0, 1.7 + 2.25 + 0.1)
    car = car_at_world(tk, x, y, h + 90.0)
    dec = replay(tk, objects=[car])
    assert dec.level == PATH_CLEAR, (dec.level, dec.reason)
    # ...and the same car a metre nearer, its nose 0.7 m from the path, is a block: the van would hit it
    x, y, h = beside_path(tk, 10.0, 0.7 + 2.25 + 0.1)
    assert replay(tk, objects=[car_at_world(tk, x, y, h + 90.0)]).level == PATH_BLOCKED


def test_case_d_a_car_crossing_on_the_other_branch_is_left_to_the_moving_rules():
    tk = tick(0)
    x, y, h = beside_path(tk, 10.0, 2.8)
    mover = car_at_world(tk, x, y, h + 90.0, speed=6.0)
    dec = replay(tk, objects=[mover])
    assert dec.level != PATH_BLOCKED, (dec.level, dec.reason)


def test_case_d_a_tall_roadside_thing_that_is_not_a_vehicle_keeps_the_old_rules():
    """The steered-path sweep reads vehicles: a 3.6 m tall canopy box a corner drives under stays with the
    bands (two single-tick blocks in the 61-record replay when every class was admitted)."""
    tk = tick(0)
    x, y, h = beside_path(tk, 10.0, 2.8)
    canopy = car_at_world(tk, x, y, h + 90.0, length=3.6, width=2.8, kind="obstacle")
    assert replay(tk, objects=[canopy]).level != PATH_BLOCKED


def test_case_i_a_car_stopped_square_across_the_path_in_the_junction_blocks():
    tk = tick(0)
    x, y, h = beside_path(tk, 10.0, 0.0)
    car = car_at_world(tk, x, y, h + 90.0)
    dec = replay(tk, objects=[car])
    assert dec.level == PATH_BLOCKED and dec.reason == BLOCKED_SWEPT_PATH and dec.blocker_touches_body


# ---- CASE E: a curved turn — the body is swept along the arc the van is steering -------------------------

def turning_route():
    """A road running +x that turns left (to +y) through a junction at x = 20, radius 12 m."""
    wps = [Waypoint(x=float(x), y=0.0) for x in range(-10, 21, 2)]
    for k in range(1, 13):
        a = math.radians(90.0 * k / 12.0)
        wps.append(Waypoint(x=20.0 + 12.0 * math.sin(a), y=12.0 - 12.0 * math.cos(a), is_junction=True))
    wps += [Waypoint(x=32.0, y=12.0 + 2.0 * k, is_junction=(k < 2)) for k in range(1, 12)]
    return Route(waypoints=wps)


def arc_path(ego_x):
    """The intended path from ego_x on the straight, round the same arc, 25 m long."""
    pts = [(x, 0.0) for x in np.arange(ego_x, 20.0, 0.5)]
    for a in np.arange(0.0, math.pi / 2 + 1e-9, 0.5 / 12.0):
        pts.append((20.0 + 12.0 * math.sin(a), 12.0 - 12.0 * math.cos(a)))
    pts += [(32.0, 12.0 + 0.5 * k) for k in range(1, 20)]
    out, s = [pts[0]], 0.0
    for p in pts[1:]:
        s += math.hypot(p[0] - out[-1][0], p[1] - out[-1][1])
        out.append(p)
        if s > 25.0:
            break
    return out


def obj_on_turn(wx, wy, world_yaw_deg, ego, length=4.5, width=1.9, speed=0.0):
    ex, ey, eyaw = ego
    c, s = math.cos(eyaw), math.sin(eyaw)
    dx, dy = wx - ex, wy - ey
    yaw_rel = world_yaw_deg - math.degrees(eyaw)
    return DetectedObject(object_type=ObjectType.VEHICLE, x=c * dx + s * dy, y=-s * dx + c * dy, distance=math.hypot(dx, dy),
                          speed=speed, id=77, stationary=speed < 0.5, length_m=length, width_m=width, height_m=1.6,
                          yaw_deg=yaw_rel, box_length_m=length, box_width_m=width, box_yaw_deg=yaw_rel, clearance_radius_m=1.2)


def run_turn(objects, ego=(12.0, 0.0, 0.0), path=True):
    p = PerceptionOutput(objects=objects, closest_obstacle_distance=min(o.distance for o in objects), closest_obstacle_type=ObjectType.VEHICLE)
    return planner().filter_to_route_corridor(p, turning_route(), *ego, danger_m=8.0, footprint=FOOT,
                                              intended_path=(arc_path(ego[0]) if path else None))


def test_case_e_a_stopped_body_lying_across_the_arc_blocks_along_the_steered_path():
    # a car standing across the arc 20 degrees round it (heading tangent + 90), its centre 1.5 m outside the
    # arc line so that half its body lies over the path -- 10 m along the route from the van at x = 14
    a = math.radians(20.0)
    px, py = 20.0 + 12.0 * math.sin(a), 12.0 - 12.0 * math.cos(a)
    nx, ny = math.sin(a), -math.cos(a)                                          # outward normal of the arc
    car = obj_on_turn(px + 1.5 * nx, py + 1.5 * ny, 20.0 + 90.0, (14.0, 0.0, 0.0))
    dec = run_turn([car], ego=(14.0, 0.0, 0.0))
    assert dec.level == PATH_BLOCKED and dec.reason == BLOCKED_SWEPT_PATH, (dec.level, dec.reason)
    # the same car with its centre 4.0 m outside the arc: its near end 1.75 m off the line, clear
    far = obj_on_turn(px + 4.0 * nx, py + 4.0 * ny, 20.0 + 90.0, (14.0, 0.0, 0.0))
    assert run_turn([far], ego=(14.0, 0.0, 0.0)).level != PATH_BLOCKED


def test_case_e_the_same_body_straight_ahead_beyond_the_corner_is_not_in_the_arcs_way():
    # dead ahead of the nose, 20 m on, where the road no longer goes: a straight sweep would hit it, the arc does not
    car = obj_on_turn(33.0, 0.0, 90.0, (12.0, 0.0, 0.0))
    dec = run_turn([car])
    assert dec.level != PATH_BLOCKED, (dec.level, dec.reason)


def test_case_e_with_no_steered_path_the_turn_keeps_the_old_rules():
    a = math.radians(20.0)
    px, py = 20.0 + 12.0 * math.sin(a), 12.0 - 12.0 * math.cos(a)
    car = obj_on_turn(px + 1.5 * math.sin(a), py - 1.5 * math.cos(a), 110.0, (14.0, 0.0, 0.0))
    assert run_turn([car], ego=(14.0, 0.0, 0.0), path=False).level != PATH_BLOCKED


# ---- CASE F: the plain road is untouched ----------------------------------------------------------------

def straight_route():
    return Route(waypoints=[Waypoint(x=float(x), y=0.0) for x in range(-10, 60, 2)])


def run_straight(objects, ego=(0.0, 0.0, 0.0)):
    p = PerceptionOutput(objects=objects, closest_obstacle_distance=min(o.distance for o in objects), closest_obstacle_type=ObjectType.VEHICLE)
    return planner().filter_to_route_corridor(p, straight_route(), *ego, danger_m=8.0, footprint=FOOT,
                                              intended_path=[(x, 0.0) for x in np.arange(0.0, 25.5, 0.5)])


def test_case_f_open_road_verdicts_are_unchanged():
    ego = (0.0, 0.0, 0.0)
    assert run_straight([obj_on_turn(10.0, 1.6, 0.0, ego)]).level == PATH_BLOCKED       # a parked car reaching into the margin: blocks (run 62)
    assert run_straight([obj_on_turn(10.0, 3.5, 0.0, ego)]).level == PATH_CLEAR         # the next lane over: clear
    across = obj_on_turn(10.0, 3.3, 90.0, ego)                                          # lying across, centre 3.3 m off: still the centre bound on a plain road
    dec = run_straight([across])
    assert dec.level != PATH_BLOCKED, "on a plain road the sweep's candidates are unchanged (see the decision record: not this phase)"


# ---- Step 14: the control chain, planner BLOCK -> behaviour -> controller ---------------------------------

from test_v2a_end_to_end import place, see, start_mission, chain  # noqa: E402, F401


def test_the_recorded_collision_tick_now_ends_in_a_stop_command(van):
    """WAV-0386 run 3, the first tick inside the sweep reach (4.9 m/s, truth gap 7.4 m): the record is
    clear/clear and throttle 0.24. Through the real stack the same inputs are BLOCKED -> STOPPED_VEHICLE ->
    throttle 0, brake 1.0 -- on the tick the path is first steered (the first tick has no intended path)."""
    tk = tick(0)
    start_mission(van, route=recorded_route(), dest=(FIX["route"][-1]["x"], FIX["route"][-1]["y"]))
    van.mission_manager.set_executing()
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    objs = [to_ego(o, pose) for o in tk["objects"]]
    for _ in range(3):
        place(van, x=pose[0], y=pose[1], yaw=pose[2], speed=tk["pose"]["speed"])
        see(van, objs)
        van.tick()
        van.clock.advance(0.2)
    c = chain(van)
    st = van._current_state["planner"]
    assert c["planner_level"] == PATH_BLOCKED and c["planner_reason"] == BLOCKED_SWEPT_PATH, c
    assert st["blocker_id"] == truck_id(tk) and st["blocker_touches_body"] is True, st
    assert van.behavior.current_behavior.value == "stopped_vehicle", van.behavior.current_behavior
    assert c["throttle"] == 0.0 and c["brake"] >= 0.9, c
