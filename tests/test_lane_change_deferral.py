"""Lane-change gap wait / rear-end safety (2026-09-18, WAV-0148).

Three runs on stack 48f3573 (scratch/runs/sweep_2, sweep_6, sweep_7): six rear-end hits, five of them with the van
at 0-1 m/s in "waiting for a gap to move over". Four of those five were waits for a lane change the route did not
contain: next_lane_change read the change off the route's offset beside the van's NOSE, so a 38 degree bend and a
5 degree heading error both looked like a lane change, and the van stopped in its live lane for the car in front
of it. The fifth was the real change the map draws in the first 19 m of the route: the target lane was busy, so
the van stood 8.6 s at its spawn point and was hit from behind at 10 m/s -- with lane -2 running on for 140 m to
the next junction.

Now: smooth_lane_changes RECORDS each real sideways step on the route (LaneChange), next_lane_change reads that
record (a change under way is not asked again), and a busy target lane is answered by DEFERRING the change --
the move redrawn LANE_CHANGE_LOOK_M further on, the van driving its own lane at its own speed -- wherever the switch
still completes before the route's next junction and the old lane is there. Where it cannot be deferred, the old
crawl-and-stop stays.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput
from warp_av.planning.planner import (LaneChange, Route, RoutePlanner, Waypoint, lane_change_blocker,
                                      _route_arcs, _route_offset)
from test_v2a_end_to_end import van, place, see, start_mission, chain  # noqa: F401
from warp_av.behavior import transitions as T

FIX = json.load(open(Path(__file__).parent / "fixtures" / "gapwait" / "wav0148_spawn.json", encoding="utf-8"))
LEFT, RIGHT = -1, +1


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def stepped_route(change_at=40.0, over=3.5, n=80, junction_from=None):
    """Straight along +x, stepping `over` metres to the right at `change_at` (a map-style step)."""
    wps = []
    for i in range(n):
        x = i * 2.0
        wps.append(Waypoint(x=x, y=(over if x >= change_at else 0.0), yaw=0.0,
                            road_id=1, lane_id=(-2 if x >= change_at else -1),
                            is_junction=(junction_from is not None and x >= junction_from)))
    return Route(waypoints=wps)


def smoothed(route):
    planner().smooth_lane_changes(route)
    return route


def old_lane_offset(route, x, y):
    """How far (x, y) sits to the right of the ORIGINAL lane line y = 0 of stepped_route."""
    return y


# ---- the record --------------------------------------------------------------------------------------------

def test_the_smoother_records_what_it_spread():
    r = smoothed(stepped_route(change_at=40.0, over=3.5))
    assert len(r.lane_changes) == 1
    lc = r.lane_changes[0]
    assert r.waypoints[lc.index].x == 40.0 and lc.lateral_m == pytest.approx(3.5) and 16.0 <= lc.over_m <= 18.0


def test_the_request_is_read_from_the_record_not_from_the_nose():
    r = smoothed(stepped_route(change_at=40.0))
    p = planner()
    start_m, side = p.next_lane_change(r, 0.0, 0.0, 0.0, within_m=40.0)
    assert side == RIGHT and 20.0 <= start_m <= 24.0                        # switch at 40 m, spread over ~18 m
    # the van's heading does not enter into it any more: 15 degrees off, same answer
    assert p.next_lane_change(r, 0.0, 0.0, math.radians(15.0), within_m=40.0) == (pytest.approx(start_m), side)


def test_a_bend_is_not_a_lane_change():
    """A 12 m radius left turn: the route is 7 m beside the nose line within 25 m, and used to be a lane change."""
    wps = [Waypoint(x=float(x), y=0.0, yaw=0.0, road_id=3, lane_id=-1) for x in range(-10, 21, 2)]
    for k in range(1, 20):
        a = math.radians(90.0 * k / 12.0)
        wps.append(Waypoint(x=20.0 + 12.0 * math.sin(a), y=12.0 - 12.0 * math.cos(a), yaw=a, road_id=3, lane_id=-1))
    r = Route(waypoints=wps)
    planner().smooth_lane_changes(r)
    assert r.lane_changes == []
    assert planner().next_lane_change(r, 14.0, 0.0, 0.0, within_m=25.0) is None


def test_a_heading_error_on_a_straight_is_not_a_lane_change():
    straight = Route(waypoints=[Waypoint(x=i * 2.0, y=0.0, yaw=0.0, road_id=1, lane_id=-1) for i in range(40)])
    for yaw_deg in (0.0, 5.0, 11.0, -13.0):
        assert planner().next_lane_change(straight, 4.0, 0.0, math.radians(yaw_deg), within_m=25.0) is None


def test_a_change_under_way_is_not_asked_again():
    r = smoothed(stepped_route(change_at=40.0, over=3.5))
    p = planner()
    assert p.next_lane_change(r, 30.0, 0.9, 0.0) is None          # 0.9 m over toward the new lane: committed
    assert p.next_lane_change(r, 30.0, 0.2, 0.0) is not None       # barely moved: still asked
    assert p.next_lane_change(r, 50.0, 3.5, 0.0) is None           # past the switch: done


# ---- deferral -----------------------------------------------------------------------------------------------

def test_case_b_a_deferred_change_keeps_the_van_in_its_lane_and_the_switch_moves_on():
    r = smoothed(stepped_route(change_at=40.0, over=3.5, n=120))
    p = planner()
    assert p.defer_lane_change(r, 20.0, 0.0, start_ahead_m=25.0) is True
    lc = r.lane_changes[0]
    wanted = 20.0 + 25.0 + lc.over_m                     # the ramp's end, as asked: 25 m on, then the ramp
    # the switch is the first point past that: never sooner, and within two waypoint spacings
    assert wanted <= r.waypoints[lc.index].x <= wanted + 4.0, r.waypoints[lc.index].x
    # the next 25 m of route lie on the OLD lane's line, and the ramp only after that
    for w in r.waypoints:
        if 20.0 <= w.x <= 44.0:
            assert abs(old_lane_offset(r, w.x, w.y)) < 0.15, (w.x, w.y)
            assert w.lane_id == -1
        if w.x >= r.waypoints[lc.index].x:
            assert old_lane_offset(r, w.x, w.y) == pytest.approx(3.5) and w.lane_id == -2
    # asked again from here: the move begins 25 m on, beyond the look -- no wait, no merge
    assert p.next_lane_change(r, 20.0, 0.0, 0.0, within_m=25.0) is None
    got = p.next_lane_change(r, 20.0, 0.0, 0.0, within_m=40.0)
    assert got is not None and 25.0 <= got[0] <= 29.0 and got[1] == RIGHT


def test_case_e_no_deferral_when_the_switch_would_pass_the_next_junction():
    """The turn may need the new lane: the old stop-and-wait stays."""
    r = smoothed(stepped_route(change_at=40.0, over=3.5, n=80, junction_from=60.0))
    assert planner().defer_lane_change(r, 20.0, 0.0, start_ahead_m=25.0) is False
    assert planner().next_lane_change(r, 20.0, 0.0, 0.0, within_m=25.0) is not None


def test_case_e_no_deferral_into_the_next_lane_changes_ramp_or_across_a_junction_before_the_switch():
    # right at 40 m, back left at 70 m: deferring the first by 25 m would run its ramp into the second's
    wps = [Waypoint(x=i * 2.0, y=(3.5 if 40.0 <= i * 2.0 < 70.0 else 0.0), yaw=0.0, road_id=1,
                    lane_id=(-2 if 40.0 <= i * 2.0 < 70.0 else -1)) for i in range(120)]
    r = Route(waypoints=wps)
    assert planner().smooth_lane_changes(r) == 2
    assert planner().defer_lane_change(r, 20.0, 0.0, start_ahead_m=25.0) is False
    # a junction between the van and the switch (30-36 m): the redrawn ramp would cross it
    r2 = smoothed(stepped_route(change_at=40.0, over=3.5, n=120))
    for i, w in enumerate(r2.waypoints):
        if 30.0 <= w.x <= 36.0:
            r2.waypoints[i] = Waypoint(x=w.x, y=w.y, yaw=w.yaw, road_id=w.road_id, lane_id=w.lane_id, is_junction=True)
    assert planner().defer_lane_change(r2, 20.0, 0.0, start_ahead_m=25.0) is False


def test_case_e_no_deferral_once_the_move_has_begun_or_past_the_route_end():
    r = smoothed(stepped_route(change_at=40.0, over=3.5, n=80))
    assert planner().defer_lane_change(r, 30.0, 0.9, start_ahead_m=25.0) is False       # under way
    short = smoothed(stepped_route(change_at=40.0, over=3.5, n=30))                      # route ends at 58 m
    assert planner().defer_lane_change(short, 20.0, 0.0, start_ahead_m=25.0) is False


class _Lane:
    """A stand-in CARLA map: driving lanes at y = 0 and y = 3.5 only, up to x_end."""
    def __init__(self, x_end):
        self.x_end = x_end

    def get_waypoint(self, loc, project_to_road=True, lane_type=None):
        import types
        y = 0.0 if abs(loc.y) < abs(loc.y - 3.5) else 3.5
        x = min(loc.x, self.x_end)
        return types.SimpleNamespace(transform=types.SimpleNamespace(location=types.SimpleNamespace(x=x, y=y)))


def test_case_e_no_deferral_where_the_old_lane_ends_on_the_map(monkeypatch):
    import warp_av.planning.planner as PL
    import types
    monkeypatch.setattr(PL, "carla", types.SimpleNamespace(Location=lambda x, y, z: types.SimpleNamespace(x=x, y=y, z=z),
                                                          LaneType=types.SimpleNamespace(Driving="driving")), raising=False)
    r = smoothed(stepped_route(change_at=40.0, over=3.5, n=120))
    p = planner(); p.carla_map = _Lane(x_end=200.0)
    assert p.defer_lane_change(r, 20.0, 0.0, start_ahead_m=25.0) is True
    r2 = smoothed(stepped_route(change_at=40.0, over=3.5, n=120))
    p2 = planner(); p2.carla_map = _Lane(x_end=50.0)              # the old lane stops at x = 50
    assert p2.defer_lane_change(r2, 20.0, 0.0, start_ahead_m=25.0) is False


def test_a_deferral_holds_tick_after_tick_with_a_map_that_only_knows_lane_centres(monkeypatch):
    """First live run (2026-09-18): the deferral held once at the spawn and was refused on every later tick.
    The map check sampled the points the redraw moved -- and a redraw that slides the ramp on by a metre
    moves only ramp points, which sit between the lanes by design. With a map that projects everything to
    a lane centre, the redraw must keep holding as the van drives on."""
    import warp_av.planning.planner as PL
    import types
    monkeypatch.setattr(PL, "carla", types.SimpleNamespace(Location=lambda x, y, z: types.SimpleNamespace(x=x, y=y, z=z),
                                                          LaneType=types.SimpleNamespace(Driving="driving")), raising=False)
    r = smoothed(stepped_route(change_at=40.0, over=3.5, n=120))
    p = planner(); p.carla_map = _Lane(x_end=200.0)
    assert p.defer_lane_change(r, 20.0, 0.0, start_ahead_m=25.0) is True
    first_switch = r.lane_changes[0].index
    for x in (21.0, 22.0, 24.0, 27.0, 31.0):                      # the van drives on, the lane still busy
        assert p.defer_lane_change(r, x, 0.0, start_ahead_m=25.0) is True, x
        assert p.next_lane_change(r, x, 0.0, 0.0, within_m=25.0) is None, x
    assert r.lane_changes[0].index > first_switch                 # and the switch kept moving on
    for w in r.waypoints:                                         # every point still on one of the two lanes, or the ramp between
        assert -0.2 <= w.y <= 3.7


def test_a_deferral_on_a_bend_keeps_the_lanes_parallel():
    """A route that turns: the redrawn old-lane stretch must sit one lane-width off the new-lane arc all along."""
    R = 30.0
    wps = []
    for k in range(0, 61):
        a = math.radians(k * 1.5)                                     # 1.5 deg per point, ~0.8 m spacing... use arc 2 m
        a = 2.0 * k / R
        x, y = R * math.sin(a), R - R * math.cos(a)
        wps.append(Waypoint(x=x, y=y, yaw=a, road_id=5, lane_id=-1))
    # a right step of 3.5 m at k = 25 (lane -2 is the outer lane, radius R + 3.5)
    for k in range(25, 61):
        a = 2.0 * k / R
        wps[k] = Waypoint(x=(R + 3.5) * math.sin(a), y=R - (R + 3.5) * math.cos(a), yaw=a, road_id=5, lane_id=-2)
    r = Route(waypoints=wps)
    planner().smooth_lane_changes(r)
    assert len(r.lane_changes) == 1 and r.lane_changes[0].lateral_m == pytest.approx(-3.5, abs=0.2)   # the outer lane is to the left
    p = planner()
    # from point 4 (8 m along) the ramp begins 20 m on; asked to begin 40 m on, it moves by 20 m
    assert p.defer_lane_change(r, 2.0 * 4, r.waypoints[4].y, start_ahead_m=40.0) is True
    lc = r.lane_changes[0]
    assert lc.index >= 33, lc.index                                   # 8 + 40 + 20 m of ramp = 68 m along, 2 m apart
    for k in range(6, lc.index - 10):                                # the deferred stretch, before the new ramp
        w = r.waypoints[k]
        radius = math.hypot(w.x, w.y - R)
        assert radius == pytest.approx(R, abs=0.15), (k, radius)  # back on the inner lane's arc


# ---- the recorded spawn tick (WAV-0148 run 3, t+0.8-9.4) through the real stack ------------------------------

def recorded_route():
    wps = [Waypoint(x=w["x"], y=w["y"], yaw=float(w.get("yaw") or 0.0), is_junction=bool(w["is_junction"]),
                    road_id=w.get("road_id"), lane_id=w.get("lane_id")) for w in FIX["route"]]
    r = Route(waypoints=wps)
    r.lane_changes = [LaneChange(**lc) for lc in FIX["lane_changes"]]
    return r


def to_ego(o, pose):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = o["x"] - px, o["y"] - py
    return DetectedObject(object_type=ObjectType(o["type"]), x=c * dx + s * dy, y=-s * dx + c * dy, distance=o["distance"],
                          speed=o.get("speed", 0.0), confidence=o.get("confidence", 1.0), id=o["id"],
                          length_m=o.get("length_m", 0.0), width_m=o.get("width_m", 0.0), height_m=o.get("height_m", 0.0),
                          yaw_deg=o.get("yaw_deg", 0.0), stationary=o.get("stationary", True), size_uncertain=o.get("size_uncertain", False),
                          clearance_radius_m=o.get("clearance_radius_m", 0.4), motion_class=o.get("motion_class", "dynamic"),
                          static_rule=o.get("static_rule", ""), motion_why=o.get("motion_why", ""),
                          vx_world=o.get("vx_world", 0.0), vy_world=o.get("vy_world", 0.0),
                          box_dx=o.get("box_dx", 0.0), box_dy=o.get("box_dy", 0.0), box_length_m=o.get("box_length_m", 0.0),
                          box_width_m=o.get("box_width_m", 0.0), box_yaw_deg=o.get("box_yaw_deg", 0.0))


def test_case_a_the_recorded_spawn_tick_is_a_real_change_with_a_busy_lane_that_used_to_stop_the_van():
    tk = FIX["tick"]
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    assert tk["live"]["brake"] >= 0.9 and tk["pose"]["speed"] < 0.1 and "waiting for a gap" in tk["live"]["reason"]
    r = recorded_route(); p = planner()
    nxt = p.next_lane_change(r, *pose, within_m=25.0)
    assert nxt is not None and nxt[0] < 3.0 and nxt[1] == LEFT            # the map's change begins at the spawn
    objs = [to_ego(o, pose) for o in tk["objects"]]
    assert lane_change_blocker(objs, nxt[1], pose[2]) is not None          # and the target lane really is busy
    assert lane_change_blocker(objs, nxt[1], pose[2]) == tk["live"]["blocker"] or "in that lane" in lane_change_blocker(objs, nxt[1], pose[2])


def test_case_a_and_d_through_the_stack_the_van_now_drives_on_in_its_lane(van):
    """The same inputs through WarpAV: no stop, throttle on, the move deferred; the car closing behind in our lane
    (tm_13, hit the standing van at t+6.1 in the record) is no longer given a stationary target."""
    tk = FIX["tick"]
    r = recorded_route()
    van.planner.carla_map = None                                             # the harness map has no lanes to ask
    start_mission(van, route=r, dest=(FIX["route"][-1]["x"], FIX["route"][-1]["y"]))
    van.mission_manager.set_executing()
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    objs = [to_ego(o, pose) for o in tk["objects"]]
    reasons = []
    for _ in range(3):
        place(van, x=pose[0], y=pose[1], yaw=pose[2], speed=0.0)
        see(van, objs)
        van.tick()
        van.clock.advance(0.2)
        reasons.append(van._current_state["behavior_reason"])
    c = chain(van)
    # tick 1 defers the move; from then on it lies beyond the look and is not asked about again
    assert "lane change deferred" in reasons[0], reasons
    assert not any("waiting for a gap" in why for why in reasons), reasons
    assert c["throttle"] > 0.0 and c["brake"] == 0.0, c
    assert r.lane_changes[0].index > FIX["lane_changes"][0]["index"]        # the route's switch moved on
    assert any(c.kind == T.MOVE and c.why == T.LANE_CHANGE_DEFERRED for c in van.behavior.transitions.changes), \
        "said so on the record"


def test_case_c_a_clear_target_lane_changes_lane_as_before(van):
    tk = FIX["tick"]
    r = recorded_route()
    van.planner.carla_map = None
    start_mission(van, route=r, dest=(FIX["route"][-1]["x"], FIX["route"][-1]["y"]))
    van.mission_manager.set_executing()
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    for _ in range(3):
        place(van, x=pose[0], y=pose[1], yaw=pose[2], speed=0.0)
        see(van, [])                                                         # nobody in the lane
        van.tick()
        van.clock.advance(0.2)
    reason = van._current_state["behavior_reason"]
    assert "deferred" not in reason and "waiting for a gap" not in reason
    assert r.lane_changes[0].index == FIX["lane_changes"][0]["index"]        # untouched: the move proceeds


def test_case_e_the_stop_stays_where_the_change_cannot_be_deferred(van):
    """The same busy lane, but the route's next junction 30 m on: the old crawl-and-stop."""
    tk = FIX["tick"]
    r = recorded_route()
    arcs = _route_arcs(r.waypoints)
    ego_arc, _ = _route_offset(r.waypoints, tk["pose"]["x"], tk["pose"]["y"])
    for i, w in enumerate(r.waypoints):
        if arcs[i] > ego_arc + 30.0:
            r.waypoints[i] = Waypoint(x=w.x, y=w.y, yaw=w.yaw, is_junction=True, road_id=w.road_id, lane_id=w.lane_id)
    van.planner.carla_map = None
    start_mission(van, route=r, dest=(FIX["route"][-1]["x"], FIX["route"][-1]["y"]))
    van.mission_manager.set_executing()
    pose = (tk["pose"]["x"], tk["pose"]["y"], math.radians(tk["pose"]["yaw_deg"]))
    objs = [to_ego(o, pose) for o in tk["objects"]]
    for _ in range(3):
        place(van, x=pose[0], y=pose[1], yaw=pose[2], speed=0.0)
        see(van, objs)
        van.tick()
        van.clock.advance(0.2)
    c = chain(van)
    reason = van._current_state["behavior_reason"]
    assert "waiting for a gap" in reason and "deferred" not in reason, reason
    assert c["throttle"] == 0.0 and c["brake"] >= 0.9, c


def test_case_g_the_wait_is_never_asked_during_a_pass():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text(encoding="utf-8")
    i = src.index("def _wait_for_a_gap")
    body = src[i:i + 900]
    assert "self._overtake_point is not None" in body and "return" in body


def test_the_deferral_is_on_the_record():
    assert T.LANE_CHANGE_DEFERRED in T.ALL_MOVES
