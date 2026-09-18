"""Go-around rejoin (2026-09-18): a pass is extended past whatever stands where its ramp back was going to land.

The stopped-car scenario, three batches the same: blocked 10 s by a dead car 11 m ahead, LANE_LEFT accepted, the
way round drawn on the route with its ramp back at 8-16 m past that car, checked once against the parked row 20-27 m
ahead that the tracker still reported as far partial boxes. The row (N-Mustang at x 10.3, Nissan Patrol at x 17.7,
0.84 m from the lane's body line) stands exactly on the ramp: the van followed it to +28.7 deg, stopped with its nose
0.72 m from the Patrol, and stayed there with the pass active until the run timed out. Now, while a pass is active,
the rest of the way round is asked the same question the acceptance asked (planner.pull_in_blocker) and the pass is
extended past what it would touch: planned again from the current pose on the road as it was before the pass, already
at the full shift, the blocker distance set to the far end of the thing in the way, swapped in only if that same
check finds the new tail clear.

Frames: the start road as the runner saved it (test_patrol_tight_pass.ROUTE_SEG, x along the road, y 140.6-141.1 the
lane's line; the parked row and the lane to the LEFT at y 137.45 run the same way). The real WarpAV tick drives the
pass; the map is absent in the harness, so the lane tests are stubbed to "yes" and the van is moved along its own
rewritten route by hand at 2 m/s (the controller's steering is recorded, not applied).
"""
import math
import pytest

from test_v2a_end_to_end import van, start_mission, place, see, chain  # noqa: F401
from test_patrol_tight_pass import ROUTE_SEG, route as base_route
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.footprint import VehicleFootprint, ObstacleBox, sweep_conflict, _project, _point_at_arc
from warp_av.planning.planner import RoutePlanner, Route, Waypoint

FOOT = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)
TRUTH = {"Patrol": (17.73, 144.0, 5.57, 2.44), "Mustang": (10.33, 144.0, 4.72, 1.89)}      # map, scoring only
ACC_POSE = (-10.98, 140.64, math.radians(0.4))                                              # recorded acceptance pose
STOPPED = (0.52, 140.64)                                                                    # the dead car's centre, 11.5 m ahead


def van_frame(pose, wx, wy):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = wx - pose[0], wy - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def seen(pose, centroid, box_centre, length, width, spread, oid, yaw_world=0.0, kind=ObjectType.VEHICLE):
    """A stationary thing as the tracker publishes it: the points' centroid, the rectangle's centre offset from it."""
    x, y = van_frame(pose, *centroid)
    ox, oy = van_frame(pose, *box_centre)
    yaw = math.degrees(yaw_world - pose[2])
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=0.0, id=oid, stationary=True,
                          length_m=spread[0], width_m=spread[1], height_m=1.6, yaw_deg=yaw, box_dx=ox - x, box_dy=oy - y,
                          box_length_m=length, box_width_m=width, box_yaw_deg=yaw,
                          clearance_radius_m=0.5 * math.hypot(length, width), motion_class="static")


def stopped_car(pose):
    return seen(pose, STOPPED, STOPPED, 4.5, 1.9, (4.4, 1.8), oid=8571)


ROW = {   # centroid when far (a partial, thin look) and when near (the honest body), as recorded
    "Mustang": dict(far=((8.0, 143.9), (8.0, 143.9), 1.75, 1.13, (1.31, 0.6)), near=((9.6, 143.3), (10.4, 143.9), 4.52, 1.77, (4.55, 1.75)), oid=8494),
    "Patrol": dict(far=((15.4, 143.6), (15.4, 143.6), 1.80, 0.46, (1.62, 0.42)), near=((16.4, 143.25), (17.5, 143.8), 5.14, 1.74, (5.2, 1.69)), oid=8503),
}


def row_objects(pose, extra_row=()):
    out = []
    for name, spec in ROW.items():
        cx, cy = TRUTH[name][:2]
        look = spec["near"] if math.hypot(cx - pose[0], cy - pose[1]) < 15.0 else spec["far"]
        out.append(seen(pose, look[0], look[1], look[2], look[3], look[4], oid=spec["oid"]))
    for i, x in enumerate(extra_row):
        near = math.hypot(x - pose[0], 144.0 - pose[1]) < 15.0
        out.append(seen(pose, (x - 0.9, 143.3) if near else (x - 2.0, 143.8), (x, 143.9) if near else (x - 2.0, 143.8),
                        4.6 if near else 1.8, 1.8 if near else 0.5, (4.6, 1.75) if near else (1.6, 0.4), oid=9000 + i))
    return out


def advance(wps, pose, step):
    pts = [(w.x, w.y) for w in wps]
    arc = _project(pose[0], pose[1], pts)[0] + step
    x, y, heading = _point_at_arc(arc, pts)
    return (x, y, heading)


def lane_stubs(monkeypatch, v):
    monkeypatch.setattr(v, "_same_way_lane_ok", lambda x, y, yaw: True)
    monkeypatch.setattr(v, "_lane_ok", lambda x, y: True)
    monkeypatch.setattr(v, "_shoulder_ok", lambda x, y: False)
    monkeypatch.setattr(v, "_lane_width", lambda pose, fallback_m=3.5: 3.5)


def drive(v, monkeypatch, ticks=220, row=True, extra_row=(), extension=True):
    """Stand behind the dead car until the gate takes a way round, then drive the rewritten route at 2 m/s until the
    pass completes or the planner stops the van. Returns the tick records."""
    lane_stubs(monkeypatch, v)
    if not extension:
        monkeypatch.setattr(v, "_extend_pass_past", lambda pose, perception: None)
    start_mission(v, route=base_route(), dest=(49.33, 141.13))
    pose = ACC_POSE; recs = []; moving = False
    for k in range(ticks):
        place(v, x=pose[0], y=pose[1], yaw=pose[2], speed=1.9 if moving else 0.0)
        see(v, [stopped_car(pose)] + (row_objects(pose, extra_row) if row else []))
        v.tick()
        v.clock.advance(0.25)
        st = v._current_state; c = chain(v); rp = v._overtake_point
        recs.append(dict(t=k * 0.25, x=pose[0], y=pose[1], hd=math.degrees(pose[2]), active=rp is not None,
                         rejoin=(round(rp.x, 2), round(rp.y, 2)) if rp else None, level=st["planner"]["level"],
                         reason=st["planner"]["reason"], gate=((st.get("go_around") or {}).get("gate") or {}).get("reason_code"),
                         steer=c["steer"], lat=_project(pose[0], pose[1], [(x, y, *_)[0:2] for x, y, *_ in ROUTE_SEG])[1]))
        if rp is not None or (moving and recs[-1]["level"] != "blocked"):
            if st["planner"]["level"] != "blocked":
                pose = advance(v._route.waypoints, pose, 0.5); moving = True
        if moving and rp is None and pose[0] > 44.0:
            break
    return recs


# ---------------------------------------------------------------- Case A: the recorded stopped-car rejoin
def test_case_a_the_old_ramp_back_lands_on_the_patrol_and_the_body_cannot_drive_it():
    trial = base_route()
    rp = RoutePlanner.__new__(RoutePlanner)
    rejoin = rp.plan_overtake(trial, ACC_POSE[0], ACC_POSE[1], 11.0, shift_m=3.6, lane_ok=None)
    assert (round(rejoin.x, 1), round(rejoin.y, 1)) == (19.3, 141.0)               # 27 m along, on the line: the old target
    pts = [(w.x, w.y) for w in trial.waypoints]
    tangents = [math.degrees(math.atan2(y2 - y1, x2 - x1)) for (x1, y1), (x2, y2) in zip(pts, pts[1:]) if 8.0 < x2 < 18.0]
    assert max(tangents) > 28.0, tangents                                          # the ramp back turns the nose 28-34 deg
    px, py, L, W = TRUTH["Patrol"]
    hit = sweep_conflict(trial.waypoints, ACC_POSE[:2], FOOT, (px, py), horizon_m=60.0, obstacle_box=ObstacleBox(L / 2, W / 2, 0.0))
    assert hit is not None and 24.0 < hit.along_m < 30.0 and math.degrees(hit.heading) > 25.0   # ...into the Patrol


def test_case_a_before_the_fix_the_van_stops_in_the_row_with_the_pass_still_active(van, monkeypatch):
    recs = drive(van, monkeypatch, extension=False)
    accepted = next(r for r in recs if r["active"])
    assert accepted["rejoin"] == (19.33, 140.96)
    # the harness measures the row honestly from 15 m, so the block on the ramp back comes while the van is
    # still on the plateau beside the Mustang (live, with the row measured later, it came at 28.7 deg beside
    # the Patrol); either way the way round is blocked with the pass active, and nothing re-plans
    stuck = [r for r in recs if r["active"] and r["level"] == "blocked" and r["x"] > 3.0]
    assert stuck, "the way round is blocked on the row"
    assert all(r["level"] == "blocked" for r in recs[recs.index(stuck[0]):]), "and stays blocked"
    assert recs[-1]["active"] and recs[-1]["x"] < 8.0, "and the pass never completes"


def test_case_a_and_c_after_the_fix_the_pass_is_extended_past_the_row_and_completes(van, monkeypatch):
    recs = drive(van, monkeypatch)
    accepted = next(r for r in recs if r["active"])
    ext = [r for r in recs if r["gate"] == "PASS_EXTENDED"]
    assert ext, "the extension fires once the row is measured"
    assert ext[0]["rejoin"][0] >= 17.73 + 5.57 / 2 + 16.0 - 1.0, ext[0]           # rejoin past the Patrol's far end + 16 m
    assert ext[0]["x"] < 9.0, ext[0]                                              # decided before the ramp back began
    beside = [r for r in recs if 8.0 <= r["x"] <= 21.0]
    assert all(r["level"] != "blocked" for r in beside), [r for r in beside if r["level"] == "blocked"][:2]
    assert max(abs(r["hd"]) for r in beside) < 12.0, max(abs(r["hd"]) for r in beside)     # nose kept along the lane
    assert min(r["y"] for r in beside) < 137.8 and max(r["y"] for r in beside) < 138.4     # stays in the passing lane
    assert not recs[-1]["active"], "the pass completes"
    assert abs(recs[-1]["lat"]) < 0.6, recs[-1]                                    # back on the original line
    # Step 12: the steering the corrected route asks for -- no full lock, no spike
    steer = [abs(r["steer"]) for r in recs if r["active"]]
    turns = [abs(b["hd"] - a["hd"]) for a, b in zip(recs, recs[1:]) if a["active"]]
    print(f"Step 12: max |steer| {max(steer):.2f}, max heading change per tick {max(turns):.1f} deg, "
          f"extension at x {ext[0]['x']:.1f} m, rejoin moved {accepted['rejoin'][0]:.1f} -> {ext[0]['rejoin'][0]:.1f} m")
    assert max(steer) < 0.9, max(steer)


# ---------------------------------------------------------------- Case B / E: a normal pass, nothing beside the line
def test_case_b_and_e_a_plain_pass_rejoins_at_16_m_and_the_extension_never_speaks(van, monkeypatch):
    recs = drive(van, monkeypatch, row=False)
    assert next(r for r in recs if r["active"])["rejoin"] == (19.33, 140.96)
    assert not any(r["gate"] == "PASS_EXTENDED" for r in recs)
    assert not recs[-1]["active"] and abs(recs[-1]["lat"]) < 0.6
    # the acceptance tick itself still carries the block the planner saw before the swap; once moving, never
    assert not any(r["level"] == "blocked" for r in recs if r["active"] and r["x"] > ACC_POSE[0] + 1.0)


# ---------------------------------------------------------------- Case D: the rejoin is always ahead; extension only further
def test_case_d_a_rejoin_is_always_ahead_along_the_route_and_extensions_only_move_it_on():
    rp = RoutePlanner.__new__(RoutePlanner)
    for along in (0.6, 5.0, 11.0, 20.0):
        trial = base_route()
        rejoin = rp.plan_overtake(trial, ACC_POSE[0], ACC_POSE[1], along, shift_m=3.6, lane_ok=None)
        pts = [(x, y) for x, y, *_ in ROUTE_SEG]
        ahead = _project(rejoin.x, rejoin.y, pts)[0] - _project(ACC_POSE[0], ACC_POSE[1], pts)[0]
        assert ahead >= along + 16.0 - 2.0, (along, ahead)
    # already at the full shift, planned again from mid-pass: the tail starts at the offset, no ramp out
    trial = base_route(); pose = (7.0, 137.3, 0.0)
    rejoin = rp.plan_overtake(trial, pose[0], pose[1], 13.5, shift_m=3.6, lane_ok=None, already_over=True)
    first = next(w for w in trial.waypoints if w.x > pose[0] + 0.5)
    assert abs(first.y - 137.3) < 0.4 and rejoin.x > pose[0] + 27.0, (first.y, rejoin.x)


# ---------------------------------------------------------------- Case F: a genuinely unsafe rejoin is not forced
def test_case_f_a_row_to_the_end_of_the_route_leaves_the_van_in_the_lane_not_steered_into_it(van, monkeypatch):
    recs = drive(van, monkeypatch, extra_row=(24.0, 30.0, 36.0, 42.0, 48.0))
    assert recs[-1]["active"], "no way round the row exists on this route: the pass is not forced closed"
    assert recs[-1]["level"] == "blocked" and recs[-1]["x"] < 44.0, recs[-1]      # stopped in the lane before the next car
    assert max(abs(r["hd"]) for r in recs if r["x"] >= 8.0) < 12.0                # never aimed into the row


# ---------------------------------------------------------------- Case G: no go-around, nothing changes
def test_case_g_a_clear_road_never_starts_a_pass(van, monkeypatch):
    lane_stubs(monkeypatch, van)
    start_mission(van, route=base_route(), dest=(49.33, 141.13))
    pose = ACC_POSE
    for _ in range(40):
        place(van, x=pose[0], y=pose[1], yaw=pose[2], speed=4.0)
        see(van, row_objects(pose))
        van.tick(); van.clock.advance(0.25)
        assert van._overtake_point is None
        pose = advance(van._route.waypoints, pose, 1.0)
