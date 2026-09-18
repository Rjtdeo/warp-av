"""Patrol tight-pass block (2026-09-17): the van stood 13-18 s beside the Nissan Patrol parked on the start-road
shoulder, a pass that is physically safe (0.84 m body to body, 0.54 m beyond the block margin; the map in
scratch/patrol/truth.json) and that P-B05 already drives at 2 m/s.

Root cause, from the record: during the approach a frame with kerb/ground/roof points bridged into the Patrol's
cluster fits a 5.7 x 2.7 m rectangle; three such frames agree, the rectangle wins the track's best look on area,
and it lives until 60 honest views have gone by. The planner refuses it (its area is 1.6-1.8 times the track's
median point spread) and falls back to the spread on the centroid at the median heading, which reaches 0.75-1.7 m
into the swept body: BLOCKED_SWEPT_PATH. Ten seconds later the overtake gate opens and accepts a 0.71 m nudge.
Now the tracker asks the same question where the look is chosen (tracking.MERGED_AREA_SLACK): a merged rectangle is
never the best look nor the reported sighting, so the planner sees the honest 5.1 x 1.7 m picture with its offset
and lets the van pass slowly.

Frames: recorded ticks are in the map frame with the run's own saved route; the tracker cases use the V3A helpers.
"""
import math
import pytest

from test_v3a_parked_car_box import sighting, run as run_frames, reported, track
from test_v2a_end_to_end import van, start_mission, place, see, chain  # noqa: F401
from warp_av.perception import tracking
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import PATH_SLOW, PATH_CLEAR, PATH_BLOCKED
from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.behavior import transitions as T

FOOT = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)

# ---- the start road as the runner saved it (pb05_after/WAV-0888__2, meta.route): straight, lane centre y 140.6-141.1
ROUTE_SEG = [(-18.58, 140.6, 0.006, False), (-16.58, 140.61, 0.006, False), (-14.58, 140.62, 0.006, False),
             (-12.58, 140.63, 0.006, False), (-10.58, 140.65, 0.006, False), (-8.58, 140.66, 0.006, False),
             (-6.58, 140.67, 0.006, False), (-4.58, 140.68, 0.006, False), (-2.58, 140.7, 0.006, False),
             (-0.21, 140.71, 0.006, False), (1.78, 140.72, 0.007, False), (3.76, 140.74, 0.011, False),
             (5.74, 140.77, 0.015, False), (7.72, 140.8, 0.019, False), (9.71, 140.84, 0.021, False),
             (11.73, 140.88, 0.018, False), (13.75, 140.91, 0.014, False), (15.77, 140.94, 0.01, False),
             (19.33, 140.96, 0.006, False), (21.33, 140.97, 0.006, False), (23.33, 140.98, 0.006, False),
             (25.33, 140.99, 0.006, False), (27.33, 141.0, 0.006, False), (29.33, 141.02, 0.006, False),
             (31.33, 141.03, 0.006, False), (33.33, 141.04, 0.006, False), (35.33, 141.05, 0.006, False),
             (37.33, 141.06, 0.006, False), (39.33, 141.07, 0.006, False), (41.33, 141.08, 0.006, False),
             (43.33, 141.09, 0.006, False), (45.33, 141.11, 0.006, False), (47.33, 141.12, 0.006, False),
             (49.33, 141.13, 0.006, False)]
# ---- tick 605 (t+16.9): the bridged look is the best look; live: BLOCKED_SWEPT_PATH on 2757, lateral 2.42
BLOCK_POSE = (12.26, 140.89, math.radians(0.9))
BLOCK_INTENDED = [(11.6, 140.88), (13.6, 140.91), (15.6, 140.93), (17.6, 140.95), (19.6, 140.96), (21.6, 140.98),
                  (23.6, 140.99), (25.6, 141.0), (27.6, 141.01), (29.6, 141.02), (31.6, 141.03), (33.6, 141.04),
                  (35.6, 141.05), (36.6, 141.06)]
BRIDGED_LOOK = {"x": 15.76, "y": 143.36, "distance": 4.3, "length_m": 5.15, "width_m": 1.84, "height_m": 2.02, "yaw_deg": -19.2,
                "box_dx": 0.93, "box_dy": 0.9, "box_length_m": 5.74, "box_width_m": 2.7, "box_yaw_deg": 178.5,
                "clearance_radius_m": 2.73, "confidence": 0.93, "id": 2757, "motion_class": "dynamic", "size_uncertain": True,
                "speed": 0.0, "stationary": True}
MUSTANG_605 = {"x": 10.34, "y": 143.67, "distance": 3.4, "length_m": 4.68, "width_m": 2.01, "height_m": 1.06, "yaw_deg": -21.5,
               "box_dx": 0.16, "box_dy": 0.32, "box_length_m": 4.82, "box_width_m": 1.72, "box_yaw_deg": 4.1,
               "clearance_radius_m": 2.55, "confidence": 0.93, "id": 2781, "motion_class": "dynamic", "size_uncertain": True,
               "speed": 0.0, "stationary": True}
# ---- tick 659 (t+31.0): the honest look; live: slow, edge_hold, no block, and the van moved on
HONEST_POSE = (14.36, 140.91, math.radians(0.6))
HONEST_INTENDED = [(14.33, 140.91), (16.32, 140.7), (18.29, 140.4), (20.29, 140.26), (22.29, 140.24), (24.29, 140.3),
                   (26.28, 140.46), (28.27, 140.69), (30.25, 140.9), (32.25, 141.02), (34.25, 141.06), (36.25, 141.07),
                   (38.25, 141.07), (39.25, 141.07)]
HONEST_LOOK = {"x": 16.39, "y": 143.25, "distance": 3.1, "length_m": 5.2, "width_m": 1.69, "height_m": 1.93, "yaw_deg": -9.1,
               "box_dx": 1.14, "box_dy": 0.54, "box_length_m": 5.14, "box_width_m": 1.74, "box_yaw_deg": 179.3,
               "clearance_radius_m": 2.74, "confidence": 0.93, "id": 2757, "motion_class": "dynamic", "size_uncertain": False,
               "speed": 0.0, "stationary": True}


def route():
    return Route(waypoints=[Waypoint(x=x, y=y, yaw=yaw, is_junction=j) for x, y, yaw, j in ROUTE_SEG])


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def from_api(o, pose, dy=0.0):
    """A published object back into the van's frame, as the planner takes it (the P-B05 replay's inverse)."""
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    dx_, dy_ = o["x"] - px, o["y"] + dy - py
    return DetectedObject(object_type=ObjectType.VEHICLE, x=c * dx_ + s * dy_, y=-s * dx_ + c * dy_,
                          **{k: v for k, v in o.items() if k not in ("x", "y")})


def decide(pl, pose, objects, intended):
    p = PerceptionOutput(objects=objects)
    p.path_blocked = False
    p.closest_obstacle_distance = 999.0
    return pl.filter_to_route_corridor(p, route(), *pose, danger_m=8.0, footprint=FOOT, intended_path=intended)


# ---------------------------------------------------------------- Case A: the recorded geometry reproduces the stall
def test_case_a_the_bridged_best_look_is_a_hard_block_on_the_recorded_tick():
    out = decide(planner(), BLOCK_POSE, [from_api(BRIDGED_LOOK, BLOCK_POSE), from_api(MUSTANG_605, BLOCK_POSE)], BLOCK_INTENDED)
    assert out.level == PATH_BLOCKED and out.reason == "blocked_swept_path" and out.blocker_id == 2757, out.one_line()
    assert 2.3 <= out.blocker_lateral_m <= 2.5


def test_case_a_the_tracker_no_longer_keeps_a_merged_frame_as_the_best_look(monkeypatch):
    def honest(k):
        j = ((k * 7) % 5 - 2) * 0.01
        return sighting(16.4 + j, 143.25, 5.10 + j, 1.75 + j, 179.0 + j * 10, spread=(5.00 + j, 1.70 + j), off=(1.10, 0.55))
    bridged = sighting(16.4, 143.25, 5.74, 2.70, 179.0, spread=(5.40, 2.40), off=(0.95, 0.90))
    frames = [[honest(k)] for k in range(4)] + [[bridged]] * 3 + [[honest(k)] for k in range(4, 40)]
    # the old rule: the three bridged frames agree, win on area and stand while the van waits
    monkeypatch.setattr(tracking, "MERGED_AREA_SLACK", 1e9)
    tracker, snaps = run_frames(frames)
    tid = next(iter(snaps[-1]))
    widths_old = [s[tid][1] for s in snaps[15:] if tid in s]
    assert max(widths_old) == pytest.approx(2.70), widths_old
    assert track(tracker, tid).width_m < 1.9              # the track's own median never said so
    # the rule: a rectangle claiming 1.3 times the median spread is a merge, never the look
    monkeypatch.setattr(tracking, "MERGED_AREA_SLACK", 1.3)
    tracker, snaps = run_frames(frames)
    tid = next(iter(snaps[-1]))
    widths_new = [s[tid][1] for s in snaps[15:] if tid in s]
    assert max(widths_new) < 1.9 and min(widths_new) > 1.6, widths_new
    assert all(s[tid][0] < 5.4 for s in snaps[15:] if tid in s)


# ---------------------------------------------------------------- Case B: the honest look is a safe low-speed pass
def test_case_b_the_honest_look_is_a_required_slow_not_a_block():
    for pose, intended in ((HONEST_POSE, HONEST_INTENDED), (BLOCK_POSE, BLOCK_INTENDED)):
        out = decide(planner(), pose, [from_api(HONEST_LOOK, pose)], intended)
        assert out.level == PATH_SLOW and out.blocked is False and out.edge_hold is True, out.one_line()


# ---------------------------------------------------------------- Case C: a car really in the way stays blocked
def test_case_c_the_same_car_half_a_metre_nearer_the_lane_is_still_a_hard_block():
    out = decide(planner(), HONEST_POSE, [from_api(HONEST_LOOK, HONEST_POSE, dy=-0.6)], HONEST_INTENDED)
    assert out.level == PATH_BLOCKED and out.blocked is True, out.one_line()


# ---------------------------------------------------------------- Case D / E: the gate, offline, through the real stack
def _stand_beside(van, obj_fn, seconds):
    from test_v2a_end_to_end import straight_road
    start_mission(van, route=straight_road())
    place(van, x=10.0, y=0.0, yaw=0.0, speed=0.0)
    see(van)
    van.tick()
    seen_blocked = seen_go_around = False; weighed_at = None; t = 0.0
    for _ in range(int(seconds / 0.25)):
        place(van, x=10.0, y=0.0, yaw=0.0, speed=0.0)
        see(van, [obj_fn()])
        van.tick()
        van.clock.advance(0.25)
        t += 0.25
        st = van._current_state
        seen_blocked = seen_blocked or st["planner"]["level"] == "blocked"
        ga = st.get("go_around") or {}
        if ga.get("options") and weighed_at is None:      # the gate opened and the ways round were weighed
            seen_go_around, weighed_at = True, t
    return seen_blocked, seen_go_around, weighed_at, van._current_state


def bridged_beside():
    # the recorded tick 605 in the van's frame: the Patrol's track 3.54 m ahead, 2.41 m to the right
    return DetectedObject(object_type=ObjectType.VEHICLE, x=3.54, y=2.41, distance=4.3, length_m=5.15, width_m=1.84, height_m=2.02,
                          yaw_deg=-19.2, box_dx=0.93, box_dy=0.9, box_length_m=5.74, box_width_m=2.7, box_yaw_deg=178.5,
                          clearance_radius_m=2.73, id=2757, stationary=True, size_uncertain=True, motion_class="dynamic")


def honest_beside():
    return DetectedObject(object_type=ObjectType.VEHICLE, x=2.06, y=2.32, distance=3.1, length_m=5.2, width_m=1.69, height_m=1.93,
                          yaw_deg=-9.1, box_dx=1.14, box_dy=0.54, box_length_m=5.14, box_width_m=1.74, box_yaw_deg=179.3,
                          clearance_radius_m=2.74, id=2757, stationary=True, size_uncertain=False, motion_class="dynamic")


def test_case_d_the_recorded_block_opens_the_gate_after_ten_seconds(van):
    # the harness has no map, so every way round is refused (MAP_ERROR); the transition itself -- ten
    # seconds of the recorded block, then the gate weighing the ways round -- is what live did
    blocked, go_around, weighed_at, st = _stand_beside(van, bridged_beside, 13.0)
    assert blocked and go_around and 10.0 <= weighed_at <= 12.0, (blocked, go_around, weighed_at, st.get("go_around"))
    assert {o["option"] for o in st["go_around"]["options"]} >= {"NUDGE_LEFT", "LANE_LEFT"}


def test_case_d_and_e_the_honest_look_never_blocks_so_the_gate_never_opens(van):
    blocked, go_around, weighed_at, st = _stand_beside(van, honest_beside, 13.0)
    assert not blocked and not go_around and weighed_at is None, (blocked, go_around, weighed_at, st.get("go_around"))
    assert st["planner"]["level"] == PATH_SLOW and st["planner"].get("edge_hold") is True     # P-B05 keeps the crawl
    assert st["overtaking"] is False
