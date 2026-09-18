"""V3A / P-B02: the tracker's remembered box for a parked car (tracking.best_box).

Recorded live (WAV-0001, run v18_after, 2026-09-17): a Mercedes 4.67 x 2.00 m parked on the
shoulder was reported as 4.50 x 2.56 m, heading 12 deg off, for 82.6 s -- one frame in which the
car's cluster took in a chip of the kerb beyond it (fixture rc1_merc_C5, forced merge: 4.5 x
2.6-2.7 m, centre 0.6 m over) was confirmed by the normal views (agreement ignored width) and,
being the largest, became the car for the rest of the mission. The planner blocked on it beside
0.86 m of real room.

These drive the REAL ObjectTracker with sightings built from the recorded numbers. The van
stands at the origin facing +x, so the van frame is the world frame."""
import math
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_v2a_end_to_end import van  # noqa: F401  (the real WarpAV, outside world stubbed)

import pytest

from warp_av.perception.tracking import (ObjectTracker, _agree, BOX_AGREE_NEEDED, BOX_VIEWS_KEPT,
                                         BOX_AGREE_WIDTH_M, BOX_SUPPORT_KEPT)

DT = 0.15                                    # a perception frame at the live 6-7 Hz


def sighting(wx, wy, length, width, yaw_deg, spread=None, cls="vehicle", off=(0.0, 0.0),
             height=1.4, weak=False):
    """One observation as camera_lidar_perception hands it to the tracker: the cluster's
    centroid (wx, wy), the points' spread, and the fitted rectangle (its centre `off` from
    the centroid, its heading on the map)."""
    sl, sw = spread if spread else (length, width)
    return {"static_shapes": (), "road_gap_fn": None, "wx": wx, "wy": wy, "cls": cls,
            "cls_source": "camera" if cls else None, "confidence": 0.9 if cls else 0.0,
            "weak": weak, "distance": math.hypot(wx, wy),
            "length_m": sl, "width_m": sw, "height_m": height, "yaw_deg": yaw_deg,
            "yaw_world_deg": yaw_deg, "box_wx": wx + off[0], "box_wy": wy + off[1],
            "box_len": length, "box_wid": width, "box_yaw_world_deg": yaw_deg}


def reported(tr):
    """(length, width, heading) as camera_lidar_perception reports it: the best confirmed
    look while the thing stands still, else the box of the sighting nearest the median."""
    if tr.stationary and tr.best_box is not None:
        return tr.best_box[3], tr.best_box[4], tr.best_box[5]
    return tr.box_len, tr.box_wid, tr.box_yaw_world_deg


def run(frames, tracker=None, t0=0.0):
    """Feed one observation list per frame. Returns (tracker, snaps): snaps[k] maps track id to
    the box REPORTED at frame k -- a copy taken then, since the Track objects live on."""
    tracker = tracker or ObjectTracker()
    snaps = []
    for k, obs in enumerate(frames):
        tracks = tracker.update([dict(o) for o in obs], t0 + k * DT)
        snaps.append({tr.tid: reported(tr) for tr in tracks})
    return tracker, snaps


def track(tracker, tid):
    return next(tr for tr in tracker._tracks if tr.tid == tid)


def widths_of(snaps, tid):
    return [s[tid][1] for s in snaps if tid in s]


# ---- the recorded views of the Mercedes, van frame at the 5.4 m pose (fixture rc1_merc_C5)
def normal_view(k=0):
    """The car as it was fitted in 10 of 14 frames: 4.1-4.3 x 1.4-1.5 m, heading 170 deg."""
    jitter = [(0.0, 0.0), (0.05, -0.03), (-0.04, 0.02), (0.03, 0.04)][k % 4]
    return sighting(5.4 + jitter[0], 2.3 + jitter[1], 4.16 + jitter[0], 1.42 + jitter[1], 170.5,
                    spread=(4.29, 1.45), off=(1.05, 0.16))          # the recorded fitted centre


def merged_view():
    """The frame that took in the kerb chip: the same length and heading, 2.6 m wide, its
    fitted centre 0.48 m from the normal one (recorded: (+0.87, +0.63) against (+1.05, +0.16),
    inside the 0.5 m the views agree within); its own spread is wide too (self-consistent)."""
    return sighting(5.4, 2.3, 4.50, 2.62, 171.0, spread=(4.96, 2.42), off=(0.90, 0.62))


# ================================================================ CASE A: the recorded oversized fit

def test_case_a_one_merged_frame_does_not_become_the_car():
    frames = [[normal_view(k)] for k in range(4)] + [[merged_view()]] + [[normal_view(k)] for k in range(10)]
    _, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    widths = widths_of(snaps, tid)
    assert all(w <= 1.7 for w in widths), widths           # never the 2.62 m view, not even once
    assert widths[-1] == pytest.approx(1.42, abs=0.1)


def test_a_merged_view_is_not_agreed_with_by_the_normal_views():
    v_normal = (4.16 * 1.42, 5.7, 2.5, 4.16, 1.42, 170.5)
    v_merged = (4.50 * 2.62, 5.66, 3.14, 4.50, 2.62, 171.0)
    assert not _agree(v_merged, v_normal)                  # width: 1.2 m apart
    v_same = (4.2 * 1.5, 5.75, 2.55, 4.2, 1.5, 172.0)
    assert _agree(v_same, v_normal)                        # 0.08 m wider, 1.5 deg off: the same look
    assert BOX_AGREE_WIDTH_M < 1.2


# ================================================================ CASE B: several good views

def test_case_b_consistent_views_give_a_stable_box():
    tracker, snaps = run([[normal_view(k)] for k in range(14)])
    tid = next(iter(snaps[-1]))
    boxes = [s[tid] for s in snaps[2:] if tid in s]
    widths = [b[1] for b in boxes]
    assert all(1.3 <= w <= 1.6 for w in widths), widths
    assert max(abs(a - b) for a, b in zip(widths, widths[1:])) <= 0.15, widths
    yaws = [b[2] for b in boxes]
    assert max(abs(a - b) for a, b in zip(yaws, yaws[1:])) <= 1.5, yaws
    assert track(tracker, tid).best_box is not None           # confirmed, and it stands


# ================================================================ CASE C: reasonable, then bad, then good

def test_case_c_a_run_of_merged_frames_does_not_show_when_the_track_itself_says_merge():
    """V3A let three agreeing merged frames show as the best look for up to 63 frames, on the
    grounds that they were genuinely observed. The Patrol tight-pass record (2026-09-17) showed
    what that costs: three kerb-bridged 5.7 x 2.7 m frames of a 5.6 x 2.4 m Patrol stood as the
    look for 13-18 s while the van waited beside a safe 0.84 m pass. A rectangle covering more
    than 1.3 times the track's own median point spread (tracking.MERGED_AREA_SLACK, the planner's
    FIT_AREA_SLACK) is a merge: never the look, and let go if it has become one."""
    frames = ([[normal_view(k)] for k in range(4)]
              + [[merged_view()] for _ in range(BOX_AGREE_NEEDED)]         # three: they confirm each other
              + [[normal_view(k)] for k in range(BOX_SUPPORT_KEPT + 6)])
    _, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    widths = widths_of(snaps, tid)
    assert all(w <= 2.0 for w in widths[3:]), [i for i, w in enumerate(widths) if w > 2.0]
    assert widths[-1] == pytest.approx(1.42, abs=0.1), widths
    assert all(w <= 1.7 for w in widths[-5:]), widths


# ================================================================ CASE D: a partial look must not shrink a good body

def test_case_d_a_few_narrow_partial_views_do_not_shrink_a_well_supported_car():
    full = [[sighting(8.0, 2.0, 4.60, 1.85, 0.0, off=(0.1, 0.0))] for _ in range(8)]   # seen from behind
    partial = [[sighting(8.0, 2.0, 4.20, 1.30, 0.5, off=(0.1, 0.3))] for _ in range(3)]  # alongside: one side
    _, snaps = run(full + partial)
    tid = next(iter(snaps[-1]))
    assert snaps[-1][tid][1] == pytest.approx(1.85, abs=0.05)
    assert snaps[-1][tid][0] == pytest.approx(4.60, abs=0.05)


def test_case_d2_a_sustained_change_of_view_is_believed_after_the_window():
    """Sixty-odd sightings of a narrower look is not a blink, nor a pass: the box follows
    the evidence -- and not before (test_planning_fix2_pass_parked: twenty alongside views
    must leave the fuller look from behind standing)."""
    full = [[sighting(8.0, 2.0, 4.60, 1.85, 0.0, off=(0.1, 0.0))] for _ in range(6)]
    partial = [[sighting(8.0, 2.0, 4.20, 1.30, 0.5, off=(0.1, 0.3))] for _ in range(BOX_SUPPORT_KEPT + 2)]
    _, snaps = run(full + partial)
    tid = next(iter(snaps[-1]))
    assert snaps[-1][tid][1] == pytest.approx(1.30, abs=0.05)


# ================================================================ CASE E: two close vehicles

def test_case_e_two_cars_side_by_side_keep_their_own_boxes_and_ids():
    left = lambda k: sighting(9.0, -1.2, 4.5, 1.9, 0.0, off=(0.0, 0.0))
    right = lambda k: sighting(9.0, 1.4, 4.4, 1.8, 2.0, off=(0.0, 0.0))     # 2.6 m apart, centre to centre
    frames = [[left(k), right(k)] for k in range(12)] + [[left(0), merged_view_at(9.0, 1.4)]] + [[left(k), right(k)] for k in range(6)]
    tracker, snaps = run(frames)
    ids = sorted(snaps[-1])
    assert len(ids) == 2, ids
    assert all(sorted(s) == ids for s in snaps[3:]), "the same two tracks throughout"
    by_y = sorted((track(tracker, tid) for tid in ids), key=lambda tr: tr.wy)
    assert snaps[-1][by_y[0].tid][1] == pytest.approx(1.9, abs=0.1) and snaps[-1][by_y[1].tid][1] == pytest.approx(1.8, abs=0.1)
    assert by_y[0].wy < 0 < by_y[1].wy
    assert all(s[by_y[1].tid][1] <= 2.0 for s in snaps[3:] if by_y[1].tid in s), "the blink never became the car"


def merged_view_at(wx, wy):
    return sighting(wx, wy, 4.5, 2.62, 1.0, spread=(4.9, 2.4), off=(0.0, 0.6))


# ================================================================ CASE F: a small non-vehicle

def test_case_f_a_barrel_is_not_expanded_by_one_car_sized_blink():
    barrel = [[sighting(7.0, 1.5, 0.6, 0.55, 0.0, cls=None, height=1.0)] for _ in range(8)]
    blink = [[sighting(7.0, 1.5, 4.2, 1.8, 0.0, cls=None, height=1.0)]]
    _, snaps = run(barrel + blink + barrel[:4])
    tid = next(iter(snaps[-1]))
    assert all(s[tid][1] <= 0.7 and s[tid][0] <= 0.8 for s in snaps[2:] if tid in s), [s[tid][:2] for s in snaps if tid in s]


# ================================================================ CASE G: a clean, straight vehicle

def test_case_g_a_parked_car_seen_squarely_reports_its_measured_box():
    frames = [[sighting(12.0, 0.4, 4.55, 1.88, 0.0, off=(0.0, 0.0))] for _ in range(10)]
    _, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    length, width, yaw = snaps[-1][tid]
    assert length == pytest.approx(4.55, abs=0.05) and width == pytest.approx(1.88, abs=0.05)
    assert abs(yaw) < 0.5


def test_case_g2_a_moving_car_is_untouched_by_the_remembered_box_rule():
    frames = [[sighting(10.0 + 1.5 * k, 0.5, 4.5, 1.9, 0.0)] for k in range(10)]
    tracker, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    tr = track(tracker, tid)
    assert tr.stationary is False and tr.best_box is None
    assert snaps[-1][tid][1] == pytest.approx(1.9, abs=0.05)


# ================================================================ the downstream effect (V2A/V2B harness)

def test_the_box_the_tracker_now_reports_lets_the_planner_clear_the_car(van):
    """Step 12: the same recorded sequence, then the tracker's reported box through the REAL
    chain at the stall pose. Before V3A the reported box was the merged frame (4.50 x 2.56)
    and the chain blocked (tests/test_v2b_recorded_failures.py); now it is the confirmed
    normal look, and the chain drives on."""
    from warp_av.perception.perception import DetectedObject, ObjectType
    from test_v2b_recorded_failures import rc1_chain, recorded_mercedes
    from warp_av.planning.instrumentation import PATH_BLOCKED
    frames = [[normal_view(k)] for k in range(4)] + [[merged_view()]] + [[normal_view(k)] for k in range(10)]
    tracker, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    tr = track(tracker, tid)
    length, width, yaw = snaps[-1][tid]
    bx, by = tr.best_box[1], tr.best_box[2]
    # at the recorded stall pose the centroid sat 1.88 m ahead, 2.54 m to the right (V2B)
    reported_now = DetectedObject(object_type=ObjectType.VEHICLE, x=1.88, y=2.54, distance=3.2, id=5573,
                                  stationary=True, speed=0.0, length_m=tr.length_m, width_m=tr.width_m,
                                  height_m=1.14, yaw_deg=-13.0, box_dx=bx - tr.wx, box_dy=by - tr.wy,
                                  box_length_m=length, box_width_m=width, box_yaw_deg=yaw,
                                  motion_class="dynamic", size_uncertain=tr.size_uncertain, confidence=0.93)
    assert width <= 1.7 and abs(bx - tr.wx - 1.05) < 0.15 and abs(by - tr.wy - 0.16) < 0.15
    before = rc1_chain(van, recorded_mercedes())
    after = rc1_chain(van, reported_now)
    print("Step 12 BEFORE:", {k: before[k] for k in ("planner_level", "planner_reason", "closest_m", "behaviour", "brake")})
    print("Step 12 AFTER: ", {k: after[k] for k in ("planner_level", "planner_reason", "closest_m", "behaviour", "throttle", "brake")})
    assert before["planner_level"] == PATH_BLOCKED and before["brake"] == 1.0
    assert after["planner_level"] != PATH_BLOCKED and after["stop"] is False and after["throttle"] > 0.0, after
