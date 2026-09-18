"""V3B / P-B01: a far, sparse look at a parked car must not outlive the better looks that follow.

Recorded live (WAV-0888, run v18_after, 2026-09-17): the Mustang (4.72 x 1.89 m) was one LiDAR
ring's line, 1.0-1.3 x 0.00-0.05 m, from 32 m in. Confirmed as the track's best look, that
line stayed the reported box from 21 m down to 5.5 m while the track's own median spread grew
1.50 x 0.35 -> 1.87 x 1.18 -> 2.53 x 1.49 m; the lane-edge hold fired at 1.6 m instead of the
9 m the spread would have given. These drive the REAL ObjectTracker with sightings built from
the recorded numbers (van at the origin, facing +x: van frame = world frame)."""
import math
import sys

import pytest

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_v3a_parked_car_box import sighting, reported, run, track, widths_of, normal_view, merged_view, DT  # noqa
from test_v2a_end_to_end import van  # noqa: F401  (the real WarpAV, outside world stubbed)
from warp_av.perception.tracking import _degenerate, DEGENERATE_WIDTH_FRACTION, DEGENERATE_MIN_GAP_M, BOX_AGREE_NEEDED

CAR = (26.0, 2.8)        # where the Mustang sat on the approach: ahead and to the right


def far_line(k=0):
    """One ring's line at 21-32 m, as recorded: 1.1-1.3 m long, a few centimetres wide, no
    rectangle fitted beyond 20 m (the box IS the spread). The recorded line views drifted in
    heading by a degree or so a frame (58.5, 61.0, 66.0, 68.4, 70.9 ...), which is why they
    confirmed each other as the track's best look."""
    l, w = [(1.30, 0.04), (1.11, 0.00), (1.33, 0.05), (1.20, 0.02)][k % 4]
    return sighting(CAR[0], CAR[1], l, w, 60.0 + 1.0 * k, spread=(l, w), off=(0.0, 0.0), cls="vehicle")


def transitional(k=0):
    """The looks that followed on the recorded approach, 16 m to 5 m: the spread becomes a body
    while the rectangle fits (now within 20 m) swing in heading and centre from frame to frame
    (recorded box headings 66, 72.7, 116, 74.4, 30 ...), so no three of them agree and none is
    confirmed before the car is close."""
    l, w = [(1.50, 0.35), (1.69, 1.01), (1.87, 1.18), (2.00, 1.22), (2.53, 1.49), (2.61, 1.50)][k % 6]
    yaw = [28.8, 72.7, 116.1, 74.4, 30.4, 150.0, 95.0, 55.0, 10.0, 130.0, 170.0, 40.0][k % 12]
    off = [(-0.18, 0.36), (0.24, -0.01), (0.18, 0.15), (-0.17, 0.24), (0.11, 0.12), (-0.21, 0.17)][k % 6]
    return sighting(CAR[0], CAR[1], l, w, yaw, spread=(l, w), off=off, cls="vehicle")


def good_body(k=0):
    """A confirmed near look: 4.4-4.7 x 1.8 m within 5 deg of the truth."""
    jitter = [(0.0, 0.0), (0.05, 0.02), (-0.04, -0.02)][k % 3]
    return sighting(CAR[0], CAR[1], 4.49 + jitter[0], 1.81 + jitter[1], 177.0, spread=(4.5, 1.85), off=(0.9, 0.4), cls="vehicle")


# ================================================================ CASE A: a far line stays a line

def test_case_a_a_far_thin_line_is_reported_as_what_it_is():
    _, snaps = run([[far_line(k)] for k in range(8)])
    tid = next(iter(snaps[-1]))
    for length, width, _ in (s[tid] for s in snaps if tid in s):
        assert width <= 0.1 and length <= 1.4, (length, width)          # nothing invented


def test_degenerate_is_relative_to_the_tracks_own_median():
    class T: width_m = 1.18
    assert _degenerate(0.04, T()) and _degenerate(0.5, T()) and not _degenerate(0.6, T())
    class L: width_m = 0.05                                             # a track that IS a line
    assert not _degenerate(0.04, L()) and not _degenerate(0.0, L())     # one ring's noise, not evidence
    class M: width_m = 0.35                                             # the recorded median at 15.8 m
    assert _degenerate(0.04, M()) and not _degenerate(0.2, M())
    assert DEGENERATE_WIDTH_FRACTION == 0.5 and DEGENERATE_MIN_GAP_M == 0.2


# ================================================================ CASE B: thin, thin, then a body-like spread

def test_case_b_the_line_stops_being_reported_once_the_spread_is_a_body():
    frames = [[far_line(k)] for k in range(6)] + [[transitional(k)] for k in range(12)]
    tracker, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    widths = widths_of(snaps, tid)
    tr = track(tracker, tid)
    assert tr.width_m > 1.0, tr.width_m                                 # the median says: a body
    assert widths[-1] >= 1.0, widths                                     # ...and so does the reported box
    assert all(w >= 0.9 for w in widths[-4:]), widths                    # and stays one
    assert widths[-1] <= 1.6, widths                                     # nothing invented either
    assert tr.best_box is None or tr.best_box[4] >= 0.9, tr.best_box    # the line is no longer the best look


def test_case_b_before_the_fix_the_line_would_have_stood():
    """The mechanism the fix removes: the line views agree with each other, so they confirm
    a best look of area 0.07; with the fix that look is degenerate against the median."""
    frames = [[far_line(k)] for k in range(6)]
    tracker, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    tr = track(tracker, tid)
    assert tr.best_box is not None and tr.best_box[4] <= 0.06           # confirmed: the line IS the best look at 25 m
    for k in range(12):
        tracker.update([dict(transitional(k))], 6 * DT + k * DT)
    assert tr.width_m > 1.0
    assert tr.best_box is None or tr.best_box[4] >= 0.9, tr.best_box    # ...and not once the evidence is a body


# ================================================================ CASE C: thin, then a confirmed good fit

def test_case_c_a_confirmed_good_body_replaces_the_stale_line():
    frames = [[far_line(k)] for k in range(6)] + [[good_body(k)] for k in range(BOX_AGREE_NEEDED + 2)]
    tracker, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    length, width, yaw = snaps[-1][tid]
    assert width == pytest.approx(1.81, abs=0.1) and length == pytest.approx(4.49, abs=0.15)
    assert track(tracker, tid).best_box is not None and track(tracker, tid).best_box[4] > 1.7


# ================================================================ CASE D: one wide noisy look

def test_case_d_one_body_like_blink_among_lines_does_not_inflate_the_box():
    frames = [[far_line(k)] for k in range(8)] + [[transitional(4)]] + [[far_line(k)] for k in range(6)]
    _, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    widths = widths_of(snaps, tid)
    assert all(w <= 0.5 for w in widths), widths                        # the median never budged


# ================================================================ CASE E: P-B02 stays fixed

def test_case_e_the_v3a_merged_frame_is_still_never_the_car():
    frames = [[normal_view(k)] for k in range(4)] + [[merged_view()]] + [[normal_view(k)] for k in range(10)]
    _, snaps = run(frames)
    tid = next(iter(snaps[-1]))
    assert all(w <= 1.7 for w in widths_of(snaps, tid))


def test_case_e2_the_fuller_look_from_behind_still_stands_alongside():
    full = [[sighting(8.0, 2.0, 4.60, 1.85, 0.0, off=(0.1, 0.0))] for _ in range(6)]
    partial = [[sighting(8.0, 2.0, 4.20, 1.30, 0.5, off=(0.1, 0.3))] for _ in range(20)]
    _, snaps = run(full + partial)
    tid = next(iter(snaps[-1]))
    assert snaps[-1][tid][1] == pytest.approx(1.85, abs=0.05)         # 1.85 is not degenerate against 1.3


# ================================================================ CASE F: genuinely thin things

def test_case_f_a_kerb_line_and_a_pole_keep_their_thin_boxes():
    kerb = [[sighting(9.0, 3.2, 3.5, 0.1, 0.0, cls=None, height=0.15)] for _ in range(10)]
    _, snaps = run(kerb)
    tid = next(iter(snaps[-1]))
    assert all(abs(s[tid][1] - 0.1) < 0.05 and abs(s[tid][0] - 3.5) < 0.1 for s in snaps[2:] if tid in s)
    pole = [[sighting(12.0, -2.5, 0.12, 0.1, 0.0, cls=None, height=3.5)] for _ in range(10)]
    _, snaps = run(pole)
    tid = next(iter(snaps[-1]))
    assert all(s[tid][1] <= 0.15 for s in snaps[2:] if tid in s)


# ================================================================ CASE G: two close vehicles

def test_case_g_two_close_cars_keep_their_ids_through_the_far_to_near_transition():
    left = lambda k: (far_line(k) if k < 5 else transitional(k)) | {"wx": 26.0, "wy": -1.0, "box_wx": 26.0, "box_wy": -1.0}
    right = lambda k: (far_line(k + 1) if k < 5 else transitional(k + 1)) | {"wx": 26.0, "wy": 1.6, "box_wx": 26.0, "box_wy": 1.6}
    tracker, snaps = run([[left(k), right(k)] for k in range(14)])
    ids = sorted(snaps[-1])
    assert len(ids) == 2 and all(sorted(s) == ids for s in snaps[3:]), "the same two tracks throughout"
    by_y = sorted((track(tracker, tid) for tid in ids), key=lambda tr: tr.wy)
    assert by_y[0].wy < 0 < by_y[1].wy


# ================================================================ Step 10: the lane-edge hold downstream (V2A harness)

def _mustang_at_9m(van_yaw_deg, lat=2.85, ahead=9.4):
    yaw = math.radians(van_yaw_deg)
    return ahead * math.cos(yaw) + lat * math.sin(yaw), -ahead * math.sin(yaw) + lat * math.cos(yaw)


def recorded_line_report(x, y):
    """The planner-facing Mustang recorded at 9.4 m (v18_after WAV-0888, t+49.7): the remembered
    line 1.30 x 0.04 @ 74.4 deg, fitted centre (+0.18, +0.15) from the centroid, while the
    track's spread was already 2.00 x 1.22."""
    from warp_av.perception.perception import DetectedObject, ObjectType
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y), id=3310, stationary=True,
                          speed=0.0, length_m=2.00, width_m=1.22, height_m=1.15, yaw_deg=-13.0, box_dx=0.18, box_dy=0.15,
                          box_length_m=1.30, box_width_m=0.04, box_yaw_deg=74.4, motion_class="dynamic",
                          size_uncertain=True, confidence=0.86)


def body_report(x, y):
    """What the fixed tracker reports at the same moment: the median-length sighting's own box,
    2.00 x 1.22 at its centroid, along the object's axis."""
    from warp_av.perception.perception import DetectedObject, ObjectType
    return DetectedObject(object_type=ObjectType.VEHICLE, x=x, y=y, distance=math.hypot(x, y), id=3310, stationary=True,
                          speed=0.0, length_m=2.00, width_m=1.22, height_m=1.15, yaw_deg=-13.0, box_dx=0.0, box_dy=0.0,
                          box_length_m=2.00, box_width_m=1.22, box_yaw_deg=-13.0, motion_class="dynamic",
                          size_uncertain=True, confidence=0.86)


def _approach(van, obj_fn, van_yaw_deg=8.0):
    from test_v2a_end_to_end import start_mission, place, see, chain
    start_mission(van)
    yaw = math.radians(van_yaw_deg)                  # the live drift: the nose a few degrees toward the shoulder
    place(van, x=0.0, y=0.0, yaw=yaw, speed=6.4)
    see(van)
    van.tick()                                       # a cruise tick: the intended path from this pose
    ox, oy = _mustang_at_9m(van_yaw_deg)
    place(van, x=0.0, y=0.0, yaw=yaw, speed=6.4)
    see(van, [obj_fn(ox, oy)])
    van.tick()
    c = chain(van)
    c["edge_hold"] = bool(getattr(van._path, "edge_hold", False))
    return c


def test_step10_the_remembered_line_is_a_required_slow_too_under_the_pb05_tier(van):
    """V3B (2026-09-17) recorded this case as "the line hides the car from the hold": under the V1.7
    rule (the block margin, 0.30 m) the line's box cleared the body and nothing was required. P-B05
    made the early slow a tier of its own (EDGE_SLOW_CLEARANCE_M, 1.0 m along the intended path), so
    a stationary vehicle track this close asks for the pass speed whatever its remembered shape. The
    body case below still stands; what P-B01 fixed is the geometry the block and the pass are
    judged by, not whether this slow happens."""
    from warp_av.planning.instrumentation import PATH_SLOW
    from warp_av.behavior.transitions import OBJECT_AHEAD_SLOW
    before = _approach(van, recorded_line_report)
    print("Step 10 (line, P-B05 tier):", {k: before[k] for k in ("planner_level", "edge_hold", "why", "safety_required", "brake")})
    assert before["planner_level"] == PATH_SLOW and before["edge_hold"] is True, before
    assert before["why"] == OBJECT_AHEAD_SLOW and before["safety_required"] is True and before["brake"] > 0.0, before


def test_step10_the_body_the_tracker_now_reports_fires_the_hold_at_9m(van):
    from warp_av.planning.instrumentation import PATH_SLOW
    from warp_av.behavior.transitions import OBJECT_AHEAD_SLOW
    after = _approach(van, body_report)
    print("Step 10 AFTER (body):", {k: after[k] for k in ("planner_level", "edge_hold", "why", "safety_required", "brake")})
    assert after["planner_level"] == PATH_SLOW and after["edge_hold"] is True, after
    assert after["why"] == OBJECT_AHEAD_SLOW and after["safety_required"] is True, after
    assert after["brake"] == pytest.approx(0.6) and after["throttle"] == 0.0, after
