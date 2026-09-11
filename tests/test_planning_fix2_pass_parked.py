"""Fix 2: the van would not pass a car parked beside its path.

Live, 2026-09-10 (WAV-0228): it waited 65 s beside a parked car. CARLA's truth for that car:
centre (108.50, 112.20), heading -70 deg, 4.67 x 2.00 m, its near side 0.96 m from the van's.
What the van measured: its middle 0.9 m nearer the van, heading 9 deg off. Two measuring
faults, both fixed here, and a slow pass for when only the safety margin is in the way.
The lane there turns LEFT 14 deg over 12 m with the car on the right, so the van's front
corner swings towards it -- all rebuilt below in the van's own frame.
"""
import math

import pytest

import warp_av.planning.planner as P
from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput
from warp_av.perception.tracking import ObjectTracker, cluster_points, merge_split_clusters
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.planner import RoutePlanner, Route, Waypoint, can_pass_with_care

SPRINTER = VehicleFootprint(half_length=2.958, half_width=0.994, safety_margin=0.30)


def lane_turning_left():
    pts, x, y, h = [], -20.0, 0.0, 0.0
    for i in range(40):
        pts.append(Waypoint(x=x, y=y, yaw=h))
        if 0.0 <= -20.0 + i < 12.0:
            h -= math.radians(14.2 / 12.0)
        x += math.cos(h)
        y += math.sin(h)
    return Route(waypoints=pts)


def car(at=(2.6, 2.5), size=(4.3, 1.8, 1.5), yaw=-16.0, box=(0.0, 0.0), **kw):
    o = DetectedObject(object_type=kw.pop("kind", ObjectType.VEHICLE), x=at[0], y=at[1],
                       distance=math.hypot(*at), id=9, stationary=kw.pop("stationary", True),
                       yaw_deg=yaw, box_dx=box[0], box_dy=box[1], **kw)
    o.length_m, o.width_m, o.height_m = size
    return o


def judge(obj, with_care=True):
    saved = P.can_pass_with_care
    if not with_care:
        P.can_pass_with_care = lambda o: False
    try:
        per = PerceptionOutput(objects=[obj])
        p = RoutePlanner.__new__(RoutePlanner)
        p.filter_to_route_corridor(per, lane_turning_left(), 0.0, 0.0, 0.0, footprint=SPRINTER)
        return per, p.last_decision
    finally:
        P.can_pass_with_care = saved


AS_MEASURED = dict()                                        # middle of the points, old heading
MEASURED_RIGHT = dict(box=(0.86, 0.34), yaw=-7.0)           # the rectangle's centre, heading now
TRUTH = dict(at=(3.46, 2.84), size=(4.67, 2.0, 1.5), yaw=-7.0)


def test_as_it_was_measured_the_van_waited():
    per, d = judge(car(**AS_MEASURED))
    assert per.path_blocked and d.reason == "blocked_swept_path"


def test_measured_right_it_passes_with_the_full_margin():
    per, d = judge(car(**MEASURED_RIGHT), with_care=False)
    assert not per.path_blocked


def test_the_truth_itself_has_room():
    per, _ = judge(car(**TRUTH), with_care=False)
    assert not per.path_blocked


def test_a_car_really_sticking_out_still_stops_the_van():
    """Its near side 0.2 m from the van's: no margin, no slow pass."""
    per, d = judge(car(at=(3.46, 2.2), size=(4.67, 2.0, 1.5), yaw=-7.0))
    assert per.path_blocked and d.passing_id is None


def test_only_the_margin_in_the_way_means_a_slow_pass():
    """Box put right, heading still old: the full margin says touch, the car is passed slowly."""
    per, d = judge(car(box=(0.86, 0.34)))
    assert not per.path_blocked and d.passing_id == 9
    assert per.closest_obstacle_type == ObjectType.VEHICLE, "reported, so the 3 m/s slow zone applies"
    per, _ = judge(car(box=(0.86, 0.34)), with_care=False)
    assert per.path_blocked


def test_a_moving_car_is_not_passed_this_way():
    assert not can_pass_with_care(car(stationary=False))


@pytest.mark.parametrize("kind", [ObjectType.PEDESTRIAN, ObjectType.CYCLIST, ObjectType.OBSTACLE])
def test_people_and_unknown_things_keep_the_full_margin(kind):
    assert not can_pass_with_care(car(kind=kind, size=(0.6, 0.5, 1.7)))


def test_a_static_post_keeps_the_full_margin():
    """It stands at the height of the van's mirrors; a car's roof is below them."""
    assert not can_pass_with_care(car(kind=ObjectType.OBSTACLE, size=(0.3, 0.3, 3.5),
                                      motion_class="static", static_rule="pole"))


# ---- the two measuring faults ----------------------------------------------------------

def side_and_end_of_a_car(cx=4.0, cy=2.9, length=4.6, width=2.0, step=0.1):
    """What the LiDAR sees of a car ahead and to the right: its near (left) side and its
    rear, an L of points piled up at the corner nearest the van."""
    hl, hw = length / 2.0, width / 2.0
    side = [(cx - hl + k * step, cy - hw) for k in range(int(length / step) + 1)]
    rear = [(cx - hl, cy - hw + k * step) for k in range(1, int(width / step) + 1)]
    return side + rear


def test_the_rectangle_is_centred_on_the_car_not_on_its_points():
    c = cluster_points(side_and_end_of_a_car(), cell=0.8, min_points=3)[0]
    assert math.hypot(c["x"] - 4.0, c["y"] - 2.9) > 0.5, "the average sits on the van's side"
    assert math.hypot(c["box_x"] - 4.0, c["box_y"] - 2.9) < 0.1, "the rectangle does not"
    assert abs(c["box_yaw_deg"]) < 2.0 and c["box_len"] == pytest.approx(4.6, abs=0.15)
    assert abs(c["yaw_deg"]) > 8.0, "the spread of an L runs diagonally -- left as it was"


def test_a_single_face_fits_along_it():
    side = [(2.0 + k * 0.1, 3.0) for k in range(40)]
    c = cluster_points(side, cell=0.8, min_points=3)[0]
    assert abs(c["box_yaw_deg"]) < 2.0 and c["box_wid"] < 0.05


def test_far_off_things_are_not_fitted():
    far = [(p[0] + 30.0, p[1]) for p in side_and_end_of_a_car()]
    c = cluster_points(far, cell=0.8, min_points=3)[0]
    assert c["box_yaw_deg"] == pytest.approx(c["yaw_deg"])


def test_a_car_seen_as_two_halves_is_centred_between_them():
    a = {"x": 2.0, "y": 3.0, "n": 20, "extent": 1.0, "distance": 3.6, "height": 1.5,
         "length_m": 2.0, "width_m": 1.8, "box_x": 2.1, "box_y": 3.3,
         "box_len": 2.0, "box_wid": 1.8, "box_yaw_deg": 0.0}
    b = {"x": 4.1, "y": 3.0, "n": 18, "extent": 1.0, "distance": 5.1, "height": 1.5,
         "length_m": 2.0, "width_m": 1.8, "box_x": 4.1, "box_y": 3.3,
         "box_len": 2.0, "box_wid": 1.8, "box_yaw_deg": 1.0}
    merged = merge_split_clusters([a, b])
    assert len(merged) == 1
    assert merged[0]["box_x"] == pytest.approx(3.1) and merged[0]["box_len"] >= 3.6


def test_the_tracker_keeps_the_heading_on_the_map_and_the_box_with_it():
    tr = ObjectTracker()
    for t in range(6):
        tracks = tr.update([{"wx": 10.0, "wy": 3.0, "length_m": 4.5, "width_m": 1.8, "height_m": 1.5,
                             "yaw_deg": -16.0 - t, "yaw_world_deg": -70.0,
                             "box_wx": 10.8, "box_wy": 3.3, "box_len": 4.7, "box_wid": 2.0,
                             "box_yaw_world_deg": -69.0, "distance": 10.0}], t * 0.1)
    track = tracks[0]
    assert track.yaw_world_deg == pytest.approx(-70.0)
    assert track.box_off == pytest.approx((0.8, 0.3))
    assert (track.box_len, track.box_wid, track.box_yaw_world_deg) == pytest.approx((4.7, 2.0, -69.0))


def test_the_planner_judges_the_fitted_rectangle_when_there_is_one():
    """The live car as perception now reports it: spread values as they were (4.3 x 1.8 at
    -16 deg), the fitted rectangle beside them (4.67 x 2.0 at -7 deg, centred right)."""
    obj = car(box=(0.86, 0.34), box_length_m=4.67, box_width_m=2.0, box_yaw_deg=-7.0)
    per, _ = judge(obj, with_care=False)
    assert not per.path_blocked
    assert P.obstacle_box_for(obj, 0.0).heading == pytest.approx(math.radians(-7.0))


def test_perception_hands_both_on():
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "perception" / "camera_lidar_perception.py").read_text()
    assert '"yaw_world_deg": float(c.get("yaw_deg", 0.0) or 0.0) + tf.rotation.yaw' in src
    assert "yaw_now = (tr.yaw_world_deg - tf.rotation.yaw + 180.0) % 360.0 - 180.0" in src
    assert "box_dx=ox * cy + oy * sy, box_dy=-ox * sy + oy * cy" in src
    assert '"box_yaw_world_deg": float(c.get("box_yaw_deg", c.get("yaw_deg", 0.0)) or 0.0) + tf.rotation.yaw' in src


def test_a_still_thing_keeps_its_best_look():
    """Alongside, the van's own body hides a parked car's side: the best view stands."""
    tr = ObjectTracker()
    good = {"wx": 10.0, "wy": 3.0, "length_m": 4.3, "width_m": 1.7, "height_m": 1.5, "yaw_deg": 0.0,
            "yaw_world_deg": 0.0, "box_wx": 10.3, "box_wy": 3.4, "box_len": 4.3, "box_wid": 1.7,
            "box_yaw_world_deg": 0.5, "distance": 6.0}
    poor = dict(good, box_wx=10.1, box_wy=3.1, box_wid=1.4, box_yaw_world_deg=6.5, distance=1.5)
    t = 0.0
    for _ in range(5):
        tracks = tr.update([dict(good)], t); t += 0.1
    for _ in range(20):
        tracks = tr.update([dict(poor)], t); t += 0.1
    best = tracks[0].best_box
    assert best is not None and best[5] == pytest.approx(0.5) and best[4] == pytest.approx(1.7)


def test_the_best_look_is_dropped_when_it_moves():
    tr = ObjectTracker()
    t = 0.0
    for k in range(30):
        x = 10.0 + (0.0 if k < 10 else (k - 10) * 0.6)          # parked, then drives off
        tracks = tr.update([{"wx": x, "wy": 3.0, "length_m": 4.3, "width_m": 1.7, "height_m": 1.5,
                             "yaw_deg": 0.0, "yaw_world_deg": 0.0, "box_wx": x + 0.3, "box_wy": 3.4,
                             "box_len": 4.3, "box_wid": 1.7, "box_yaw_world_deg": 0.0, "distance": 8.0}], t)
        t += 0.1
    assert not tracks[0].stationary and tracks[0].best_box is None


def test_one_odd_bigger_view_does_not_become_the_best():
    """Live: a parked car measured 2.3 degrees off, then ONE merged, bigger view 19 off."""
    tr = ObjectTracker()
    good = {"wx": 10.0, "wy": 3.0, "length_m": 4.3, "width_m": 1.7, "height_m": 1.5, "yaw_deg": 0.0,
            "yaw_world_deg": 0.0, "box_wx": 10.3, "box_wy": 3.4, "box_len": 4.3, "box_wid": 1.7,
            "box_yaw_world_deg": 2.3, "distance": 11.0}
    odd = dict(good, box_wx=10.6, box_wy=3.2, box_len=5.1, box_wid=2.3, box_yaw_world_deg=19.0)
    t = 0.0
    for view in [good] * 6 + [odd] + [good] * 4:
        tracks = tr.update([dict(view)], t)
        t += 0.1
    assert tracks[0].best_box[5] == pytest.approx(2.3), "the odd view was never confirmed"
