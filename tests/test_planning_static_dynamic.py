"""Static or dynamic: can this thing move? (Planning V2, perception/motion_class.py)

Everything starts dynamic. Static is earned: a learned shape-and-place rule, holding over
several sightings, on a track nothing has ever named a person, cyclist or vehicle.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from warp_av.perception import motion_class as mc
from warp_av.perception.motion_class import (LOW, POLE, STRUCTURE, MotionMemory, RoadGap,
                                             rule_holds, shape_rules)
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.perception.tracking import ObjectTracker
from warp_av.planning.prediction import _could_use_a_road

FIX = Path(__file__).parent / "fixtures" / "static_truth"
ROAD_USERS = ("pedestrian", "rider", "car", "truck", "bus", "motorcycle", "bicycle")
STATIC_CLASSES = ("building", "wall", "fence", "pole", "traffic light", "traffic sign",
                  "vegetation", "static prop", "sidewalk/kerb", "terrain", "ground",
                  "bridge", "rail track", "guard rail")


# ---- the shape part -----------------------------------------------------------------

def test_a_lamp_post_has_the_shape_of_a_pole():
    assert POLE in shape_rules(3.6, 0.2, 0.45, 20)


def test_the_tallest_person_is_not_a_pole():
    # 1.92 m was the tallest person in the recordings; the floor is 2.3 m
    assert POLE not in shape_rules(1.92, 0.3, 0.0, 40)
    assert POLE not in shape_rules(2.25, 0.3, 0.4, 40), "nothing under 2.3 m is a pole"


def test_a_person_with_one_point_of_the_pole_behind_them_is_not_a_pole():
    # the fresh recording's one mistake before the 30 % rule: 4 points of a person and one
    # of a pole, "3.46 m tall"
    assert POLE not in shape_rules(3.46, 0.37, 0.20, 5)


def test_a_person_beside_a_pole_is_too_wide_for_one():
    assert POLE not in shape_rules(3.8, 0.77, 0.28, 25)


def test_a_building_front_has_the_shape_of_a_structure_and_a_car_does_not():
    assert STRUCTURE in shape_rules(3.9, 12.0, 0.75, 400)
    assert STRUCTURE not in shape_rules(1.5, 4.5, 0.0, 300)
    # a bus side-on: tall, and much of it high -- but always in a lane (see the place part)
    assert STRUCTURE in shape_rules(3.8, 9.8, 0.72, 600)


def test_a_kerb_is_low_and_a_crouching_person_is_not():
    assert LOW in shape_rules(0.14, 1.1, 0.0, 30)
    assert LOW not in shape_rules(0.9, 0.5, 0.0, 30)


def test_a_two_point_blob_is_never_judged():
    assert shape_rules(3.6, 0.2, 0.5, 2) == ()


# ---- the place part -----------------------------------------------------------------

def test_poles_and_structures_must_be_clear_of_every_lane():
    assert rule_holds(POLE, 0.3) and rule_holds(STRUCTURE, 5.0)
    assert not rule_holds(POLE, 0.2)
    assert not rule_holds(STRUCTURE, -1.6), "a bus IN a lane is never a building"


def test_a_kerb_may_sit_on_the_lane_edge_but_not_inside_the_lane():
    assert rule_holds(LOW, -0.2)
    assert not rule_holds(LOW, -0.7), "a car half-hidden behind another: 3 low points IN the lane"


def test_unknown_or_far_off_places_are_never_good_enough():
    assert not rule_holds(POLE, None)
    assert not rule_holds(POLE, float("nan"))
    assert not rule_holds(STRUCTURE, 30.0), "a lorry in a yard 30 m off looks like a wall"


def test_the_gap_to_the_road_is_taken_from_the_blobs_nearest_point():
    # a straight lane along x, centred on y = 0, 3.5 m wide: its edge is at y = 1.75
    gap = RoadGap(lambda x, y, z: (x, 0.0, 3.5))
    assert gap.gap_m([(0.0, 2.25)]) == pytest.approx(0.5)
    assert gap.gap_m([(0.0, 6.0), (1.0, 2.0), (2.0, 9.0)]) == pytest.approx(0.25)
    assert gap.gap_m([(0.0, 0.5)]) == pytest.approx(-1.25), "inside the lane: negative"
    assert RoadGap(lambda x, y, z: None).gap_m([(0.0, 5.0)]) is None


# ---- earning it, and losing it ------------------------------------------------------

def _see(m, t, shapes=(POLE,), gap=1.0, named=False, stationary=True, x=0.0, y=0.0):
    return m.note(shapes, (lambda: gap), named=named, stationary=stationary, wx=x, wy=y, t=t)


def test_static_needs_three_sightings_over_a_second():
    m = MotionMemory()
    assert m.state == "dynamic", "everything starts dynamic"
    assert _see(m, 0.0) == "dynamic"
    assert _see(m, 0.1) == "dynamic"
    assert _see(m, 0.2) == "dynamic", "three sightings, but only 0.2 s"
    assert _see(m, 1.0) == "static"
    assert m.rule == POLE


def test_a_miss_before_it_is_earned_starts_the_count_again():
    m = MotionMemory()
    _see(m, 0.0); _see(m, 0.5)
    _see(m, 0.7, shapes=())
    assert _see(m, 1.0) == "dynamic"
    _see(m, 1.5)
    assert _see(m, 2.0) == "static"


def test_once_static_it_rides_out_two_misses_and_drops_on_the_third():
    m = MotionMemory()
    for t in (0.0, 0.5, 1.0):
        _see(m, t)
    assert _see(m, 1.1, shapes=()) == "static"
    assert _see(m, 1.2, shapes=()) == "static"
    assert _see(m, 1.3, shapes=()) == "dynamic"


def test_one_road_user_name_and_it_is_dynamic_for_good():
    m = MotionMemory()
    for t in (0.0, 0.5, 1.0):
        _see(m, t)
    assert m.state == "static"
    assert _see(m, 1.1, named=True) == "dynamic"
    for t in (1.5, 2.0, 2.5, 3.0):
        assert _see(m, t) == "dynamic", "the name is gone but the memory of it is not"


def test_a_thing_that_moves_is_dynamic_at_once():
    m = MotionMemory()
    for t in (0.0, 0.5, 1.0):
        _see(m, t)
    assert _see(m, 1.1, stationary=False) == "dynamic"


def test_the_map_is_asked_once_and_the_answer_reused():
    calls = []

    def gap():
        calls.append(1)
        return 1.0

    m = MotionMemory()
    for t in (0.0, 0.3, 0.6, 0.9, 1.2):
        m.note((POLE,), gap, named=False, stationary=True, wx=0.0, wy=0.0, t=t)
    assert len(calls) == 1
    m.note((POLE,), gap, named=False, stationary=True, wx=0.0, wy=0.0, t=3.5)
    assert len(calls) == 2, "stale after 2 s"
    m.note((POLE,), gap, named=False, stationary=True, wx=0.8, wy=0.0, t=3.6)
    assert len(calls) == 3, "asked again once the track has moved"


def test_the_map_is_not_asked_at_all_when_no_shape_fits():
    m = MotionMemory()
    m.note((), lambda: pytest.fail("looked up a blob with no static shape"),
           named=False, stationary=True, wx=0.0, wy=0.0, t=0.0)


# ---- inside the tracker ---------------------------------------------------------------

def _obs(x, y, shapes=(POLE,), gap=1.0, cls=None):
    o = {"wx": x, "wy": y, "static_shapes": shapes, "road_gap_fn": (lambda: gap),
         "height_m": 3.5, "length_m": 0.2, "width_m": 0.2, "distance": 10.0}
    if cls:
        o["cls"], o["cls_source"], o["confidence"] = cls, "camera", 0.8
    return o


def test_the_tracker_carries_the_answer_on_each_track():
    tr = ObjectTracker()
    t = 0.0
    for _ in range(12):
        tracks = tr.update([_obs(10.0, 3.0)], t)
        t += 0.1
    assert len(tracks) == 1 and tracks[0].motion.state == "static"
    tracks = tr.update([_obs(10.0, 3.0, cls="pedestrian")], t)
    assert tracks[0].motion.state == "dynamic", "the camera saw a person"


def test_a_blob_in_the_lane_never_becomes_static_in_the_tracker():
    tr = ObjectTracker()
    for i in range(15):
        tracks = tr.update([_obs(10.0, 0.0, shapes=(STRUCTURE,), gap=-1.6)], i * 0.1)
    assert tracks[0].motion.state == "dynamic"


# ---- what static changes, and what it does not ---------------------------------------

def _thing(motion_class):
    return DetectedObject(object_type=ObjectType.OBSTACLE, x=8.0, y=3.0, distance=8.5,
                          height_m=1.5, length_m=0.4, width_m=0.4, motion_class=motion_class)


def test_prediction_leaves_static_things_alone():
    assert _could_use_a_road(_thing("dynamic")) is True
    assert _could_use_a_road(_thing("static")) is False


def test_everything_starts_dynamic():
    assert DetectedObject(object_type=ObjectType.OBSTACLE, x=1.0, y=0.0, distance=1.0).motion_class == "dynamic"


def test_static_in_the_path_still_blocks():
    """A static thing in the lane stops the van exactly as before: the corridor rules never
    read motion_class (only the parking give-up does, to decide when waiting is pointless)."""
    import inspect
    from warp_av.perception.perception import PerceptionOutput
    from warp_av.planning.planner import RoutePlanner, Route, Waypoint
    assert "motion_class" not in inspect.getsource(RoutePlanner.filter_to_route_corridor)
    for label in ("dynamic", "static"):
        post = DetectedObject(object_type=ObjectType.OBSTACLE, x=6.0, y=0.2, distance=6.0,
                              length_m=0.3, width_m=0.3, height_m=3.5, id=7,
                              motion_class=label, static_rule="pole" if label == "static" else "")
        per = PerceptionOutput(objects=[post])
        road = Route(waypoints=[Waypoint(x=-20.0 + i * 2.0, y=0.0) for i in range(61)])
        RoutePlanner.__new__(RoutePlanner).filter_to_route_corridor(per, road, 0.0, 0.0, 0.0)
        assert per.path_blocked, f"a {label} post in the lane must stop the van"


def test_the_switch_turns_it_off():
    assert mc.static_dynamic_wanted({}) is True
    assert mc.static_dynamic_wanted({"WARP_STATIC_DYNAMIC": "0"}) is False


def test_perception_hands_the_answer_on():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "perception" / "camera_lidar_perception.py").read_text()
    assert "motion_class=tr.motion.state" in src
    assert '"static_shapes": shapes' in src and '"road_gap_fn": gap_fn' in src


# ---- the rules against CARLA's answer key -------------------------------------------

def _judge(path):
    d = np.load(path)
    static = np.array([
        any(rule_holds(r, g) for r in shape_rules(h if np.isfinite(h) else 0.0, max(l, w),
                                                  f if np.isfinite(f) else 0.0, int(n)))
        for h, l, w, f, n, g in zip(d["height"], d["length"], d["width"], d["frac_high"],
                                    d["n"], d["road_gap"])])
    holds_user = np.array([sum(v for c, v in json.loads(js).items() if c in ROAD_USERS) >= 3
                           for js in d["tags_seen"]])
    user = np.isin(d["tag_name"], ROAD_USERS) | holds_user
    real_static = np.isin(d["tag_name"], STATIC_CLASSES) & ~holds_user & (d["road_gap"] <= 12.0)
    return static, user, real_static, d


@pytest.mark.parametrize("name", ["town10_seed7.npz", "town10_seed23_fresh.npz"])
def test_no_person_or_vehicle_is_called_static_on_a_single_sighting(name):
    static, user, real_static, d = _judge(FIX / name)
    wrong = np.where(static & user)[0]
    assert len(wrong) == 0, [(d["tag_name"][i], d["tags_seen"][i]) for i in wrong]
    assert user.sum() > 1000, "the recording must actually hold people and vehicles"


@pytest.mark.parametrize("name", ["town10_seed7.npz", "town10_seed23_fresh.npz"])
def test_about_a_quarter_of_static_things_are_caught(name):
    static, user, real_static, d = _judge(FIX / name)
    share = (static & real_static).sum() / real_static.sum()
    assert share >= 0.20, f"only {share:.0%} of static things called static"
    kerb = d["tag_name"] == "sidewalk/kerb"
    assert (static & kerb).sum() / kerb.sum() >= 0.75, "kerbs are the easy case"
