"""
Perception V2 day 7: one sheet that says what the van knows.

The world model is a view, not a second opinion: every number in it must be
the number perception produced. These tests pin that, pin the frame maths
(metres ahead and to the right of the van versus a place on the map), and pin
the small questions the rest of the stack asks.
"""
import math
import sys
import time
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.perception import DetectedObject, ObjectType, PerceptionOutput  # noqa: E402
from warp_av.world_model import (DEFAULT_PATH_HALF_WIDTH_M, WorldModel, WorldObject,  # noqa: E402
                                 build_world_model)


def pose(x=0.0, y=0.0, yaw=0.0, speed=0.0, healthy=True):
    return types.SimpleNamespace(x=x, y=y, yaw=yaw, speed=speed, healthy=healthy)


def obj(x, y, kind=ObjectType.OBSTACLE, **kw):
    kw.setdefault("distance", math.hypot(x, y))
    return DetectedObject(object_type=kind, x=x, y=y, **kw)


def test_an_empty_world_is_still_a_world():
    wm = build_world_model(PerceptionOutput(), pose())
    assert wm.objects == []
    assert wm.path.blocked is False
    assert wm.path.closest_distance_m is None, "999 means nothing there, not an object 999 m away"
    assert wm.as_dict()["counts"]["total"] == 0


def test_a_thing_ahead_keeps_its_place_in_both_frames():
    p = PerceptionOutput(objects=[obj(10.0, 2.0)])
    wm = build_world_model(p, pose(x=100.0, y=50.0, yaw=0.0))
    o = wm.objects[0]
    assert (o.x, o.y) == (10.0, 2.0), "ahead and to the right of the van, unchanged"
    assert (o.world_x, o.world_y) == pytest.approx((110.0, 52.0))


def test_the_map_place_turns_with_the_van():
    p = PerceptionOutput(objects=[obj(10.0, 0.0)])
    wm = build_world_model(p, pose(x=0.0, y=0.0, yaw=math.pi / 2))
    o = wm.objects[0]
    assert (o.world_x, o.world_y) == pytest.approx((0.0, 10.0), abs=1e-9), "facing north, 10 m ahead is 10 m north"


def test_every_number_comes_straight_from_perception():
    d = obj(8.0, -1.0, ObjectType.VEHICLE, speed=4.5, confidence=0.82, id=7,
            length_m=4.4, width_m=1.9, height_m=1.5, yaw_deg=12.0, stationary=False)
    wm = build_world_model(PerceptionOutput(objects=[d]), pose())
    o = wm.objects[0]
    assert (o.id, o.kind, o.speed_mps, o.confidence) == (7, "vehicle", 4.5, 0.82)
    assert (o.length_m, o.width_m, o.height_m, o.yaw_deg) == (4.4, 1.9, 1.5, 12.0)
    assert o.moving and not o.stationary and o.is_vehicle


def test_the_path_summary_is_carried_over_and_gets_an_id():
    d = obj(12.0, 0.3, ObjectType.PEDESTRIAN, id=3)
    p = PerceptionOutput(objects=[d], path_blocked=True, closest_obstacle_distance=12.0,
                         closest_obstacle_type=ObjectType.PEDESTRIAN, closest_obstacle_speed=1.2,
                         closest_obstacle_lateral_m=0.3)
    wm = build_world_model(p, pose())
    assert wm.path.blocked and wm.path.closest_kind == "pedestrian"
    assert wm.path.closest_distance_m == 12.0 and wm.path.closest_speed_mps == 1.2
    assert wm.path.closest_id == 3, "the summary should point at the object it came from"


def test_unhealthy_perception_is_reported_not_hidden():
    p = PerceptionOutput(healthy=False, reason="LIDAR_STALE_3.0s")
    wm = build_world_model(p, pose())
    assert wm.sensors_healthy is False and "LIDAR_STALE" in wm.sensors_reason


def test_questions_the_rest_of_the_stack_asks():
    p = PerceptionOutput(objects=[
        obj(5.0, 0.2, ObjectType.PEDESTRIAN, id=1),
        obj(25.0, 0.0, ObjectType.VEHICLE, id=2, speed=6.0, stationary=False),
        obj(9.0, 6.0, ObjectType.OBSTACLE, id=3),          # off to the side
        obj(-4.0, 0.0, ObjectType.OBSTACLE, id=4),         # behind
    ])
    wm = build_world_model(p, pose())
    assert [o.id for o in wm.in_corridor()] == [1, 2], "only the two straight ahead"
    assert wm.nearest_ahead().id == 1
    assert [o.id for o in wm.moving_objects()] == [2]
    assert [o.id for o in wm.parked_objects()] == [1, 3, 4]
    assert [o.id for o in wm.of_kind(ObjectType.VEHICLE)] == [2]
    assert wm.by_id(3).y == 6.0 and wm.by_id(99) is None


def test_the_corridor_is_the_width_of_the_van_s_lane():
    p = PerceptionOutput(objects=[obj(10.0, DEFAULT_PATH_HALF_WIDTH_M - 0.1, id=1),
                                  obj(10.0, DEFAULT_PATH_HALF_WIDTH_M + 0.1, id=2)])
    wm = build_world_model(p, pose())
    assert [o.id for o in wm.in_corridor()] == [1]


def test_age_says_how_stale_the_answer_is():
    now = time.time()
    p = PerceptionOutput(objects=[obj(10.0, 0.0)])
    p.timestamp = now - 0.4
    p.objects[0].timestamp = now - 0.25
    wm = build_world_model(p, pose(), now=now)
    assert wm.objects[0].age_s == pytest.approx(0.25, abs=0.01)
    assert wm.age_s(now) == pytest.approx(0.4, abs=0.01)


def test_a_lost_pose_does_not_invent_map_places():
    p = PerceptionOutput(objects=[obj(10.0, 0.0)])
    wm = build_world_model(p, pose(healthy=False))
    o = wm.objects[0]
    assert (o.x, o.y) == (10.0, 0.0), "what the sensors saw is still true"
    assert math.isnan(o.world_x) and math.isnan(o.world_y), "where that is on the map is not known"
    d = o.as_dict()
    assert d["x"] is None and d["y"] is None, "an unknown place is null, not a number the page cannot read"
    assert d["ego_x"] == 10.0


def test_it_can_be_written_out_for_the_web_page():
    p = PerceptionOutput(objects=[obj(10.0, 1.0, ObjectType.VEHICLE, id=5, speed=3.0, stationary=False)],
                         path_blocked=True, closest_obstacle_distance=10.0,
                         closest_obstacle_type=ObjectType.VEHICLE, closest_obstacle_speed=3.0)
    d = build_world_model(p, pose(x=20.0, y=30.0)).as_dict()
    assert d["counts"] == {"total": 1, "moving": 1, "vehicles": 1, "pedestrians": 0, "cyclists": 0,
                           "static": 0}
    assert d["objects"][0]["motion_class"] == "dynamic", "everything starts dynamic"
    assert d["objects"][0]["stationary"] is False and d["objects"][0]["type"] == "vehicle"
    assert d["path"]["blocked"] is True and d["van"]["x"] == 20.0
    import json
    json.dumps(d)          # it must survive the trip to the browser


def test_the_junction_question_gives_the_same_answer_as_the_old_loop():
    """The behaviour layer used to walk perception's object list itself. The world
    model answers the same question; both must agree, object for object."""
    from warp_av.behavior.behavior import BehaviorSystem

    cases = [
        obj(20.0, 8.0, ObjectType.VEHICLE, id=1, speed=6.0, stationary=False),   # crossing
        obj(10.0, 0.5, ObjectType.VEHICLE, id=2, speed=6.0, stationary=False),   # our lane
        obj(15.0, 7.0, ObjectType.VEHICLE, id=3, speed=0.2),                     # parked
        obj(-8.0, 5.0, ObjectType.VEHICLE, id=4, speed=6.0, stationary=False),   # behind
        obj(60.0, 9.0, ObjectType.VEHICLE, id=5, speed=6.0, stationary=False),   # far away
        obj(12.0, 6.0, ObjectType.PEDESTRIAN, id=6, speed=1.4, stationary=False),  # not a vehicle
        obj(14.0, 6.0, ObjectType.VEHICLE, id=7, speed=6.0, stationary=False),   # the nearest crossing one
    ]
    p = PerceptionOutput(objects=cases)
    bp = BehaviorSystem()
    old = bp._junction_conflict(p)
    new = bp._junction_conflict(p, build_world_model(p, pose()))
    assert old == new == pytest.approx(cases[-1].distance), "both should pick the vehicle at 14 m"

    empty = PerceptionOutput()
    assert bp._junction_conflict(empty) is None
    assert bp._junction_conflict(empty, build_world_model(empty, pose())) is None


def test_the_web_payload_matches_the_old_hand_written_maths():
    """The state endpoint used to work out map coordinates inline. Whatever it
    publishes now must be the same numbers."""
    p_ = pose(x=10.0, y=-5.0, yaw=0.7)
    objects = [obj(12.0, 1.0, ObjectType.VEHICLE, id=4, speed=5.5, stationary=False,
                   length_m=4.3, width_m=1.8, height_m=1.5, yaw_deg=3.0),
               obj(6.0, -2.0, ObjectType.OBSTACLE, id=9)]
    wm = build_world_model(PerceptionOutput(objects=objects), p_)
    for o, w in zip(objects, wm.objects):
        d = w.as_dict()
        assert d["x"] == round(p_.x + math.cos(p_.yaw) * o.x - math.sin(p_.yaw) * o.y, 2)
        assert d["y"] == round(p_.y + math.sin(p_.yaw) * o.x + math.cos(p_.yaw) * o.y, 2)
        assert d["distance"] == round(o.distance, 1)
        assert (d["id"], d["type"]) == (o.id, o.object_type.value)
        assert d["speed"] == round(o.speed, 2) and d["stationary"] == o.stationary
        assert d["length_m"] == round(o.length_m, 2) and d["height_m"] == round(o.height_m, 2)
