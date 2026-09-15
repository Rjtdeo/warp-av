"""
Planning V2 task 2: one honest record of what is in the way.

Before it, three writers scribbled on the same PerceptionOutput fields (path_blocked,
closest_obstacle_*): perception itself, then the route corridor wrote its verdict over that,
then the ground map's second opinion wrote over that again. Afterwards nothing could say what
perception had actually seen, planner.blocked and perception.path_blocked could disagree,
and "slow" or "not sure" had no word of their own.

Now perception says what it saw and is never written. The planner RETURNS the path record
(PlannerDecision: level, the nearest thing, the reason, what the ground said), the ground
map writes that record, and the behaviour, the world model, the log and the API read it.
"""
import math
from types import SimpleNamespace

import pytest

from warp_av.planning.planner import RoutePlanner, Route, Waypoint
from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.instrumentation import (PlannerDecision, PATH_CLEAR, PATH_SLOW, PATH_UNSURE,
                                              PATH_BLOCKED, GROUND_RELEASED, CLEAR, NO_ROUTE,
                                              BLOCKED_TRACKED_OBJECT, BLOCKED_SWEPT_PATH,
                                              BLOCKED_OCCUPANCY, UNKNOWN_SPACE)
from warp_av.perception.perception import PerceptionOutput, ObjectType, DetectedObject
from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
from warp_av.localization.localization import Pose
from warp_av.world_model import build_world_model

FOOT = VehicleFootprint(half_length=2.96, half_width=0.99, safety_margin=0.30)


def planner():
    return RoutePlanner.__new__(RoutePlanner)


def road():
    return Route(waypoints=[Waypoint(x=-20.0 + i * 2.0, y=0.0) for i in range(61)])


def obj(x, y=0.0, kind=ObjectType.VEHICLE, speed=0.0, ident=1):
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), speed=speed, id=ident)


def snapshot(p):
    return (p.path_blocked, p.closest_obstacle_distance, p.closest_obstacle_type,
            p.closest_obstacle_speed, p.closest_obstacle_lateral_m)


# ---- the planner returns the record and leaves perception alone -------------------------

@pytest.mark.parametrize("things", [
    [obj(5.0, 0.2)],                                  # a car dead ahead: blocked
    [obj(15.0, 1.0, kind=ObjectType.OBSTACLE)],       # a thing in the corridor, further: slow
    [obj(10.0, 6.0)],                                 # a car well off the line: clear
    [obj(5.0, 0.3, kind=ObjectType.PEDESTRIAN)],      # a person: blocked, its own reason
])
def test_perception_is_never_written(things):
    per = PerceptionOutput(objects=things, path_blocked=True, closest_obstacle_distance=1.0,
                           closest_obstacle_type=ObjectType.CYCLIST, closest_obstacle_speed=2.0)
    before = snapshot(per)
    out = planner().filter_to_route_corridor(per, road(), 0.0, 0.0, 0.0, footprint=FOOT)
    assert isinstance(out, PlannerDecision)
    assert snapshot(per) == before, "the corridor check wrote into perception"


def test_the_record_is_the_decision_and_carries_the_numbers():
    p = planner()
    out = p.filter_to_route_corridor(PerceptionOutput(objects=[obj(5.0, 0.2, ident=17)]),
                                     road(), 0.0, 0.0, 0.0, footprint=FOOT)
    assert out is p.last_decision
    assert out.level == PATH_BLOCKED and out.blocked is True
    assert out.closest_distance_m == pytest.approx(5.0)
    assert out.closest_kind == ObjectType.VEHICLE
    assert out.blocker_id == 17 and out.reason in (BLOCKED_TRACKED_OBJECT, BLOCKED_SWEPT_PATH)
    # the names the verdict had while it lived in perception still read, and agree
    assert out.path_blocked is out.blocked
    assert out.closest_obstacle_distance == out.closest_distance_m
    assert out.closest_obstacle_type == out.closest_kind


def test_levels_clear_slow_blocked():
    run = lambda things: planner().filter_to_route_corridor(
        PerceptionOutput(objects=things), road(), 0.0, 0.0, 0.0, footprint=FOOT)
    assert run([obj(10.0, 6.0)]).level == PATH_CLEAR
    slow = run([obj(15.0, 1.0, kind=ObjectType.OBSTACLE)])
    assert slow.level == PATH_SLOW and slow.blocked is False
    assert slow.closest_distance_m == pytest.approx(15.0), "seen, so the slow zone applies"
    assert run([obj(5.0, 0.2)]).level == PATH_BLOCKED
    nothing = run([obj(10.0, 6.0)])
    assert nothing.closest_distance_m == 999.0 and nothing.closest_kind is None


def test_no_route_means_perceptions_own_verdict_is_the_record():
    per = PerceptionOutput(objects=[obj(5.0)], path_blocked=True, closest_obstacle_distance=5.0,
                           closest_obstacle_type=ObjectType.VEHICLE, closest_obstacle_speed=0.0)
    out = planner().filter_to_route_corridor(per, Route(waypoints=[]), 0.0, 0.0, 0.0)
    assert out.reason == NO_ROUTE and out.level == PATH_BLOCKED
    assert out.closest_distance_m == 5.0 and out.closest_kind == ObjectType.VEHICLE
    assert PlannerDecision.from_perception(PerceptionOutput()).level == PATH_CLEAR
    slow = PlannerDecision.from_perception(PerceptionOutput(closest_obstacle_distance=15.0))
    assert slow.level == PATH_SLOW and slow.blocked is False
    assert PlannerDecision.from_perception(PerceptionOutput(closest_obstacle_distance=0.0,
                                                            path_blocked=True)).closest_distance_m == 0.0


def test_blocked_follows_the_level_not_the_reason():
    # the old pair could disagree: planner.blocked from the reason, path_blocked from the release
    d = PlannerDecision(reason=BLOCKED_SWEPT_PATH)
    assert d.level == PATH_BLOCKED and d.blocked is True          # worked out from the reason
    d.level = PATH_SLOW                                            # ...until the ground releases it
    assert d.blocked is False and d.reason == BLOCKED_SWEPT_PATH
    assert d.as_dict()["blocked"] is False and d.as_dict()["level"] == PATH_SLOW
    assert PlannerDecision(reason=CLEAR).blocked is False
    assert PlannerDecision(reason=NO_ROUTE).blocked is False


# ---- the ground map's second opinion writes the record ----------------------------------

class Grid:
    def __init__(self, counts, nearest=None):
        self.counts, self.nearest, self.updated = counts, nearest, True

    def strip_ahead(self, a, b, half):
        return self.counts

    def nearest_block_ahead(self, a, b, half):
        return self.nearest


def system(grid):
    # main.py needs the stack's web modules; where they are not installed (the Mac) these
    # four tests skip and run on the CARLA laptop, which has them
    pytest.importorskip("flask_socketio")
    from warp_av.main import WarpAV
    s = SimpleNamespace(perception=SimpleNamespace(grid=grid),
                        behavior=SimpleNamespace(front_offset_m=2.95),
                        footprint_blocking=SimpleNamespace(footprint=FOOT),
                        _overtake_point=None, _ground_block_at=0.0, _ground_seen_free_at=0.0,
                        _road_edge_ahead=lambda pose: False,
                        _note_move=lambda *a, **k: None,
                        logger=SimpleNamespace(log_event=lambda *a, **k: None))
    s.second_opinion = lambda path, pose: WarpAV._second_opinion_on_the_ground(s, path, pose)
    s._drive_on_if_the_ground_is_seen_free = (
        lambda path, pose, g, f: WarpAV._drive_on_if_the_ground_is_seen_free(s, path, pose, g, f))
    return s


def test_solid_ground_blocks_on_the_record_and_never_in_perception():
    per = PerceptionOutput(objects=[])
    path = PlannerDecision(reason=CLEAR)
    s = system(Grid(counts=(300, 6, 0), nearest=4.2))
    s.second_opinion(path, Pose(healthy=True))
    assert path.level == PATH_BLOCKED and path.blocked is True
    assert path.reason == BLOCKED_OCCUPANCY and path.second_opinion == BLOCKED_OCCUPANCY
    assert path.closest_distance_m == 4.2 and path.closest_kind == ObjectType.UNKNOWN
    assert per.path_blocked is False and per.closest_obstacle_distance == 999.0


def test_ground_seen_empty_releases_the_record_to_slow_and_keeps_the_reason():
    path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=6.0,
                           closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0)
    s = system(Grid(counts=(400, 0, 0)))                   # every square seen empty
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is False and path.level == PATH_SLOW
    assert path.reason == BLOCKED_SWEPT_PATH, "the planner's reason stays beside what the ground said"
    assert path.second_opinion == GROUND_RELEASED
    assert path.closest_distance_m == 6.0, "still the nearest thing ahead: the slow zone applies"


def test_a_person_is_never_released_by_the_ground():
    path = PlannerDecision(reason=BLOCKED_TRACKED_OBJECT, closest_distance_m=6.0,
                           closest_kind=ObjectType.PEDESTRIAN)
    s = system(Grid(counts=(400, 0, 0)))
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is True and path.second_opinion is None


def test_unseen_ground_is_said_on_the_record_as_unsure():
    path = PlannerDecision(reason=CLEAR)
    s = system(Grid(counts=(100, 0, 300)))                 # mostly never seen
    s.second_opinion(path, Pose(healthy=True))
    assert path.level == PATH_UNSURE and path.blocked is False
    assert path.second_opinion == UNKNOWN_SPACE


# ---- the behaviour and the world model read the record ----------------------------------

def test_the_behaviour_obeys_the_record_over_perceptions_own_fields():
    b = BehaviorSystem(); b.set_mission()
    pose = Pose(healthy=True)
    saw_clear = PerceptionOutput()                          # perception: nothing in its strip
    record = PlannerDecision(reason=BLOCKED_TRACKED_OBJECT, closest_distance_m=5.0,
                             closest_kind=ObjectType.OBSTACLE)
    out = b.update(saw_clear, pose, 500, True, path=record)
    assert out.behavior == DrivingBehavior.STOPPED_OBSTACLE and out.should_stop
    assert "5.0m" in out.reason

    saw_blocked = PerceptionOutput(path_blocked=True, closest_obstacle_distance=3.0,
                                   closest_obstacle_type=ObjectType.VEHICLE)
    released = PlannerDecision(reason=CLEAR, closest_distance_m=999.0)
    out = b.update(saw_blocked, pose, 500, True, path=released)
    assert not out.should_stop, "perception's own strip verdict must not stop the van once the record says clear"


def test_without_a_record_the_behaviour_uses_perceptions_own_verdict_as_before():
    b = BehaviorSystem(); b.set_mission()
    out = b.update(PerceptionOutput(path_blocked=True, closest_obstacle_distance=4.0,
                                    closest_obstacle_type=ObjectType.VEHICLE), Pose(healthy=True), 500, True)
    assert out.behavior == DrivingBehavior.STOPPED_VEHICLE


def test_the_world_model_path_comes_from_the_record():
    per = PerceptionOutput(path_blocked=True, closest_obstacle_distance=3.0,
                           closest_obstacle_type=ObjectType.VEHICLE)
    record = PlannerDecision(reason=CLEAR, closest_distance_m=12.0, closest_kind=ObjectType.OBSTACLE)
    wm = build_world_model(per, SimpleNamespace(x=0.0, y=0.0, yaw=0.0, speed=0.0, healthy=True),
                           path=record)
    assert wm.path.blocked is False and wm.path.closest_distance_m == 12.0
    assert wm.path.closest_kind == "obstacle"
    assert build_world_model(per, SimpleNamespace(x=0.0, y=0.0, yaw=0.0)).path.blocked is True


def test_the_record_says_it_all_in_one_dict():
    d = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=6.25,
                        closest_kind=ObjectType.VEHICLE, closest_lateral_m=0.816,
                        blocker_id=3, blocker_kind="vehicle", blocker_distance_m=6.25)
    out = d.as_dict()
    assert out["level"] == PATH_BLOCKED and out["blocked"] is True
    assert out["closest_distance_m"] == 6.2 and out["closest_kind"] == "vehicle"
    assert out["closest_lateral_m"] == 0.82 and out["second_opinion"] is None
    assert d.one_line().startswith("blocked reason=blocked_swept_path blocker=3")


# ---- the ground can only vouch for ground it looked at (E3, 2026-09-15) ------------------

def test_a_blocker_nearer_than_the_nose_line_is_never_released_by_the_ground():
    """The strip starts at the van's nose, 2.95 m out, and runs forward. A barrel 1.9 m ahead
    is nearer than that, so the laser never looked at where it is -- and said the ground was
    empty, which it was. Live in E3 the swept check blocked on exactly that barrel while the
    van turned into a bay, this released the block, and the van drove into it at 1.8 m/s."""
    path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=1.9,
                           closest_kind=ObjectType.OBSTACLE, closest_speed_mps=0.0)
    path.level = PATH_BLOCKED
    s = system(Grid(counts=(400, 0, 0)))          # every square the strip saw was empty
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is True, "the block stands: that ground was never looked at"
    assert path.second_opinion != GROUND_RELEASED


def test_a_blocker_the_strip_does_cover_is_still_released():
    """No change to the case this rule was written for."""
    path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=6.0,
                           closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0)
    s = system(Grid(counts=(400, 0, 0)))
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is False and path.second_opinion == GROUND_RELEASED


def test_the_near_boundary_is_the_nose_line_itself():
    for at_m, released in ((2.90, False), (3.10, True)):
        path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=at_m,
                               closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0)
        path.level = PATH_BLOCKED
        s = system(Grid(counts=(400, 0, 0)))
        s.second_opinion(path, Pose(healthy=True))
        got = path.second_opinion == GROUND_RELEASED
        assert got is released, f"{at_m} m: released={got}, expected {released}"


def test_ground_that_stops_short_of_the_blocker_cannot_vouch_for_it_either():
    """The strip reaches 7 m past the nose at most. A thing 40 m off is beyond it."""
    path = PlannerDecision(reason=BLOCKED_SWEPT_PATH, closest_distance_m=40.0,
                           closest_kind=ObjectType.VEHICLE, closest_speed_mps=0.0)
    path.level = PATH_BLOCKED
    s = system(Grid(counts=(400, 0, 0)))
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is True and path.second_opinion != GROUND_RELEASED


def test_a_person_is_still_never_released_however_far_off():
    path = PlannerDecision(reason=BLOCKED_TRACKED_OBJECT, closest_distance_m=6.0,
                           closest_kind=ObjectType.PEDESTRIAN)
    s = system(Grid(counts=(400, 0, 0)))
    s.second_opinion(path, Pose(healthy=True))
    assert path.blocked is True and path.second_opinion is None
