"""
Perception V2 day 7: someone on a bicycle.

A live run showed a cyclist reported as a pedestrian 0.65 m long, against a real
1.66 m body. The camera model has no cyclist class: it reports a person and a
bicycle in the same place. So the van now looks for that pair, and treats the
result as what it is, a person who needs a vehicle's room and a person's right
of way.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.camera_model import box_overlap  # noqa: E402
from warp_av.perception.perception import ObjectType, PerceptionOutput, VULNERABLE_TYPES  # noqa: E402
from warp_av.perception.tracking import CLASS_SIZE_LIMITS, clearance_radius_m, plausible_size  # noqa: E402


def test_box_overlap_measures_the_share_of_the_first_box():
    person = (100, 100, 40, 120)          # left, top, width, height
    bike = (95, 180, 90, 50)              # under them, overlapping the lower part
    share = box_overlap(person, bike)
    assert 0.0 < share < 1.0
    assert box_overlap(person, person) == pytest.approx(1.0)
    assert box_overlap(person, (500, 500, 40, 40)) == 0.0
    inside = (110, 110, 10, 10)
    assert box_overlap(inside, person) == pytest.approx(1.0), "a box wholly inside another is 1"


def test_a_person_beside_a_bike_is_not_riding_it():
    person = (100, 100, 40, 120)
    parked_bike = (200, 180, 90, 50)      # a few metres away in the picture
    assert box_overlap(person, parked_bike) < 0.35


def test_a_person_over_a_bike_is_riding_it():
    person = (100, 60, 40, 160)           # standing tall over the bike
    bike = (90, 140, 70, 80)
    assert box_overlap(person, bike) >= 0.35


def test_a_cyclist_gets_room_for_the_bicycle_and_a_person_s_minimum():
    small = clearance_radius_m("cyclist", 0.0, 0.0)
    assert small >= 0.8, "even a badly measured cyclist gets more room than a pedestrian"
    assert small > clearance_radius_m("pedestrian", 0.0, 0.0)
    measured = clearance_radius_m("cyclist", 1.8, 0.6)
    assert measured > small, "a well measured bicycle asks for its own length"


def test_a_cyclist_may_be_bicycle_sized_but_not_wall_sized():
    assert plausible_size("cyclist", 1.8, 0.6, 1.7) is True
    assert plausible_size("cyclist", 2.6, 0.7, 1.8) is True, "a motorbike is longer"
    assert plausible_size("cyclist", 8.0, 0.4, 4.0) is False
    assert CLASS_SIZE_LIMITS["cyclist"][0] > CLASS_SIZE_LIMITS["pedestrian"][0]


def test_the_van_gives_way_to_a_cyclist_as_to_a_person():
    from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
    from warp_av.localization.localization import Pose
    assert ObjectType.CYCLIST in VULNERABLE_TYPES
    for kind in (ObjectType.PEDESTRIAN, ObjectType.CYCLIST):
        p = PerceptionOutput(path_blocked=True, closest_obstacle_distance=6.0,
                             closest_obstacle_type=kind)
        bs = BehaviorSystem()
        bs.set_mission()
        pose = Pose(x=0.0, y=0.0, yaw=0.0, speed=2.0)
        out = bs.update(perception=p, pose=pose, destination_distance=50.0, safety_ok=True)
        assert out.behavior == DrivingBehavior.STOPPED_PEDESTRIAN, f"{kind} should stop the van"
        assert kind.value.upper() in out.reason
        assert out.desired_speed_mps == 0.0 and out.should_stop


def test_a_cyclist_never_makes_the_road_count_as_blocked():
    """A cyclist moves on. Declaring the road blocked would send the van round them."""
    from warp_av.behavior.behavior import BehaviorSystem, DrivingBehavior
    from warp_av.localization.localization import Pose
    bs = BehaviorSystem()
    bs.set_mission()
    pose = Pose(x=0.0, y=0.0, yaw=0.0, speed=0.0)
    p = PerceptionOutput(path_blocked=True, closest_obstacle_distance=6.0,
                         closest_obstacle_type=ObjectType.CYCLIST)
    for _ in range(3):
        out = bs.update(perception=p, pose=pose, destination_distance=50.0, safety_ok=True)
    assert bs._blocked_since is None, "a cyclist must not start the blocked-road clock"
    assert out.behavior == DrivingBehavior.STOPPED_PEDESTRIAN
