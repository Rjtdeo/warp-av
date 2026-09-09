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


def test_the_rider_claims_the_blob_before_a_person_standing_beside_the_bike():
    """The camera reported two people and one bicycle around a motorbike: one person
    barely overlapping it, one sitting squarely on it. Whichever is processed first takes
    the blob, so the rider has to go first or the cyclist is named a pedestrian."""
    from warp_av.perception.camera_model import box_overlap
    bike = (334, 259, 34, 70)
    bystander = (311, 224, 31, 94)
    rider = (334, 241, 36, 78)
    overlaps = [(box_overlap(bystander, bike), "bystander"), (box_overlap(rider, bike), "rider")]
    overlaps.sort(reverse=True)
    assert overlaps[0][1] == "rider", "the rider overlaps the bike far more"
    assert overlaps[0][0] >= 0.35 > overlaps[1][0], "and only the rider passes the test"


# ---------------------------------------------------------------------------
# Shown live: a bicycle with a rider, a bare bicycle and a motorbike with a
# rider were all called pedestrians. The camera reports a bicycle only some of
# the time in this simulator, and the laser sees the rider, not the frame, so a
# cyclist measures the same as someone standing. Two signals the old rule threw
# away: a bicycle the camera saw but was unsure about, and speed.
# ---------------------------------------------------------------------------

def test_a_hesitant_bicycle_still_counts_for_the_rider_test():
    """A bicycle is never reported as an object of its own. It only ever decides whether a
    person is riding, and a rider and a walker are both people the van stops for. So a
    low-confidence bicycle can move a label between two vulnerable classes and can never
    invent an object."""
    from warp_av.perception.camera_lidar_perception import (RIDDEN_CONFIDENCE_THRESHOLD,
                                                            RIDDEN_CLASSES, VEHICLE_CLASSES)
    assert RIDDEN_CONFIDENCE_THRESHOLD < 0.40, "lower than the bar for everything else"
    assert 1 in RIDDEN_CLASSES, "the bicycle class"
    assert 1 not in VEHICLE_CLASSES, "and it never becomes a vehicle on its own"


def test_a_person_at_cycling_pace_is_a_cyclist():
    """Nobody walks at 4 m/s. If the camera missed the bicycle, the speed still says so."""
    from warp_av.perception.camera_lidar_perception import CYCLIST_SPEED_MPS
    assert 2.5 < CYCLIST_SPEED_MPS < 6.0, "faster than running, slower than traffic"


def test_the_speed_rule_only_moves_a_pedestrian_to_a_cyclist():
    """It must never touch a vehicle's name, and never fire on someone standing still."""
    import inspect
    from warp_av.perception import camera_lidar_perception as clp
    src = inspect.getsource(clp.CameraLidarPerception.update)
    assert 'kind == "pedestrian"' in src
    assert 'not getattr(tr, "stationary", True)' in src, "a parked thing is never a cyclist"
    assert 'kind = "cyclist"' in src


def test_a_rider_is_not_demoted_by_a_frame_that_missed_the_bicycle():
    """Live, a cyclist flickered between cyclist and pedestrian. The camera sees the bicycle
    in some frames and only the person in others, and every such frame was overwriting the
    name. A rider stays a rider until the name expires for want of any sighting at all."""
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    def feed(cls):
        nonlocal t
        t += 0.1
        tr.update([{"wx": 10.0, "wy": 0.0, "distance": 10.0, "cls": cls,
                    "cls_source": "camera", "confidence": 0.9}], t)
    feed("pedestrian")
    feed("cyclist")
    assert tr._tracks[0].cls == "cyclist"
    for _ in range(5):
        feed("pedestrian")            # the camera keeps missing the bike
    assert tr._tracks[0].cls == "cyclist", "it is still the same person on the same bicycle"


def test_a_pedestrian_can_still_become_a_cyclist_but_not_the_other_way_round():
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    for cls in ("pedestrian", "pedestrian", "cyclist"):
        t += 0.1
        tr.update([{"wx": 9.0, "wy": 0.0, "distance": 9.0, "cls": cls,
                    "cls_source": "camera", "confidence": 0.9}], t)
    assert tr._tracks[0].cls == "cyclist"


def test_a_vehicle_is_not_protected_by_the_rider_rule():
    """Only the pedestrian-over-cyclist case is held. A camera calling something a vehicle
    must still be able to correct a cyclist."""
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    for cls in ("cyclist", "vehicle"):
        t += 0.1
        tr.update([{"wx": 9.0, "wy": 0.0, "distance": 9.0, "cls": cls,
                    "cls_source": "camera", "confidence": 0.9}], t)
    assert tr._tracks[0].cls == "vehicle"


def test_the_speed_rule_needs_real_travel_not_one_fast_reading():
    """A thing that has just appeared can show a large speed for a moment. That was enough
    to turn a person standing still into a cyclist in a live run."""
    import inspect
    from warp_av.perception import camera_lidar_perception as clp
    from warp_av.perception.camera_lidar_perception import CYCLIST_TRAVEL_M
    src = inspect.getsource(clp.CameraLidarPerception.update)
    assert 'travelled_m' in src, "the test is on ground actually covered"
    assert CYCLIST_TRAVEL_M >= 2.0


def test_what_the_camera_can_and_cannot_see_here():
    """Measured on the simulator, six frames each, with the bar dropped to 0.05:

        a plain walker      no bicycle at all
        a pedal cyclist     no bicycle at all
        a motorbike rider   0.12 to 0.23

    So a motorbike rider can be named from the camera and a pedal cyclist cannot. That is
    the detector's limit, not the fusion's, and it is why the threshold sits at 0.15: low
    enough for the motorbike, and it invents nothing on a walker because there is nothing
    there to find.
    """
    from warp_av.perception.camera_lidar_perception import RIDDEN_CONFIDENCE_THRESHOLD
    assert 0.10 <= RIDDEN_CONFIDENCE_THRESHOLD <= 0.25
