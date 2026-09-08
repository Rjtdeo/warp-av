"""
Perception V2 day 7: how big is it, when things really do come in different sizes?

The van used to keep the largest view it had ever had of a thing. One frame that
glued a person to a wall then set that person's size for the life of the track,
and a live run reported a person 8.6 m long and 4 m tall.

People genuinely vary, though, from a child to an adult with a bag and a bike, so
the answer cannot be a fixed size. These tests pin the rule that replaced it: keep
the middle of the recent sightings, let a size that several frames agree on come
through, and use limits only to throw away the impossible.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.tracking import (CLASS_SIZE_LIMITS, ObjectTracker,  # noqa: E402
                                         clearance_radius_m, plausible_size)

DT = 0.1


def run(sizes, cls=None, x=10.0):
    """Feed a track one sighting per step; sizes is a list of (length, width, height)."""
    tr = ObjectTracker()
    t = 0.0
    track = None
    for (l, w, h) in sizes:
        t += DT
        o = {"wx": x, "wy": 0.0, "distance": x, "length_m": l, "width_m": w, "height_m": h,
             "yaw_deg": 0.0}
        if cls:
            o["cls"] = cls
            o["confidence"] = 0.9
        got = tr.update([o], t)
        if got:
            track = got[0]
    return track


def test_one_merged_frame_does_not_set_the_size_for_ever():
    """Nine honest sightings of a person and one frame merged with a wall."""
    person = [(0.5, 0.3, 1.8)] * 9
    merged = [(8.6, 0.2, 4.0)]
    t = run(person[:4] + merged + person[4:], cls="pedestrian")
    assert t.length_m == pytest.approx(0.5, abs=0.1), f"one bad frame took over: {t.length_m:.2f} m"
    assert t.height_m == pytest.approx(1.8, abs=0.2)


def test_people_of_different_sizes_are_reported_as_they_are():
    child = run([(0.35, 0.3, 1.15)] * 8, cls="pedestrian")
    adult = run([(0.55, 0.4, 1.85)] * 8, cls="pedestrian")
    with_a_bike = run([(1.7, 0.6, 1.9)] * 8, cls="pedestrian")
    assert child.height_m == pytest.approx(1.15, abs=0.1)
    assert adult.height_m == pytest.approx(1.85, abs=0.1)
    assert with_a_bike.length_m == pytest.approx(1.7, abs=0.2), "a person with a bike needs their room"
    assert child.length_m < adult.length_m < with_a_bike.length_m


def test_a_size_several_frames_agree_on_is_adopted():
    """A car reveals more of itself as the van approaches; that is not a merge."""
    t = run([(1.8, 1.0, 1.5)] * 4 + [(4.4, 1.9, 1.5)] * 8, cls="vehicle")
    assert t.length_m == pytest.approx(4.4, abs=0.3), "sustained agreement should win"


def test_an_impossible_person_is_thrown_away_not_believed():
    assert plausible_size("pedestrian", 0.5, 0.4, 1.9) is True
    assert plausible_size("pedestrian", 1.5, 0.7, 2.3) is True, "a person with a bag and a bike"
    assert plausible_size("pedestrian", 8.6, 0.2, 4.0) is False
    assert plausible_size("vehicle", 12.0, 2.6, 3.4) is True, "a bus is still a vehicle"
    assert plausible_size("vehicle", 25.0, 8.0, 4.0) is False
    assert plausible_size(None, 25.0, 8.0, 4.0) is True, "with no name, nothing is impossible"


def test_the_limits_never_rewrite_a_real_measurement():
    t = run([(1.4, 0.6, 2.0)] * 8, cls="pedestrian")
    assert t.length_m == pytest.approx(1.4, abs=0.15), "a large person keeps their measured size"


def test_it_says_when_it_does_not_know():
    steady = run([(0.5, 0.3, 1.8)] * 8, cls="pedestrian")
    jumpy = run([(0.4, 0.3, 1.8), (1.5, 0.5, 2.0), (0.5, 0.3, 1.7),
                 (1.4, 0.6, 2.1), (0.45, 0.3, 1.8), (1.5, 0.5, 2.2)], cls="pedestrian")
    assert steady.size_uncertain is False
    assert jumpy.size_uncertain is True, "wildly different sightings should be admitted as unsure"


def test_the_normal_wobble_of_a_person_is_not_called_unsure():
    """A ratio test called every pedestrian unsure, because any natural swing is a big share
    of something 0.4 m across. The test is an absolute swing instead, and it grows with
    range, because the LiDAR's points spread out."""
    from warp_av.perception.tracking import expected_size_spread_m
    assert expected_size_spread_m(6.0) < expected_size_spread_m(22.0)
    # measured on the recordings: a walker at 6 m swings 0.17 m, a bin at 12 m 0.27 m,
    # a car at 22 m 0.32 m; a blob merged with its neighbour swings over a metre
    assert expected_size_spread_m(6.0) > 0.17
    assert expected_size_spread_m(12.0) > 0.27
    assert expected_size_spread_m(22.0) > 0.32
    assert expected_size_spread_m(15.0) < 1.0
    person = run([(0.42, 0.3, 1.8), (0.5, 0.28, 1.75), (0.38, 0.31, 1.82),
                  (0.47, 0.29, 1.78), (0.44, 0.3, 1.8), (0.4, 0.3, 1.77)], cls="pedestrian", x=10.0)
    assert person.size_uncertain is False, "a person's ordinary wobble is not a warning"


def test_a_merged_frame_is_admitted_as_unsure_at_any_range():
    near = run([(0.5, 0.3, 1.8)] * 3 + [(2.4, 0.4, 1.9)] + [(0.5, 0.3, 1.8)] * 3,
               cls="pedestrian", x=8.0)
    assert near.size_uncertain is True
    assert near.length_m == pytest.approx(0.5, abs=0.15), "and the size itself is unharmed"


def test_room_to_leave_is_never_less_than_the_kind_deserves():
    small_person = clearance_radius_m("pedestrian", 0.35, 0.3)
    assert small_person >= 0.6, "a small person can still step sideways without warning"
    big_person = clearance_radius_m("pedestrian", 1.7, 0.6)
    assert big_person > small_person, "a bigger measurement asks for more room"
    car = clearance_radius_m("vehicle", 4.4, 1.9)
    assert car == pytest.approx(0.5 * math.hypot(4.4, 1.9), abs=0.01)
    assert clearance_radius_m(None, 0.0, 0.0) > 0.0, "an unmeasured thing still gets room"
