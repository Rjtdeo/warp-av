"""
One thing seen as two. (Perception V2, day 10 follow-up.)

Live, a car eleven metres ahead was reported twice, at 8.3 m and 8.6 m, both
correctly sized. The grid that groups laser points is a fixed 0.8 m, and the
laser's points are not fixed: they spread apart with distance and they land on
whichever face of a thing is turned towards the van. A car close by comes back
as its back and its side, with a gap between them wider than the grid.

Two blobs are put back together when the space between them is small for their
range, they stand at the same height, and what they make together is still a
plausible size. All three guards matter: the whole of day 4 was about stopping
a barrel from merging with the planter beside it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.tracking import (MERGE_MAX_LENGTH_M, merge_split_clusters)  # noqa: E402


def blob(x, y, n=20, extent=0.8, height=1.5):
    import math
    return {"x": x, "y": y, "n": n, "extent": extent, "height": height,
            "distance": math.hypot(x, y), "length_m": extent * 2, "width_m": extent,
            "yaw_deg": 0.0, "axis_deg": 0.0, "weak": False}


def test_the_back_and_the_side_of_one_car_become_one_thing():
    got = merge_split_clusters([blob(8.3, 0.2, extent=0.7), blob(9.5, 0.8, extent=0.7)])
    assert len(got) == 1
    assert got[0]["n"] == 40
    assert 8.0 < got[0]["distance"] < 10.0
    assert got[0]["length_m"] > 1.4, "the joined thing is longer than either half"


def test_two_things_standing_apart_stay_two():
    got = merge_split_clusters([blob(8.0, 0.0), blob(14.0, 0.0)])
    assert len(got) == 2


def test_things_of_different_heights_are_never_joined():
    """The day-4 case: a barrel 0.8 m tall beside a planter 0.17 m tall."""
    barrel = blob(9.0, 0.0, extent=0.3, height=0.8)
    planter = blob(9.6, 0.4, extent=0.3, height=0.17)
    assert len(merge_split_clusters([barrel, planter])) == 2


def test_two_blobs_that_would_make_something_impossible_stay_apart():
    a = blob(10.0, -2.0, extent=1.5)
    b = blob(10.0, 2.0, extent=1.5)
    span = 4.0 + 1.5 + 1.5
    assert span > MERGE_MAX_LENGTH_M
    assert len(merge_split_clusters([a, b])) == 2, "that would be a nine-metre car"


def test_the_allowed_gap_grows_with_distance():
    """Close up the laser's points are dense, so a gap means a real gap. Far off the same
    gap is just the beams spreading out."""
    near = [blob(6.0, 0.0, extent=0.4), blob(7.6, 0.0, extent=0.4)]
    far = [blob(28.0, 0.0, extent=0.4), blob(29.6, 0.0, extent=0.4)]
    assert len(merge_split_clusters(near)) == 2, "0.8 m of clear air at 6 m is two things"
    assert len(merge_split_clusters(far)) == 1, "the same gap at 28 m is one thing"


def test_three_pieces_of_one_thing_all_come_together():
    got = merge_split_clusters([blob(9.0, 0.0, extent=0.5), blob(9.9, 0.2, extent=0.5),
                                blob(10.8, 0.4, extent=0.5)])
    assert len(got) == 1 and got[0]["n"] == 60


def test_nothing_is_lost_when_there_is_nothing_to_join():
    one = [blob(5.0, 0.0)]
    assert merge_split_clusters(one)[0]["x"] == 5.0
    assert merge_split_clusters([]) == []


def test_the_joined_thing_keeps_which_points_it_came_from():
    a = dict(blob(9.0, 0.0, extent=0.5), members=[1, 2, 3])
    b = dict(blob(9.8, 0.1, extent=0.5), members=[4, 5])
    got = merge_split_clusters([a, b])
    assert sorted(got[0]["members"]) == [1, 2, 3, 4, 5]
