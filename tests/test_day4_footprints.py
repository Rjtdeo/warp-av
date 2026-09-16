"""
Perception V2 day 4: the van measures how big a thing is, not just where it is.

A cluster now reports the footprint it has actually seen (long side, short side,
heading), the track keeps the largest view of each side, and DetectedObject
carries length, width, height and heading through to whoever plans the drive.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.perception import DetectedObject, ObjectType  # noqa: E402
from warp_av.perception.tracking import ObjectTracker, cluster_points  # noqa: E402


def rect(cx, cy, length, width, yaw_deg=0.0, step=0.1):
    """Points on the outline of a rectangle, in the sensor frame."""
    a = math.radians(yaw_deg)
    pts = []
    n_l, n_w = int(length / step) + 1, int(width / step) + 1
    for i in range(n_l):
        u = -length / 2 + i * step
        for v in (-width / 2, width / 2):
            pts.append((cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)))
    for j in range(n_w):
        v = -width / 2 + j * step
        for u in (-length / 2, length / 2):
            pts.append((cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)))
    return pts


@pytest.mark.parametrize("yaw", [0.0, 30.0, -45.0, 80.0])
def test_footprint_of_a_known_rectangle(yaw):
    pts = rect(12.0, 1.0, 4.0, 1.8, yaw)
    c = cluster_points(pts, cell=1.0)[0]
    assert c["length_m"] == pytest.approx(4.0, abs=0.15)
    assert c["width_m"] == pytest.approx(1.8, abs=0.15)
    diff = (c["yaw_deg"] - yaw + 90.0) % 180.0 - 90.0
    assert abs(diff) < 6.0, f"heading {c['yaw_deg']:.1f} for a box at {yaw}"


def test_long_side_is_the_length():
    c = cluster_points(rect(10.0, 0.0, 1.0, 3.0, 0.0), cell=1.0)[0]
    assert c["length_m"] > c["width_m"]
    assert c["length_m"] == pytest.approx(3.0, abs=0.15)


def test_a_single_point_has_no_footprint():
    c = cluster_points([(8.0, 0.0), (8.05, 0.0), (8.1, 0.0)], cell=1.0)[0]
    assert c["length_m"] == pytest.approx(0.1, abs=0.05)
    assert c["width_m"] == pytest.approx(0.0, abs=0.05)


def test_track_takes_the_biggest_view_until_there_is_something_to_vote_on():
    """With one or two sightings there is no majority, so the larger view wins: the LiDAR
    only ever catches part of a thing. From the third sighting the middle value takes over
    (day 7), so one frame that glues the thing to a wall cannot set its size for ever."""
    tr = ObjectTracker()
    obs = lambda l, w, h, yaw: [{"wx": 10.0, "wy": 0.0, "length_m": l, "width_m": w, "height_m": h, "yaw_deg": yaw}]
    tr.update(obs(1.0, 0.5, 1.2, 10.0), 0.0)
    t = tr.update(obs(3.5, 1.6, 1.5, 80.0), 0.1)[0]
    assert t.length_m == pytest.approx(3.5), "two sightings: the bigger one wins"
    assert t.width_m == pytest.approx(1.6) and t.height_m == pytest.approx(1.5)


def test_track_without_sizes_stays_zero():
    tr = ObjectTracker()
    tr.update([{"wx": 5.0, "wy": 0.0}], 0.0)
    t = tr.update([{"wx": 5.0, "wy": 0.0}], 0.1)[0]
    assert (t.length_m, t.width_m, t.height_m) == (0.0, 0.0, 0.0)


def test_detected_object_defaults_to_unmeasured():
    o = DetectedObject(object_type=ObjectType.OBSTACLE, x=5.0, y=0.0, distance=5.0)
    assert (o.length_m, o.width_m, o.height_m, o.yaw_deg) == (0.0, 0.0, 0.0, 0.0)


def test_two_things_a_metre_apart_are_two_things_whatever_the_grid():
    """This used to read "a coarse grid glues them together", and it did: the flood fill joins
    CELLS, so at 1.2 m two squares 2 m apart came out as one blob and only the van's own 0.8 m
    grid kept them apart. Since 2026-09-15 the points inside a blob are re-joined by DISTANCE
    (BODY_GAP_M), so how coarse the grid is no longer decides whether two bodies are one."""
    pts = list(rect(10.0, 0.0, 0.4, 0.4)) + list(rect(10.0, 2.0, 0.4, 0.4))
    assert len(cluster_points(pts, cell=0.8)) == 2, "the van's grid keeps them apart, as before"
    assert len(cluster_points(pts, cell=1.2)) == 2, "and a coarse one no longer glues them"


# ---- scraps: the pieces too small to be a body of their own -------------------------------

def test_a_scrap_beside_one_body_is_measured_as_part_of_it():
    """The split returns groups of points, and a group below BODY_MIN_POINTS is not a body.
    Those used to be thrown away, which shortened whatever they came off: on the two-parked
    fixture the 4x4 lost twelve of its own points and measured 1.53 m wide against a true
    2.15. A scrap with one body near it now goes back to that body."""
    body = rect(10.0, 0.0, 2.0, 1.6)
    scrap = [(11.7, 0.0), (11.7, 0.2)]          # 0.7 m off the nose: past BODY_GAP_M, inside reach
    apart = cluster_points(body, cell=0.8)
    whole = cluster_points(body + scrap, cell=0.8)
    assert len(whole) == 1, "the scrap does not become a body of its own"
    assert whole[0]["length_m"] > apart[0]["length_m"] + 0.5, \
        f"the scrap was dropped: {whole[0]['length_m']:.2f} m against {apart[0]['length_m']:.2f}"


def test_a_scrap_between_two_bodies_joins_neither():
    """Which is what keeps this from gluing the pair back together. A scrap in the space
    between two vehicles is near both of them, so it belongs to neither with any confidence
    and stays out -- it must not drag one body across the gap into the other."""
    left = rect(10.0, 0.0, 2.0, 1.6)
    right = rect(10.0, 3.0, 2.0, 1.6)           # 1.4 m of air between the two bodies
    # in the middle: 0.7 m from each, so past the gap rule for both and inside reach of both
    scrap = [(10.0, 1.5), (10.2, 1.5)]
    got = cluster_points(left + right + scrap, cell=0.8)
    assert len(got) == 2, f"{len(got)} bodies: the scrap in the gap joined them up"
    for c in got:
        assert c["width_m"] < 2.2, \
            f"a body measures {c['width_m']:.2f} m across: it reached over the gap"
