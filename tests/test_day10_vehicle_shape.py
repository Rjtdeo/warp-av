"""
Is it car-shaped, or merely big? (Perception V2, day 10 follow-up.)

Rajat noticed the van labelling things "VEHICLE" where no vehicle was. Measured
on the four recordings with the camera off, so the shape rule did all the
naming: of 1,727 things called a vehicle, 23 were vehicles. Fifty-four per cent
were buildings and twenty-three per cent were walls.

The rule was written before the van could measure anything. It asked only:
wider than 0.9 m, twelve laser points, taller than half a metre. A building
wall passes all three.

Since day 4 the van measures each blob's length, width and height. A car is
bounded in all three. A building is too tall, a wall too long, a post too thin,
a bin too short. So the rule now asks whether the thing would fit in a parking
space, and the count of false vehicles fell roughly eight-fold with no change
to how often the real car is named.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp_av.perception.tracking import (VEHICLE_MAX_HEIGHT_M, VEHICLE_MAX_LENGTH_M,  # noqa: E402
                                         VEHICLE_MIN_LENGTH_M, VEHICLE_MIN_WIDTH_M,
                                         vehicle_shaped)


def blob(length, width, height, n=40, distance=12.0):
    """One clustered thing, as the pipeline describes it."""
    return {"n": n, "extent": max(length, width) / 2.0, "height": height,
            "length_m": length, "width_m": width, "distance": distance}


def test_a_car_is_a_vehicle():
    assert vehicle_shaped(blob(4.4, 1.9, 1.5)) is True
    assert vehicle_shaped(blob(3.9, 1.8, 1.6, distance=20.0)) is True


def test_a_van_is_still_a_vehicle():
    assert vehicle_shaped(blob(5.9, 2.0, 2.5)) is True, "the Sprinter itself is 2.5 m tall"


def test_a_building_is_not():
    """The commonest mistake: 54 % of the false labels were buildings."""
    assert vehicle_shaped(blob(6.0, 3.0, 3.9)) is False
    assert vehicle_shaped(blob(9.0, 2.0, 4.0, n=90)) is False


def test_a_wall_is_not():
    assert vehicle_shaped(blob(18.0, 0.4, 2.0, n=80)) is False


def test_a_pole_is_not():
    assert vehicle_shaped(blob(0.4, 0.3, 2.4, n=20)) is False


def test_a_bin_or_a_post_beside_the_van_is_not():
    assert vehicle_shaped(blob(0.7, 0.6, 1.1, distance=8.0)) is False


def test_a_hedge_row_is_not():
    assert vehicle_shaped(blob(7.0, 0.5, 1.4, n=70)) is False


def test_the_far_end_on_view_of_a_car_is_still_allowed():
    """A car 30 m away shows a thin near face. It is too far for the width test to mean
    anything, and if it is really too sparse it fails the point count instead."""
    assert vehicle_shaped(blob(2.4, 0.4, 1.5, n=14, distance=30.0)) is True
    assert vehicle_shaped(blob(2.4, 0.4, 1.5, n=8, distance=30.0)) is False, "too few points"


def test_a_blob_with_no_measurements_falls_back_to_the_old_test():
    """Ground-truth and camera-only modes never measure a footprint. They keep the rule
    they had, since there is nothing better to judge them on."""
    assert vehicle_shaped({"n": 20, "extent": 1.4, "height": 1.5}) is True
    assert vehicle_shaped({"n": 4, "extent": 1.4, "height": 1.5}) is False


def test_the_limits_are_the_size_of_real_vehicles():
    assert 2.0 <= VEHICLE_MAX_HEIGHT_M <= 3.0, "a car or a van, not a building"
    assert VEHICLE_MAX_LENGTH_M >= 12.0, "a bus is still a vehicle"
    assert VEHICLE_MIN_WIDTH_M >= 0.7, "narrower than this is a post or a fence"
    assert VEHICLE_MIN_LENGTH_M >= 1.5


def test_height_alone_decides_the_common_case():
    """The measurement that mattered: false labels stood 3.8 to 4.0 m tall, the real car
    1.5 m. Everything else is a refinement on top of that."""
    car_ish = blob(4.0, 1.8, 1.5)
    assert vehicle_shaped(car_ish) is True
    tall = dict(car_ish, height=3.9)
    assert vehicle_shaped(tall) is False


# ---------------------------------------------------------------------------
# Found live, after the shape rule was fixed: the van was still calling 3.9 m
# tall things vehicles. The rule was right; the name was stuck. A track took a
# name once and kept it for the rest of its life, so one frame in which a blob
# looked car-shaped labelled that thing a vehicle for ever.
# ---------------------------------------------------------------------------

def test_a_name_from_the_shape_is_dropped_when_the_shape_stops_agreeing():
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    named = {"wx": 10.0, "wy": 0.0, "distance": 10.0, "cls": "vehicle",
             "cls_source": "shape", "confidence": 0.45}
    plain = {"wx": 10.0, "wy": 0.0, "distance": 10.0}
    for _ in range(3):
        t += 0.1
        tr.update([named], t)
    assert tr._tracks[0].cls == "vehicle"
    for _ in range(4):
        t += 0.1
        tr.update([plain], t)
    assert tr._tracks[0].cls is None, "the shape stopped agreeing, so the name must go"


def test_a_name_from_the_camera_survives_a_few_missed_frames():
    """A detector missing one frame is ordinary. A shape that stops matching is not."""
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    seen = {"wx": 8.0, "wy": 0.0, "distance": 8.0, "cls": "pedestrian",
            "cls_source": "camera", "confidence": 0.9}
    plain = {"wx": 8.0, "wy": 0.0, "distance": 8.0}
    for _ in range(3):
        t += 0.1
        tr.update([seen], t)
    for _ in range(5):
        t += 0.1
        tr.update([plain], t)
    assert tr._tracks[0].cls == "pedestrian", "five missed frames is not proof it left"


def test_a_name_comes_straight_back_when_it_is_confirmed_again():
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    named = {"wx": 10.0, "wy": 0.0, "distance": 10.0, "cls": "vehicle",
             "cls_source": "shape", "confidence": 0.45}
    plain = {"wx": 10.0, "wy": 0.0, "distance": 10.0}
    for _ in range(2):
        t += 0.1
        tr.update([named], t)
    for _ in range(4):
        t += 0.1
        tr.update([plain], t)
    assert tr._tracks[0].cls is None
    t += 0.1
    tr.update([named], t)
    assert tr._tracks[0].cls == "vehicle"


def test_a_rail_far_away_is_not_a_vehicle():
    """Found live at 37.8 m: 1.9 m long and 0.2 m wide, called a vehicle. The width test
    was switched off beyond 25 m, because a distant car shows only its thin near face.
    The bar drops out there; it does not disappear."""
    from warp_av.perception.tracking import VEHICLE_MIN_WIDTH_FAR_M
    assert vehicle_shaped(blob(1.9, 0.2, 2.3, n=20, distance=37.8)) is False
    assert VEHICLE_MIN_WIDTH_FAR_M < VEHICLE_MIN_WIDTH_M, "the far bar is lower, not absent"


def test_a_real_car_far_away_still_passes():
    assert vehicle_shaped(blob(2.6, 1.1, 1.5, n=14, distance=30.0)) is True
    assert vehicle_shaped(blob(2.0, 0.5, 1.5, n=14, distance=30.0)) is True, (
        "its near face is thin, but not rail-thin")


def test_a_name_is_rechecked_against_what_the_track_has_learned():
    """Seen live: a thing 3.7 m tall and a thing with no width were both still labelled
    vehicles. Each had had one frame that looked car-shaped, and the shape rule only ever
    sees one frame. The track knows the middle of the last twelve, which is better."""
    from warp_av.perception.tracking import ObjectTracker

    def feed(tr, sizes, cls_frames):
        t = 0.0
        for k, (l, w, h) in enumerate(sizes):
            t += 0.1
            o = {"wx": 20.0, "wy": 0.0, "distance": 20.0,
                 "length_m": l, "width_m": w, "height_m": h, "yaw_deg": 0.0}
            if k in cls_frames:
                o["cls"] = "vehicle"
                o["cls_source"] = "shape"
                o["confidence"] = 0.45
            tr.update([o], t)
        return tr._tracks[0]

    tall = feed(ObjectTracker(), [(4.0, 1.8, 3.7)] * 8, {0, 1})
    assert tall.cls is None, "a 3.7 m tall thing is not a vehicle, whatever one frame said"

    thin = feed(ObjectTracker(), [(1.4, 0.05, 2.1)] * 8, {0, 1})
    assert thin.cls is None, "and neither is a rail"

    # a real car keeps looking car-shaped, so the rule keeps naming it every frame
    car = feed(ObjectTracker(), [(4.2, 1.8, 1.5)] * 8, set(range(8)))
    assert car.cls == "vehicle", "a real car keeps its name"
    # and even if the shape rule falls silent for a couple of frames, the re-check has no
    # complaint about it: what the track has learned is a perfectly ordinary car
    blinking = feed(ObjectTracker(), [(4.2, 1.8, 1.5)] * 8, {0, 1, 2, 4, 6})
    assert blinking.cls == "vehicle"


def test_a_camera_name_is_left_alone_by_the_recheck():
    """The camera saw the thing itself, not just its outline; a poor LiDAR view of a
    person must not overrule it."""
    from warp_av.perception.tracking import ObjectTracker
    tr = ObjectTracker()
    t = 0.0
    for k in range(8):
        t += 0.1
        o = {"wx": 12.0, "wy": 0.0, "distance": 12.0, "length_m": 0.5, "width_m": 0.05,
             "height_m": 1.8, "yaw_deg": 0.0, "cls": "pedestrian", "cls_source": "camera",
             "confidence": 0.9}
        tr.update([o], t)
    assert tr._tracks[0].cls == "pedestrian"


def test_a_box_lorry_is_a_vehicle():
    """Live, a CARLA box lorry measured 2.7 m tall and the limit was 2.6, so it came out as
    an obstacle. Allowing 3.0 m costs four extra wrong labels across the recordings; 3.4 m
    costs seventy-three, which is where the buildings start."""
    assert vehicle_shaped(blob(4.0, 2.0, 2.7, n=60)) is True
    assert vehicle_shaped(blob(4.0, 2.0, 3.9, n=60)) is False, "and a building still is not"
