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
