"""The first fused estimate: gyro and wheels to carry it, GNSS to hold it.

Phase L3, 2026-09-16. These pin the arithmetic and, more importantly, the things that make a
filter trustworthy rather than merely plausible: that it refuses time it cannot use, that its
covariance grows when it is guessing and shrinks when it is told something, and that it never
quietly claims more confidence than it has earned.
"""
import math

import pytest

from warp_av.localization import geo
from warp_av.localization.ekf import LocalizationEKF, MAX_PREDICT_S


# ---- the GNSS conversion (measured in L2-GAP) ---------------------------------------------

def test_a_fix_at_the_origin_is_the_origin():
    assert geo.to_xy(0.0, 0.0) == (0.0, 0.0)


def test_latitude_runs_the_OPPOSITE_way_to_carla_y():
    """The sign that mirrors the whole map if it is wrong. CARLA's y axis points SOUTH, so
    driving to +y must make the latitude go DOWN."""
    lat, lon = geo.to_latlon(0.0, 100.0)
    assert lat < 0.0, "100 m of +y must be south of the origin"
    assert lon == pytest.approx(0.0)


def test_longitude_runs_the_same_way_as_carla_x():
    lat, lon = geo.to_latlon(100.0, 0.0)
    assert lon > 0.0 and lat == pytest.approx(0.0)


@pytest.mark.parametrize("x,y", [(0.0, 0.0), (100.0, -50.0), (-250.0, 300.0), (12.5, 7.25)])
def test_the_round_trip_returns_the_same_metres(x, y):
    lat, lon = geo.to_latlon(x, y)
    bx, by = geo.to_xy(lat, lon)
    assert bx == pytest.approx(x, abs=1e-6)
    assert by == pytest.approx(y, abs=1e-6)


def test_one_degree_is_about_a_hundred_and_eleven_kilometres():
    """A sanity check on the constant itself, so a wrong one cannot pass unnoticed."""
    assert geo.METRES_PER_DEGREE == pytest.approx(111320.0, rel=1e-3)


def test_metres_convert_to_degrees_for_setting_sensor_noise():
    """CARLA wants GNSS noise in degrees; every requirement we have is in metres."""
    assert geo.DEFAULT.degrees_lat_for(0.02) == pytest.approx(0.02 * 8.983e-06)


def test_the_compass_is_ninety_degrees_off_the_yaw():
    """Measured exactly in L2-GAP: compass_deg - yaw_deg = 90.000."""
    assert math.degrees(geo.bearing_to_yaw(math.radians(90.0))) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(geo.bearing_to_yaw(math.radians(0.0))) == pytest.approx(-90.0)


# ---- the filter ---------------------------------------------------------------------------

def ekf(x=0.0, y=0.0, yaw=0.0, t=100.0, speed=0.0):
    f = LocalizationEKF()
    f.seed(x, y, yaw, t)
    f.set_speed(speed, False, t)
    return f


def test_it_says_nothing_before_it_is_seeded():
    f = LocalizationEKF()
    assert f.pose() is None
    assert f.predict_to(1.0, 0.0) is False
    assert f.state() == {"seeded": False}


def test_seeding_starts_unsure_rather_than_certain():
    """A filter that begins certain refuses every correction that would have corrected it."""
    f = ekf()
    c = f.pose().cov
    assert c.sigma_x > 0.1, "a seed is a guess, not a measurement"
    assert c.is_truth is False


def test_driving_straight_moves_along_the_heading():
    f = ekf(speed=10.0)
    for i in range(10):
        f.predict_to(100.0 + 0.1 * (i + 1), 0.0)
    p = f.pose()
    assert p.x == pytest.approx(10.0, abs=1e-6)
    assert p.y == pytest.approx(0.0, abs=1e-9)


def test_a_yaw_rate_turns_it():
    f = ekf(speed=0.0)
    f.predict_to(100.5, math.radians(20.0))      # 20 deg/s for half a second
    assert math.degrees(f.pose().yaw) == pytest.approx(10.0)


def test_uncertainty_grows_while_it_is_only_guessing():
    f = ekf(speed=10.0)
    before = f.pose().cov.sigma_x
    for i in range(20):
        f.predict_to(100.0 + 0.1 * (i + 1), 0.0)
    assert f.pose().cov.sigma_x > before, "dead reckoning must get less sure, never more"


def test_a_fix_pulls_it_back_and_makes_it_surer():
    f = ekf(speed=0.0)
    f.x[0] = 5.0                                  # pretend it has drifted 5 m
    grown = f.pose().cov.sigma_x
    lat, lon = geo.to_latlon(0.0, 0.0)            # the truth says it is at the origin
    assert f.correct_gnss(lat, lon, 100.1) is True
    p = f.pose()
    assert 0.0 <= p.x < 5.0, "the fix must pull it back towards the origin"
    assert p.cov.sigma_x < grown, "and being told something must make it surer"


def test_many_fixes_converge_on_the_measurement():
    f = ekf(speed=0.0)
    f.x[0] = 5.0
    lat, lon = geo.to_latlon(0.0, 0.0)
    for i in range(50):
        f.predict_to(100.0 + 0.1 * (i + 1), 0.0)
        f.correct_gnss(lat, lon, 100.0 + 0.1 * (i + 1))
    assert f.pose().x == pytest.approx(0.0, abs=0.05)


def test_the_covariance_stays_symmetric_over_a_long_run():
    """Joseph form is used for exactly this: the short update quietly loses symmetry."""
    f = ekf(speed=8.0)
    lat, lon = geo.to_latlon(0.0, 0.0)
    for i in range(400):
        t = 100.0 + 0.05 * (i + 1)
        f.predict_to(t, math.radians(5.0))
        if i % 2 == 0:
            f.correct_gnss(lat, lon, t)
    P = f.P
    for i in range(3):
        for j in range(3):
            assert P[i][j] == pytest.approx(P[j][i], rel=1e-9, abs=1e-12)
        assert P[i][i] > 0.0, "a variance may never go negative"


# ---- time: the part that makes it usable on real measurements -----------------------------

def test_a_duplicate_timestamp_is_dropped_not_integrated():
    f = ekf(speed=10.0)
    f.predict_to(100.1, 0.0)
    x = f.pose().x
    assert f.predict_to(100.1, 0.0) is False
    assert f.pose().x == x
    assert f.rejected_old == 1


def test_a_measurement_that_arrives_late_is_dropped():
    """Out of order, not out of mind: integrating backwards would move the van the wrong way."""
    f = ekf(speed=10.0)
    f.predict_to(100.2, 0.0)
    assert f.predict_to(100.1, 0.0) is False
    assert f.rejected_old == 1


def test_a_long_gap_is_skipped_rather_than_guessed_through():
    f = ekf(speed=10.0)
    x = f.pose().x
    assert f.predict_to(100.0 + MAX_PREDICT_S + 0.1, 0.0) is False
    assert f.pose().x == x, "no motion invented across a gap"
    assert f.rejected_gap == 1
    # ...and the clock moved on, so the next step is a normal one rather than a huge one
    assert f.predict_to(100.0 + MAX_PREDICT_S + 0.2, 0.0) is True


def test_jitter_is_fine_because_the_real_dt_is_used():
    """L2-GAP measured IMU samples 0.0065 s to 0.15 s apart. Uneven is not a problem;
    ASSUMING even would be."""
    f = ekf(speed=10.0)
    t = 100.0
    for dt in (0.0065, 0.09, 0.011, 0.15, 0.04):
        t += dt
        f.predict_to(t, 0.0)
    assert f.pose().x == pytest.approx(10.0 * (t - 100.0), abs=1e-6)


def test_missing_gnss_is_not_an_error_it_is_just_less_certain():
    f = ekf(speed=10.0)
    for i in range(50):
        f.predict_to(100.0 + 0.1 * (i + 1), 0.0)
    assert f.corrections == 0
    assert f.pose().cov.sigma_x > 0.5
    assert f.pose() is not None, "it keeps answering; it just says it is unsure"


def test_a_nonsense_fix_is_refused():
    f = ekf()
    assert f.correct_gnss(float("nan"), 0.0, 100.1) is False
    assert f.gnss_rejected == 1


# ---- what it publishes ----------------------------------------------------------------------

def test_the_covariance_comes_from_the_filter_not_from_a_growth_model():
    """The dead reckoner's sigma is a curve fitted to three drives. This one is computed."""
    f = ekf(speed=10.0)
    f.predict_to(100.1, 0.0)
    c = f.pose().cov
    assert c.xx > 0 and c.yy > 0 and c.yaw > 0
    assert c.as_matrix()[0][1] == c.as_matrix()[1][0]


def test_it_never_claims_to_be_the_truth():
    f = ekf(speed=1.0)
    f.predict_to(100.1, 0.0)
    p = f.pose()
    assert p.reason == "EKF"
    assert p.cov.is_truth is False


def test_the_error_report_says_how_far_off_and_how_sure_it_claimed_to_be():
    """The pairing that matters: a large error beside a small sigma is the dangerous case."""
    from warp_av.localization.localization import Pose
    f = ekf(speed=10.0)
    for i in range(10):
        f.predict_to(100.0 + 0.1 * (i + 1), 0.0)
    e = f.error_against(Pose(x=9.0, y=0.0, yaw=0.0))
    assert e["along_m"] == pytest.approx(1.0, abs=1e-6)
    assert e["sigma_m"] > 0.0
    assert e["err_over_sigma"] == pytest.approx(e["error_m"] / e["sigma_m"])


def test_the_published_state_counts_what_it_refused():
    f = ekf(speed=1.0)
    f.predict_to(100.1, 0.0)
    f.predict_to(100.1, 0.0)                 # duplicate
    st = f.state()
    assert st["predicts"] == 1 and st["rejected_old"] == 1
    assert st["corrections"] == 0
    assert "sigma_x_m" in st and "sigma_yaw_deg" in st
