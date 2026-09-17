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
    for i in range(4):
        for j in range(4):
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


# ---- L6: rotation measured against the world, and the gyro offset it reveals --------------
#
# The compass is GONE from this filter. L4 measured its injected 0.30-degree offset as a hard
# floor under heading accuracy -- twice the entire budget -- and L4.1 showed that estimating
# that offset made heading WORSE, because the offset state absorbed the sideslip too. What
# replaces it is LiDAR scan matching, which measures how far the van actually swung against
# the standing world: no north, no offset, nothing to absorb.

def test_the_compass_is_an_anchor_not_a_heading_source():
    """The compass is fused again after L6 first removed it -- but weakly, on purpose.

    Removing it cost 2.5-3.9 degrees of heading on turn-heavy routes, because LiDAR measures
    ROTATION and a rotation says nothing about where the angle started, while GNSS anchors
    heading only through the direction of travel and that coupling is gated off below 2 m/s --
    which is exactly when a van turns. What it must NOT do is dominate: its offset is twice the
    heading budget, so one reading has to barely move the estimate.
    """
    from warp_av.localization.ekf import COMPASS_SIGMA_RAD
    assert math.degrees(COMPASS_SIGMA_RAD) >= 1.0, \
        "the weight must reflect the offset under it, not just the noise"
    # Settle the way the van actually runs -- with GNSS coming in. That matters: the compass
    # only counts for little once something ELSE has pinned the heading down. Left at the
    # seed's two degrees of doubt it would dominate, and rightly so, because at that point it
    # really is the best information available.
    f = ekf(speed=6.0)
    t = 100.0
    for i in range(200):
        t += 0.05
        f.set_speed(6.0, False, t)
        f.predict_to(t, 0.0)
        if i % 5 == 0:
            lat, lon = geo.to_latlon(f.x[0], f.x[1])
            f.correct_gnss(lat, lon, t)
    assert math.degrees(f.pose().cov.sigma_yaw) < 0.5, "GNSS has pinned the heading first"
    before = f.x[2]
    f.correct_compass(math.radians(10.0), t + 0.05)   # a wildly wrong reading
    moved = abs(math.degrees(f.x[2] - before))
    assert moved < 0.5, "one compass reading may nudge the heading, never steer it"


def test_the_compass_offset_starts_at_zero_and_unknown():
    """It is estimated again in L7, but never handed the answer."""
    from warp_av.localization.ekf import COMPASS_BIAS_SIGMA0_RAD
    f = ekf()
    assert f.x.shape == (5,), "x, y, yaw, gyro bias, compass bias"
    assert f.compass_bias_rad == 0.0
    assert f.compass_bias_sigma_rad == pytest.approx(COMPASS_BIAS_SIGMA0_RAD)


def test_nothing_in_the_filter_knows_the_injected_compass_offset():
    """Checked against the module's CONSTANTS rather than its prose: the docstrings discuss
    the injected value at length, which is the point of them, and a text search would either
    ban that discussion or be trivially evaded by writing the number differently."""
    from warp_av.localization import ekf as mod
    answer = math.radians(0.30)
    for name in dir(mod):
        v = getattr(mod, name)
        if isinstance(v, float) and abs(abs(v) - answer) < 1e-9:
            raise AssertionError("%s carries the injected offset" % name)
    f = ekf()
    assert f.compass_bias_rad == 0.0, "and the state starts at zero, not at the answer"


def test_lidar_keeps_the_compass_offset_from_swallowing_the_sideslip():
    """THE experiment of this phase, and the thing L4.1 got wrong.

    A van crabs: it travels a little to the side of where its nose points. GNSS sees where it
    WENT, the compass sees where it POINTS plus an offset. In L4.1 those were the only two
    voices, so the offset state had no way to tell the two disagreements apart and settled on
    offset-minus-sideslip -- roughly twice the injected value, with heading worse for it.

    L7 was built on the expectation that LiDAR would break that tie. IT DOES NOT, and this
    test records the fact rather than the hope. LiDAR measures how the heading CHANGES; it says
    nothing about where the heading IS. The compass gives yaw + offset and GNSS gives
    yaw + sideslip: two equations in three unknowns, and a constraint on the derivative of one
    of them adds no fourth. Measured with LiDAR on and with LiDAR off, the offset converges to
    the same wrong number to four decimal places.

    Separating these needs an independent ABSOLUTE heading -- a second GNSS antenna, or
    matching the LiDAR against a map -- not another state.
    """
    offset = math.radians(0.30)
    slip = math.radians(0.75)                      # L2-GAP measured 0.756 on this van
    f = ekf(speed=8.0)
    t, x, y, yaw = 100.0, 0.0, 0.0, 0.0
    for i in range(4000):
        t += 0.05
        f.set_speed(8.0, False, t)
        f.predict_to(t, 0.0)
        x += 8.0 * 0.05 * math.cos(yaw + slip)     # travelling off the nose
        y += 8.0 * 0.05 * math.sin(yaw + slip)
        f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.06))
        if i % 2 == 0:
            f.correct_compass(yaw + offset, t)
        if i % 5 == 0:
            lat, lon = geo.to_latlon(x, y)
            f.correct_gnss(lat, lon, t)
    got = math.degrees(f.compass_bias_rad)
    assert got == pytest.approx(-0.45, abs=0.05), \
        "it lands on offset-minus-sideslip, not on the offset"
    assert math.degrees(f.x[2]) == pytest.approx(0.75, abs=0.05), \
        "because the heading has been pulled onto the COURSE"


def test_lidar_makes_no_difference_to_that_at_all():
    """The sharpest statement of the limit: run the same drive with the rotation measurement
    and without it, and the offset lands in the same place."""
    def settle(with_lidar):
        offset, slip = math.radians(0.30), math.radians(0.75)
        f = ekf(speed=8.0)
        t, x, y = 100.0, 0.0, 0.0
        for i in range(3000):
            t += 0.05
            f.set_speed(8.0, False, t)
            f.predict_to(t, 0.0)
            x += 8.0 * 0.05 * math.cos(slip)
            y += 8.0 * 0.05 * math.sin(slip)
            if with_lidar:
                f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.06))
            if i % 2 == 0:
                f.correct_compass(offset, t)
            if i % 5 == 0:
                lat, lon = geo.to_latlon(x, y)
                f.correct_gnss(lat, lon, t)
        return math.degrees(f.compass_bias_rad)
    assert settle(True) == pytest.approx(settle(False), abs=0.02)


def test_with_no_sideslip_it_finds_the_offset_exactly():
    """The other half of the proof: the estimator is not broken. Take the sideslip away and it
    recovers the injected offset precisely. What defeats it is the ambiguity, not the maths."""
    offset = math.radians(0.30)
    f = ekf(speed=8.0)
    t, x = 100.0, 0.0
    for i in range(3000):
        t += 0.05
        f.set_speed(8.0, False, t)
        f.predict_to(t, 0.0)
        x += 8.0 * 0.05
        f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.06))
        if i % 2 == 0:
            f.correct_compass(offset, t)
        if i % 5 == 0:
            lat, lon = geo.to_latlon(x, 0.0)
            f.correct_gnss(lat, lon, t)
    assert math.degrees(f.compass_bias_rad) == pytest.approx(0.30, abs=0.03)
    assert abs(math.degrees(f.x[2])) < 0.05


def test_the_two_offsets_do_not_trade_error_with_each_other():
    """A gyro offset and a compass offset can look alike over a short window: both push the
    heading one way. They are separable because one acts on the RATE and the other on the
    ANGLE, and this checks the filter keeps them apart rather than swapping error between."""
    gyro_off = 5e-4
    comp_off = math.radians(0.30)
    f = ekf(speed=8.0)
    t = 100.0
    for i in range(4000):
        t += 0.05
        f.set_speed(8.0, False, t)
        f.predict_to(t, gyro_off)                  # the gyro reads high by a constant
        f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.06))
        if i % 2 == 0:
            f.correct_compass(comp_off, t)
    assert f.gyro_bias_rad_s == pytest.approx(gyro_off, rel=0.5), "the rate offset is the gyro's"
    assert math.degrees(f.compass_bias_rad) == pytest.approx(0.30, abs=0.20), \
        "and the angle offset is the compass's"


def test_the_compass_offset_cannot_jump_in_one_reading():
    f = ekf(speed=8.0)
    t = 100.0
    for i in range(400):
        t += 0.05
        f.predict_to(t, 0.0)
        f.correct_compass(math.radians(0.3), t)
    settled = f.compass_bias_rad
    f.correct_compass(math.radians(45.0), t + 0.05)
    assert abs(math.degrees(f.compass_bias_rad - settled)) < 1.0


def test_the_compass_does_hold_the_heading_over_a_long_run():
    """The job it is actually there for: with only rotation measurements the angle wanders,
    and with the anchor it does not."""
    def drift(anchor):
        f = ekf(speed=6.0)
        t = 100.0
        for i in range(2000):
            t += 0.05
            f.predict_to(t, 5e-4)                  # a small unmodelled offset, integrating up
            f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.06))
            if anchor and i % 4 == 0:
                f.correct_compass(0.0, t)
        return abs(math.degrees(f.x[2]))
    assert drift(False) > drift(True), "the anchor must hold the angle down"
    assert drift(True) < 0.5


def test_lidar_rotation_teaches_it_the_gyro_offset():
    """The whole point of the fourth state. The gyro reads a constant amount high; LiDAR keeps
    saying the van turned less than that; the difference is the offset."""
    f = ekf(speed=6.0)
    bias = 5e-4                                    # rad/s: one prior sigma, not twenty
    true_rate = math.radians(4.0)
    t = 100.0
    # A sharper LiDAR than the real one, on purpose: this test is about the MECHANISM. How
    # fast the real instrument can resolve a real offset is a separate question, measured in
    # test_how_long_it_takes_to_see_a_real_gyro_offset below.
    for _ in range(2000):
        t += 0.05
        f.predict_to(t, true_rate + bias)          # the gyro, reading high
        f.correct_lidar_yaw_rate(true_rate * 0.05, 0.05, t, math.radians(0.006))
    assert f.gyro_bias_rad_s == pytest.approx(bias, rel=0.25)


def test_learning_the_offset_is_what_keeps_the_heading_straight():
    """Converging on the offset is only worth anything if the HEADING gets better for it."""
    bias = 5e-4
    drifting = ekf(speed=6.0)
    corrected = ekf(speed=6.0)
    t = 100.0
    for _ in range(2000):
        t += 0.05
        drifting.predict_to(t, bias)               # gyro only: the offset integrates up
        corrected.predict_to(t, bias)
        corrected.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(0.006))
    assert abs(math.degrees(drifting.x[2])) > 2.5, "unaided, the offset runs away"
    assert abs(math.degrees(corrected.x[2])) < 0.5, "with LiDAR it does not"


def test_nothing_in_the_filter_knows_the_injected_gyro_offset():
    """The easiest way to fake this result would be to write 5e-05 somewhere in the filter."""
    import inspect
    from warp_av.localization import ekf as mod
    src = inspect.getsource(mod)
    for banned in ("5e-5", "5e-05", "0.00005"):
        assert banned not in src, "the filter must not carry the answer"


def test_without_lidar_it_never_claims_to_know_the_offset():
    """Observability, stated as a test. The gyro cannot audit itself: nothing but a second
    opinion on rotation can separate a real turn from an offset."""
    f = ekf(speed=6.0)
    before = f.gyro_bias_sigma_rad_s
    t = 100.0
    for _ in range(400):
        t += 0.05
        f.predict_to(t, math.radians(4.0))
    assert f.gyro_bias_sigma_rad_s >= before, "with no LiDAR the doubt must not shrink"


def test_the_offset_doubt_grows_without_lidar_and_shrinks_with_it():
    """Knowing when the van is unsure, at the state LiDAR actually observes.

    The gyro cannot audit itself, so while nothing is measuring rotation independently the
    filter must not get any surer of the offset -- and must get surer again when it can.
    """
    f = ekf(speed=6.0)
    t = 100.0
    for _ in range(200):                           # LiDAR present: the doubt comes down
        t += 0.05
        f.predict_to(t, math.radians(2.0))
        f.correct_lidar_yaw_rate(math.radians(2.0) * 0.05, 0.05, t, math.radians(0.06))
    settled = f.gyro_bias_sigma_rad_s
    for _ in range(200):                           # ten seconds with none
        t += 0.05
        f.predict_to(t, math.radians(2.0))
    during = f.gyro_bias_sigma_rad_s
    for _ in range(200):                           # and it comes back
        t += 0.05
        f.predict_to(t, math.radians(2.0))
        f.correct_lidar_yaw_rate(math.radians(2.0) * 0.05, 0.05, t, math.radians(0.06))
    after = f.gyro_bias_sigma_rad_s
    assert settled < 5e-4, "with LiDAR it learns something"
    assert during >= settled, "with none it must not get surer"
    assert after < during, "and it learns again when LiDAR returns"


def test_lidar_slows_the_heading_drift_but_cannot_stop_it():
    """The honest limit of a RELATIVE measurement, pinned so nobody expects more of it.

    With no absolute reference, heading uncertainty grows whatever LiDAR says -- a measurement
    of how far the van turned carries nothing about where it started. What it does is make the
    growth slower. Bounding heading outright is GNSS's job, tested below.
    """
    def grow(with_lidar):
        f = ekf(speed=6.0)
        t = 100.0
        before = f.pose().cov.sigma_yaw
        for _ in range(400):
            t += 0.05
            f.predict_to(t, math.radians(2.0))
            if with_lidar:
                f.correct_lidar_yaw_rate(math.radians(2.0) * 0.05, 0.05, t, math.radians(0.06))
        return f.pose().cov.sigma_yaw - before
    assert grow(True) > 0.0, "it grows even with LiDAR: that is what relative means"
    assert grow(True) < grow(False), "but slower than without it"


def test_gnss_is_what_actually_bounds_the_heading():
    """The companion to the test above: with position fixes coming in while moving, absolute
    heading uncertainty settles instead of running away."""
    f = ekf(speed=6.0)
    t = 100.0
    for i in range(400):
        t += 0.05
        f.set_speed(6.0, False, t)
        f.predict_to(t, math.radians(2.0))
        if i % 5 == 0:
            lat, lon = geo.to_latlon(f.x[0], f.x[1])
            f.correct_gnss(lat, lon, t)
        f.correct_lidar_yaw_rate(math.radians(2.0) * 0.05, 0.05, t, math.radians(0.06))
    assert math.degrees(f.pose().cov.sigma_yaw) < 0.5, "GNSS holds the heading doubt down"


def test_a_nonsense_rotation_is_refused():
    f = ekf()
    assert f.correct_lidar_yaw_rate(float("nan"), 0.1, 100.1, 0.01) is False
    assert f.correct_lidar_yaw_rate(0.01, 0.0, 100.1, 0.01) is False, "no time, no rate"
    assert f.correct_lidar_yaw_rate(0.01, 0.1, 100.1, 0.0) is False, "no zero uncertainty"
    assert f.lidar_corrections == 0
    assert f.lidar_rejected == 3


def test_the_rotation_innovation_is_wrapped():
    """A measurement that crosses the wrap must be two degrees, not three hundred and fifty."""
    f = ekf(speed=6.0)
    for i in range(6):                             # the gyro history the window needs
        f.predict_to(100.0 + 0.05 * (i + 1), 0.0)
    ok = f.correct_lidar_yaw_rate(math.radians(359.0), 0.1, 100.3, math.radians(0.06))
    assert ok
    assert abs(math.degrees(f.gyro_bias_rad_s)) < 30.0, "wrapped to -1 deg, not +359"


def test_a_worse_measurement_moves_it_less():
    """The calibrated uncertainty has to actually do something."""
    def settle(sigma_deg):
        f = ekf(speed=6.0)
        t = 100.0
        for _ in range(40):
            t += 0.05
            f.predict_to(t, math.radians(3.0))
            f.correct_lidar_yaw_rate(0.0, 0.05, t, math.radians(sigma_deg))
        return abs(f.gyro_bias_rad_s)
    assert settle(0.06) > settle(2.0), "a doubtful measurement must pull less"


def test_below_two_metres_a_second_gnss_moves_x_and_y_but_not_the_heading():
    """L3's 11.3 degree startup error, prevented. The fix is still used -- it is a good fix --
    only its indirect pull on the heading is suppressed."""
    f = ekf(speed=1.0)                            # crawling
    f.x[0] = 5.0
    f.x[2] = math.radians(3.0)
    yaw_before = f.pose().yaw
    lat, lon = geo.to_latlon(0.0, 0.0)
    assert f.correct_gnss(lat, lon, 100.1) is True
    assert f.pose().x < 5.0, "position must still be corrected"
    assert f.pose().yaw == pytest.approx(yaw_before), "heading must not be rotated"
    assert f.gnss_yaw_suppressed == 1


def test_above_two_metres_a_second_the_normal_coupling_resumes():
    f = ekf(speed=8.0)
    f.x[0] = 5.0
    f.P[0, 2] = f.P[2, 0] = 0.05                  # a real position-heading correlation
    yaw_before = f.pose().yaw
    lat, lon = geo.to_latlon(0.0, 0.0)
    f.correct_gnss(lat, lon, 100.1)
    assert f.pose().yaw != pytest.approx(yaw_before), "above the gate, GNSS may inform heading"
    assert f.gnss_yaw_suppressed == 0


def test_the_gate_does_not_throw_the_fix_away():
    """The danger of a clumsy fix would be disabling GNSS at low speed altogether."""
    f = ekf(speed=0.5)
    f.x[0] = 5.0
    lat, lon = geo.to_latlon(0.0, 0.0)
    for i in range(30):
        f.correct_gnss(lat, lon, 100.0 + 0.1 * (i + 1))
    assert f.pose().x == pytest.approx(0.0, abs=0.1), "position still converges while crawling"
    assert f.corrections == 30


def test_the_covariance_stays_sound_with_a_suppressed_gain():
    """Zeroing a row of K makes the gain sub-optimal, which is fine -- but only because the
    Joseph form is valid for ANY gain. This checks it really does stay symmetric and positive."""
    f = ekf(speed=1.0)
    lat, lon = geo.to_latlon(0.0, 0.0)
    for i in range(200):
        t = 100.0 + 0.05 * (i + 1)
        f.predict_to(t, math.radians(3.0))
        f.correct_gnss(lat, lon, t)
    P = f.P
    for i in range(4):
        for j in range(4):
            assert P[i][j] == pytest.approx(P[j][i], rel=1e-9, abs=1e-12)
        assert P[i][i] > 0.0


def test_how_long_it_takes_to_see_a_real_gyro_offset():
    """A characterisation, not a pass mark: how sharply can the REAL instrument pin the offset?

    At the calibrated standstill sigma of 0.06 degrees over a tenth of a second, each LiDAR
    measurement is worth about 0.6 deg/s of rate. Information accumulates as time x dt over
    that squared, so a three-minute run leaves the offset known to a few times 1e-4 rad/s.
    That is COARSER than a small physical offset, which is worth knowing before anyone expects
    the gyro-bias state to do heavy lifting inside a single drive.
    """
    f = ekf(speed=6.0)
    t = 100.0
    for _ in range(1800):                          # 180 s at 10 Hz
        t += 0.1
        f.predict_to(t, math.radians(2.0))
        f.correct_lidar_yaw_rate(math.radians(2.0) * 0.1, 0.1, t, math.radians(0.06))
    sigma = f.gyro_bias_sigma_rad_s
    assert sigma < 5e-4, "three minutes must teach it something"
    assert sigma > 5e-5, "but not enough to resolve a small offset inside one drive"


def test_muting_lidar_odometry_does_not_blind_the_van():
    """The fault hook L6 Part G needs, pinned so it stays a LOCALIZATION fault.

    Cutting the sensor is a different failure: it blinds perception and the safety supervisor
    stops the van, so the run measures a parked vehicle instead of an estimator riding a gap.
    The mute must leave the sensor, perception and the odometer all running.
    """
    from warp_av.localization.lidar_odometry import LidarOdometry
    odo = LidarOdometry()
    assert hasattr(odo, "update"), "the odometer itself is untouched by muting"
    import inspect
    from warp_av import main as mod
    src = inspect.getsource(mod.WarpAV._run_lidar_odometry)
    assert "lidar_odometry_muted" in src, "the mute is applied where measurements are queued"
    assert "lidar_enabled" not in src, "and never by switching the sensor off"


def test_a_turn_inside_the_window_does_not_look_like_a_gyro_offset():
    """The bug that made L6's first scoring run worse than the phase before it.

    Comparing the LiDAR's average rate against the newest single gyro sample is fine at a
    constant rate and badly wrong while the rate is changing -- which is exactly what a turn
    is. The offset estimate swung between -0.20 and +0.08 deg/s on turn-heavy routes and
    dragged the heading seven degrees. Here the gyro accelerates hard through the window and
    the LiDAR reports the truth; the offset must stay put.
    """
    f = ekf(speed=6.0)
    t = 100.0
    rate = 0.0
    for _ in range(200):
        prev_rate, prev_t = rate, t
        t += 0.05
        rate = math.radians(20.0) * math.sin((t - 100.0) * 2.0)      # a rate that never sits still
        f.predict_to(t, rate)
        turned = 0.5 * (prev_rate + rate) * (t - prev_t)             # what really happened
        f.correct_lidar_yaw_rate(turned, t - prev_t, t, math.radians(0.06))
    assert abs(math.degrees(f.gyro_bias_rad_s)) < 0.02, \
        "a changing rate must not be read as an offset"
    assert abs(math.degrees(f.x[2] - 0.0)) < 90.0                     # sanity: it did turn


def test_it_refuses_when_the_gyro_never_covered_the_window():
    """No extrapolating across a hole in the gyro trace."""
    f = ekf(speed=6.0)
    f.predict_to(100.05, 0.0)
    f.predict_to(100.10, 0.0)
    assert f.correct_lidar_yaw_rate(0.01, 0.1, 140.0, math.radians(0.06)) is False
    assert f.lidar_rejected >= 1


# ---- L7: reported uncertainty must contain what the filter cannot estimate -----------------

def test_the_reported_sigma_never_drops_below_the_gnss_offset():
    """A filter fed fixes that all lean the same way converges onto the lean and goes quiet.
    It cannot estimate that offset -- nothing else measures absolute position -- so it must at
    least not claim to be surer than the offset it cannot see."""
    from warp_av.localization.ekf import GNSS_BIAS_M
    f = ekf(speed=0.0)
    lat, lon = geo.to_latlon(0.0, 0.0)
    t = 100.0
    for _ in range(600):                           # long enough for P to bottom out
        t += 0.05
        f.predict_to(t, 0.0)
        f.correct_gnss(lat, lon, t)
    assert f.pose().cov.sigma_x >= GNSS_BIAS_M
    assert f.pose().cov.sigma_y >= GNSS_BIAS_M


def test_a_stale_pose_is_reported_as_less_certain():
    """The largest single contributor to this project's position error turned out to be twelve
    milliseconds of age. At 4 m/s that is four centimetres, and it is a real error for whoever
    reads the pose -- so it is charged, not explained away."""
    f = ekf(speed=8.0)
    t = 100.0
    for _ in range(200):
        t += 0.05
        f.set_speed(8.0, False, t)
        f.predict_to(t, 0.0)
    fresh = f.covariance(now=t).sigma_x
    stale = f.covariance(now=t + 0.1).sigma_x
    assert stale > fresh
    assert stale == pytest.approx(math.hypot(fresh, 8.0 * 0.1), rel=0.01)


def test_age_is_only_charged_forwards():
    """A timestamp from before the filter's own clock must not shrink the uncertainty."""
    f = ekf(speed=8.0)
    for i in range(20):
        f.set_speed(8.0, False, 100.0 + 0.05 * (i + 1))
        f.predict_to(100.0 + 0.05 * (i + 1), 0.0)
    assert f.covariance(now=100.0).sigma_x == pytest.approx(f.covariance().sigma_x)


def test_standing_still_age_costs_nothing():
    """The charge is speed times age. Parked, a stale pose is still a correct pose."""
    f = ekf(speed=0.0)
    for i in range(20):
        f.predict_to(100.0 + 0.05 * (i + 1), 0.0)
    assert f.covariance(now=102.0).sigma_x == pytest.approx(f.covariance().sigma_x)
