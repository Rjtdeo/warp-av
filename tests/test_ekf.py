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


# ---- L4: heading as a measurement, and the low-speed yaw gate ------------------------------

def test_the_compass_makes_the_heading_observable():
    """Without it, the only evidence about heading is the direction the van appears to have
    travelled -- which is no evidence at all when it is barely moving."""
    f = ekf(yaw=0.0, speed=0.0)
    f.x[2] = math.radians(10.0)                   # pretend the heading has drifted 10 deg
    before = abs(math.degrees(f.pose().yaw))
    assert f.correct_heading(0.0, 100.1) is True  # the compass says straight ahead
    after = abs(math.degrees(f.pose().yaw))
    assert after < before, "being told the heading must move the estimate towards it"
    assert f.heading_corrections == 1


def test_repeated_headings_converge_and_make_it_surer():
    """PARKED, the compass closes the gap but cannot say whose fault it was.

    L4 (three states) drove yaw straight to the measurement, which was right by luck and wrong
    by reasoning: it assumed the compass had no offset. Standing still there is no second
    opinion on heading, so a 10 degree disagreement is 10 degrees of heading error, or 10
    degrees of compass offset, or any split of the two -- and the filter must close the gap
    without pretending to know which. It does still get surer of the SUM, which is the part it
    was actually told about.
    """
    f = ekf(yaw=0.0, speed=0.0)
    f.x[2] = math.radians(10.0)
    grown = f.pose().cov.sigma_yaw
    for i in range(60):
        f.correct_heading(0.0, 100.0 + 0.05 * (i + 1))
    assert abs(math.degrees(f.x[2] + f.x[3])) < 0.5, "the gap must close"
    assert f.pose().cov.sigma_yaw < grown
    assert abs(math.degrees(f.x[2])) > 1.0, "but it may not claim to have resolved the heading"


def test_driving_resolves_what_parking_could_not():
    """The other half of the pair above: give it motion and a second opinion, and the split it
    refused to guess at standing still becomes knowable."""
    f = ekf(yaw=0.0, speed=8.0)
    f.x[2] = math.radians(10.0)                   # a real heading error, no compass offset
    drive(f, seconds=30.0, compass_bias_deg=0.0)
    assert abs(math.degrees(f.x[2])) < 0.5, "GNSS and the motion model pin the heading"
    assert abs(math.degrees(f.x[3])) < 0.5, "and correctly find no compass offset"


def test_the_heading_innovation_is_wrapped():
    """+179 measured against -179 held is two degrees apart, not three hundred and fifty
    eight. Unwrapped, this sends the estimate the long way round."""
    f = ekf(yaw=math.radians(-179.0), speed=0.0)
    f.correct_heading(math.radians(179.0), 100.1)
    got = math.degrees(f.pose().yaw)
    assert got < -178.0 or got > 178.0, f"went the long way round: {got}"


def test_a_nonsense_heading_is_refused():
    f = ekf()
    assert f.correct_heading(float("nan"), 100.1) is False
    assert f.heading_corrections == 0


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


# ---- the compass offset the filter works out for itself (L4.1) ----------------------------
#
# L4 left the heading error sitting on a floor: mean 0.32-0.37 degrees, near constant across
# every run, because a fixed compass offset is exactly what averaging cannot remove. These pin
# the fourth state that estimates it -- and, just as importantly, pin the fact that it is
# never told the answer and cannot learn one while parked.

BIAS_DEG = 0.30            # what noisy_sim injects. Used ONLY to score, never as an input.


def drive(f, seconds, speed=8.0, yaw_rate=0.0, compass_bias_deg=BIAS_DEG,
          gnss=True, t0=100.0, jitter=None):
    """Run the filter down a straight track with a compass that reads `compass_bias_deg` high.

    GNSS is fed from the TRUE path, so the only thing the filter can blame a persistent
    compass residual on is the offset. That is the whole observability argument, in a loop.
    """
    t, x, yaw = t0, 0.0, 0.0
    bias = math.radians(compass_bias_deg)
    for i in range(int(seconds / 0.05)):
        t += 0.05
        yaw += yaw_rate * 0.05
        x += speed * 0.05
        f.set_speed(speed, False, t)
        f.predict_to(t, yaw_rate)
        noise = 0.0 if jitter is None else jitter(i)
        f.correct_heading(yaw + bias + noise, t)
        if gnss and i % 5 == 0:
            lat, lon = geo.to_latlon(x * math.cos(yaw), x * math.sin(yaw))
            f.correct_gnss(lat, lon, t)
    return f


def test_the_offset_starts_at_zero_and_unknown():
    """Not seeded from truth, not pre-loaded with the injected value: zero, and unsure."""
    f = ekf()
    assert f.heading_bias_rad == 0.0
    assert math.degrees(f.bias_sigma_rad) == pytest.approx(2.0, abs=1e-9)


def test_nothing_in_the_filter_knows_the_injected_offset():
    """The guard against the easiest way to fake this result. If 0.30 appears anywhere in the
    estimator as a constant, the experiment is measuring its own answer."""
    import inspect
    from warp_av.localization import ekf as mod
    src = inspect.getsource(mod)
    assert "0.30" not in src and "0.3)" not in src, "the filter must not carry the answer"


def test_driving_teaches_it_the_offset():
    """The point of the whole state. Compass reads 0.30 high, GNSS says where the van really
    went, and the difference between the two is the offset."""
    f = drive(ekf(speed=8.0), seconds=30.0)
    assert math.degrees(f.heading_bias_rad) == pytest.approx(BIAS_DEG, abs=0.05)


def test_learning_the_offset_is_what_removes_the_heading_error():
    """Converging on the offset is only interesting if the HEADING gets better for it."""
    f = drive(ekf(speed=8.0), seconds=30.0)
    assert abs(math.degrees(f.x[2])) < 0.02, "heading should land on the true zero"


def test_it_grows_more_sure_of_the_offset_by_driving():
    before = ekf().bias_sigma_rad
    after = drive(ekf(speed=8.0), seconds=30.0).bias_sigma_rad
    assert after < before / 4.0, "30 s of driving should sharpen the offset a lot"


def test_parked_it_learns_nothing_and_says_so():
    """Observability, stated as a test. Standing still there is no second opinion on heading,
    so offset and heading cannot be told apart -- and the filter must NOT pretend otherwise."""
    f = drive(ekf(speed=0.0), seconds=30.0, speed=0.0, gnss=True)
    assert math.degrees(f.bias_sigma_rad) > 1.0, "it may not claim to have learned an offset"


def test_the_compass_alone_only_constrains_the_SUM():
    """One equation, two unknowns. Whatever it does to heading and offset individually, their
    sum must move to meet the measurement -- and neither may be pinned by it alone."""
    f = ekf(speed=0.0)
    for i in range(50):
        f.correct_heading(math.radians(1.0), 100.0 + 0.05 * (i + 1))
    total = math.degrees(f.x[2] + f.x[3])
    assert total == pytest.approx(1.0, abs=0.05), "the sum must meet the measurement"
    assert math.degrees(f.bias_sigma_rad) > 1.0, "but the offset itself stays unknown"


def test_the_offset_cannot_jump_in_one_frame():
    """It is a slow-moving physical quantity, not a per-frame free parameter. A single wild
    reading must barely move it."""
    f = drive(ekf(speed=8.0), seconds=30.0)
    settled = f.heading_bias_rad
    f.correct_heading(math.radians(45.0), f.t + 0.05)       # one absurd reading
    assert abs(math.degrees(f.heading_bias_rad - settled)) < 0.5


def test_a_crawl_lets_gnss_move_x_and_y_but_neither_heading_state():
    """The L4 gate, extended. Zeroing only the yaw row would leave the same bad inference a
    door into the offset, where it would persist long after the van sped up."""
    f = ekf(speed=1.0)
    f.x[0] = 5.0
    f.x[2] = math.radians(3.0)
    f.x[3] = math.radians(0.2)
    yaw_before, bias_before = f.x[2], f.x[3]
    lat, lon = geo.to_latlon(0.0, 0.0)
    f.P[0, 2] = f.P[2, 0] = 0.05                  # real correlations for GNSS to pull on
    f.P[0, 3] = f.P[3, 0] = 0.05
    assert f.correct_gnss(lat, lon, 100.1) is True
    assert f.pose().x < 5.0, "position must still be corrected"
    assert f.x[2] == pytest.approx(yaw_before), "heading must not be rotated"
    assert f.x[3] == pytest.approx(bias_before), "nor may the offset be moved"


def test_above_the_gate_gnss_may_inform_the_offset_again():
    f = ekf(speed=8.0)
    f.x[0] = 5.0
    f.P[0, 3] = f.P[3, 0] = 0.05
    bias_before = f.x[3]
    lat, lon = geo.to_latlon(0.0, 0.0)
    f.correct_gnss(lat, lon, 100.1)
    assert f.x[3] != pytest.approx(bias_before), "at speed the coupling is legitimate"


def test_it_survives_a_noisy_compass_rather_than_chasing_it():
    """With jitter ten times the offset, it must still find the offset and not follow the noise."""
    rng = __import__("random").Random(12345)
    f = drive(ekf(speed=8.0), seconds=60.0,
              jitter=lambda i: rng.gauss(0.0, math.radians(3.0)))
    assert math.degrees(f.heading_bias_rad) == pytest.approx(BIAS_DEG, abs=0.15)


def test_the_offset_is_reported_so_a_run_can_be_scored():
    st = drive(ekf(speed=8.0), seconds=10.0).state()
    assert "heading_bias_deg" in st and "sigma_bias_deg" in st
    assert st["sigma_bias_deg"] < 2.0
