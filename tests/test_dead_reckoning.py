"""The first estimator: where the van would think it is from wheels and a gyro alone.

Phase L2, 2026-09-16. These tests pin the arithmetic, because every later phase is measured
against how fast this drifts, and a drift number produced by a broken integrator is worse than
no number at all.

The turn input is the sensor adapter's RUNNING TOTAL of how far the van has rotated, not this
step's turn. That is on purpose: the IMU arrives at 20 Hz and the driving loop runs at 9-10 Hz,
so passing a rate would throw away every other sample. Several tests below exist only to make
sure the difference-since-last-look is what gets used.
"""
import math

import pytest

from warp_av.localization.dead_reckoning import DeadReckoning, MAX_STEP_S


def dr(x=0.0, y=0.0, yaw=0.0, t=100.0, turn=0.0):
    d = DeadReckoning()
    d.seed(x, y, yaw, now=t, turn_rad=turn)
    return d


def drive(d, speed, seconds, turn_to=None, reverse=False, t0=100.0, step=0.1):
    """Run the estimator the way the stack does: small steps at roughly loop rate.

    Steps are 0.1 s because that is what a 9-10 Hz loop gives it, and because anything at or
    beyond MAX_STEP_S is deliberately thrown away as a stall rather than integrated.
    `turn_to` is the RUNNING TOTAL to reach by the end, ramped evenly across the steps.
    """
    n = max(1, int(round(seconds / step)))
    start = d._last_turn if d._last_turn is not None else 0.0
    out = None
    for i in range(n):
        turn = None if turn_to is None else start + (turn_to - start) * (i + 1) / n
        out = d.update(speed, reverse, turn_rad=turn, now=t0 + step * (i + 1))
    return out


# ---- before it is told where it starts ----------------------------------------------------

def test_it_says_nothing_until_it_is_seeded():
    d = DeadReckoning()
    assert d.update(speed_mps=5.0, reverse=False, turn_rad=0.0, now=1.0) is None
    assert d.pose() is None
    assert d.state() == {"seeded": False}


def test_forgetting_stops_it_again():
    d = dr()
    d.update(5.0, False, 0.0, now=101.0)
    d.forget()
    assert d.update(5.0, False, 0.0, now=102.0) is None


# ---- driving in a straight line -----------------------------------------------------------

def test_a_straight_second_at_five_metres_a_second_moves_five_metres():
    d = dr()
    p = drive(d, 5.0, seconds=1.0, turn_to=0.0)
    assert p.x == pytest.approx(5.0)
    assert p.y == pytest.approx(0.0)
    assert d.distance_m == pytest.approx(5.0)
    assert d.elapsed_s == pytest.approx(1.0)


def test_it_drives_along_its_heading_not_along_x():
    d = dr(yaw=math.radians(90.0))
    p = drive(d, 4.0, seconds=1.0, turn_to=0.0)
    assert p.x == pytest.approx(0.0, abs=1e-9)
    assert p.y == pytest.approx(4.0)


def test_reverse_goes_backwards_and_still_counts_as_distance():
    d = dr()
    p = drive(d, 3.0, seconds=1.0, turn_to=0.0, reverse=True)
    assert p.x == pytest.approx(-3.0)
    assert d.distance_m == pytest.approx(3.0), "distance travelled, not displacement"


def test_a_negative_speed_reading_does_not_double_the_reverse():
    """A speedometer has no sign; the gear does. Handing in both must not cancel out."""
    d = dr()
    p = drive(d, -3.0, seconds=1.0, turn_to=0.0, reverse=True)
    assert p.x == pytest.approx(-3.0)


def test_standing_still_goes_nowhere():
    d = dr()
    p = drive(d, 0.0, seconds=1.0, turn_to=0.0)
    assert (p.x, p.y) == (0.0, 0.0)
    assert d.distance_m == 0.0


# ---- turning --------------------------------------------------------------------------------

def test_the_turn_input_is_a_running_total_not_this_step_s_turn():
    d = dr(turn=10.0)                      # the adapter has already summed 10 rad since boot
    p = d.update(0.0, False, turn_rad=10.5, now=100.1)
    assert math.degrees(p.yaw) == pytest.approx(math.degrees(0.5)), "only the 0.5 rad is new"
    p = d.update(0.0, False, turn_rad=10.5, now=100.2)
    assert math.degrees(p.yaw) == pytest.approx(math.degrees(0.5)), "no new turn, no new yaw"


def test_a_turn_with_no_speed_rotates_on_the_spot():
    d = dr()
    p = d.update(0.0, False, turn_rad=math.radians(30.0), now=100.1)
    assert math.degrees(p.yaw) == pytest.approx(30.0)
    assert (p.x, p.y) == (0.0, 0.0)


def test_heading_stays_inside_plus_or_minus_pi():
    d = dr(yaw=math.radians(179.0))
    p = d.update(0.0, False, turn_rad=math.radians(5.0), now=100.1)
    assert -math.pi <= p.yaw <= math.pi
    assert math.degrees(p.yaw) == pytest.approx(-176.0)


def test_a_quarter_circle_lands_where_the_arc_says_it_should():
    """The reason position is integrated along the AVERAGE heading of a step rather than the
    old or the new one. Driving a 90 degree bend of radius 10 m in 20 steps should land on the
    arc's end, not several centimetres inside it."""
    R, V = 10.0, 5.0
    total = math.pi / 2.0                       # radians of turn
    arc = R * total
    steps, dt = 20, (arc / V) / 20.0
    d = dr()
    turn = 0.0
    for i in range(steps):
        turn += total / steps
        d.update(V, False, turn_rad=turn, now=100.0 + dt * (i + 1))
    # a quarter circle of radius R starting at the origin heading +x ends at (R, R)
    assert d.x == pytest.approx(R, abs=0.05)
    assert d.y == pytest.approx(R, abs=0.05)
    assert math.degrees(d.yaw) == pytest.approx(90.0)


def test_turning_the_other_way_is_symmetric():
    d = dr()
    drive(d, 5.0, seconds=1.0, turn_to=math.radians(-45.0))
    assert math.degrees(d.yaw) == pytest.approx(-45.0)
    assert d.y < 0.0


# ---- steps that should not be integrated ---------------------------------------------------

def test_a_long_gap_is_dropped_rather_than_guessed_through():
    """A stalled loop or a paused simulator must not become a straight line through whatever
    really happened."""
    d = dr()
    before = (d.x, d.y)
    d.update(10.0, False, 0.0, now=100.0 + MAX_STEP_S + 0.1)
    assert (d.x, d.y) == before
    assert d.skipped == 1


def test_the_turn_total_is_rebased_across_a_dropped_step():
    """Otherwise the whole of a long gap's rotation lands in one lump on the next step."""
    d = dr(turn=0.0)
    d.update(0.0, False, turn_rad=1.0, now=100.0 + MAX_STEP_S + 0.1)   # dropped
    d.update(0.0, False, turn_rad=1.1, now=100.0 + MAX_STEP_S + 0.2)   # only 0.1 is new
    assert d.yaw == pytest.approx(0.1)


def test_time_going_backwards_is_dropped():
    d = dr(t=100.0)
    d.update(5.0, False, 0.0, now=99.0)
    assert (d.x, d.y) == (0.0, 0.0)


def test_no_gyro_at_all_still_dead_reckons_in_a_straight_line():
    """A dead sensor should degrade, not crash."""
    d = dr()
    p = drive(d, 5.0, seconds=1.0, turn_to=None)
    assert p.x == pytest.approx(5.0) and p.yaw == 0.0


# ---- seeding ---------------------------------------------------------------------------------

def test_seeding_again_forgets_everything_that_came_before():
    d = dr()
    drive(d, 5.0, seconds=1.0, turn_to=0.0)
    d.seed(50.0, 60.0, math.radians(90.0), now=200.0, turn_rad=7.0)
    assert (d.x, d.y) == (50.0, 60.0)
    assert d.distance_m == 0.0 and d.elapsed_s == 0.0 and d.steps == 0
    p = drive(d, 2.0, seconds=1.0, turn_to=7.0, t0=200.0)
    assert p.y == pytest.approx(62.0), "moving along the NEW heading from the NEW place"


# ---- scoring ----------------------------------------------------------------------------------

def test_the_error_splits_into_along_track_and_cross_track():
    """A metre late and a metre sideways are not the same mistake, and the estimator has to be
    able to say which it made."""
    from warp_av.localization.localization import Pose
    d = dr()
    drive(d, 10.0, seconds=1.0, turn_to=0.0)       # estimate is at (10, 0)
    truth = Pose(x=9.0, y=0.5, yaw=0.0)            # really 1 m further back and 0.5 m left
    e = d.error_against(truth)
    assert e["along_m"] == pytest.approx(1.0)
    assert e["cross_m"] == pytest.approx(-0.5)
    assert e["error_m"] == pytest.approx(math.hypot(1.0, 0.5))


def test_the_error_is_measured_in_the_true_heading_s_frame():
    from warp_av.localization.localization import Pose
    d = dr(yaw=math.radians(90.0))
    drive(d, 10.0, seconds=1.0, turn_to=0.0)       # estimate at (0, 10)
    truth = Pose(x=0.0, y=9.0, yaw=math.radians(90.0))
    e = d.error_against(truth)
    assert e["along_m"] == pytest.approx(1.0), "1 m ahead ALONG a northward heading"
    assert e["cross_m"] == pytest.approx(0.0, abs=1e-9)


def test_error_per_hundred_metres_waits_until_it_has_gone_somewhere():
    from warp_av.localization.localization import Pose
    d = dr()
    drive(d, 0.5, seconds=1.0, turn_to=0.0)
    assert d.error_against(Pose())["error_per_100m"] is None, "half a metre proves nothing"


def test_yaw_error_is_reported_in_degrees_and_wrapped():
    from warp_av.localization.localization import Pose
    d = dr(yaw=math.radians(179.0))
    e = d.error_against(Pose(yaw=math.radians(-179.0)))
    assert e["yaw_err_deg"] == pytest.approx(-2.0), "2 degrees apart across the wrap, not 358"


# ---- what it publishes -------------------------------------------------------------------------

def test_the_covariance_grows_with_distance_at_the_measured_rate():
    """Set from the L2 runs: 1.17 % of distance travelled, which is the worst of three scored
    drives. Standard deviation proportional to distance, so variance as distance squared."""
    from warp_av.localization.dead_reckoning import DRIFT_SIGMA_PER_M
    d = dr()
    drive(d, 10.0, seconds=10.0, turn_to=0.0)          # 100 m
    assert d.distance_m == pytest.approx(100.0)
    c = d.pose().cov
    assert c.sigma_x == pytest.approx(DRIFT_SIGMA_PER_M * 100.0), "1.17 m of sigma at 100 m"
    assert c.is_truth is False, "an estimate is never ground truth"


def test_the_uncertainty_is_linear_in_distance_not_in_its_square_root():
    """The shape matters more than the constant: a square-root growth would say the estimate
    gets relatively BETTER the further it goes, which is the opposite of what happens."""
    a, b = dr(), dr()
    drive(a, 10.0, seconds=10.0, turn_to=0.0)          # 100 m
    drive(b, 10.0, seconds=20.0, turn_to=0.0)          # 200 m
    assert b.pose().cov.sigma_x == pytest.approx(2.0 * a.pose().cov.sigma_x, rel=1e-6)


def test_an_estimate_that_has_not_moved_claims_no_error_yet():
    d = dr()
    assert d.pose().cov.sigma_x == 0.0
    assert d.pose().cov.is_truth is False


def test_the_published_state_says_how_far_and_how_long():
    d = dr()
    drive(d, 5.0, seconds=1.0, turn_to=math.radians(10.0))
    st = d.state()
    assert st["seeded"] is True
    assert st["distance_m"] == pytest.approx(5.0)
    assert st["elapsed_s"] == pytest.approx(1.0)
    assert st["yaw_deg"] == pytest.approx(10.0)
    assert st["steps"] == 10, "ten loop ticks in a second, not one"
    assert st["skipped"] == 0


def test_it_never_claims_to_be_the_truth():
    d = dr()
    p = drive(d, 5.0, seconds=1.0, turn_to=0.0)
    assert p.reason == "DEAD_RECKONING"
    assert p.cov.is_truth is False
