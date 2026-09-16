"""One pose source, an uncertainty that says what it means, and a heading fault.

Phase L1, 2026-09-16. Until now the van asked the simulator where it was in three
independent places (localization, the LiDAR sweep, perception's free-space map), and a Pose
carried no uncertainty at all. L0 had also shown that the one thing it could NOT measure was
heading error, because the fault injector could bend the van's idea of WHERE it was but not
of WHICH WAY IT FACED.

These tests pin the three things L1 added, and -- just as importantly -- pin that the pose
still comes out of CarlaTruthPoseSource exactly as it came out of the old code, because L1 is
a structural change that is supposed to move no numbers.
"""
import math
import types

import pytest

from warp_av.localization.localization import (FAULT_RAMP_S, LocalizationQuality,
                                               LocalizationSystem, Pose, PoseCovariance)
from warp_av.localization.pose_source import CarlaTruthPoseSource


def actor(x=1.0, y=2.0, z=0.5, yaw_deg=30.0, vx=3.0, vy=4.0, vz=0.0, velocity=True):
    """A stand-in for CARLA's vehicle actor: a transform, and usually a velocity."""
    loc = types.SimpleNamespace(x=x, y=y, z=z)
    rot = types.SimpleNamespace(yaw=yaw_deg, pitch=0.0, roll=0.0)
    tf = types.SimpleNamespace(location=loc, rotation=rot)
    a = types.SimpleNamespace(get_transform=lambda: tf)
    if velocity:
        a.get_velocity = lambda: types.SimpleNamespace(x=vx, y=vy, z=vz)
    return a


# ---- the source itself --------------------------------------------------------------------

def test_the_truth_source_reports_what_the_simulator_says():
    p = CarlaTruthPoseSource(actor()).pose()
    assert (p.x, p.y, p.z) == (1.0, 2.0, 0.5)
    assert p.yaw == pytest.approx(math.radians(30.0))
    assert p.speed == pytest.approx(5.0)          # 3-4-5
    assert p.healthy is True


def test_yaw_is_radians_not_degrees():
    """The one unit mistake that would quietly rotate the whole world."""
    assert CarlaTruthPoseSource(actor(yaw_deg=180.0)).pose().yaw == pytest.approx(math.pi)


def test_a_stand_in_without_a_velocity_is_a_standstill_not_a_crash():
    """The replay harness and several unit tests hand over a transform and nothing else."""
    p = CarlaTruthPoseSource(actor(velocity=False)).pose()
    assert p.healthy is True and p.speed == 0.0


def test_a_broken_actor_gives_an_unhealthy_pose_rather_than_raising():
    class Broken:
        def get_transform(self):
            raise RuntimeError("no rpc")
    p = CarlaTruthPoseSource(Broken()).pose()
    assert p.healthy is False and p.confidence == 0.0
    assert "POSE_SOURCE_ERROR" in p.reason
    assert p.cov.xx == float("inf"), "no pose is unbounded error, not zero error"


# ---- sensor_to_world: the capture-time question ------------------------------------------

def test_the_sweep_gets_the_pose_the_simulator_stamped_at_capture():
    m = [[1.0, 0, 0, 7.0], [0, 1.0, 0, 8.0], [0, 0, 1.0, 9.0], [0, 0, 0, 1.0]]
    stamped = types.SimpleNamespace(get_matrix=lambda: m)
    out = CarlaTruthPoseSource(actor()).sensor_to_world(123.4, stamped)
    assert out.shape == (4, 4)
    assert out[0, 3] == 7.0 and out[1, 3] == 8.0 and out[2, 3] == 9.0


def test_no_capture_pose_is_none_so_the_caller_can_fall_back():
    """None means 'I cannot place this delivery' -- the adapter then keeps the raw delivery
    rather than de-skewing it with a pose that does not belong to it."""
    assert CarlaTruthPoseSource(actor()).sensor_to_world(1.0, None) is None


def test_the_capture_pose_is_not_the_pose_now():
    """The whole reason the interface takes a timestamp. The van is at x=1; the sweep asks
    about a delivery captured when the sensor was at x=7, and must get 7."""
    src = CarlaTruthPoseSource(actor(x=1.0))
    m = [[1.0, 0, 0, 7.0], [0, 1.0, 0, 0.0], [0, 0, 1.0, 0.0], [0, 0, 0, 1.0]]
    got = src.sensor_to_world(99.0, types.SimpleNamespace(get_matrix=lambda: m))
    assert got[0, 3] == 7.0 and src.pose().x == 1.0


# ---- covariance ---------------------------------------------------------------------------

def test_truth_publishes_zero_variance_and_says_it_is_truth():
    c = CarlaTruthPoseSource(actor()).pose().cov
    assert (c.xx, c.yy, c.yaw) == (0.0, 0.0, 0.0)
    assert c.is_truth is True, "zeros must be readable as 'this is truth', not 'I am certain'"


def test_the_matrix_is_three_by_three_and_symmetric():
    c = PoseCovariance(xx=0.04, yy=0.09, yaw=0.01, xy=0.02, x_yaw=0.003, y_yaw=0.004)
    m = c.as_matrix()
    assert len(m) == 3 and all(len(r) == 3 for r in m)
    for i in range(3):
        for j in range(3):
            assert m[i][j] == m[j][i]


def test_sigma_is_the_square_root_of_the_variance():
    """Variance in, standard deviation out. Mixing these is the classic silent error, so
    the names are different and this test says which is which."""
    c = PoseCovariance(xx=0.25, yy=0.04, yaw=0.01)
    assert c.sigma_x == pytest.approx(0.5)      # metres
    assert c.sigma_y == pytest.approx(0.2)      # metres
    assert c.sigma_yaw == pytest.approx(0.1)    # radians


def test_a_default_pose_still_works_for_every_caller_that_never_heard_of_covariance():
    assert Pose().cov.xx == 0.0


# ---- the heading fault --------------------------------------------------------------------

def sys_at(yaw_deg=0.0, **kw):
    return LocalizationSystem(CarlaTruthPoseSource(actor(yaw_deg=yaw_deg, **kw)))


def test_by_default_there_is_no_heading_error():
    assert sys_at(yaw_deg=10.0).update().yaw == pytest.approx(math.radians(10.0))


@pytest.mark.parametrize("deg", [0.5, 1.0, 2.0, 5.0, -1.0, -5.0])
def test_a_yaw_fault_bends_the_heading_by_exactly_that_many_degrees(deg):
    s = sys_at(yaw_deg=10.0)
    s.inject_fault("yaw", deg=deg)
    assert math.degrees(s.update().yaw) == pytest.approx(10.0 + deg)


def test_a_yaw_fault_does_not_move_x_or_y():
    """The two faults have to be separable or neither measurement means anything."""
    s = sys_at(yaw_deg=10.0)
    before = s.update()
    s.inject_fault("yaw", deg=5.0)
    after = s.update()
    assert (after.x, after.y, after.z) == (before.x, before.y, before.z)


def test_a_position_fault_does_not_bend_the_heading():
    s = sys_at(yaw_deg=10.0)
    s.inject_fault("noise", offset_m=1.0)
    assert math.degrees(s.update().yaw) == pytest.approx(10.0)


def test_clearing_restores_the_baseline_exactly():
    s = sys_at(yaw_deg=10.0)
    base = s.update()
    s.inject_fault("yaw", deg=5.0)
    s.update()
    s.enable()                      # the only full reset the class offers
    back = s.update()
    assert back.yaw == pytest.approx(base.yaw)
    assert (back.x, back.y) == (base.x, base.y)


def test_the_heading_stays_inside_plus_or_minus_pi():
    """179 degrees plus 5 is -176, not 184: everything downstream does trigonometry on this."""
    s = sys_at(yaw_deg=179.0)
    s.inject_fault("yaw", deg=5.0)
    got = s.update().yaw
    assert -math.pi <= got <= math.pi
    assert math.degrees(got) == pytest.approx(-176.0)


def test_a_drifting_heading_starts_at_nothing_and_grows():
    s = sys_at(yaw_deg=0.0)
    s.inject_fault("yaw", deg=10.0, mode="drift")
    s._fault["yaw_t0"] = __import__("time").time()          # start the clock now
    assert abs(math.degrees(s.update().yaw)) < 1.0, "a drift has not arrived yet"
    s._fault["yaw_t0"] -= FAULT_RAMP_S                      # ...and a ramp later
    assert math.degrees(s.update().yaw) == pytest.approx(10.0, abs=0.2)


def test_a_drifting_heading_ramps_the_same_way_in_both_directions():
    """A negative angle must ramp like a positive one. (The POSITION drift does not -- its
    min() jumps straight to a negative offset. That is pre-existing and left alone in L1;
    this test exists so the new fault is not written with the same bug.)"""
    s = sys_at(yaw_deg=0.0)
    s.inject_fault("yaw", deg=-10.0, mode="drift")
    s._fault["yaw_t0"] = __import__("time").time() - FAULT_RAMP_S / 2.0
    assert math.degrees(s.update().yaw) == pytest.approx(-5.0, abs=0.5)


def test_an_unknown_mode_is_refused_rather_than_silently_accepted():
    s = sys_at()
    assert s.inject_fault("yaw", deg=1.0, mode="sideways") is False
    assert s.update().yaw == pytest.approx(0.0)


def test_the_older_faults_still_answer_the_same_way():
    """Backward compatibility: L1 must not have changed the fault surface L0 measured with."""
    s = sys_at()
    assert s.inject_fault("noise", offset_m=0.5) is True
    assert s.inject_fault("low_confidence", value=0.1) is True
    assert s.inject_fault("stale", age_s=2.0) is True
    assert s.inject_fault("freeze") is True
    assert s.inject_fault("nonsense") is False


def test_a_position_fault_still_moves_the_pose_sideways_by_exactly_the_offset():
    """The mechanism L0 measured with, pinned so a later phase cannot quietly change it."""
    s = sys_at(yaw_deg=0.0)
    before = s.update()
    s.inject_fault("noise", offset_m=1.0)
    after = s.update()
    assert math.hypot(after.x - before.x, after.y - before.y) == pytest.approx(1.0)


def test_confidence_is_still_one_while_the_pose_is_wrong():
    """L0's finding, kept visible: the van can be a metre out and still claim certainty.
    L1 deliberately does NOT fix this -- it only builds the place where a real estimator
    will put a real number."""
    s = sys_at()
    s.inject_fault("noise", offset_m=1.0)
    s.inject_fault("yaw", deg=5.0)
    p = s.update()
    assert p.confidence == 1.0
    assert p.quality is LocalizationQuality.GOOD
    assert p.cov.is_truth is True


# ---- the system still accepts a bare actor (sim/ demos do) --------------------------------

def test_a_raw_actor_is_wrapped_so_the_standalone_demos_still_run():
    s = LocalizationSystem(actor(x=4.0))
    assert s.update().x == 4.0
