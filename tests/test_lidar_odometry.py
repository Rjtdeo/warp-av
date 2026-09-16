"""How far the van turned, measured against the world instead of asked of a sensor.

Phase L5, 2026-09-16. The heading problem in L3/L4/L4.1 was never noise -- it was that the two
things that could speak about heading (GNSS course and a compass with an unknown offset) could
not be told apart. This is the third opinion, and these pin the properties that make it worth
having: that it is truth-free, that it measures rotation rather than assuming it, that moving
traffic does not drag it, and above all that it REFUSES when the geometry cannot support an
answer. A registration that quietly returns a confident wrong number is worse than none.
"""
import math

import numpy as np
import pytest

from warp_av.localization.lidar_odometry import (
    LidarOdometry, MIN_YAW_INFORMATION, prepare, register)


# ---- a world to drive through -------------------------------------------------------------

def town(n_walls=4, spacing=0.12, seed=3, poles=True):
    """A street: building faces, plus the posts, kerb stubs and parked shapes that give a real
    scene its texture.

    The texture matters and is not decoration. Point-to-point ICP finds the nearest point, and
    on a FEATURELESS straight wall the nearest point slides along the wall under a rotation --
    so a world made only of smooth lines under-reads the turn (pinned below in
    test_a_featureless_corridor_under_reads_the_turn). Real streets have corners and posts, the
    offline fixtures have them, and this scene has them.
    """
    rng = np.random.RandomState(seed)
    pts = []
    for (ax, ay, bx, by) in [(-25, -8, 25, -8), (-25, 8, 25, 8),
                             (-20, -8, -20, 8), (30, -8, 30, 8)][:n_walls]:
        n = int(math.hypot(bx - ax, by - ay) / spacing)
        f = np.linspace(0, 1, max(2, n))
        x = ax + (bx - ax) * f + rng.normal(0, 0.01, len(f))
        y = ay + (by - ay) * f + rng.normal(0, 0.01, len(f))
        z = rng.uniform(-1.5, 0.5, len(f))
        pts.append(np.c_[x, y, z, np.full(len(f), 0.5)])
    if poles:
        # parked cars along both kerbs. Boxes, because it is CORNERS that pin a rotation:
        # a corner can only sit one way, whereas a point on a smooth wall slides freely along
        # it and lets the fit under-read the turn.
        for (cx, cy) in [(6.0, -6.4), (14.0, 6.6), (-9.0, -6.5), (21.0, -6.6),
                         (-16.0, 6.7), (27.0, 6.3)]:
            for (ax, ay, bx, by) in [(-2.2, -0.8, 2.2, -0.8), (-2.2, 0.8, 2.2, 0.8),
                                     (-2.2, -0.8, -2.2, 0.8), (2.2, -0.8, 2.2, 0.8)]:
                n = max(2, int(math.hypot(bx - ax, by - ay) / spacing))
                f = np.linspace(0, 1, n)
                pts.append(np.c_[cx + ax + (bx - ax) * f + rng.normal(0, 0.01, n),
                                 cy + ay + (by - ay) * f + rng.normal(0, 0.01, n),
                                 rng.uniform(-1.4, 0.1, n), np.full(n, 0.5)])
    return np.concatenate(pts, axis=0)


def move(points, dyaw=0.0, dx=0.0, dy=0.0):
    """The same world as seen after the van moves by (dx, dy, dyaw) -- i.e. the inverse."""
    p = np.asarray(points, dtype=float).copy()
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    x, y = p[:, 0] - dx, p[:, 1] - dy
    p[:, 0] = c * x - s * y
    p[:, 1] = s * x + c * y
    return p


def run_pair(a, b, t0=100.0, dt=0.1):
    odo = LidarOdometry()
    assert odo.update(a, t0) is None, "the first sweep has nothing to register against"
    return odo, odo.update(b, t0 + dt)


# ---- does it measure what actually happened? ----------------------------------------------

@pytest.mark.parametrize("deg", [0.0, 0.5, 1.5, 3.0, -2.0])
def test_it_recovers_a_pure_rotation(deg):
    w = town()
    odo, m = run_pair(w, move(w, dyaw=math.radians(deg)))
    assert m.valid, m.reason
    assert m.delta_yaw_deg == pytest.approx(deg, abs=0.05)


@pytest.mark.parametrize("dx,dy", [(0.0, 0.0), (1.0, 0.0), (0.5, 0.3), (-0.8, 0.2)])
def test_it_recovers_a_pure_translation(dx, dy):
    w = town()
    odo, m = run_pair(w, move(w, dx=dx, dy=dy))
    assert m.valid, m.reason
    assert m.delta_x == pytest.approx(dx, abs=0.05)
    assert m.delta_y == pytest.approx(dy, abs=0.05)


def test_it_recovers_a_turn_while_driving():
    """The real case: a van at 10 m/s taking a 30 deg/s bend, over one sweep."""
    w = town()
    odo, m = run_pair(w, move(w, dyaw=math.radians(3.0), dx=1.0, dy=0.03))
    assert m.valid, m.reason
    assert m.delta_yaw_deg == pytest.approx(3.0, abs=0.05)
    assert m.delta_x == pytest.approx(1.0, abs=0.05)


def test_standing_still_reads_as_standing_still():
    """Parked, it must say zero -- and say it confidently, not refuse."""
    w = town()
    odo, m = run_pair(w, w.copy())
    assert m.valid, m.reason
    assert abs(m.delta_yaw_deg) < 0.01
    assert math.hypot(m.delta_x, m.delta_y) < 0.01


# ---- the thing that has to be true for the whole phase ------------------------------------

def test_it_takes_no_pose_no_transform_and_no_truth():
    """A source-level guard. The measurement must be computable from points and a clock; if any
    of these names can reach it, the registration is reading the simulator's own answer."""
    import inspect
    from warp_av.localization import lidar_odometry as mod
    src = inspect.getsource(mod)
    for banned in ("get_transform", "CarlaTruthPoseSource", "sensor_to_world",
                   "pose_source", "get_matrix", "import carla"):
        assert banned not in src, "%s must not appear in the odometry" % banned


def test_update_only_needs_points_and_a_timestamp():
    """The signature IS the contract: no pose argument exists to pass one through."""
    import inspect
    sig = inspect.signature(LidarOdometry.update)
    assert list(sig.parameters) == ["self", "points", "sim_time"]


# ---- moving traffic must not drag it ------------------------------------------------------

def test_a_passing_car_does_not_drag_the_answer():
    """No labels, no classifier: a minority of points that do not fit the rigid motion end up
    in the trimmed tail. This is the whole dynamic-object story."""
    w = town()
    turned = move(w, dyaw=math.radians(2.0))
    # a car-sized slab of points sweeping past at a completely different velocity
    rng = np.random.RandomState(11)
    car = np.c_[rng.uniform(4, 9, 120), rng.uniform(-3.0, -1.5, 120),
                rng.uniform(-1.2, 0.2, 120), np.full(120, 0.5)]
    car2 = car.copy()
    car2[:, 0] += 3.0                       # it moved 3 m relative to us in one sweep
    odo, m = run_pair(np.r_[w, car], np.r_[turned, car2])
    assert m.valid, m.reason
    assert m.delta_yaw_deg == pytest.approx(2.0, abs=0.1), "the static world must win"


# ---- and it must know when it cannot answer -----------------------------------------------

def test_a_bare_scene_is_refused_not_guessed():
    odo = LidarOdometry()
    thin = town()[:40]
    odo.update(thin, 100.0)
    m = odo.update(thin, 100.1)
    assert not m.valid and m.reason == "TOO_FEW_POINTS"


def test_a_featureless_corridor_refuses_instead_of_guessing():
    """The aperture problem, stated as a test, and the reason the information matrix is built
    from surface normals rather than from point counts.

    Between two long parallel walls and nothing else, sliding along the corridor changes the
    scan not at all -- along-track motion is genuinely unmeasurable. A point-to-point
    information matrix cannot see this (it credits every correspondence with pinning both
    axes) and the first build of this module duly reported 0.42 m of a 1.00 m step with four
    millimetres of claimed uncertainty. Built from the normals, the along-track direction is
    empty, and the answer is refused.
    """
    w = town(n_walls=2, poles=False)                 # two parallel walls, no ends, no cars
    odo, m = run_pair(w, move(w, dx=1.0))
    assert m.yaw_information < 0.01, "an unobservable direction must show as no information"
    assert not m.valid and m.reason == "WEAK_GEOMETRY"


def test_a_featureless_corridor_under_reads_the_turn():
    """The other half: with no corners to pin it, the fit is not merely uncertain, it is wrong.
    Worth pinning so nobody later reads the refusal above as excessive caution."""
    w = town(n_walls=2, poles=False)
    odo, m = run_pair(w, move(w, dyaw=math.radians(1.5)))
    assert abs(m.delta_yaw_deg - 1.5) > 0.05 or not m.valid


def test_an_informative_minority_is_not_thrown_away():
    """The lesson that replaced hard trimming with a robust weight.

    Down a corridor, the only points that can see an along-track error are the few facing the
    way you are going -- and those are exactly the points with the LARGEST residuals while the
    estimate is still wrong. Ranking by residual and discarding the tail removes them, and the
    fit then settles into the error it started with. The robust weight keeps them.
    """
    w = town()                                        # corridor plus parked cars: ends exist
    odo, m = run_pair(w, move(w, dx=1.0))
    assert m.valid, m.reason
    assert m.delta_x == pytest.approx(1.0, abs=0.05), "the minority that could see it must pull"


def test_a_scene_that_changed_completely_is_refused():
    odo = LidarOdometry()
    odo.update(town(seed=1), 100.0)
    m = odo.update(town(seed=99) + np.array([40.0, 40.0, 0.0, 0.0]), 100.1)
    assert not m.valid


def test_a_silly_time_step_is_refused():
    w = town()
    odo = LidarOdometry()
    odo.update(w, 100.0)
    assert odo.update(w, 100.0).reason == "BAD_DT"        # no time passed
    odo2 = LidarOdometry()
    odo2.update(w, 100.0)
    assert odo2.update(w, 105.0).reason == "BAD_DT"       # five seconds is not a sweep pair


def test_the_refusals_are_counted_and_reported():
    odo = LidarOdometry()
    thin = town()[:40]
    odo.update(thin, 100.0)
    odo.update(thin, 100.1)
    st = odo.state()
    assert st["attempts"] == 1 and st["valid"] == 0 and st["failure_rate"] == 1.0
    assert st["refusals"] == {"TOO_FEW_POINTS": 1}


# ---- the uncertainty has to mean something ------------------------------------------------

def test_it_reports_a_finite_positive_uncertainty():
    w = town()
    odo, m = run_pair(w, move(w, dyaw=math.radians(1.0)))
    assert m.cov is not None and np.isfinite(m.cov).all()
    assert all(m.cov[i, i] > 0 for i in range(3))
    assert np.allclose(m.cov, m.cov.T)
    assert 0.0 < math.degrees(m.sigma_yaw_rad) < 1.0


def test_a_richer_scene_is_claimed_more_confidently_than_a_sparse_one():
    """The uncertainty must respond to the evidence, not be a constant dressed up as one."""
    rich, sparse = town(n_walls=4), town(n_walls=2)
    _, mr = run_pair(rich, move(rich, dyaw=math.radians(1.0)))
    _, ms = run_pair(sparse, move(sparse, dyaw=math.radians(1.0)))
    assert mr.valid and ms.valid
    assert mr.yaw_information > ms.yaw_information


def test_the_yaw_information_floor_is_actually_reachable():
    """A guard on the constant itself: if MIN_YAW_INFORMATION were set below anything real, the
    weak-geometry gate would never fire and the test above would pass for the wrong reason."""
    w = town()
    _, m = run_pair(w, move(w, dyaw=math.radians(1.0)))
    assert m.yaw_information > MIN_YAW_INFORMATION


# ---- the preparation step -----------------------------------------------------------------

def test_the_road_and_the_sky_are_dropped_by_height():
    pts = np.array([[10.0, 0.0, -5.0, 0.5],      # below the band: road
                    [10.0, 1.0, 3.0, 0.5],       # above the band: wires, foliage
                    [10.0, 2.0, 0.0, 0.5],       # in the band: a wall
                    [1.0, 0.0, 0.0, 0.5],        # too near: the van's own body
                    [90.0, 0.0, 0.0, 0.5]])      # too far: too sparse to correspond
    out = prepare(pts)
    assert out.shape == (1, 2)
    assert out[0] == pytest.approx([10.0, 2.0])


def test_registration_of_nothing_says_nothing():
    got = register(np.zeros((0, 2)), np.zeros((0, 2)))
    assert got["correspondences"] == 0 and not got["converged"]
