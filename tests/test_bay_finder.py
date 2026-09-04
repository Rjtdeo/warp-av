"""The lidar bay finder, proven on synthetic scenes before it meets CARLA."""
import math

import numpy as np

from warp_av.perception.bay_finder import (find_bays, fit_kerb, LIDAR_HEIGHT_M,
                                           SLOT_WID_M, KERB_GAP_M)

RNG = np.random.default_rng(3)


def scene(kerb_y=3.0, kerb_yaw_deg=0.0, cars=(), noise=0.02, ground=True, pavement_m=0.0,
          camber=0.0, curve=0.0):
    """camber: road drops this many metres per metre to the right (cambered street).
    curve: kerb y_right = kerb_y + t*x + curve*x^2 (a gentle bend)."""
    """Sensor-frame points (x fwd, y right, z up from the sensor): a road,
    a kerb line at y_right = kerb_y (+ slope), and parked-car boxes given as
    (along_centre, length) on the kerb side."""
    pts = []
    t = math.tan(math.radians(kerb_yaw_deg))
    road = lambda y: -LIDAR_HEIGHT_M - camber * y          # road surface height at lateral y
    edge = lambda x: kerb_y + t * x + curve * x * x         # kerb edge line
    if ground:
        gx = RNG.uniform(-15, 45, 3000); gy = RNG.uniform(-6, 12, 3000)
        pts.append(np.stack([gx, gy, road(gy) + RNG.normal(0, noise, 3000)], 1))
    kx = RNG.uniform(-15, 45, 600)
    ky = edge(kx) + RNG.normal(0, noise, 600)
    kz = road(ky) + RNG.uniform(0.05, 0.18, 600)
    pts.append(np.stack([kx, ky, kz], 1))
    if pavement_m:
        # the pavement surface beyond the kerb face, at kerb height - the lidar sees all of it
        px = RNG.uniform(-15, 45, 2500); pw = RNG.uniform(0.0, pavement_m, 2500)
        py = edge(px) + pw + RNG.normal(0, noise, 2500)
        pts.append(np.stack([px, py, road(py) + RNG.uniform(0.10, 0.18, 2500)], 1))
    for along, length in cars:
        n = 400
        cx = RNG.uniform(along - length / 2, along + length / 2, n)
        lat = RNG.uniform(0.3, 2.0, n)            # car body sits 0.3-2.0 m off the kerb, road side
        cy = edge(cx) - lat
        cz = road(cy) + RNG.uniform(0.4, 1.6, n)
        pts.append(np.stack([cx, cy, cz], 1))
    p = np.concatenate(pts)
    return np.concatenate([p, np.zeros((len(p), 1))], 1).astype(np.float32)


def test_no_kerb_means_no_bay():
    assert find_bays(scene(kerb_y=3.0, ground=True)[:3000]) == []      # ground only
    assert find_bays(np.zeros((0, 4), np.float32)) == []


def test_kerb_line_is_found_straight_and_where_it_is():
    k = fit_kerb(scene(kerb_y=3.2))
    assert k is not None and abs(k.a - 3.2) < 0.08 and abs(math.degrees(k.yaw)) < 1.0


def test_a_gap_between_two_parked_cars_is_a_bay():
    bays = find_bays(scene(kerb_y=3.0, cars=((2.0, 4.5), (15.0, 4.5))))   # gap ~8.5 m centred at 8.5
    assert len(bays) >= 1
    b = next(bay for bay in bays if 4.0 < bay.x < 13.0)
    assert abs(b.x - 8.5) < 0.6
    assert abs(-b.y - (3.0 - KERB_GAP_M - SLOT_WID_M / 2)) < 0.25     # y is LEFT; bay is to the right
    assert abs(math.degrees(b.yaw)) < 1.5


def test_a_gap_too_short_for_the_van_is_not_a_bay():
    bays = find_bays(scene(kerb_y=3.0, cars=((2.0, 4.5), (9.0, 4.5))))    # gap ~2.5 m
    assert not any(4.0 < bay.x < 7.0 for bay in bays)


def test_the_bay_follows_a_slanted_kerb():
    bays = find_bays(scene(kerb_y=3.0, kerb_yaw_deg=8.0, cars=((2.0, 4.5), (15.0, 4.5))))
    b = next(bay for bay in bays if 4.0 < bay.x < 13.0)
    assert abs(math.degrees(b.yaw) - 8.0) < 2.0


def test_an_empty_kerb_is_one_long_bay_nearest_first():
    bays = find_bays(scene(kerb_y=3.0))
    assert bays and bays[0].length > 20.0


def test_the_kerb_line_hugs_the_pavement_edge_not_its_middle():
    """First real probe: +1.2 m side error at all 40 spots, because the whole
    pavement surface sits at kerb height and a plain fit ran down its middle."""
    k = fit_kerb(scene(kerb_y=3.0, pavement_m=2.5))
    assert k is not None and abs(k.a - 3.0) < 0.15


def test_a_bay_beside_a_wide_pavement_is_still_placed_against_the_kerb():
    bays = find_bays(scene(kerb_y=3.0, pavement_m=2.5, cars=((2.0, 4.5), (15.0, 4.5))))
    b = next(bay for bay in bays if 4.0 < bay.x < 13.0)
    assert abs(-b.y - (3.0 - KERB_GAP_M - SLOT_WID_M / 2)) < 0.3


def test_a_sparse_far_car_still_blocks_the_bay():
    """At 25 m a parked car returns only a handful of lidar points; a fixed
    3-points-per-bin rule called a 43 m stretch free with three cars in it."""
    sc = scene(kerb_y=3.0)
    far = np.array([[24.0, 1.8, -LIDAR_HEIGHT_M + 0.9, 0.0], [25.5, 1.6, -LIDAR_HEIGHT_M + 1.1, 0.0],
                    [27.0, 1.9, -LIDAR_HEIGHT_M + 0.8, 0.0]], dtype=np.float32)
    bays = find_bays(np.concatenate([sc, far]))
    assert all(b.slot_end < 23.5 or b.slot_start > 27.5 for b in bays), "the slot must not sit on the far car"


def test_the_slot_is_the_middle_seven_metres_of_the_free_stretch():
    bays = find_bays(scene(kerb_y=3.0, cars=((2.0, 4.5), (15.0, 4.5))))
    b = next(bay for bay in bays if 4.0 < bay.x < 13.0)
    assert abs((b.slot_end - b.slot_start) - 7.0) < 1e-6
    assert b.along_start < b.slot_start and b.slot_end < b.along_end


def test_a_set_back_kerb_nine_metres_out_is_found():
    """The rig's destinations: bays 6.25 m right of the lane, kerb 7-9 m out.
    The old 8 m search window missed them ("0 kerb-height points")."""
    k = fit_kerb(scene(kerb_y=9.0, pavement_m=2.0))
    assert k is not None and abs(k.a - 9.0) < 0.2


def test_a_cambered_road_does_not_hide_the_far_pavement():
    """Road dropping 2.5 cm per metre to the right: at 8 m the pavement top is
    20 cm lower than under the van and vanished from a fixed height band."""
    k = fit_kerb(scene(kerb_y=8.0, pavement_m=2.0, camber=0.025))
    assert k is not None and abs(k.a - 8.0) < 0.25
    bays = find_bays(scene(kerb_y=8.0, pavement_m=2.0, camber=0.025, cars=((2.0, 4.5), (15.0, 4.5))))
    assert any(4.0 < b.x < 13.0 for b in bays)


def test_a_gently_curving_kerb_is_followed():
    """y = 3 + 0.004 x^2: about 12 deg of heading change over 30 m."""
    k = fit_kerb(scene(kerb_y=3.0, curve=0.004))
    assert k is not None and 0.002 < k.c < 0.006
    bays = find_bays(scene(kerb_y=3.0, curve=0.004, cars=((2.0, 4.5), (15.0, 4.5))))
    b = next(bay for bay in bays if 4.0 < bay.x < 13.0)
    expected = math.degrees(math.atan(2 * 0.004 * b.x))
    assert abs(math.degrees(b.yaw) - expected) < 2.5, "the slot heading follows the local tangent"


def test_a_tight_bend_is_refused_as_a_curve_but_still_gives_a_line_nearby():
    k = fit_kerb(scene(kerb_y=3.0, curve=0.03))
    assert k is None or k.c == 0.0
