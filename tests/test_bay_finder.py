"""The lidar bay finder, proven on synthetic scenes before it meets CARLA."""
import math

import numpy as np

from warp_av.perception.bay_finder import (find_bays, fit_kerb, LIDAR_HEIGHT_M,
                                           SLOT_WID_M, KERB_GAP_M)

RNG = np.random.default_rng(3)


def scene(kerb_y=3.0, kerb_yaw_deg=0.0, cars=(), noise=0.02, ground=True):
    """Sensor-frame points (x fwd, y right, z up from the sensor): a road,
    a kerb line at y_right = kerb_y (+ slope), and parked-car boxes given as
    (along_centre, length) on the kerb side."""
    pts = []
    t = math.tan(math.radians(kerb_yaw_deg))
    if ground:
        gx = RNG.uniform(-15, 45, 3000); gy = RNG.uniform(-6, 9, 3000)
        pts.append(np.stack([gx, gy, np.full_like(gx, -LIDAR_HEIGHT_M) + RNG.normal(0, noise, 3000)], 1))
    kx = RNG.uniform(-15, 45, 600)
    ky = kerb_y + t * kx + RNG.normal(0, noise, 600)
    kz = -LIDAR_HEIGHT_M + RNG.uniform(0.05, 0.18, 600)
    pts.append(np.stack([kx, ky, kz], 1))
    for along, length in cars:
        n = 400
        cx = RNG.uniform(along - length / 2, along + length / 2, n)
        lat = RNG.uniform(0.3, 2.0, n)            # car body sits 0.3-2.0 m off the kerb, road side
        cy = kerb_y + t * cx - lat
        cz = -LIDAR_HEIGHT_M + RNG.uniform(0.4, 1.6, n)
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
