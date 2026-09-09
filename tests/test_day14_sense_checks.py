"""Perception V2 day 14: a sense that is broken, not merely missing.

Before this the van asked one question of each sense: is data still arriving? A frozen
camera, a lens under a carrier bag, a laser with half its beams dead -- all of them keep
data arriving perfectly, and all of them read as healthy while the van drove on.

Every limit is set from the van's own measurements, taken in six weathers on 2026-09-09.
The worst HEALTHY readings were: brightness 27 (night), contrast 30 (rain), frame-to-frame
change 0.49 (fog, parked), and 32 of 32 beams every single time. The tests below use those
real numbers, so a limit can never creep up until bad weather trips it.
"""
import numpy as np
import pytest

from warp_av.adapters.carla_sensor_adapter import (
    CarlaSensorAdapter, CAMERA_MIN_BRIGHTNESS, CAMERA_MIN_CONTRAST, CAMERA_MIN_CHANGE,
    CAMERA_BAD_FRAMES, LIDAR_MIN_BEAMS, LIDAR_POINT_COLLAPSE, LIDAR_BAD_SCANS)


def bare():
    """The checks only, without a simulator behind them."""
    a = object.__new__(CarlaSensorAdapter)
    a._last_thumb = None
    a._camera_bad = 0
    a._camera_bad_why = ""
    a._lidar_bad = 0
    a._lidar_bad_why = ""
    a._point_history = []
    return a


def picture(brightness=120, contrast=50, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(brightness, contrast, size=(600, 800, 4))
    return np.clip(img, 0, 255).astype(np.uint8)


def feed(a, frames):
    for f in frames:
        a._check_the_picture(f)
    return a.camera_fault()


# ---------------------------------------------------------------- the camera


def test_a_normal_picture_is_fine():
    assert feed(bare(), [picture(seed=i) for i in range(10)]) == ""


def test_a_black_picture_is_a_covered_lens():
    a = bare()
    assert feed(a, [picture(brightness=1, contrast=1, seed=i) for i in range(CAMERA_BAD_FRAMES)])
    assert "black" in a.camera_fault()


def test_a_blank_picture_is_a_blocked_lens():
    """Bright but featureless -- a carrier bag, or pointed at a white wall."""
    a = bare()
    assert feed(a, [picture(brightness=200, contrast=1, seed=i) for i in range(CAMERA_BAD_FRAMES)])
    assert "blank" in a.camera_fault()


def test_the_same_picture_over_and_over_is_a_frozen_camera():
    a = bare()
    frozen = picture(seed=7)
    assert feed(a, [frozen] * (CAMERA_BAD_FRAMES + 1))
    assert "frozen" in a.camera_fault()


def test_one_bad_frame_never_slows_the_van():
    a = bare()
    a._check_the_picture(picture(seed=1))
    for _ in range(CAMERA_BAD_FRAMES - 1):
        a._check_the_picture(picture(brightness=1, contrast=1))
    assert a.camera_fault() == "", "a fault must persist before it counts"


def test_it_clears_the_moment_the_picture_comes_back():
    a = bare()
    feed(a, [picture(brightness=1, contrast=1, seed=i) for i in range(CAMERA_BAD_FRAMES + 2)])
    assert a.camera_fault() != ""
    a._check_the_picture(picture(seed=99))
    assert a.camera_fault() == ""


@pytest.mark.parametrize("weather,brightness,contrast", [
    ("bright noon", 121, 48), ("overcast", 92, 37), ("heavy rain", 107, 28),
    ("thick fog", 140, 34), ("dusk", 34, 24), ("night", 19, 21),
    # the worst the camera can be while still working, all pushed together
    ("night + storm", 22, 16),
    ("night + storm + fog", 22, 14),
    ("no moon, thickest fog", 22, 13),
])
def test_real_bad_weather_never_trips_it(weather, brightness, contrast):
    """The actual figures measured on the van. None of these is a fault."""
    assert brightness > CAMERA_MIN_BRIGHTNESS, weather
    assert contrast > CAMERA_MIN_CONTRAST, weather
    a = bare()
    assert feed(a, [picture(brightness, contrast, seed=i) for i in range(12)]) == "", weather


def test_the_worst_weather_has_room_to_spare():
    """The real worst figures, measured with no moon, 100 % rain and thickest fog together.

    An earlier version of this test used 27 and 30, which came from testing night and rain
    SEPARATELY. Pushed together the camera goes darker and flatter than either alone --
    contrast falls to 13.9, not 30 -- so the earlier margins were overstated.
    """
    WORST_BRIGHTNESS, WORST_CONTRAST, WORST_CHANGE = 19.5, 13.9, 0.48
    assert CAMERA_MIN_BRIGHTNESS <= WORST_BRIGHTNESS / 2.0
    assert CAMERA_MIN_CONTRAST <= WORST_CONTRAST / 2.0
    assert CAMERA_MIN_CHANGE <= WORST_CHANGE / 2.0


# ---------------------------------------------------------------- the laser


def scan(n_points=6800, beams=32):
    pts = np.zeros((n_points, 5), dtype=np.float32)
    pts[:, 4] = np.arange(n_points) % max(1, beams)
    return pts


def feed_lidar(a, scans):
    for s in scans:
        a._check_the_laser(s)
    return a.lidar_fault()


def test_a_healthy_laser_is_fine():
    assert feed_lidar(bare(), [scan() for _ in range(20)]) == ""


def test_losing_beams_is_a_broken_laser():
    a = bare()
    feed_lidar(a, [scan() for _ in range(12)])
    assert feed_lidar(a, [scan(beams=LIDAR_MIN_BEAMS - 4) for _ in range(LIDAR_BAD_SCANS)])
    assert "beams" in a.lidar_fault()


def test_points_falling_off_a_cliff_is_a_fault():
    a = bare()
    feed_lidar(a, [scan(6800) for _ in range(12)])
    collapsed = int(6800 * LIDAR_POINT_COLLAPSE * 0.5)
    assert feed_lidar(a, [scan(collapsed) for _ in range(LIDAR_BAD_SCANS)])
    assert "collapsed" in a.lidar_fault()


def test_the_normal_wobble_in_point_count_is_not_a_fault():
    """Measured live: 6,692 to 6,912 points a turn. That is not a collapse."""
    a = bare()
    counts = [6692, 6800, 6912, 6750, 6880, 6700, 6905, 6733, 6812, 6690, 6858, 6721]
    assert feed_lidar(a, [scan(n) for n in counts]) == ""


def test_it_says_nothing_until_it_knows_what_normal_is():
    """A cold start must not report a collapse on its first few turns."""
    a = bare()
    assert feed_lidar(a, [scan(500) for _ in range(4)]) == ""


# ---------------------------------------------------------------- something on the glass

from warp_av.adapters.carla_sensor_adapter import (TILE_ROWS, TILE_COLS, MAX_DEAD_TILES,
                                                   SCENE_MOVING_FLOOR)


def moving_scene(seed):
    """A picture of a world going past: every part of it changes frame to frame."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 45, size=(600, 800, 4)), 0, 255).astype(np.uint8)


def with_droplets(img, n):
    """Patches STUCK to the glass: the same grey every frame, whatever is behind them."""
    out = img.copy()
    spots = [(60, 100), (300, 420), (420, 60), (60, 620), (300, 60), (420, 420)]
    for k in range(n):
        y, x = spots[k % len(spots)]
        # about 140 px across, the size the live test used -- a 60 px speck covers less
        # than one tile and rightly does not count
        out[y:y + 140, x:x + 140] = 100 + k      # fixed, so it never changes
    return out


def test_a_clean_lens_reports_nothing_on_it():
    a = bare()
    for i in range(12):
        a._check_the_picture(moving_scene(i))
    assert a.camera_blocked_tiles == 0
    assert a.camera_fault() == ""


def test_drops_stuck_to_the_lens_are_noticed():
    """The rain case: drops build up, block patches, and every other check still passes."""
    a = bare()
    for i in range(CAMERA_BAD_FRAMES + 4):
        a._check_the_picture(with_droplets(moving_scene(i), 5))
    assert a.camera_blocked_tiles > MAX_DEAD_TILES
    assert "on the lens" in a.camera_fault()


def test_the_other_checks_would_have_missed_it():
    """Proof that this check earns its place: a drop-covered picture is bright, has
    contrast, and is still changing. All three earlier checks pass it."""
    a = bare()
    img = with_droplets(moving_scene(3), 5)
    a._check_the_picture(with_droplets(moving_scene(2), 5))
    a._check_the_picture(img)
    assert a.camera_brightness > CAMERA_MIN_BRIGHTNESS
    assert a.camera_contrast > CAMERA_MIN_CONTRAST
    assert a.camera_change > CAMERA_MIN_CHANGE


def test_a_perfectly_still_picture_gets_no_opinion():
    """If NOTHING at all is moving, every patch looks stuck and the answer would be
    nonsense, so the check says nothing.

    Note what this does and does not mean. On the real van, PARKED, it turns out there is
    still enough going on -- passing traffic, changing light, sensor noise -- that the check
    does judge, and judges correctly: blocking the lens while parked was caught, and thirty
    seconds parked with a CLEAN lens gave 0 false alarms in 15 readings (measured live,
    2026-09-09). So this guard is for the truly frozen case, not for "the van is stopped".
    """
    a = bare()
    still = moving_scene(1)
    for _ in range(CAMERA_BAD_FRAMES + 4):
        a._check_the_picture(still + np.uint8(0))
    assert a.camera_blocked_tiles == 0, "a completely frozen picture deserves no opinion"


def test_a_single_raindrop_does_not_hold_the_van_to_walking_pace():
    """Measured: one droplet freezes about 4 tiles, four droplets about 6. The bar sits
    above one on purpose -- crying wolf here means a van that crawls in drizzle."""
    a = bare()
    for i in range(12):
        a._check_the_picture(with_droplets(moving_scene(i), 1))
    assert a.camera_fault() == ""
