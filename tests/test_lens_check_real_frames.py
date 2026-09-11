"""The "something on the lens" check, on REAL pictures from the van's front camera.

The day-14 tests feed it random noise, where every patch of the picture changes every frame.
A real picture is not like that: plain sky, and the far end of the road straight ahead,
hardly change while the van drives. On 2026-09-10 that made a clean lens read "broken" on
18-75 % of the daytime driving, and the safety supervisor stopped the van each time.
"""
from pathlib import Path

import numpy as np
import pytest

from warp_av.adapters import carla_sensor_adapter as A
from warp_av.adapters.carla_sensor_adapter import CAMERA_BAD_FRAMES, CarlaSensorAdapter

CLIPS = np.load(Path(__file__).parent / "fixtures" / "lens_check" / "clean_lens_town10.npz")


def bare():
    a = object.__new__(CarlaSensorAdapter)
    a._last_thumb = None
    a._camera_bad = 0
    a._camera_bad_why = ""
    return a


def full_size(thumb):
    """Back to an 800 x 600 BGRA picture whose every-eighth-pixel thumbnail is `thumb`."""
    big = np.repeat(np.repeat(thumb, 8, axis=0), 8, axis=1)
    return np.concatenate([big, np.full(big.shape[:2] + (1,), 255, np.uint8)], axis=2)


def with_drops(img, n):
    """The fault injector's drops (carla_sensor_adapter._on_camera, camera_drops)."""
    out = img.copy()
    h, w = out.shape[0], out.shape[1]
    for k in range(n):
        y = int(h * (0.15 + 0.22 * (k % 3)))
        x = int(w * (0.12 + 0.19 * (k % 4)))
        out[y:y + h // 5, x:x + w // 6] = 100 + 7 * k
    return out


def verdict(clip, drops=0):
    a = bare()
    for thumb in clip:
        a._check_the_picture(with_drops(full_size(thumb), drops))
    return a.camera_fault()


@pytest.mark.parametrize("name", ["clear_noon", "wet_noon"])
def test_these_clips_really_fooled_the_old_rule(name, monkeypatch):
    """Without the frozen floor the clips reproduce the false alarm -- so they test it."""
    monkeypatch.setattr(A, "TILE_FROZEN_BELOW", float("inf"))
    assert "on the lens" in verdict(CLIPS[name])


@pytest.mark.parametrize("name", ["clear_noon", "wet_noon"])
def test_plain_sky_is_not_something_on_the_lens(name):
    assert verdict(CLIPS[name]) == ""


@pytest.mark.parametrize("name", ["clear_noon", "wet_noon"])
def test_drops_on_a_real_picture_are_still_caught(name):
    assert "on the lens" in verdict(CLIPS[name], drops=5)


@pytest.mark.parametrize("name", ["clear_noon", "wet_noon"])
def test_one_drop_on_a_real_picture_does_not_slow_the_van(name):
    assert verdict(CLIPS[name], drops=1) == ""


def test_the_clips_are_long_enough_to_judge():
    assert len(CLIPS["clear_noon"]) > CAMERA_BAD_FRAMES + 1
