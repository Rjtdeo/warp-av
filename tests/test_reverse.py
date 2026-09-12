"""Backing out of somewhere there is no way forward from.

The van had no reverse gear at all: the parking failure even said so -- "a fixed object blocks
the way in (no reverse gear)" -- and a spot it had turned into and could not reach ended the
mission where the van happened to be standing.

Reverse is deliberately the smallest thing that gets it out: straight back, slowly, a few
metres, and only over ground the laser has SEEN empty behind.
"""
import math
from pathlib import Path

from warp_av.control.controller import VehicleController
from warp_av.vehicle_interface import GearState
from warp_av.behavior import transitions as T


def van():
    c = VehicleController()
    c.enable()
    return c


def command(desired, speed=0.0, stop=False):
    return van().compute_command(current_x=0.0, current_y=0.0, current_yaw=0.0,
                                 current_speed=speed, target_x=10.0, target_y=0.0,
                                 desired_speed=desired, should_stop=stop)


def test_a_negative_wanted_speed_puts_it_in_reverse():
    cmd = command(-0.8)
    assert cmd.gear == GearState.REVERSE and cmd.throttle > 0.0 and cmd.brake == 0.0


def test_backing_out_is_straight_and_slow():
    assert command(-0.8).steering == 0.0, "nothing can be swung into on the way out"
    fast = command(-5.0)
    assert fast.throttle <= VehicleController.REVERSE_THROTTLE
    assert command(-0.8, speed=VehicleController.REVERSE_MAX_MPS + 0.5).throttle == 0.0


def test_a_stop_still_beats_it():
    cmd = command(-0.8, stop=True)
    assert cmd.gear != GearState.REVERSE and cmd.brake == 1.0


def test_going_forwards_is_untouched():
    cmd = command(4.0)
    assert cmd.gear == GearState.DRIVE


def test_the_van_only_backs_over_ground_it_has_seen_empty():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("def _rear_is_clear")
    body = src[i:i + 1200]
    assert "grid.strip_ahead(-(rear + self.REVERSE_LOOK_M)" in body
    assert "what_the_ground_says(counts) == CLEAR" in body


def test_it_stops_the_moment_anything_is_wrong():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("def _keep_backing_out")
    body = src[i:i + 2200]
    assert "gone >= self.REVERSE_MAX_M" in body
    assert "REVERSE_TIMEOUT_S" in body
    assert "not self._rear_is_clear(pose)" in body, "it checks behind every tick, not once"


def test_a_spot_it_cannot_reach_is_backed_out_of_before_the_mission_is_failed():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("def _maybe_give_up_on_the_spot")
    body = src[i:i + 4000]
    assert "self._start_backing_out(pose" in body
    assert '"rechoose_parking"' in body
    assert "_backed_out_for" in body, "and it only tries that once per spot"
    assert "no reverse gear" not in src, "that excuse is gone"
    assert T.REVERSING in T.ALL_WHY and T.BACKED_OUT in T.ALL_MOVES
