"""RC-2: the free-space STOP check must read the ground under the van's PATH, not straight ahead.

Recorded live (WAV-V15-BEND5, run v17_hold, 2026-09-17, Town10HD north-east bend): the van came
to rest at (99.27, 123.86) heading -47.8 deg, its nose on the rear of a Mustang parked on the
outside of the curve -- centre 9.4 m ahead and 0.9 m to the right, 3.96 m box to box -- while
its own planned path bent LEFT past that car with 0.86 m to spare. The straight strip from the
nose (2.95-7.95 m, body + 10 cm) held 9-13 of the car's solid squares every tick; the tracked
corridor was clear (the car sits 2.9 m off the route line); nothing else held the van. It stood
there 33 s and the mission was cancelled. Four V1 missions died at the same spot.

Numbers below are the recorded ones, in the van's frame (forward, right), metres."""
import math
from pathlib import Path

import numpy as np

from warp_av.perception.occupancy import OccupancyGrid, path_in_van_frame
from warp_av.planning.instrumentation import BLOCKED_OCCUPANCY, CLEAR
from warp_av.planning.planner import GROUND_KEEP_M, GROUND_LOOK_M, what_the_ground_says

NOSE_M = 2.95
HALF_M = 0.99 + GROUND_KEEP_M

#: the controller's path from the frozen pose (trajectory points, van frame), bending left
RECORDED_PATH = [(0.01, 0.0), (2.0, -0.19), (3.95, -0.63), (5.87, -1.18), (7.78, -1.78),
                 (9.68, -2.4), (11.57, -3.05), (13.45, -3.75)]
STRAIGHT_PATH = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)]


def car_returns(centre=(9.44, 0.93), yaw_deg=-12.2, half_len=2.36, half_wid=0.95):
    """Laser returns off a parked car's body, as it stood in the recording."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for u in np.arange(-half_len, half_len + 1e-6, 0.1):
        for v in np.arange(-half_wid, half_wid + 1e-6, 0.1):
            pts.append((centre[0] + u * c - v * s, centre[1] + u * s + v * c))
    return pts


def grid_with(solid=None, reach=25.0):
    """Open road seen to `reach` metres on every bearing, plus solid returns."""
    ang = np.radians(np.arange(0.0, 360.0, 0.25))
    ground = np.stack([reach * np.cos(ang), reach * np.sin(ang)], axis=1)
    pts, occ = [ground], [np.zeros(len(ground), dtype=bool)]
    if solid:
        pts.append(np.asarray(solid, dtype=float))
        occ.append(np.ones(len(solid), dtype=bool))
    return OccupancyGrid().update(np.concatenate(pts), np.concatenate(occ))


def read(grid, path):
    return grid.strip_along(path, NOSE_M, NOSE_M + GROUND_LOOK_M, HALF_M)


# ---- the recorded freeze ------------------------------------------------------------------

def test_the_recorded_bend_car_blocks_the_straight_strip_but_not_the_path():
    grid = grid_with(car_returns())
    straight = grid.strip_ahead(NOSE_M, NOSE_M + GROUND_LOOK_M, HALF_M)
    assert what_the_ground_says(straight) == BLOCKED_OCCUPANCY, "this is what froze the van"
    at_straight = grid.nearest_block_ahead(NOSE_M, NOSE_M + GROUND_LOOK_M, HALF_M)
    assert at_straight is not None and 6.5 <= at_straight <= 7.5      # recorded: 6.9 m

    counts, at = read(grid, RECORDED_PATH)
    assert counts[1] == 0 and at is None, "the car is beside the path, not on it"
    assert what_the_ground_says(counts) == CLEAR


def test_the_rc2_sequence_stop_stand_then_the_path_bends_away():
    """1. solid ground on the path -> a valid STOP; 2. the van stands still and the map is
    re-read; 3. the path from the standing pose bends away from the car; 4. the block clears;
    5. the ground says CLEAR, so the planner may say SLOW/CLEAR again. The car and the two
    paths are the recorded ones: straight on, the car's rear is on the path 7 m out."""
    car = car_returns()
    grid = grid_with(car)
    counts, at = read(grid, STRAIGHT_PATH)                     # 1. heading straight at it
    assert what_the_ground_says(counts) == BLOCKED_OCCUPANCY and at is not None and at < 7.5
    grid.update(np.asarray(car, dtype=float), np.ones(len(car), dtype=bool), moved=(0.0, 0.0, 0.0))
    counts, _ = read(grid, STRAIGHT_PATH)                      # 2. standing: still there
    assert what_the_ground_says(counts) == BLOCKED_OCCUPANCY
    counts, at = read(grid, RECORDED_PATH)                     # 3. the route turns left of it
    assert counts[1] == 0 and at is None                       # 4. nothing stale holds it
    assert what_the_ground_says(counts) == CLEAR               # 5.


def test_a_car_on_the_bending_path_still_stops_the_van():
    """We must not solve freezing by driving through what is actually on the path."""
    on_path = car_returns(centre=(7.78, -1.78), yaw_deg=-17.0)   # parked right on the path's 7.8 m point
    counts, at = read(grid_with(on_path), RECORDED_PATH)
    assert what_the_ground_says(counts) == BLOCKED_OCCUPANCY
    assert at is not None and 4.0 <= at <= 6.5


def test_a_straight_path_reads_the_same_ground_as_the_straight_strip():
    car = car_returns(centre=(7.0, 0.0), yaw_deg=0.0)
    grid = grid_with(car)
    counts, at = read(grid, STRAIGHT_PATH)
    straight = grid.strip_ahead(NOSE_M, NOSE_M + GROUND_LOOK_M, HALF_M)
    assert what_the_ground_says(counts) == what_the_ground_says(straight) == BLOCKED_OCCUPANCY
    assert abs(at - grid.nearest_block_ahead(NOSE_M, NOSE_M + GROUND_LOOK_M, HALF_M)) <= 0.3
    open_counts, open_at = read(grid_with(), STRAIGHT_PATH)
    assert what_the_ground_says(open_counts) == CLEAR and open_at is None


def test_unseen_ground_under_the_path_is_still_not_free():
    counts, at = read(grid_with(reach=5.0), RECORDED_PATH)      # the laser only saw 5 m out
    assert what_the_ground_says(counts) != CLEAR and at is None


def test_no_map_or_no_path_reads_nothing():
    assert OccupancyGrid().strip_along(RECORDED_PATH, NOSE_M, 7.95, HALF_M) is None
    grid = grid_with()
    assert grid.strip_along(None, NOSE_M, 7.95, HALF_M) is None
    assert grid.strip_along([(0.0, 0.0)], NOSE_M, 7.95, HALF_M) is None
    assert grid.strip_along([(6.0, 4.0), (12.0, 4.0)], NOSE_M, 7.95, HALF_M) is None, \
        "a path the van is not on (4 m off) is not read"


def test_the_frame_change_matches_the_recording():
    """World -> van frame, checked on the recorded pose and the Mustang's centre."""
    fwd, right = path_in_van_frame([(106.3, 117.5)], 99.267, 123.859, -0.8335)[0]
    assert abs(fwd - 9.44) < 0.05 and abs(right - 0.93) < 0.05


def test_the_stop_half_reads_the_path_and_falls_back_to_straight_ahead():
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text(encoding="utf-8")
    i = src.index("def _second_opinion_on_the_ground")
    body = src[i:i + 3500]
    assert "ground_under_the_path(" in body
    j = src.index("def ground_under_the_path")
    helper = src[j:j + 2000]
    assert "strip_along(" in helper and "path_in_van_frame(" in helper
    assert "strip_ahead(" in helper, "no path to read -> the straight strip, exactly as before"
