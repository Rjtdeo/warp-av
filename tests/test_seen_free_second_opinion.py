"""The free-space map as a second opinion on what the corridor check says is in the way.

Live on 2026-09-11, one long run, two faults with one cause -- a body drawn on empty ground:

  * a line of street furniture beside a parking bay (the laser measured 6.3 x 0.84 m, the
    fit called it 8.0 x 2.3 m, the camera named it a vehicle) stopped the van 3.6 m short of
    its spot and failed the mission, while the laser's own map showed the ground ahead clear
    for 21 m;
  * a building's two faces at a corner, points spread 51 x 2.8 m, fitted as one 22.6 x 12.4 m
    block centred inside the building and reaching across the pavement, made the van give up
    the spot it had chosen;
  * and a "VEHICLE blocking path at 11.6 m" that CARLA says was never there held it 33 s.
"""
import math

import numpy as np

from warp_av.perception.occupancy import OccupancyGrid
from warp_av.perception.perception import DetectedObject, ObjectType
from warp_av.planning.planner import (SEEN_FREE_SHARE, fit_is_believable, obstacle_box_for,
                                      nothing_is_standing_there)


def thing(x, y, size, box=None, box_at=(0.0, 0.0), yaw=0.0, kind=ObjectType.VEHICLE):
    L, W, H = size
    bl, bw = box if box else (0.0, 0.0)
    return DetectedObject(object_type=kind, x=x, y=y, distance=math.hypot(x, y), id=7,
                          stationary=True, length_m=L, width_m=W, height_m=H, yaw_deg=yaw,
                          box_length_m=bl, box_width_m=bw, box_yaw_deg=yaw,
                          box_dx=box_at[0], box_dy=box_at[1])


# ---- a fitted rectangle may not claim ground the points never touched -------------------

def test_the_building_corner_fit_is_not_believed():
    """The live facade: 144 square metres of points, a 281 square metre "body"."""
    facade = thing(0.26, 7.45, (51.06, 2.82, 4.0), box=(22.64, 12.43), box_at=(-21.66, 3.96),
                   yaw=-40.0, kind=ObjectType.OBSTACLE)
    assert not fit_is_believable(facade)
    box = obstacle_box_for(facade, 0.0)
    assert 7.45 - box.half_width > 3.5               # its body stays off the road, 7.45 m away
    assert (box.dx, box.dy) == (0.0, 0.0)            # and centred on its points, not inside the building


def test_a_car_seen_from_its_corner_is_still_believed():
    """The 2026-09-10 parked car: the fit is what the planner should judge it by."""
    car = thing(2.6, 2.5, (4.3, 1.8, 1.5), box=(4.67, 2.0), box_at=(0.86, 0.34), yaw=-7.0)
    assert fit_is_believable(car)
    box = obstacle_box_for(car, 0.0)
    assert box.heading == math.radians(-7.0) and (box.dx, box.dy) == (0.86, 0.34)


def test_nothing_measured_still_has_no_box_at_all():
    """Never measured means the caller falls back to the circle, which is the cautious answer."""
    assert obstacle_box_for(thing(5.0, 0.0, (0.0, 0.0, 0.0)), 0.0) is None


def test_a_kerb_line_is_not_swung_across_the_road_by_a_heading_allowance():
    """A 3.5 x 0.1 m kerb strip: it can only tilt within its own thickness."""
    kerb = thing(8.0, 1.9, (3.5, 0.1, 0.15), kind=ObjectType.OBSTACLE)
    assert obstacle_box_for(kerb, 0.0).half_width < 0.35


# ---- the ground the van is about to cover -----------------------------------------------

def swept_grid(extra=None, reach=25.0):
    """A laser map around a van at the origin facing +x: every bearing seen to `reach` metres
    of open road, plus any solid returns (van frame: x forward, y right)."""
    ang = np.radians(np.arange(0.0, 360.0, 0.25))
    ground = np.stack([reach * np.cos(ang), reach * np.sin(ang)], axis=1)
    pts, occ = [ground], [np.zeros(len(ground), dtype=bool)]
    if extra is not None:
        pts.append(np.asarray(extra, dtype=float))
        occ.append(np.ones(len(extra), dtype=bool))
    return OccupancyGrid().update(np.concatenate(pts), np.concatenate(occ))


def strip(grid):
    return grid.strip_ahead(2.95, 2.95 + 7.0, 1.29)          # bumper to 7 m on, body plus margin


def test_open_road_ahead_is_seen_free():
    free, blocked, unseen = strip(swept_grid())
    assert blocked == 0 and free >= SEEN_FREE_SHARE * (free + unseen)


def test_a_real_car_ahead_is_not():
    car = [(x, y) for x in np.arange(7.0, 9.0, 0.1) for y in np.arange(-0.9, 0.9, 0.1)]
    free, blocked, unseen = strip(swept_grid(car))
    assert blocked > 0


def test_an_unseen_map_is_never_free():
    assert OccupancyGrid().strip_ahead(3.0, 10.0, 1.3) is None


# ---- when that overrules the corridor check ---------------------------------------------

def test_a_body_drawn_on_ground_seen_empty_does_not_stop_the_van():
    assert nothing_is_standing_there("vehicle", 0.0, strip(swept_grid()))


def test_but_a_single_blocked_square_does():
    assert not nothing_is_standing_there("vehicle", 0.0, (300, 1, 0))


def test_and_so_does_ground_the_laser_has_not_seen():
    assert not nothing_is_standing_there("vehicle", 0.0, (900, 0, 100))


def test_a_person_is_never_overruled():
    assert not nothing_is_standing_there("pedestrian", 0.0, strip(swept_grid()))
    assert not nothing_is_standing_there("cyclist", 0.0, strip(swept_grid()))


def test_nor_is_anything_moving():
    assert not nothing_is_standing_there("vehicle", 1.2, strip(swept_grid()))


def test_no_map_at_all_changes_nothing():
    assert not nothing_is_standing_there("vehicle", 0.0, None)


def test_the_van_asks_before_it_gives_up_a_parking_spot():
    """main.py must ask this of every corridor block, not only while parking."""
    from pathlib import Path
    src = (Path(__file__).parents[1] / "src" / "warp_av" / "main.py").read_text()
    i = src.index("filter_to_route_corridor(")
    assert "_unblock_if_the_ground_is_seen_free" in src[i:i + 900]
