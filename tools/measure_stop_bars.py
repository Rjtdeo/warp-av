#!/usr/bin/env python3
"""
Measure where the PAINTED stop bar is on every traffic light's approach lane, from overhead
pictures, and write it down as the answer key for red-light runs.

    python tools/measure_stop_bars.py [--pictures DIR]      # the stack must be DOWN

Why: CARLA's map has no painted stop bars. `get_stop_waypoints()` is the centre of a light's
trigger box, 1 to 7.8 m short of the paint on Town10HD, so scoring against it calls a correct
stop a red-light run. The van itself stops at a line worked out from the map
(traffic_lights.stop_line_for_lane); judging it against that same line would agree by
construction. So the judge is the paint, found in a picture.

For each lane: a camera straight above it, looking down, heading along the lane, so the bar
is a horizontal white band. Across the middle 2.2 m of the lane, the first run of rows that is
paint almost all the way across -- well above the brightness of the road just around it, and
not yellow -- from 2 m behind CARLA's point to just past the first zebra, is the bar.
  * ALL the way across, because a turn arrow or a hatched island is paint too, but not across
    the whole lane (the first try, at 60 %, called two arrows stop bars). Zebra stripes run
    along the lane, so a row through them is about half paint.
  * The road AROUND it, because a bar in a building's shadow is darker than sunlit road
    (the first try missed light 10's bar that way).
  * From 12 m up first; where a tree hides the road, again from 4.5 m, under the leaves.

Writes tools/data/<town>_stop_bars.json: metres along the lane from CARLA's stop point to the
light's map position, the junction entry, the first zebra, and the paint (null where no bar
was found). With --pictures DIR, one marked picture per lane: red CARLA's point, blue the map
position, yellow the junction, green the zebra, white the bar found.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import carla  # noqa: E402
from scratch_world import ScratchWorld, _Grabber  # noqa: E402
from warp_av.perception.traffic_lights import (  # noqa: E402
    crosswalk_polygons, crosswalk_test, _next_along_lane)

VIEWS = ((12.0, 6.0), (4.5, 5.0))   # (camera height, metres past CARLA's point): high, then under trees
PIXELS = 800             # square picture, 90 degree lens -> 2 * height across
LANE_HALF_M = 1.1        # rows are judged across the middle 2.2 m of the lane
AROUND_M = 1.5           # "the road around it": the median of this far before and after
ROW_BRIGHTER_BY = 35     # a bar row's middle brightness stands this far above the road around it
BRIGHTER_BY = 30         # ...and this share of its pixels are at least this much brighter
ROW_PAINT_SHARE = 0.75   #    (worn paint in shadow, light 10: 57-72 % at +45, rows +50 overall)
NOT_YELLOW = 45          # paint's colour channels are this close together
BAR_DEPTH_M = 0.09       # ...for at least this deep
WALK_STEP_M = 0.25


class View:
    """Where a picture was taken from, to turn rows into metres along the lane and back."""

    def __init__(self, height_m: float, ahead_m: float):
        self.height_m, self.ahead_m = height_m, ahead_m
        self.m_per_px = 2.0 * height_m / PIXELS

    def row(self, along_m: float) -> int:
        """Metres along the lane from CARLA's point -> picture row (row 0 is furthest ahead)."""
        return int(round(PIXELS / 2 - (along_m - self.ahead_m) / self.m_per_px))

    def along(self, row: float) -> float:
        return self.ahead_m + (PIXELS / 2 - row) * self.m_per_px


def find_bar(bgr: np.ndarray, view: View, start_m: float, stop_m: float):
    """Near edge of the first painted bar between start_m and stop_m along the lane, or None.
    Also None when leaves cover the lane (no road to compare with)."""
    half = int(LANE_HALF_M / view.m_per_px)
    c0, c1 = PIXELS // 2 - half, PIXELS // 2 + half
    strip = bgr[:, c0:c1, :].astype(int)
    grey = strip.mean(axis=2)
    colourful = (strip.max(axis=2) - strip.min(axis=2)) >= NOT_YELLOW
    if colourful.mean() > 0.35:
        return None                                  # green and brown: a tree, not a road
    row_grey = np.median(grey, axis=1)
    reach = max(1, int(AROUND_M / view.m_per_px))
    lo, hi = max(0, view.row(stop_m)), min(PIXELS - 1, view.row(start_m))
    need = max(2, int(round(BAR_DEPTH_M / view.m_per_px)))
    run = 0
    for r in range(hi, lo - 1, -1):          # from behind, going forwards
        around = np.concatenate([row_grey[max(0, r - reach):max(0, r - 3)],
                                 row_grey[min(PIXELS, r + 4):min(PIXELS, r + reach)]])
        road = float(np.median(around)) if len(around) else float(np.median(row_grey))
        paint = (grey[r] > road + BRIGHTER_BY) & ~colourful[r]
        if row_grey[r] > road + ROW_BRIGHTER_BY and paint.mean() >= ROW_PAINT_SHARE:
            run += 1
            if run >= need:
                return view.along(r + run - 1)
        else:
            run = 0
    return None


def lane_marks(stop_wp, affected, on_zebra):
    """(map position, junction entry, first zebra) in metres along the lane from stop_wp."""
    loc = stop_wp.transform.location
    h = math.radians(stop_wp.transform.rotation.yaw)
    fx, fy = math.cos(h), math.sin(h)
    mine = [b for b in affected if b.road_id == stop_wp.road_id and b.lane_id == stop_wp.lane_id]
    near = mine or list(affected)
    d_map = None
    if near:
        b = min(near, key=lambda w: (w.transform.location.x - loc.x) ** 2 + (w.transform.location.y - loc.y) ** 2)
        d_map = (b.transform.location.x - loc.x) * fx + (b.transform.location.y - loc.y) * fy
    wp, d, d_junction, d_zebra = stop_wp, 0.0, None, None
    while d <= 25.0 and (d_junction is None or d_zebra is None):
        here = wp.transform.location
        if d_zebra is None and on_zebra(here.x, here.y):
            d_zebra = d
        if d_junction is None and wp.is_junction:
            d_junction = d
        nxt = _next_along_lane(wp, WALK_STEP_M)
        if nxt is None:
            break
        wp, d = nxt, d + WALK_STEP_M
    return d_map, d_junction, d_zebra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="default tools/data/<town>_stop_bars.json")
    ap.add_argument("--pictures", default=None, help="write a marked picture per lane here")
    a = ap.parse_args()
    if a.pictures:
        import cv2
        os.makedirs(a.pictures, exist_ok=True)

    rows = []
    with ScratchWorld() as sim:
        world = sim.world
        cmap = world.get_map()
        town = cmap.name.split("/")[-1]
        on_zebra = crosswalk_test(crosswalk_polygons(cmap))
        bp = sim.blueprint("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(PIXELS))
        bp.set_attribute("image_size_y", str(PIXELS))
        bp.set_attribute("fov", "90")
        for tl in sorted(world.get_actors().filter("traffic.traffic_light*"), key=lambda t: t.id):
            affected = list(tl.get_affected_lane_waypoints())
            for stop in tl.get_stop_waypoints():
                loc = stop.transform.location
                yaw = stop.transform.rotation.yaw
                h = math.radians(yaw)
                d_map, d_junction, d_zebra = lane_marks(stop, affected, on_zebra)
                stop_m = (d_zebra + 0.5) if d_zebra is not None else 20.0
                d_bar, view, bgr = None, None, None
                for height_m, ahead_m in VIEWS:
                    view = View(height_m, ahead_m)
                    cam = sim.spawn(bp, carla.Transform(
                        carla.Location(loc.x + math.cos(h) * ahead_m, loc.y + math.sin(h) * ahead_m,
                                       loc.z + height_m),
                        carla.Rotation(pitch=-90.0, yaw=yaw)))
                    grab = _Grabber(sim, cam)
                    image = grab.frame(settle_ticks=4)
                    grab.stop()
                    cam.destroy()
                    sim._mine.remove(cam)
                    bgr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(PIXELS, PIXELS, 4)[:, :, :3].copy()
                    d_bar = find_bar(bgr, view, -2.0, min(stop_m, view.along(0)))
                    if d_bar is not None:
                        break
                row = {"light": tl.id, "road": stop.road_id, "lane": stop.lane_id,
                       "x": round(loc.x, 3), "y": round(loc.y, 3), "yaw": round(yaw, 2),
                       "lane_width": round(stop.lane_width, 2),
                       "map_m": None if d_map is None else round(d_map, 2),
                       "junction_m": d_junction, "zebra_m": d_zebra,
                       "paint_m": None if d_bar is None else round(d_bar, 2),
                       "seen_from_m": view.height_m}
                rows.append(row)
                print(f"light {tl.id:3d} lane {stop.lane_id:3d}: map {row['map_m']}  junction {d_junction}"
                      f"  zebra {d_zebra}  PAINT {row['paint_m']}", flush=True)
                if a.pictures:
                    for m, colour in ((0.0, (0, 0, 255)), (d_map, (255, 80, 0)), (d_junction, (0, 220, 255)),
                                      (d_zebra, (0, 255, 0)), (d_bar, (255, 255, 255))):
                        if m is not None and 0 <= view.row(m) < PIXELS:
                            cv2.line(bgr, (PIXELS // 2 - 100, view.row(m)), (PIXELS // 2 + 100, view.row(m)), colour, 1)
                    cv2.imwrite(os.path.join(a.pictures, f"stop_bar_{tl.id}_{stop.lane_id}.png"), bgr)
    out = a.out or os.path.join(HERE, "data", f"{town.lower()}_stop_bars.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"town": town, "units": "metres along the lane from CARLA's stop waypoint",
                   "lanes": rows}, fh, indent=1)
    found = sum(1 for r in rows if r["paint_m"] is not None)
    print(f"\n{found} of {len(rows)} lanes: paint found. Written to {out}")


if __name__ == "__main__":
    main()
