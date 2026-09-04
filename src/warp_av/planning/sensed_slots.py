"""
Turn what the lidar bay finder sees into the planner's parking slots.

The planner's slots are dicts in the WORLD frame: {x, y, yaw, length, width,
corners}. The bay finder returns bays in the VEHICLE frame (x forward, y
LEFT). This module does the conversion and slices a long free stretch into
consecutive 7 m slots, so choose_free_slot's "the slot before it must be free
too" rule keeps working exactly as it does with map slots.

Pure math; the caller supplies the ego pose and the lidar sweep.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from warp_av.perception.bay_finder import find_bays, fit_kerb, Bay, SLOT_LEN_M, SLOT_WID_M, KERB_GAP_M

SOURCE_SENSED = "lidar"
SOURCE_MAP = "map"


def vehicle_to_world(x_fwd: float, y_left: float, ex: float, ey: float, eyaw: float):
    """Vehicle-frame point (x forward, y LEFT) -> world, for an ego at (ex, ey, eyaw).
    CARLA's frame is y-right, so a left offset is a negative y-right offset."""
    y_right = -y_left
    c, s = math.cos(eyaw), math.sin(eyaw)
    return ex + x_fwd * c - y_right * s, ey + x_fwd * s + y_right * c


def slot_dict(cx: float, cy: float, yaw: float, width: float = SLOT_WID_M) -> Dict:
    fwd = (math.cos(yaw), math.sin(yaw))
    right = (-math.sin(yaw), math.cos(yaw))
    hl, hw = SLOT_LEN_M / 2.0, width / 2.0
    corners = [[round(cx + sx * fwd[0] * hl + sy * right[0] * hw, 2),
                round(cy + sx * fwd[1] * hl + sy * right[1] * hw, 2)]
               for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    return {"x": round(cx, 2), "y": round(cy, 2), "yaw": round(yaw, 3),
            "length": SLOT_LEN_M, "width": round(width, 2), "corners": corners,
            "source": SOURCE_SENSED}


def bays_to_slots(bays: List[Bay], ex: float, ey: float, eyaw: float,
                  min_ahead_m: float = 0.0) -> List[Dict]:
    """Every 7 m slot that fits in each free stretch, world frame. Slots whose
    centre is behind the van (x < min_ahead_m) are dropped: it has no reverse
    gear, so it cannot get into them."""
    out = []
    for bay in bays:
        usable_start = bay.along_start + 0.5
        usable_end = bay.along_end - 0.5
        n = int((usable_end - usable_start) // SLOT_LEN_M)
        if n <= 0:
            continue
        # centre the run of n slots in the free stretch
        first_mid = 0.5 * (usable_start + usable_end) - (n - 1) * SLOT_LEN_M / 2.0
        c, s = math.cos(bay.yaw), math.sin(bay.yaw)
        # the bay's slot centre is one point on the slot centre line; move along it
        for k in range(n):
            mid = first_mid + k * SLOT_LEN_M
            mid0 = 0.5 * (bay.slot_start + bay.slot_end)
            d = mid - mid0
            vx, vy_left = bay.x + d * c, bay.y - d * s       # y_left decreases as y_right grows
            if vx < min_ahead_m:
                continue
            wx, wy = vehicle_to_world(vx, vy_left, ex, ey, eyaw)
            out.append(slot_dict(wx, wy, eyaw + bay.yaw))
    return out


def sensed_parking_slots(points: Optional[np.ndarray], ex: float, ey: float, eyaw: float) -> List[Dict]:
    """World-frame slots from one lidar sweep; [] when no kerb is seen."""
    if points is None or len(points) == 0:
        return []
    bays = find_bays(points)
    if not bays:
        return []
    # far -> near, like the map slicer: sort by distance ahead descending
    slots = bays_to_slots(bays, ex, ey, eyaw)
    slots.sort(key=lambda sl: -((sl["x"] - ex) * math.cos(eyaw) + (sl["y"] - ey) * math.sin(eyaw)))
    return slots
