"""
Remove Sprinters the stack is not driving, and any props left behind by an
interrupted test.

Every stack restart spawns a fresh van; if an old process still held one, the
old van stays parked on the road, and the new van reports it as a vehicle
blocking the path. That cost two live runs on 2026-09-08. Run this before a
live test:

    python tools/clear_leftover_vans.py            # dry run: just say what it finds
    python tools/clear_leftover_vans.py --remove
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    ap.add_argument("--remove", action="store_true", help="actually delete them")
    ap.add_argument("--props", action="store_true", help="also delete static props near the van")
    ap.add_argument("--radius", type=float, default=120.0, help="how far around the van to look for props")
    a = ap.parse_args()

    try:
        st = json.load(urllib.request.urlopen(a.api + "/api/state", timeout=6))
        px, py = st["pose"]["x"], st["pose"]["y"]
    except Exception as e:
        sys.exit(f"the stack's API did not answer ({e}): start it first, it says which van is real")

    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    world = client.get_world()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if not vans:
        sys.exit("no van in the world at all")
    mine = min(vans, key=lambda v: math.hypot(v.get_transform().location.x - px,
                                              v.get_transform().location.y - py))
    off = math.hypot(mine.get_transform().location.x - px, mine.get_transform().location.y - py)
    print(f"the stack drives the van at ({px:.1f}, {py:.1f}); nearest actor is {off:.2f} m away")
    if off > 3.0:
        sys.exit("no van matches the stack's position: refusing to guess which one to delete")

    doomed = [v for v in vans if v.id != mine.id]
    for v in doomed:
        loc = v.get_transform().location
        print(f"  leftover van at ({loc.x:.1f}, {loc.y:.1f})" + ("  -> removed" if a.remove else ""))
        if a.remove:
            v.destroy()
    if a.props:
        for act in world.get_actors():
            if not act.type_id.startswith("static.prop"):
                continue
            loc = act.get_transform().location
            if math.hypot(loc.x - px, loc.y - py) <= a.radius:
                print(f"  leftover prop {act.type_id} at ({loc.x:.1f}, {loc.y:.1f})"
                      + ("  -> removed" if a.remove else ""))
                if a.remove:
                    act.destroy()
    if not doomed:
        print("  nothing left over")
    if not a.remove:
        print("dry run: add --remove to delete these")


if __name__ == "__main__":
    main()
