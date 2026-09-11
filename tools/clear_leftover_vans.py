"""
Remove Sprinters the stack is not driving, and any props left behind by an
interrupted test.

Every stack restart spawns a fresh van; if an old process still held one, the
old van stays parked on the road, and the new van reports it as a vehicle
blocking the path. That cost two live runs on 2026-09-08. Run this before a
live test:

    python tools/clear_leftover_vans.py            # dry run: just say what it finds
    python tools/clear_leftover_vans.py --remove
    python tools/clear_leftover_vans.py --remove --sensors   # also sensors attached to nothing
    python tools/clear_leftover_vans.py --remove --sensors --stack-down   # stack stopped on purpose

A leftover van's own sensors go with it. CARLA does not remove a sensor when the thing it
rides on is destroyed: found 2026-09-10, 81 sensors -- 45 of them cameras, still rendering --
left by 9 stack runs that were stopped without cleaning up. Stopping the stack's scheduled
task does not stop run.py either (it kills the batch file, not its child): stop the python
process itself.
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
    ap.add_argument("--sensors", action="store_true",
                    help="also delete sensors attached to nothing (left by a stack that was killed)")
    ap.add_argument("--stack-down", action="store_true",
                    help="the stack has been stopped on purpose: every Sprinter is left over")
    a = ap.parse_args()

    try:
        st = json.load(urllib.request.urlopen(a.api + "/api/state", timeout=6))
        px, py = st["pose"]["x"], st["pose"]["y"]
    except Exception as e:
        if not a.stack_down:
            sys.exit(f"the stack's API did not answer ({e}): start it first, it says which van is real"
                     " -- or pass --stack-down if you have stopped it on purpose")
        px = py = None

    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    world = client.get_world()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if px is None:
        # the stack is down, so no van is being driven: every Sprinter is left over
        print(f"the stack is down: all {len(vans)} Sprinters are left over")
        mine = None
    else:
        if not vans:
            sys.exit("no van in the world at all")
        mine = min(vans, key=lambda v: math.hypot(v.get_transform().location.x - px,
                                                  v.get_transform().location.y - py))
        off = math.hypot(mine.get_transform().location.x - px, mine.get_transform().location.y - py)
        print(f"the stack drives the van at ({px:.1f}, {py:.1f}); nearest actor is {off:.2f} m away")
        if off > 3.0:
            sys.exit("no van matches the stack's position: refusing to guess which one to delete")

    doomed = [v for v in vans if mine is None or v.id != mine.id]
    sensors = list(world.get_actors().filter("sensor.*"))
    for v in doomed:
        loc = v.get_transform().location
        riders = [x for x in sensors if x.parent is not None and x.parent.id == v.id]
        print(f"  leftover van at ({loc.x:.1f}, {loc.y:.1f}) with {len(riders)} sensors"
              + ("  -> removed" if a.remove else ""))
        if a.remove:
            for x in riders:                       # the sensors first: they outlive the van
                x.destroy()
            v.destroy()
    if a.sensors:
        loose = [x for x in world.get_actors().filter("sensor.*") if x.parent is None]
        print(f"  {len(loose)} sensors attached to nothing" + ("  -> removed" if a.remove and loose else ""))
        if a.remove:
            for x in loose:
                x.destroy()
    if a.props and px is not None:
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
