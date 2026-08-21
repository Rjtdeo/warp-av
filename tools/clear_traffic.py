#!/usr/bin/env python3
"""
Remove ALL traffic from the CARLA world except the Warp van itself.

Use when a spawn_traffic session died without Ctrl+C and left frozen,
driverless cars standing around (they block junctions forever).

    python tools/clear_traffic.py
"""
import json
import math
import urllib.request

import carla


def main(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()

    ego_id = None
    try:
        with urllib.request.urlopen(f"http://{host}:5000/api/state", timeout=2) as r:
            d = json.loads(r.read().decode())
        ex, ey = d["pose"]["x"], d["pose"]["y"]
        vehicles = list(world.get_actors().filter("vehicle.*"))
        if vehicles:
            ego = min(vehicles, key=lambda v: math.hypot(v.get_location().x - ex,
                                                         v.get_location().y - ey))
            ego_id = ego.id
            print(f"keeping the Warp van (actor {ego_id} at {ex:.0f}, {ey:.0f})")
    except Exception:
        print("WARNING: Warp stack not reachable — will remove EVERY vehicle, van included.")
        if input("continue? [y/N] ").strip().lower() != "y":
            return

    removed = 0
    for pattern in ("controller.ai.walker", "walker.pedestrian.*", "vehicle.*"):
        for a in world.get_actors().filter(pattern):
            if a.id == ego_id:
                continue
            try:
                if pattern.startswith("controller"):
                    a.stop()
                a.destroy()
                removed += 1
            except Exception:
                pass
    print(f"removed {removed} actors. Town is clear.")


if __name__ == "__main__":
    main()
