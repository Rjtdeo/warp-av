#!/usr/bin/env python3
"""
Park stationary cars in the bay near the ACTIVE mission's destination, so
FIND PARKING has occupied slots to avoid.

    1. start a mission (route must exist)
    2. python tools/spawn_parked_cars.py --count 4
    3. press FIND PARKING on the dashboard -> occupied slots show RED,
       the chosen free slot GREEN
    4. python tools/spawn_parked_cars.py --clear      (remove them again)

Options:
    --count N        cars to park (default 4)
    --spacing M      metres between parked cars (default 14 = every other slot)
    --fill-all       pack the bay solid to test the "all slots occupied" answer
"""
import argparse
import json
import math
import urllib.request

import carla

ROLE = "warp_parked"


def api_route(host):
    with urllib.request.urlopen(f"http://{host}:5000/api/route", timeout=3) as r:
        return json.loads(r.read().decode())


def right_bay(cmap, x, y):
    wp = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.3),
                           project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        return None
    for _ in range(3):
        r = wp.get_right_lane()
        if (r is not None and r.lane_type == carla.LaneType.Driving
                and abs((r.transform.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180) < 60):
            wp = r
        else:
            break
    bay = wp.get_right_lane()
    if (bay is not None and bay.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder)
            and bay.lane_width >= 1.8):
        t = bay.transform
        return (t.location.x, t.location.y, t.rotation.yaw)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=14.0)
    ap.add_argument("--fill-all", action="store_true")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(10.0)
    world = client.get_world()

    if a.clear:
        n = 0
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name") == ROLE:
                v.destroy()
                n += 1
        print(f"removed {n} parked cars.")
        return

    try:
        route = api_route(a.host)
    except Exception as e:
        raise SystemExit(f"Could not fetch the route ({e}) — start a mission first.")
    if len(route) < 5:
        raise SystemExit("Route too short / no active mission — start a mission first.")

    cmap = world.get_map()
    # walk the route tail (last ~70 m) and collect bay positions in order
    pts = [(p["x"], p["y"]) for p in route]
    arc_back = 0.0
    tail = [pts[-1]]
    for p, q in zip(reversed(pts[:-1]), reversed(pts)):
        arc_back += math.hypot(q[0] - p[0], q[1] - p[1])
        tail.append(p)
        if arc_back > 70.0:
            break
    tail.reverse()      # route order, ending at the destination

    bays = []
    for x, y in tail:
        b = right_bay(cmap, x, y)
        if b is not None:
            bays.append(b)
    if len(bays) < 3:
        raise SystemExit("No parking bay on the final stretch of this route.")

    spacing = 8.0 if a.fill_all else a.spacing
    count = 999 if a.fill_all else a.count
    # keep the last ~10 m nearest the destination free unless --fill-all
    usable = bays if a.fill_all else bays[:max(1, len(bays) - 5)]

    bp_lib = world.get_blueprint_library()
    models = ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.nissan.patrol", "vehicle.mini.cooper_s"]
    spawned = 0
    next_at = 0.0
    arc = 0.0
    prev = usable[0]
    for b in usable:
        arc += math.hypot(b[0] - prev[0], b[1] - prev[1])
        prev = b
        if arc < next_at or spawned >= count:
            continue
        bp = bp_lib.filter(models[spawned % len(models)])[0]
        bp.set_attribute("role_name", ROLE)
        tr = carla.Transform(carla.Location(x=b[0], y=b[1], z=0.3),
                             carla.Rotation(yaw=b[2]))
        car = world.try_spawn_actor(bp, tr)
        if car is not None:
            car.apply_control(carla.VehicleControl(hand_brake=True))
            spawned += 1
            next_at = arc + spacing
    print(f"parked {spawned} cars in the bay near the destination"
          + (" (bay packed solid)" if a.fill_all else ", end of bay left free")
          + ". Now press FIND PARKING. Remove later with --clear.")


if __name__ == "__main__":
    main()
