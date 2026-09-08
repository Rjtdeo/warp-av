"""
Live obstacle demo for watching in the CARLA window and Mission Control.

Puts one object on the lane centre ahead of the parked van, points the CARLA
spectator camera at the van from behind, starts a mission past the object,
and follows the van with the camera while printing what the stack reports.
Cleans up the object and stops the mission at the end (unless --keep).

    venv/Scripts/python.exe tools/live_obstacle_demo.py [--object barrel|cone|car|person]
        [--ahead 22] [--dest 90] [--follow 120] [--keep]
    or several at once, name@metres_ahead[:metres_right]:
    venv/Scripts/python.exe tools/live_obstacle_demo.py --objects planter@10,cone@12:1.0,barrel@14:-0.8
"""
import argparse
import math
import sys
import time
from pathlib import Path

import requests
import carla

OBJECTS = {"barrel": ("static.prop.barrel", 0.05), "cone": ("static.prop.trafficcone01", 0.05),
           "planter": ("static.prop.plantpot04", 0.05), "barrier": ("static.prop.streetbarrier", 0.05),
           "car": ("vehicle.tesla.model3", 0.3), "person": ("walker.pedestrian.0001", 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="barrel", choices=sorted(OBJECTS))
    ap.add_argument("--objects", default=None, help="several: name@ahead[:right],... overrides --object/--ahead")
    ap.add_argument("--at-route-fraction", type=float, default=None,
                    help="place the objects on the PLANNED route at this fraction of the trip (0.5 = halfway); "
                         "'ahead' in --objects is then metres beyond that point")
    ap.add_argument("--ahead", type=float, default=22.0)
    ap.add_argument("--dest", type=float, default=90.0)
    ap.add_argument("--follow", type=float, default=120.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if not vans:
        sys.exit("no van in the world: start the stack first")
    van = vans[0]
    st = requests.get(a.api + "/api/state", timeout=3).json()
    if st.get("mission", {}).get("state") not in (None, "idle", "completed"):
        sys.exit(f"stack is busy: mission {st['mission']['state']}")

    tf = van.get_transform()
    wp = cmap.get_waypoint(tf.location)
    dest_wp = wp.next(a.dest)[0]
    wanted = []                                        # (name, ahead, right)
    if a.objects:
        for item in a.objects.split(","):
            name, _, rest = item.strip().partition("@")
            ahead, _, right = rest.partition(":")
            wanted.append((name, float(ahead), float(right) if right else 0.0))
    else:
        wanted.append((a.object, a.ahead, 0.0))
    placed = []                                        # (name, actor)

    def place(name, x, y, yaw_deg, right, label):
        bp_id, z_up = OBJECTS[name]
        yaw = math.radians(yaw_deg)
        gx, gy = x - math.sin(yaw) * right, y + math.cos(yaw) * right
        z = cmap.get_waypoint(carla.Location(x=gx, y=gy, z=tf.location.z)).transform.location.z
        loc = carla.Location(x=gx, y=gy, z=z + z_up)
        bp = world.get_blueprint_library().find(bp_id)
        rot = carla.Rotation(yaw=yaw_deg)
        actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))
        if actor is None:
            loc.z += 0.5
            actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))
        if actor is None:
            print(f"{name}: spawn failed ({label})")
            return
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        placed.append((name, actor))
        print(f"{name} placed {label}, {right:+.1f} m right, at ({loc.x:.1f}, {loc.y:.1f})", flush=True)

    if a.at_route_fraction is None:
        for name, ahead, right in wanted:
            w = wp.next(ahead)[0].transform
            place(name, w.location.x, w.location.y, w.rotation.yaw, right, f"{ahead:.0f} m ahead")
        if not placed:
            sys.exit("nothing placed")
    print(f"destination {a.dest:.0f} m ahead at ({dest_wp.transform.location.x:.1f}, {dest_wp.transform.location.y:.1f})")

    spectator = world.get_spectator()

    def chase():
        t = van.get_transform()
        yaw = math.radians(t.rotation.yaw)
        cam = carla.Location(x=t.location.x - 10.0 * math.cos(yaw), y=t.location.y - 10.0 * math.sin(yaw),
                             z=t.location.z + 5.0)
        spectator.set_transform(carla.Transform(cam, carla.Rotation(pitch=-18.0, yaw=t.rotation.yaw)))

    chase()
    time.sleep(1.0)
    r = requests.post(a.api + "/api/mission/start",
                      json={"x": dest_wp.transform.location.x, "y": dest_wp.transform.location.y}, timeout=5).json()
    print("mission start:", r)
    if a.at_route_fraction is not None:
        # the objects go onto the route the van actually planned, part-way along it
        route = []
        for _ in range(30):
            try:
                route = requests.get(a.api + "/api/route", timeout=2).json()
            except Exception:
                route = []
            if len(route) >= 2:
                break
            time.sleep(0.1)
        if len(route) < 2:
            requests.post(a.api + "/api/mission/stop", timeout=5)
            sys.exit("no route from the stack")
        xs = [p["x"] for p in route]
        ys = [p["y"] for p in route]
        seg = [math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)]
        total = sum(seg)
        cum = [0.0]
        for d in seg:
            cum.append(cum[-1] + d)

        def at(arc):
            arc = max(0.0, min(total - 0.01, arc))
            for i in range(len(seg)):
                if cum[i + 1] >= arc:
                    t = (arc - cum[i]) / seg[i] if seg[i] > 0 else 0.0
                    x = xs[i] + t * (xs[i + 1] - xs[i])
                    y = ys[i] + t * (ys[i + 1] - ys[i])
                    return x, y, math.degrees(math.atan2(ys[i + 1] - ys[i], xs[i + 1] - xs[i]))
            return xs[-1], ys[-1], 0.0

        base = a.at_route_fraction * total
        print(f"route is {total:.0f} m; objects go {base:.0f} m along it", flush=True)
        for name, ahead, right in wanted:
            x, y, yaw_deg = at(base + ahead)
            place(name, x, y, yaw_deg, right, f"{base + ahead:.0f} m along the route")
        if not placed:
            requests.post(a.api + "/api/mission/stop", timeout=5)
            sys.exit("nothing placed")

    t0 = time.time()
    last_print = 0.0
    try:
        while time.time() - t0 < a.follow:
            chase()
            if time.time() - last_print >= 2.0:
                last_print = time.time()
                try:
                    s = requests.get(a.api + "/api/state", timeout=2).json()
                    objs = s.get("perception", {}).get("objects", [])
                    vloc = van.get_location()
                    seen = []
                    for name, act in placed:
                        loc = act.get_location()
                        d = loc.distance(vloc)
                        # a reported object counts as this one when it lies within the
                        # object's own size (plus a margin) of its origin: a long planter's
                        # lidar blob sits at its near face, not at its origin
                        ext = act.bounding_box.extent
                        reach = max(1.5, math.hypot(ext.x, ext.y) + 0.5)
                        near = [o for o in objs if math.hypot(o["x"] - loc.x, o["y"] - loc.y) <= reach]
                        seen.append(f"{name} {d:4.1f} m: {near[0]['type'] if near else 'not seen'}")
                    print(f"t={time.time() - t0:5.1f}s  speed {s.get('pose', {}).get('speed', 0):4.1f} m/s  "
                          f"{s.get('behavior')}: {(s.get('behavior_reason') or '')[:48]}  |  " + "  |  ".join(seen), flush=True)
                    if s.get("mission", {}).get("state") == "completed":
                        print("mission completed")
                        break
                except Exception as e:
                    print("state read failed:", e)
            time.sleep(0.05)
    finally:
        if not a.keep:
            try:
                requests.post(a.api + "/api/mission/stop", timeout=5)
            except Exception:
                pass
            for _, act in placed:
                try:
                    act.destroy()
                except Exception:
                    pass
            print("cleaned up: mission stopped, objects removed")
        else:
            print("kept: object stays, mission still running")


if __name__ == "__main__":
    main()
