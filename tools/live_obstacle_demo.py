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
    for name, ahead, right in wanted:
        bp_id, z_up = OBJECTS[name]
        w = wp.next(ahead)[0].transform
        yaw = math.radians(w.rotation.yaw)
        loc = carla.Location(x=w.location.x - math.sin(yaw) * right, y=w.location.y + math.cos(yaw) * right,
                             z=w.location.z + z_up)
        bp = world.get_blueprint_library().find(bp_id)
        actor = world.try_spawn_actor(bp, carla.Transform(loc, w.rotation))
        if actor is None:
            loc.z += 0.5
            actor = world.try_spawn_actor(bp, carla.Transform(loc, w.rotation))
        if actor is None:
            print(f"{name}: spawn failed at {ahead:.0f} m")
            continue
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        placed.append((name, actor))
        print(f"{name} placed {ahead:.0f} m ahead, {right:+.1f} m right, at ({loc.x:.1f}, {loc.y:.1f})")
    if not placed:
        sys.exit("nothing placed")
    actor = placed[0][1]
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
                        near = [o for o in objs if math.hypot(o["x"] - loc.x, o["y"] - loc.y) <= 1.5]
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
