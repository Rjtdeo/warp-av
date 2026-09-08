"""
Live obstacle demo for watching in the CARLA window and Mission Control.

Puts one object on the lane centre ahead of the parked van, points the CARLA
spectator camera at the van from behind, starts a mission past the object,
and follows the van with the camera while printing what the stack reports.
Cleans up the object and stops the mission at the end (unless --keep).

    venv/Scripts/python.exe tools/live_obstacle_demo.py [--object barrel|cone|car|person]
        [--ahead 22] [--dest 90] [--follow 120] [--keep]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import requests
import carla

OBJECTS = {"barrel": ("static.prop.barrel", 0.05), "cone": ("static.prop.trafficcone01", 0.05),
           "car": ("vehicle.tesla.model3", 0.3), "person": ("walker.pedestrian.0001", 1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="barrel", choices=sorted(OBJECTS))
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
    obj_wp = wp.next(a.ahead)[0]
    dest_wp = wp.next(a.dest)[0]
    bp_id, z_up = OBJECTS[a.object]
    loc = obj_wp.transform.location
    loc.z += z_up
    actor = world.try_spawn_actor(world.get_blueprint_library().find(bp_id),
                                  carla.Transform(loc, obj_wp.transform.rotation))
    if actor is None:
        loc.z += 0.5
        actor = world.spawn_actor(world.get_blueprint_library().find(bp_id),
                                  carla.Transform(loc, obj_wp.transform.rotation))
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    print(f"{a.object} placed {a.ahead:.0f} m ahead at ({loc.x:.1f}, {loc.y:.1f}); "
          f"destination {a.dest:.0f} m ahead at ({dest_wp.transform.location.x:.1f}, {dest_wp.transform.location.y:.1f})")

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
                    d = actor.get_location().distance(van.get_location())
                    lid = s.get("health", {}).get("lidar", {})
                    print(f"t={time.time() - t0:5.1f}s  speed {s.get('pose', {}).get('speed', 0):4.1f} m/s  "
                          f"object {d:5.1f} m away  seen at {s.get('perception', {}).get('closest_distance')} m  "
                          f"blocked={s.get('perception', {}).get('path_blocked')}  {s.get('behavior')}: "
                          f"{(s.get('behavior_reason') or '')[:70]}  | lidar {lid.get('points_per_scan')} pts "
                          f"{'full sweep' if lid.get('full_sweep') else 'wedge'}", flush=True)
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
            actor.destroy()
            print("cleaned up: mission stopped, object removed")
        else:
            print("kept: object stays, mission still running")


if __name__ == "__main__":
    main()
