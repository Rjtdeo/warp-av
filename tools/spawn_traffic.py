#!/usr/bin/env python3
"""
Fill the town with live traffic and let the van deal with it.

    python tools/spawn_traffic.py                          # 20 cars, 15 walkers, 4 cyclists
    python tools/spawn_traffic.py --cars 30 --walkers 25 --cyclists 8
    python tools/spawn_traffic.py --cars 5 --walkers 40    # pedestrian city

Keep this window OPEN while testing (the traffic brain runs inside it).
Press Ctrl+C to remove every spawned actor and exit cleanly.

Walkers roam between random points and ~35% of them will CROSS ROADS, so you
will see real pedestrian encounters. Cyclists are bicycles driven by the
traffic manager. Everything avoids spawning on top of the Warp van.
"""
import argparse
import json
import math
import random
import time
import urllib.request

import carla


def ego_position(api="http://localhost:5000"):
    """Where is the Warp van right now? (None if the stack isn't running)"""
    try:
        with urllib.request.urlopen(api + "/api/state", timeout=2) as r:
            d = json.loads(r.read().decode())
        return d["pose"]["x"], d["pose"]["y"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--walkers", type=int, default=15)
    ap.add_argument("--cyclists", type=int, default=4)
    ap.add_argument("--cross-factor", type=float, default=0.35,
                    help="fraction of walkers allowed to cross roads (0..1)")
    ap.add_argument("--near", type=float, default=120.0,
                    help="concentrate traffic within this many metres of the van (0 = whole town)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    client = carla.Client(a.host, a.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager()
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(10)

    spawned = []
    controllers = []

    try:
        # ---------------- cars ----------------
        car_bps = [bp for bp in bp_lib.filter("vehicle.*")
                   if int(bp.get_attribute("number_of_wheels").as_int()) == 4]
        points = world.get_map().get_spawn_points()
        rng.shuffle(points)
        ego = ego_position(f"http://{a.host}:5000") if a.near > 0 else None
        if ego is not None:
            near = [p for p in points
                    if math.hypot(p.location.x - ego[0], p.location.y - ego[1]) < a.near]
            far = [p for p in points if p not in near]
            points = near + far      # fill close to the van first
            print(f"van at ({ego[0]:.0f}, {ego[1]:.0f}) — {len(near)} spawn points within {a.near:.0f} m")
        else:
            print("van position unknown (stack not running?) — spawning town-wide")
        cars = 0
        for sp in points:
            if cars >= a.cars:
                break
            v = world.try_spawn_actor(rng.choice(car_bps), sp)
            if v is not None:
                v.set_autopilot(True, tm.get_port())
                spawned.append(v)
                cars += 1
        print(f"cars: {cars}")

        # ---------------- cyclists ----------------
        bike_bps = list(bp_lib.filter("vehicle.bh.crossbike")) \
            + list(bp_lib.filter("vehicle.diamondback.century")) \
            + list(bp_lib.filter("vehicle.gazelle.omafiets"))
        cyclists = 0
        for sp in points[cars:]:
            if cyclists >= a.cyclists or not bike_bps:
                break
            b = world.try_spawn_actor(rng.choice(bike_bps), sp)
            if b is not None:
                b.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(b, 55)   # bikes ride slow
                spawned.append(b)
                cyclists += 1
        print(f"cyclists: {cyclists}")

        # ---------------- walkers ----------------
        world.set_pedestrians_cross_factor(max(0.0, min(1.0, a.cross_factor)))
        walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
        ctrl_bp = bp_lib.find("controller.ai.walker")
        walkers = 0
        attempts = 0
        while walkers < a.walkers and attempts < a.walkers * 6:
            attempts += 1
            loc = world.get_random_location_from_navigation()
            if loc is None:
                continue
            if ego is not None and math.hypot(loc.x - ego[0], loc.y - ego[1]) > a.near:
                continue      # keep pedestrians near the action too
            w = world.try_spawn_actor(rng.choice(walker_bps),
                                      carla.Transform(loc))
            if w is None:
                continue
            c = world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w)
            c.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                c.go_to_location(dest)
            c.set_max_speed(rng.uniform(1.0, 1.8))
            spawned.append(w)
            controllers.append(c)
            walkers += 1
        print(f"walkers: {walkers} (cross factor {a.cross_factor})")

        print("\nTraffic is LIVE. Leave this window open; Ctrl+C removes everything.")
        while True:
            time.sleep(30)
            # keep walkers roaming: hand everyone a fresh destination
            for c in controllers:
                try:
                    dest = world.get_random_location_from_navigation()
                    if dest:
                        c.go_to_location(dest)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\nCleaning up...")
    finally:
        for c in controllers:
            try:
                c.stop()
                c.destroy()
            except Exception:
                pass
        for s in spawned:
            try:
                s.destroy()
            except Exception:
                pass
        print(f"removed {len(spawned)} actors. Town is quiet again.")


if __name__ == "__main__":
    main()
