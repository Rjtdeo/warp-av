#!/usr/bin/env python3
"""
Scripted red-light demo. Run on the CARLA machine while CARLA + Warp AV run:

    python tools/demo_red_light.py            # full demo, hands-free
    python tools/demo_red_light.py --pick     # only print light-rich destinations

It: finds the destination whose route passes the most traffic lights, starts
the mission, freezes lights RED when the van gets near one, verifies the van
stops (stopped_red_light), releases to GREEN, verifies it drives off, then
returns lights to automatic. Ctrl+C at any point also releases the lights.

No extra Python packages needed (stdlib only + carla).
"""
import argparse
import json
import math
import sys
import time
import urllib.request

import carla

API = "http://localhost:5000"


def api_get(path):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read().decode())


def api_post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(API + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def light_positions(world):
    return [(tl, tl.get_location()) for tl in world.get_actors().filter("traffic.traffic_light")]


def route_lights(grp, start, end, lights, radius=15.0):
    """How many traffic lights sit within `radius` m of the planned route?"""
    try:
        route = grp.trace_route(start, end)
    except Exception:
        return -1, 0.0
    if not route or len(route) < 5:
        return -1, 0.0
    pts = [wp.transform.location for wp, _ in route]
    length = sum(a.distance(b) for a, b in zip(pts, pts[1:]))
    seen = set()
    for _, loc in lights:
        for p in pts[::3]:
            if p.distance(loc) < radius:
                seen.add((round(loc.x, 1), round(loc.y, 1)))
                break
    return len(seen), length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true", help="only list the best destinations")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(10.0)
    world = client.get_world()
    lights = light_positions(world)
    if not lights:
        sys.exit("This map has no traffic lights — switch to Town10HD/Town03/Town05.")
    print(f"{len(lights)} traffic lights in this town.")

    state = api_get("/api/state")
    ego = carla.Location(x=state["pose"]["x"], y=state["pose"]["y"], z=0.3)

    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        grp = GlobalRoutePlanner(world.get_map(), 2.0)
    except Exception as e:
        sys.exit(f"Could not import CARLA route planner ({e}) — run from the same environment as run.py")

    print("Scoring destinations by traffic lights on the route (this takes ~30 s)...")
    best = []
    for i, sp in enumerate(world.get_map().get_spawn_points()[:40]):
        n, length = route_lights(grp, ego, sp.location, lights)
        if n > 0 and 150 < length < 900:
            best.append((n, length, i, sp))
    best.sort(key=lambda t: (-t[0], t[1]))
    if not best:
        sys.exit("No route with lights found from here — drive somewhere else and retry.")
    for n, length, i, sp in best[:5]:
        print(f"  spawn point {i:3d}: {n} lights on a {length:.0f} m route  (x={sp.location.x:.1f}, y={sp.location.y:.1f})")
    if a.pick:
        return

    n, length, idx, sp = best[0]
    print(f"\nStarting mission to spawn point {idx} — {n} lights over {length:.0f} m.")
    resp = api_post("/api/mission/start", {"x": sp.location.x, "y": sp.location.y})
    if not resp.get("success"):
        sys.exit(f"Mission refused: {resp}")

    frozen = False
    try:
        # wait until the van is near any light, then force red
        print("Waiting for the van to approach a traffic light...")
        t0 = time.time()
        while time.time() - t0 < 180:
            st = api_get("/api/state")
            pos = carla.Location(x=st["pose"]["x"], y=st["pose"]["y"], z=0.3)
            near = min(loc.distance(pos) for _, loc in lights)
            if near < 45.0:
                for tl, _ in lights:
                    tl.set_state(carla.TrafficLightState.Red)
                    tl.freeze(True)
                frozen = True
                print(f"*** LIGHTS FROZEN RED (van {near:.0f} m from a light) — watch the dashboard ***")
                break
            time.sleep(0.3)
        else:
            sys.exit("Van never came near a light — try again.")

        # verify it stops for the red
        t0 = time.time()
        while time.time() - t0 < 60:
            st = api_get("/api/state")
            if st.get("behavior") == "stopped_red_light" and st["pose"]["speed"] < 0.3:
                print(f"[OK] Van STOPPED at red. Reason: {st.get('behavior_reason')}")
                break
            time.sleep(0.3)
        else:
            sys.exit("[BAD] Van did not stop for the red light within 60 s — tell Claude.")

        print("Holding red for 6 s...")
        time.sleep(6)
        for tl, _ in lights:
            tl.set_state(carla.TrafficLightState.Green)
        print("*** LIGHTS GREEN ***")

        t0 = time.time()
        while time.time() - t0 < 30:
            st = api_get("/api/state")
            if st["pose"]["speed"] > 2.0:
                print("[OK] Van moved off on green by itself. Demo complete — mission continues.")
                break
            time.sleep(0.3)
        else:
            print("[BAD] Van did not move off on green within 30 s — tell Claude.")
    finally:
        if frozen:
            for tl, _ in lights:
                tl.freeze(False)
            print("Lights returned to automatic cycling.")


if __name__ == "__main__":
    main()
