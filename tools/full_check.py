"""
One run that exercises everything perception learned in Perception V2.

Each check places something real in the simulator, or breaks something real,
and reads the answer back through the van's own interfaces. Nothing is mocked.
Run it after any change to perception:

    python tools/full_check.py                 # the whole battery
    python tools/full_check.py --only naming   # one group
    python tools/full_check.py --list

It needs the stack running and the van parked with a clear road ahead. It puts
everything back as it found it, including any sensor it switches off.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request

try:
    import carla
except ImportError:                                   # pragma: no cover
    sys.exit("this tool runs on the simulator machine, where the carla package is installed")

API = "http://127.0.0.1:5000"
PLACE_M = 12.0


# ---------------------------------------------------------------- plumbing

def get(path, timeout=8):
    return json.load(urllib.request.urlopen(API + path, timeout=timeout))


def post(path, body=None, timeout=6):
    req = urllib.request.Request(API + path, method="POST",
                                 data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


class Report:
    def __init__(self):
        self.rows = []

    def add(self, group, what, ok, detail):
        self.rows.append((group, what, ok, detail))
        mark = "pass" if ok is True else "FAIL" if ok is False else "skip"
        print(f"  [{mark}] {what:44s} {detail}")

    def summary(self):
        passed = sum(1 for r in self.rows if r[2] is True)
        failed = [r for r in self.rows if r[2] is False]
        skipped = sum(1 for r in self.rows if r[2] is None)
        print("\n" + "=" * 78)
        print(f"  {passed} passed, {len(failed)} failed, {skipped} skipped")
        for _, what, _, detail in failed:
            print(f"     FAILED: {what} — {detail}")
        return 1 if failed else 0


class Scene:
    """Places things in front of the van and always clears up after itself."""

    def __init__(self):
        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bl = self.world.get_blueprint_library()
        st = get("/api/state")
        self.van = min(self.world.get_actors().filter("vehicle.mercedes.sprinter"),
                       key=lambda v: math.hypot(v.get_transform().location.x - st["pose"]["x"],
                                                v.get_transform().location.y - st["pose"]["y"]))
        self.spawned = []

    def chase_camera(self):
        t = self.van.get_transform()
        f = t.get_forward_vector()
        self.world.get_spectator().set_transform(carla.Transform(
            carla.Location(x=t.location.x - f.x * 12.0, y=t.location.y - f.y * 12.0,
                           z=t.location.z + 6.0),
            carla.Rotation(pitch=-15.0, yaw=t.rotation.yaw)))

    def place(self, blueprint, ahead_m=PLACE_M, lift=0.05, lateral=0.0):
        wp = self.map.get_waypoint(self.van.get_transform().location)
        nxt = wp.next(ahead_m)
        if not nxt:
            return None
        t = nxt[0].transform
        a = math.radians(t.rotation.yaw)
        loc = carla.Location(x=t.location.x - math.sin(a) * lateral,
                             y=t.location.y + math.cos(a) * lateral,
                             z=t.location.z + lift)
        actor = self.world.try_spawn_actor(self.bl.find(blueprint), carla.Transform(loc, t.rotation))
        if actor is None:
            return None
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        self.spawned.append(actor)
        return actor

    def reported_near(self, actor, reach=4.0):
        """What the van says about the thing nearest that actor, from the world sheet."""
        wm = get("/api/world")
        loc = actor.get_transform().location
        van = self.van.get_transform().location
        near = [o for o in wm["objects"] if o["x"] is not None
                and math.hypot(o["x"] - loc.x, o["y"] - loc.y) <= reach]
        if not near:
            return None
        return min(near, key=lambda o: math.hypot(o["x"] - van.x, o["y"] - van.y))

    def clear(self):
        for a in self.spawned:
            try:
                a.destroy()
            except Exception:
                pass
        self.spawned = []
        time.sleep(1.5)


# ---------------------------------------------------------------- the checks

def check_health(scene, report):
    # one reading catches whatever the van happened to be doing; take several
    rates = []
    for _ in range(5):
        rates.append(get("/api/state").get("loop_hz") or 0.0)
        time.sleep(1.0)
    rates.sort()
    hz = rates[len(rates) // 2]
    st = get("/api/state")
    report.add("health", "the van thinks at least 8 times a second", hz >= 8.0,
               f"{hz:.1f} Hz (from {rates[0]:.1f} to {rates[-1]:.1f} over five seconds)")
    sensors = st.get("sensors") or {}
    report.add("health", "every sense is working", sensors.get("worst") == "note" or not sensors.get("failed"),
               sensors.get("reason", "no sensor report"))
    per = st.get("perception") or {}
    report.add("health", "the van is seeing things", (per.get("object_count") or 0) > 0,
               f"{per.get('object_count')} things around it")


def check_world_sheet(scene, report):
    wm = get("/api/world")
    report.add("world", "the world sheet answers", wm.get("healthy") is True,
               f"{wm['counts']['total']} things, {wm['age_s']:.2f} s old")
    report.add("world", "it carries free space", (wm.get("free_space") or {}).get("free_ahead_m") is not None,
               f"free ahead {(wm.get('free_space') or {}).get('free_ahead_m')} m")


def check_free_space(scene, report):
    scene.clear()
    time.sleep(2.0)
    clear_ahead = get("/api/grid?span_m=8")["free_ahead_m"]
    if clear_ahead < 8.0:
        # the van is parked somewhere hemmed in; that is a fact about the spot, not a fault
        report.add("free space", "the van is parked with room to test in", None,
                   f"only {clear_ahead:.1f} m ahead here, so the rest of this group is skipped")
        return
    report.add("free space", "a clear road reads as clear", True, f"{clear_ahead:.1f} m ahead")
    car = scene.place("vehicle.tesla.model3", ahead_m=10.0, lift=0.3)
    if car is None:
        report.add("free space", "something in the way shortens it", None, "could not place a car")
        return
    time.sleep(3.0)
    blocked = get("/api/grid?span_m=8")["free_ahead_m"]
    report.add("free space", "something in the way shortens it", blocked < clear_ahead - 1.0,
               f"{clear_ahead:.1f} m -> {blocked:.1f} m with a car at 10 m")
    scene.clear()


def check_road_edge(scene, report):
    g = get("/api/grid?span_m=8")
    kerbs = g.get("kerbs") or {}
    found = [s for s in ("left", "right") if (kerbs.get(s) or {}).get("confident")]
    if not found:
        report.add("road edge", "a kerb is found where there is one", None,
                   "no kerb within reach at this spot")
        return
    side = found[0]
    e = kerbs[side]
    report.add("road edge", "a kerb is found where there is one", abs(e["offset_m"]) < 8.0,
               f"{side} at {e['offset_m']:+.2f} m, {e['heading_deg']:+.1f} deg, over {e['length_m']:.0f} m")


def check_naming(scene, report):
    cases = [("a person", "walker.pedestrian.0001", 1.0, "pedestrian", (1.2, 2.1)),
             ("a car", "vehicle.tesla.model3", 0.3, "vehicle", (1.2, 2.1)),
             ("a barrel", "static.prop.barrel", 0.05, "obstacle", (0.5, 1.1))]
    for label, bp, lift, want, height_range in cases:
        scene.clear()
        actor = scene.place(bp, lift=lift)
        if actor is None:
            report.add("naming", f"{label} is named a {want}", None, "could not place it")
            continue
        time.sleep(3.5)
        o = scene.reported_near(actor)
        if o is None:
            report.add("naming", f"{label} is named a {want}", False, "not reported at all")
            continue
        ok = o["type"] == want
        lo, hi = height_range
        tall_ok = lo <= o["height_m"] <= hi
        report.add("naming", f"{label} is named a {want}", ok,
                   f"called '{o['type']}', {o['length_m']:.1f} x {o['width_m']:.1f} x {o['height_m']:.1f} m")
        report.add("sizes", f"{label} is measured about the right height", tall_ok,
                   f"{o['height_m']:.2f} m (expected {lo} to {hi})")
        report.add("sizes", f"{label} gets sensible room to leave", o["clearance_radius_m"] >= 0.4,
                   f"{o['clearance_radius_m']:.2f} m")
    scene.clear()


def check_no_phantom_vehicles(scene, report):
    scene.clear()
    time.sleep(2.5)
    wm = get("/api/world")
    tall = [o for o in wm["objects"] if o["height_m"] > 3.0]
    tall_named = [o for o in tall if o["type"] == "vehicle"]
    report.add("naming", "nothing tall is called a vehicle", not tall_named,
               f"{len(tall)} things over 3 m, {len(tall_named)} of them called vehicles")
    report.add("naming", "an empty road holds few vehicle labels", wm["counts"]["vehicles"] <= 2,
               f"{wm['counts']['vehicles']} vehicles among {wm['counts']['total']} things")


def check_motion(scene, report):
    scene.clear()
    barrel = scene.place("static.prop.barrel", ahead_m=9.0, lateral=4.5)
    far = scene.map.get_waypoint(scene.van.get_transform().location).next(40.0)
    if not far:
        report.add("motion", "a moving car is seen to move", None, "no room to drive a car")
        return
    t = far[0].transform
    car = scene.world.try_spawn_actor(scene.bl.find("vehicle.tesla.model3"), carla.Transform(
        carla.Location(x=t.location.x, y=t.location.y, z=t.location.z + 0.3), t.rotation))
    if car is None:
        report.add("motion", "a moving car is seen to move", None, "could not place a car")
        return
    scene.spawned.append(car)
    car.set_simulate_physics(True)
    fwd = t.get_forward_vector()
    # Two questions, not one: how quickly does the van notice, and how good is the number
    # once it has settled? Reading the speed at the first moving frame answers neither
    # fairly: a track only two sightings old has taken its speed from those two positions
    # and it can be half as much again out.
    first = None
    readings = []
    started = time.time()
    for _ in range(16):
        car.set_target_velocity(carla.Vector3D(x=-fwd.x * 6.0, y=-fwd.y * 6.0, z=0.0))
        time.sleep(0.8)
        v = car.get_velocity()
        truth = math.hypot(v.x, v.y)
        o = scene.reported_near(car, reach=3.0)
        if o is None or truth < 4.0:
            continue
        if not o["stationary"]:
            if first is None:
                first = (time.time() - started, truth, o["speed"])
            readings.append((truth, o["speed"]))
        if len(readings) >= 4:
            break
    if first is None:
        report.add("motion", "a moving car is seen to move", False, "never reported as moving")
        return
    delay, truth0, said0 = first
    report.add("motion", "a moving car is noticed quickly", delay <= 4.0,
               f"called moving {delay:.1f} s in, first guess {said0:.1f} m/s against {truth0:.1f}")
    settled = readings[-1]
    report.add("motion", "and its speed settles on the right number",
               abs(settled[1] - settled[0]) <= 1.5,
               f"really {settled[0]:.1f} m/s, van says {settled[1]:.1f} m/s "
               f"after {len(readings)} readings")
    car.set_target_velocity(carla.Vector3D())      # stop driving it before checking the barrel
    time.sleep(1.5)
    if barrel is not None:
        # only the barrel itself: the car was driven right past this spot a moment ago
        b = scene.reported_near(barrel, reach=1.5)
        if b is None:
            report.add("motion", "a parked barrel is seen as parked", None, "the barrel was not reported")
        else:
            report.add("motion", "a parked barrel is seen as parked", b["stationary"] is True,
                       f"parked={b['stationary']}, speed {b['speed']:.1f} m/s")
    scene.clear()


def check_sensor_faults(scene, report):
    scene.clear()
    try:
        post("/api/test/disable_camera")
        time.sleep(4.0)
        st = get("/api/state")
        sensors = st.get("sensors") or {}
        ok = st["safety"]["state"] == "degraded" and st["safety"].get("driving_allowed") is True
        report.add("faults", "losing the camera slows the van, it does not freeze it", ok,
                   f"{st['safety']['state']}, cap {st['safety'].get('speed_cap_mps')} m/s: {sensors.get('reason','')}")
        report.add("faults", "the report names the camera", "camera" in sensors.get("reason", ""),
                   sensors.get("reason", ""))
    finally:
        post("/api/test/enable_camera")
        time.sleep(3.0)
    try:
        post("/api/test/disable_lidar")
        time.sleep(4.0)
        st = get("/api/state")
        sensors = st.get("sensors") or {}
        ok = st["safety"]["state"] == "intervention" and st["safety"].get("speed_cap_mps") == 0.0
        report.add("faults", "losing the laser stops the van", ok,
                   f"{st['safety']['state']}: {sensors.get('reason','')}")
        report.add("faults", "the report names the laser", "lidar" in sensors.get("reason", ""),
                   sensors.get("reason", ""))
    finally:
        post("/api/test/enable_lidar")
        time.sleep(3.0)
    st = get("/api/state")
    report.add("faults", "everything comes back afterwards", st["safety"]["state"] == "ok",
               (st.get("sensors") or {}).get("reason", ""))


def check_drive(scene, report):
    """The oldest test in the project: a barrel in the lane, and the van must not hit it."""
    scene.clear()
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    runner = root / "scenarios" / "run_scenario.py"
    if not runner.exists():
        report.add("driving", "it stops for a barrel in its lane", None, "the scenario runner is missing")
        return
    subprocess.run([sys.executable, str(runner), "WAV-0294", "--api", API,
                    "--carla-host", "127.0.0.1"], cwd=str(root), capture_output=True, timeout=300)
    result = root / "scenarios" / "results" / "WAV-0294.json"
    if not result.exists():
        report.add("driving", "it stops for a barrel in its lane", False, "the drive produced no result")
        return
    data = json.loads(result.read_text())
    checks = {c["metric"]: c for c in data.get("checks", [])}
    hit = data["metrics"].get("collision_count", -1)
    gap = checks.get("min_distance_to_actor_m", {}).get("actual")
    report.add("driving", "it stops for a barrel in its lane", data.get("verdict") == "PASS",
               f"{data.get('verdict')}, {hit} contacts, stopped {gap} m short")


GROUPS = {
    "health": check_health,
    "world": check_world_sheet,
    "free space": check_free_space,
    "road edge": check_road_edge,
    "naming": check_naming,
    "phantoms": check_no_phantom_vehicles,
    "motion": check_motion,
    "faults": check_sensor_faults,
    "driving": check_drive,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="run just this group (repeatable)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--skip-drive", action="store_true", help="skip the slow drive test")
    ap.add_argument("--at", default=None, help="park the van at x,y first, on a road with room")
    a = ap.parse_args()
    if a.list:
        print("groups:", ", ".join(GROUPS))
        return 0
    groups = a.only or [g for g in GROUPS if not (a.skip_drive and g == "driving")]
    scene = Scene()
    scene.chase_camera()
    if a.at:
        x, y = (float(v) for v in a.at.split(","))
        wp = scene.map.get_waypoint(carla.Location(x=x, y=y, z=0.0))
        scene.van.set_transform(carla.Transform(
            carla.Location(x=x, y=y, z=wp.transform.location.z + 0.3),
            carla.Rotation(yaw=wp.transform.rotation.yaw)))
        time.sleep(3.0)
        scene.chase_camera()
    report = Report()
    print(f"checking the van at ({scene.van.get_transform().location.x:.1f}, "
          f"{scene.van.get_transform().location.y:.1f})\n")
    try:
        for name in groups:
            fn = GROUPS.get(name)
            if fn is None:
                print(f"  (no such group: {name})")
                continue
            print(f"-- {name}")
            scene.chase_camera()        # keep the simulator's window on the van
            try:
                fn(scene, report)
            except Exception as e:
                report.add(name, f"the {name} checks ran", False, f"they raised {e!r}")
    finally:
        scene.clear()
        for sensor in ("camera", "lidar", "gnss", "imu"):
            try:
                post(f"/api/test/enable_{sensor}")
            except Exception:
                pass
    return report.summary()


if __name__ == "__main__":
    sys.exit(main())
