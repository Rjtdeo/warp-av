#!/usr/bin/env python3
"""
Drop a mixed obstacle course on the van's road and watch what it does with each thing.

    python tools/obstacle_course.py --at=-18,140.3 \\
        --things "car@-3.5@10,bin@2.8@18,car@0@30,cone@2.4@40,car@-3.5@62,box@1.9@70,barrel@0@80"

Each thing is  what@right@ahead[@back] : a short name (or any CARLA blueprint id), how far RIGHT of
the van's lane centre its middle stands (negative = left, -3.5 = the next lane over), and how
many metres ahead along the lane, and "back" to face it the other way (a car parked in the
oncoming lanes). Things stand still (physics off), parallel to the lane.

Then a mission runs past them all, the CARLA window follows the van, and five times a second
the van's own decisions are recorded against CARLA's truth. For every thing: did the van get
past it, how close did its body come (a real gap between the two outlines), did it stop for it
and for how long, and did it swing out to pass it. The go-around's own reasons for waiting
come from the stack log. Everything dropped is removed at the end, and the mission stopped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import carla  # noqa: E402
from check_parked_pass import corners, gap, get, post  # noqa: E402

ALIASES = {
    "car": "vehicle.tesla.model3", "suv": "vehicle.nissan.patrol", "van": "vehicle.volkswagen.t2",
    "truck": "vehicle.carlamotors.carlacola", "ambulance": "vehicle.ford.ambulance",
    "motorbike": "vehicle.yamaha.yzf", "bicycle": "vehicle.bh.crossbike",
    "box": "static.prop.box01", "boxes": "static.prop.box02", "barrel": "static.prop.barrel",
    "cone": "static.prop.trafficcone01", "workcone": "static.prop.constructioncone",
    "barrier": "static.prop.streetbarrier", "worksign": "static.prop.warningconstruction",
    "bin": "static.prop.bin", "trashcan": "static.prop.trashcan01", "garbage": "static.prop.garbage01",
    "trolley": "static.prop.shoppingcart", "chair": "static.prop.plasticchair",
    "table": "static.prop.table", "suitcase": "static.prop.travelcase", "bench": "static.prop.bench01",
}
VAN_HALF_LEN, VAN_HALF_WID = 2.95, 0.99


def yaw_gap(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def along_lane(wp, metres):
    """The waypoint `metres` further down this lane, keeping its heading at every fork (a
    plain wp.next(d)[0] can take a branch onto another street)."""
    d = 0.0
    while d < metres - 1e-6:
        step = min(2.0, metres - d)
        options = wp.next(step)
        if not options:
            return None
        wp = min(options, key=lambda n: yaw_gap(n.transform.rotation.yaw, wp.transform.rotation.yaw))
        d += step
    return wp


def box_of(actor):
    """Outline and half length of any actor, whatever its kind."""
    tf = actor.get_transform()
    bb = actor.bounding_box
    loc = tf.transform(bb.location)
    return corners(carla.Transform(loc, tf.rotation), bb.extent), float(bb.extent.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=None, help="first move the idle van to the lane at x,y")
    ap.add_argument("--things", required=True, help="what@right@ahead, comma separated")
    ap.add_argument("--beyond", type=float, default=30.0, help="the destination, metres past the last thing")
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    lib = world.get_blueprint_library()
    st = get("/api/state")
    if (st.get("mission") or {}).get("state") not in (None, "idle", "completed", "failed", "cancelled"):
        sys.exit("the stack is busy with a mission")
    van = min(world.get_actors().filter("vehicle.*"),
              key=lambda v: math.hypot(v.get_location().x - st["pose"]["x"], v.get_location().y - st["pose"]["y"]))
    if a.at:
        x, y = (float(v) for v in a.at.split(","))
        w0 = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.0))
        van.set_transform(carla.Transform(carla.Location(w0.transform.location.x, w0.transform.location.y,
                                                         w0.transform.location.z + 0.3), w0.transform.rotation))
        time.sleep(4.0)
    start_wp = cmap.get_waypoint(van.get_location())

    plan = []
    for item in a.things.split(","):
        what, right, ahead, *back = item.strip().split("@")
        plan.append((what, ALIASES.get(what, what), float(right), float(ahead), bool(back)))
    dropped = []
    try:
        for what, bp_id, right, ahead, back in plan:
            spot = along_lane(start_wp, ahead)
            if spot is None:
                sys.exit(f"the lane ends before {ahead} m")
            t = spot.transform
            h = math.radians(t.rotation.yaw)
            loc = carla.Location(t.location.x - math.sin(h) * right, t.location.y + math.cos(h) * right,
                                 t.location.z + (0.3 if bp_id.startswith("vehicle.") else 0.05))
            bp = lib.find(bp_id)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "course")
            rot = carla.Rotation(yaw=t.rotation.yaw + (180.0 if back else 0.0))
            actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))
            if actor is None:
                print(f"  could not drop {what} at {ahead:.0f} m ({right:+.1f} m) -- something is there", flush=True)
                continue
            try:
                actor.set_simulate_physics(False)
            except Exception:
                pass
            dropped.append({"what": what, "bp": bp_id, "right": right, "ahead": ahead, "actor": actor,
                            "closest_gap": None, "passed_at": None, "stopped_s": 0.0, "swung_out_m": 0.0})
            e = actor.bounding_box.extent
            into = 1.75 - (abs(right) - e.y) if right >= 0 else None   # 3.5 m lane, from its centre
            where = ("in the van's lane" if abs(right) < 1.2 else
                     "partly in the van's lane" if abs(right) < 2.2 else
                     "in the next lane over" if abs(right) >= 3.0 and right < 0 else "beside the lane")
            print(f"dropped {what:9s} {ahead:5.0f} m ahead, {right:+.1f} m from the lane centre ({where}), "
                  f"body {2 * e.x:.2f} x {2 * e.y:.2f} x {2 * e.z:.2f} m"
                  + (f", {into:+.2f} m over the lane edge" if into is not None else ""), flush=True)

        last = max(d["ahead"] for d in dropped) if dropped else 20.0
        dest = along_lane(start_wp, last + a.beyond) or along_lane(start_wp, last + 5.0)
        dest = dest.transform.location

        spectator = world.get_spectator()
        done = threading.Event()

        def chase():
            while not done.is_set():
                try:
                    vt = van.get_transform()
                    vy = math.radians(vt.rotation.yaw)
                    spectator.set_transform(carla.Transform(
                        carla.Location(vt.location.x - 13 * math.cos(vy), vt.location.y - 13 * math.sin(vy),
                                       vt.location.z + 7.0), carla.Rotation(pitch=-24, yaw=vt.rotation.yaw)))
                except Exception:
                    pass
                time.sleep(0.05)

        threading.Thread(target=chase, daemon=True).start()
        log_path = os.path.join(os.path.dirname(HERE), "logs", "stack_out.log")
        log_from = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        hits0 = (get("/api/state").get("collision") or {}).get("count", 0)
        print("mission:", post("/api/mission/start", {"x": dest.x, "y": dest.y}), flush=True)
        t0 = time.time()
        rows = []
        last_why = None
        sx, sy = start_wp.transform.location.x, start_wp.transform.location.y
        sh = math.radians(start_wp.transform.rotation.yaw)
        ended = None
        moved = False
        still_since = None
        dumped = False
        while time.time() - t0 < a.timeout:
            vt = van.get_transform()
            v = van.get_velocity()
            speed = math.hypot(v.x, v.y)
            try:
                s = get("/api/state")
            except Exception:
                time.sleep(0.2)
                continue
            van_poly = corners(vt, carla.Vector3D(VAN_HALF_LEN, VAN_HALF_WID, 1.0))
            # the van's own position along the straight it started on, and how far it swung out
            van_along = (vt.location.x - sx) * math.cos(sh) + (vt.location.y - sy) * math.sin(sh)
            van_side = -(vt.location.x - sx) * math.sin(sh) + (vt.location.y - sy) * math.cos(sh)
            ahead_things = [d for d in dropped if d["passed_at"] is None]
            nearest = min(ahead_things, key=lambda d: d["ahead"]) if ahead_things else None
            for d in dropped:
                poly, half = box_of(d["actor"])
                g = gap(van_poly, poly)
                if d["closest_gap"] is None or g < d["closest_gap"]:
                    d["closest_gap"] = g
                if d["passed_at"] is None and van_along - VAN_HALF_LEN > d["ahead"] + half:
                    d["passed_at"] = round(time.time() - t0, 1)
                if d["passed_at"] is None and abs(van_along - d["ahead"]) < 12.0:
                    d["swung_out_m"] = max(d["swung_out_m"], -van_side)
            moved = moved or speed > 0.5
            if moved and nearest is not None and speed < 0.1 and 0.0 < nearest["ahead"] - van_along < 20.0:
                nearest["stopped_s"] += 0.2
            # stopped a while: say what the van sees around it, once
            still_since = (still_since or time.time()) if (moved and speed < 0.1) else None
            if still_since and time.time() - still_since > 12.0 and not dumped:
                dumped = True
                print("  what the van sees, stopped (x ahead, y right, from the van's centre):", flush=True)
                for o in sorted((s.get("perception") or {}).get("objects") or [], key=lambda o: o.get("distance", 99)):
                    if -12 < o.get("ego_x", 99) < 45 and -9 < o.get("ego_y", 99) < 4:
                        print(f"    {o.get('type'):10s} id {o.get('id')} x {o.get('ego_x'):6.1f} y {o.get('ego_y'):5.1f} "
                              f"dist {o.get('distance'):5.1f} speed {o.get('speed'):4.1f} still {o.get('stationary')} "
                              f"size {o.get('length_m')}x{o.get('width_m')}x{o.get('height_m')} "
                              f"box {o.get('box_length_m')}x{o.get('box_width_m')} at ({o.get('box_dx')},{o.get('box_dy')}) "
                              f"{o.get('motion_class')}", flush=True)
            why = (s.get("behavior_reason") or "")[:110]
            if why != last_why:
                print(f"t={time.time() - t0:6.1f}  at {van_along:5.1f} m, {van_side:+.1f} m  {s.get('behavior')}: {why}", flush=True)
                last_why = why
            rows.append({"t": round(time.time() - t0, 2), "along": round(van_along, 2), "side": round(van_side, 2),
                         "speed": round(speed, 2), "behavior": s.get("behavior"), "why": why,
                         "planner": (s.get("planner") or {}).get("reason")})
            if (s.get("mission") or {}).get("state") in ("completed", "failed", "cancelled") or \
                    (s.get("behavior") == "no_mission" and time.time() - t0 > 5):
                if ended is None:
                    ended = time.time()
                elif time.time() - ended > 1.5:
                    break
            time.sleep(0.2)
        done.set()
    finally:
        try:
            if (get("/api/state").get("mission") or {}).get("state") == "executing":
                post("/api/mission/stop")
        except Exception:
            pass
        for d in dropped:
            try:
                d["actor"].destroy()
            except Exception:
                pass

    try:
        hits = (get("/api/state").get("collision") or {}).get("count", 0) - hits0
    except Exception:
        hits = None
    print("\nWHAT THE VAN DID WITH EACH THING")
    for d in dropped:
        verdict = ("PASSED" if d["passed_at"] is not None else "NOT PASSED")
        swung = f", swung out {d['swung_out_m']:.1f} m" if d["swung_out_m"] > 1.5 else ""
        print(f"  {d['what']:9s} {d['ahead']:5.0f} m, {d['right']:+.1f} m: {verdict}"
              + (f" at t={d['passed_at']}" if d["passed_at"] is not None else "")
              + f"; closest body gap {d['closest_gap']:.2f} m; stopped for it {d['stopped_s']:.0f} s{swung}")
    print(f"  collisions during the run: {hits}")
    try:
        with open(log_path, errors="ignore") as fh:
            fh.seek(log_from)
            why_lines = [ln.strip() for ln in fh if "Overtake" in ln or "overtake" in ln]
        if why_lines:
            print("\nTHE GO-AROUND SAID")
            for ln in why_lines[-8:]:
                print("  " + ln)
    except Exception:
        pass
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows, "things": [{k: v for k, v in d.items() if k != "actor"} for d in dropped]}, fh)


if __name__ == "__main__":
    main()
