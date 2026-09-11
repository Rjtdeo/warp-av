#!/usr/bin/env python3
"""
Live check for fix 2: does the van pass a car parked beside its lane, and does it measure
the car where it really is?

    python tools/check_parked_pass.py --right 2.9          # near side just outside the lane
    python tools/check_parked_pass.py --right 2.2          # half a metre into the lane
    python tools/check_parked_pass.py --right 2.9,2.6 --spacing 30 --show   # a demo, labelled

With --show each parked car carries a label in the CARLA window: how much room it really
leaves, and what the van has decided about it -- pass, pass slowly, or stop. CARLA text is
drawn for the spectator only; the van's cameras never see it.

Parks a car (physics off, parallel to the lane) --ahead metres along the van's lane and
--right metres from its centre, starts a mission past it, and five times a second compares
what the van reports for it with CARLA's truth, in the van's frame: the average of its points
(x, y), the fitted rectangle's centre (x + box_dx, y + box_dy), and its heading. It also
measures the real gap between the two bodies and the van's speed alongside. Removes the car
and stops the mission at the end. Read-only towards the stack otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.request

import carla

API = "http://127.0.0.1:5000"


def get(path):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def corners(tf, ext):
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append((tf.location.x + c * sx * ext.x - s * sy * ext.y, tf.location.y + s * sx * ext.x + c * sy * ext.y))
    return out


def gap(a, b):
    """Distance between two convex polygons (0 when they touch)."""
    def seg_dist(p, q, r):
        qx, qy = q[0] - p[0], q[1] - p[1]
        L2 = qx * qx + qy * qy or 1e-9
        t = max(0.0, min(1.0, ((r[0] - p[0]) * qx + (r[1] - p[1]) * qy) / L2))
        return math.hypot(r[0] - (p[0] + t * qx), r[1] - (p[1] + t * qy))
    best = float("inf")
    for P_, Q_ in ((a, b), (b, a)):
        for i in range(4):
            for r in Q_:
                best = min(best, seg_dist(P_[i], P_[(i + 1) % 4], r))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ahead", type=float, default=22.0)
    ap.add_argument("--right", default="2.9",
                    help="car centre, metres right of the lane centre; several, comma separated")
    ap.add_argument("--spacing", type=float, default=30.0, help="metres between the cars")
    ap.add_argument("--show", action="store_true", help="label the cars in the CARLA window")
    ap.add_argument("--at", default=None,
                    help="first move the (idle) van to the lane at x,y -- as tools/full_check.py --at does")
    ap.add_argument("--dest", type=float, default=70.0)
    ap.add_argument("--car", default="vehicle.tesla.model3")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--out", default=None, help="write every reading, and the route, here (json)")
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    st = get("/api/state")
    if st.get("mission", {}).get("state") not in (None, "idle", "completed", "failed", "cancelled"):
        sys.exit("stack busy")
    van = min(world.get_actors().filter("vehicle.*"),
              key=lambda v: math.hypot(v.get_location().x - st["pose"]["x"], v.get_location().y - st["pose"]["y"]))
    if a.at:
        x, y = (float(v) for v in a.at.split(","))
        w0 = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.0))
        van.set_transform(carla.Transform(carla.Location(w0.transform.location.x, w0.transform.location.y,
                                                         w0.transform.location.z + 0.3), w0.transform.rotation))
        time.sleep(4.0)                     # let it settle, and the stack's view catch up
    wp = cmap.get_waypoint(van.get_location())
    rights = [float(v) for v in str(a.right).split(",")]
    span = a.spacing * (len(rights) - 1)
    # a straight stretch round the cars -- 12 m before the first to 8 m after the last, no
    # junction, heading within 6 degrees -- so the van drives up to them straight
    ahead = a.ahead
    while True:
        ref = wp.next(ahead)[0].transform.rotation.yaw
        seg = [wp.next(d)[0] for d in range(max(2, int(ahead) - 12), int(ahead + span) + 9, 2)]
        bend = max(abs((w.transform.rotation.yaw - ref + 180) % 360 - 180) for w in seg)
        if not any(w.is_junction for w in seg) and bend < 6.0:
            break
        ahead += 4.0
        if ahead > 160:
            sys.exit("no straight stretch ahead of the van: move it first")
    lane_half = wp.lane_width / 2.0
    cars = []
    for k, right in enumerate(rights):
        car = None
        for shift in (0.0, 3.0, 6.0, -3.0, 9.0):        # something already there? a little further on
            at = wp.next(ahead + k * a.spacing + shift)[0].transform
            yaw = math.radians(at.rotation.yaw)
            loc = carla.Location(at.location.x - math.sin(yaw) * right, at.location.y + math.cos(yaw) * right,
                                 at.location.z + 0.3)
            car = world.try_spawn_actor(world.get_blueprint_library().find(a.car), carla.Transform(loc, at.rotation))
            if car is not None:
                break
        if car is None:
            for c_ in cars:
                c_.destroy()
            sys.exit("could not park a car there")
        car.set_simulate_physics(False)
        room = right - car.bounding_box.extent.y - 0.994
        cars.append(car)
        print(f"car {k + 1}: {ahead + k * a.spacing:.0f} m ahead, centre {right:.2f} m right of the lane centre "
              f"-- near side {right - car.bounding_box.extent.y - lane_half:+.2f} m outside the lane edge, "
              f"{room:.2f} m of room beside a van on the lane centre", flush=True)
    spectator = world.get_spectator()
    dest = wp.next(ahead + span + max(30.0, a.dest - a.ahead))[0].transform.location

    import threading
    done = threading.Event()

    def chase():
        while not done.is_set():
            try:
                vt = van.get_transform()
                vy = math.radians(vt.rotation.yaw)
                spectator.set_transform(carla.Transform(
                    carla.Location(vt.location.x - 12 * math.cos(vy), vt.location.y - 12 * math.sin(vy), vt.location.z + 6.5),
                    carla.Rotation(pitch=-22, yaw=vt.rotation.yaw)))
            except Exception:
                pass
            time.sleep(0.05)

    rows = []
    per_car = [[] for _ in cars]
    route = []
    try:
        threading.Thread(target=chase, daemon=True).start()
        time.sleep(1.5)
        print("mission:", post("/api/mission/start", {"x": dest.x, "y": dest.y}), flush=True)
        for _ in range(20):
            try:
                route = get("/api/route")
            except Exception:
                route = []
            if len(route) >= 2:
                break
            time.sleep(0.2)
        # does the van's route really run along the cars' lane, and how far off each car?
        for k, car in enumerate(cars):
            ct0 = car.get_transform()
            best = None
            for p0, p1 in zip(route, route[1:]):
                vx_, vy_ = p1["x"] - p0["x"], p1["y"] - p0["y"]
                L2 = vx_ * vx_ + vy_ * vy_ or 1e-9
                t_ = max(0.0, min(1.0, ((ct0.location.x - p0["x"]) * vx_ + (ct0.location.y - p0["y"]) * vy_) / L2))
                d_ = math.hypot(ct0.location.x - (p0["x"] + t_ * vx_), ct0.location.y - (p0["y"] + t_ * vy_))
                if best is None or d_ < best[0]:
                    best = (d_, math.degrees(math.atan2(vy_, vx_)))
            if best:
                print(f"car {k + 1}: the van's route passes its centre {best[0]:.2f} m off, heading "
                      f"{(best[1] - ct0.rotation.yaw + 180) % 360 - 180:+.1f} deg to it", flush=True)
        t0 = time.time()
        passed_at = None
        stopped_since = None
        overtook = False
        last_behaviour = None
        while time.time() - t0 < a.timeout:
            vt = van.get_transform()
            vy = math.radians(vt.rotation.yaw)
            s = get("/api/state")
            pl = s.get("planner") or {}
            objs = [o for o in (s.get("perception") or {}).get("objects", []) if o.get("ego_x") is not None]
            for k, car in enumerate(cars):
                ct = car.get_transform()
                dx, dy = ct.location.x - vt.location.x, ct.location.y - vt.location.y
                tx, ty = dx * math.cos(vy) + dy * math.sin(vy), -dx * math.sin(vy) + dy * math.cos(vy)
                tyaw = (ct.rotation.yaw - vt.rotation.yaw + 180) % 360 - 180
                near = [o for o in objs if math.hypot(o["ego_x"] - tx, o["ego_y"] - ty) < 3.0]
                o = min(near, key=lambda o: math.hypot(o["ego_x"] - tx, o["ego_y"] - ty)) if near else None
                oid = o.get("id") if o else None
                in_lane = (rights[k] - car.bounding_box.extent.y) < lane_half - 0.3
                if s.get("overtaking") and in_lane and tx > -12.0:
                    verdict, rgb = "going round it (overtake)", (0, 200, 255)
                elif oid is not None and oid == pl.get("blocker_id") and in_lane:
                    waited = (time.time() - stopped_since) if stopped_since else 0.0
                    verdict, rgb = f"stopped behind it: {waited:.0f} s (overtakes after 10 s)", (255, 120, 40)
                elif oid is not None and oid == pl.get("blocker_id"):
                    verdict, rgb = "too close: stopping", (255, 60, 60)
                elif oid is not None and oid == pl.get("passing_id"):
                    verdict, rgb = "passing slowly, with care", (255, 200, 0)
                elif o is not None:
                    verdict, rgb = "room to pass", (60, 220, 60)
                else:
                    verdict, rgb = "not seen yet", (170, 170, 170)
                gap_now = gap(corners(vt, van.bounding_box.extent), corners(ct, car.bounding_box.extent))
                if a.show and -12.0 < tx < 45.0:
                    room = rights[k] - car.bounding_box.extent.y - 0.994
                    what = ("stopped car IN the lane" if in_lane else f"parked car: {room:.2f} m of room")
                    if "static.prop" in a.car:
                        what = "cone IN the lane" if "cone" in a.car else "object IN the lane"
                    world.debug.draw_string(carla.Location(ct.location.x, ct.location.y, ct.location.z + 2.3),
                                            what, draw_shadow=True,
                                            color=carla.Color(255, 255, 255), life_time=0.32)
                    world.debug.draw_string(carla.Location(ct.location.x, ct.location.y, ct.location.z + 1.8),
                                            f"van: {verdict}", draw_shadow=True,
                                            color=carla.Color(*rgb), life_time=0.32)
                row = {"t": round(time.time() - t0, 1), "car": k + 1, "speed": s["pose"]["speed"],
                       "behavior": s.get("behavior"), "why": (s.get("behavior_reason") or "")[:80],
                       "reason": pl.get("reason"), "blocker": (pl.get("blocker_id"), pl.get("blocker_kind"),
                                                                pl.get("blocker_lateral_m")),
                       "passing": pl.get("passing_id") if oid is not None and pl.get("passing_id") == oid else None,
                       "blocked_by_it": oid is not None and oid == pl.get("blocker_id"),
                       "gap": round(gap_now, 2), "truth": (round(tx, 2), round(ty, 2), round(tyaw, 1))}
                if o is not None:
                    row.update({"avg_err": round(math.hypot(o["ego_x"] - tx, o["ego_y"] - ty), 2),
                                "box_err": round(math.hypot(o["ego_x"] + o.get("box_dx", 0) - tx,
                                                            o["ego_y"] + o.get("box_dy", 0) - ty), 2),
                                "yaw_err_spread": round(abs((o.get("yaw_deg", 0) - tyaw + 90) % 180 - 90), 1),
                                "yaw_err_fitted": (round(abs((o.get("box_yaw_deg", 0) - tyaw + 90) % 180 - 90), 1)
                                                   if o.get("box_length_m") else None),
                                "near_side_err": round((abs(o["ego_y"] + o.get("box_dy", 0))
                                                        - 0.5 * (o.get("box_width_m") or o.get("width_m") or 0))
                                                       - (abs(ty) - car.bounding_box.extent.y), 2)})
                per_car[k].append(row)
                rows.append(row)
            if s["pose"]["speed"] < 0.2 and s.get("behavior", "").startswith("stopped"):
                stopped_since = stopped_since or time.time()
            else:
                stopped_since = None
            if s.get("overtaking") and not overtook:
                overtook = True
                print(f"t={time.time() - t0:.0f} s: overtake started", flush=True)
            if s.get("behavior") != last_behaviour:
                last_behaviour = s.get("behavior")
                print(f"t={time.time() - t0:5.1f} s: {last_behaviour} -- {(s.get('behavior_reason') or '')[:70]}", flush=True)
            if a.show:
                tag = "  OVERTAKING" if s.get("overtaking") else ""
                world.debug.draw_string(carla.Location(vt.location.x, vt.location.y, vt.location.z + 3.4),
                                        f"{s['pose']['speed']:.1f} m/s  {s.get('behavior')}{tag}", draw_shadow=True,
                                        color=carla.Color(255, 255, 255), life_time=0.32)
            last_tx = per_car[-1][-1]["truth"][0]
            if last_tx < -8.0 and passed_at is None:
                passed_at = time.time() - t0
                print(f"passed the last car at t={passed_at:.0f} s", flush=True)
            if passed_at is not None and time.time() - t0 - passed_at > 2.0:
                break
            time.sleep(0.2)
    finally:
        done.set()
        try:
            post("/api/mission/stop")
        except Exception:
            pass
        for car in cars:
            car.destroy()

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows, "route": route}, fh)
    print("\nRESULT")
    for k in range(len(cars)):
        cr = per_car[k]
        before = [r for r in cr if 0.0 < r["truth"][0] < 20.0]
        alongside = [r for r in cr if -6.0 < r["truth"][0] < 6.0]
        fitted = [r["yaw_err_fitted"] for r in cr if r.get("yaw_err_fitted") is not None and 3.0 < r["truth"][0] < 20.0]
        print(f"  car {k + 1} ({rights[k]:.2f} m right): passed {'YES' if any(r['truth'][0] < -6 for r in cr) else 'NO'}, "
              f"closest real gap {min(r['gap'] for r in cr):.2f} m, "
              f"speed alongside {min((r['speed'] for r in alongside), default=0):.1f}-"
              f"{max((r['speed'] for r in alongside), default=0):.1f} m/s, "
              f"stopped for it on {sum(1 for r in before if r['blocked_by_it'])} of {len(before)} readings, "
              f"passing it slowly on {sum(1 for r in before if r['passing'])}; heading measured "
              f"{statistics.median(fitted) if fitted else float('nan'):.1f} deg off")


if __name__ == "__main__":
    main()
