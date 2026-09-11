#!/usr/bin/env python3
"""
Live check for fix 2: does the van pass a car parked beside its lane, and does it measure
the car where it really is?

    python tools/check_parked_pass.py --right 2.9          # near side just outside the lane
    python tools/check_parked_pass.py --right 2.2          # half a metre into the lane

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
    ap.add_argument("--right", type=float, default=2.9, help="car centre, metres right of the lane centre")
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
    wp = cmap.get_waypoint(van.get_location())
    # a straight stretch round the car -- 12 m before it to 8 m after, no junction, heading
    # within 6 degrees -- so the van drives up to it straight, whatever it crossed before
    ahead = a.ahead
    while True:
        ref = wp.next(ahead)[0].transform.rotation.yaw
        seg = [wp.next(d)[0] for d in range(max(2, int(ahead) - 12), int(ahead) + 9, 2)]
        bend = max(abs((w.transform.rotation.yaw - ref + 180) % 360 - 180) for w in seg)
        if not any(w.is_junction for w in seg) and bend < 6.0:
            break
        ahead += 4.0
        if ahead > 90:
            sys.exit("no straight stretch ahead of the van: move it first")
    at = wp.next(ahead)[0].transform
    yaw = math.radians(at.rotation.yaw)
    loc = carla.Location(at.location.x - math.sin(yaw) * a.right, at.location.y + math.cos(yaw) * a.right,
                         at.location.z + 0.3)
    car = world.try_spawn_actor(world.get_blueprint_library().find(a.car), carla.Transform(loc, at.rotation))
    if car is None:
        sys.exit("could not park the car there")
    car.set_simulate_physics(False)
    lane_half = wp.lane_width / 2.0
    print(f"car parked {ahead:.0f} m ahead, centre {a.right:.2f} m right of the lane centre "
          f"(lane half-width {lane_half:.2f}; car's near side {a.right - car.bounding_box.extent.y - lane_half:+.2f} m "
          f"outside the lane edge)", flush=True)
    spectator = world.get_spectator()
    dest = wp.next(ahead + a.dest - a.ahead)[0].transform.location
    rows = []
    try:
        time.sleep(1.5)
        print("mission:", post("/api/mission/start", {"x": dest.x, "y": dest.y}), flush=True)
        route = []
        for _ in range(20):
            try:
                route = get("/api/route")
            except Exception:
                route = []
            if len(route) >= 2:
                break
            time.sleep(0.2)
        # does the van's route really run along the car's lane, and how far off the car?
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
            print(f"the van's route passes the car's centre {best[0]:.2f} m off, heading "
                  f"{(best[1] - ct0.rotation.yaw + 180) % 360 - 180:+.1f} deg to the car", flush=True)
        t0 = time.time()
        passed_at = None
        while time.time() - t0 < a.timeout:
            vt = van.get_transform()
            vy = math.radians(vt.rotation.yaw)
            spectator.set_transform(carla.Transform(
                carla.Location(vt.location.x - 11 * math.cos(vy), vt.location.y - 11 * math.sin(vy), vt.location.z + 6),
                carla.Rotation(pitch=-22, yaw=vt.rotation.yaw)))
            s = get("/api/state")
            ct = car.get_transform()
            # truth in the van's frame (x ahead, y right)
            dx, dy = ct.location.x - vt.location.x, ct.location.y - vt.location.y
            tx, ty = dx * math.cos(vy) + dy * math.sin(vy), -dx * math.sin(vy) + dy * math.cos(vy)
            tyaw = (ct.rotation.yaw - vt.rotation.yaw + 180) % 360 - 180
            objs = [o for o in (s.get("perception") or {}).get("objects", []) if o.get("ego_x") is not None]
            near = [o for o in objs if math.hypot(o["ego_x"] - tx, o["ego_y"] - ty) < 3.0]
            o = min(near, key=lambda o: math.hypot(o["ego_x"] - tx, o["ego_y"] - ty)) if near else None
            pl = s.get("planner") or {}
            row = {"t": round(time.time() - t0, 1), "speed": s["pose"]["speed"], "behavior": s.get("behavior"),
                   "why": (s.get("behavior_reason") or "")[:80], "blocker": (pl.get("blocker_id"), pl.get("blocker_kind"),
                                                                         pl.get("blocker_lateral_m")),
                   "reason": pl.get("reason"), "passing": pl.get("passing_id"),
                   "gap": round(gap(corners(vt, van.bounding_box.extent), corners(ct, car.bounding_box.extent)), 2),
                   "truth": (round(tx, 2), round(ty, 2), round(tyaw, 1))}
            if o is not None:
                row.update({"avg_err": round(math.hypot(o["ego_x"] - tx, o["ego_y"] - ty), 2),
                            "box_err": round(math.hypot(o["ego_x"] + o.get("box_dx", 0) - tx,
                                                        o["ego_y"] + o.get("box_dy", 0) - ty), 2),
                            "yaw_err_spread": round(abs((o.get("yaw_deg", 0) - tyaw + 90) % 180 - 90), 1),
                            "yaw_err_fitted": (round(abs((o.get("box_yaw_deg", 0) - tyaw + 90) % 180 - 90), 1)
                                               if o.get("box_length_m") else None),
                            "type": o.get("type"), "size": (o.get("length_m"), o.get("width_m"))})
            rows.append(row)
            if o is not None:
                # the near side as the van measured it, against the truth: what a pass depends on
                row["near_side_err"] = round((abs(o["ego_y"] + o.get("box_dy", 0)) - 0.5 * (o.get("box_width_m") or o.get("width_m") or 0))
                                             - (abs(ty) - car.bounding_box.extent.y), 2)
            if tx < -6.0 and passed_at is None:
                passed_at = row["t"]
                print(f"passed the car at t={passed_at:.0f} s", flush=True)
            if passed_at is not None and row["t"] - passed_at > 2.0:
                break
            time.sleep(0.2)
    finally:
        try:
            post("/api/mission/stop")
        except Exception:
            pass
        car.destroy()

    seen = [r for r in rows if "avg_err" in r and 3.0 < r["truth"][0] < 20.0]
    alongside = [r for r in rows if -6.0 < r["truth"][0] < 6.0]
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"rows": rows, "route": route}, fh)
    before = [r for r in rows if 0.0 < r["truth"][0] < 20.0]
    print("\nRESULT")
    print(f"  blocked while the car was 0-20 m ahead: {sum(1 for r in before if r['reason'] not in (None, 'clear'))} "
          f"of {len(before)} readings; passing with care: {sum(1 for r in before if r['passing'])}")
    ns = [r["near_side_err"] for r in rows if "near_side_err" in r and 3.0 < r["truth"][0] < 20.0]
    if ns:
        print(f"  the car's near side as measured: {statistics.median(ns):+.2f} m from the truth (median; - = nearer the van)")
    print(f"  passed the car: {'YES' if any(r['truth'][0] < -6 for r in rows) else 'NO'}; "
          f"closest real gap between the bodies: {min(r['gap'] for r in rows):.2f} m")
    if seen:
        print(f"  where the van put the car (3-20 m ahead, {len(seen)} readings): average of points "
              f"{statistics.median(r['avg_err'] for r in seen):.2f} m off the truth, fitted rectangle "
              f"{statistics.median(r['box_err'] for r in seen):.2f} m off (medians)")
        fitted = [r["yaw_err_fitted"] for r in seen if r.get("yaw_err_fitted") is not None]
        print(f"  its heading: from the spread of points {statistics.median(r['yaw_err_spread'] for r in seen):.1f} deg off, "
              f"fitted rectangle {statistics.median(fitted) if fitted else float('nan'):.1f} deg off (medians)")
    if alongside:
        print(f"  speed alongside: {min(r['speed'] for r in alongside):.1f}-{max(r['speed'] for r in alongside):.1f} m/s")
    print(f"  readings blocked: {sum(1 for r in rows if r['reason'] not in (None, 'clear'))} of {len(rows)}; "
          f"passing with care: {sum(1 for r in rows if r['passing'])}")
    stuck = [r for r in rows if r["reason"] not in (None, "clear")]
    if stuck:
        print("  blocked by:", stuck[-1])


if __name__ == "__main__":
    main()
