#!/usr/bin/env python3
"""
Drive one mission and score it against CARLA's truth: traffic lights, and where it parked.

    python tools/record_mission_truth.py --at=78,139 --dest-x -67.3 --dest-y 28.0 --out run.json

Ten times a second it records the van as CARLA knows it (pose, speed, front bumper), every
traffic light's real colour, the light CARLA says governs the van, and what the stack said
(behaviour and reason, the signal it thinks is next and its colour, the planner, the parking
spot). Lights are left to cycle normally -- nothing is forced.

Afterwards it finds every PAINTED stop bar the van's front bumper crossed, and the light's
real colour at that moment; and, for every light it met on red, how short of the paint the
bumper stopped. The paint comes from tools/data/<town>_stop_bars.json (tools/measure_stop_bars.py
measured it from overhead pictures). It is NOT CARLA's stop waypoint: that is the centre of
the light's trigger box, 1 to 7.8 m short of the paint, and scoring against it called correct
stops red-light runs (the first version of this tool did, 2026-09-11). Nor is it the line the
van steers by, which would agree with the van by construction. A lane with no paint found
borrows the other lane of the same light, else CARLA's junction entry.

And where it ended: the lane type under it, how far it sits into the driving lane, how far
from the destination and the spot.

It never touches the van, its sensors or the lights. --at moves the IDLE van first, as
tools/full_check.py --at does.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request

import carla

API = "http://127.0.0.1:5000"
STATE = {carla.TrafficLightState.Red: "RED", carla.TrafficLightState.Yellow: "YELLOW",
         carla.TrafficLightState.Green: "GREEN", carla.TrafficLightState.Off: "OFF",
         carla.TrafficLightState.Unknown: "UNKNOWN"}


def get(path):
    with urllib.request.urlopen(API + path, timeout=3) as r:
        return json.loads(r.read().decode())


def post(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def paint_lines(cmap, lights):
    """Every light's stop waypoints with the measured paint behind each: (light id, x, y,
    heading, half lane width, metres from the waypoint to the paint, where that came from)."""
    here = os.path.dirname(os.path.abspath(__file__))
    town = cmap.name.split("/")[-1].lower()
    table = {}
    try:
        with open(os.path.join(here, "data", f"{town}_stop_bars.json")) as fh:
            for r in json.load(fh)["lanes"]:
                table[(r["light"], r["road"], r["lane"])] = r
    except FileNotFoundError:
        print(f"WARNING: no measured paint for {town}; run tools/measure_stop_bars.py. "
              f"Judging against CARLA's junction entry instead.")
    out = []
    for tl in lights.values():
        mine = [r for (lid, _, _), r in table.items() if lid == tl.id and r.get("paint_m") is not None]
        for w in tl.get_stop_waypoints():
            t = w.transform
            r = table.get((tl.id, w.road_id, w.lane_id)) or {}
            if r.get("paint_m") is not None:
                paint, source = r["paint_m"], "measured"
            elif mine:
                paint, source = mine[0]["paint_m"], "measured on the light's other lane"
            elif r.get("junction_m") is not None:
                paint, source = r["junction_m"], "CARLA's junction entry (no paint measured)"
            else:
                paint, source = 0.0, "CARLA's stop waypoint (nothing better known)"
            out.append((tl.id, t.location.x, t.location.y, math.radians(t.rotation.yaw),
                        w.lane_width / 2.0, float(paint), source))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=None, help="move the idle van to the lane at x,y first")
    ap.add_argument("--dest-x", type=float, required=True)
    ap.add_argument("--dest-y", type=float, required=True)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--out", default="mission_truth.json")
    ap.add_argument("--take-chosen", action="store_true",
                    help="once the mission has chosen its parking spot, park a car in it "
                         "(/api/test/park_cars take_chosen) -- the van must see it and choose again")
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    st = get("/api/state")
    if st.get("mission", {}).get("state") not in (None, "idle", "completed", "failed", "cancelled"):
        sys.exit("the stack is busy with a mission")
    van = min(world.get_actors().filter("vehicle.*"),
              key=lambda v: math.hypot(v.get_location().x - st["pose"]["x"], v.get_location().y - st["pose"]["y"]))
    if a.at:
        x, y = (float(v) for v in a.at.split(","))
        w0 = cmap.get_waypoint(carla.Location(x=x, y=y, z=0.0))
        van.set_transform(carla.Transform(carla.Location(w0.transform.location.x, w0.transform.location.y,
                                                         w0.transform.location.z + 0.3), w0.transform.rotation))
        time.sleep(4.0)
    front = van.bounding_box.location.x + van.bounding_box.extent.x
    lights = {tl.id: tl for tl in world.get_actors().filter("traffic.traffic_light*")}
    stops = paint_lines(cmap, lights)     # (light_id, x, y, heading, half lane width, paint_m, source)
    spectator = world.get_spectator()
    rows, crossings = [], []
    prev_along = {}
    closest_on_red = {}         # light id -> how short of the paint the bumper got while red
    try:
        print("mission:", post("/api/mission/start", {"x": a.dest_x, "y": a.dest_y}), flush=True)
        route = []
        for _ in range(20):
            try:
                route = get("/api/route")
            except Exception:
                route = []
            if len(route) >= 2:
                break
            time.sleep(0.2)
        if a.take_chosen:
            chosen = (get("/api/state").get("parking_spot") or {})
            print("parking spot chosen at the start:", chosen, flush=True)
            print("a car is parked in it:", post("/api/test/park_cars", {"take_chosen": True, "count": 0}),
                  flush=True)
        t0 = time.time()
        ended = None
        while time.time() - t0 < a.timeout:
            vt = van.get_transform()
            vy = math.radians(vt.rotation.yaw)
            spectator.set_transform(carla.Transform(
                carla.Location(vt.location.x - 12 * math.cos(vy), vt.location.y - 12 * math.sin(vy), vt.location.z + 7),
                carla.Rotation(pitch=-25, yaw=vt.rotation.yaw)))
            fx, fy = vt.location.x + math.cos(vy) * front, vt.location.y + math.sin(vy) * front
            try:
                s = get("/api/state")
            except Exception:
                time.sleep(0.1)
                continue
            gov = van.get_traffic_light()
            sig = s.get("signal_ahead") or {}
            v = van.get_velocity()
            row = {"t": round(time.time() - t0, 2), "x": round(vt.location.x, 2), "y": round(vt.location.y, 2),
                   "spot": s.get("parking_spot"),
                   "yaw": round(vt.rotation.yaw, 1), "speed": round(math.sqrt(v.x * v.x + v.y * v.y), 2),
                   "behavior": s.get("behavior"), "why": (s.get("behavior_reason") or "")[:100],
                   "van_sig_id": sig.get("light_id"), "van_sig_state": sig.get("state"),
                   "van_sig_dist": sig.get("distance_m"),
                   "van_light": s.get("traffic_light"),
                   "carla_light": gov.id if gov is not None else None,
                   "carla_state": STATE.get(van.get_traffic_light_state(), "?") if gov is not None else None,
                   "planner": (s.get("planner") or {}).get("reason"),
                   "mission": (s.get("mission") or {}).get("state")}
            # painted stop bars the front bumper just crossed, going forwards, in that lane;
            # and how near to the paint it came while that light was red
            for (lid, sx, sy, sh, half, paint_m, source) in stops:
                if math.hypot(sx - fx, sy - fy) > 25.0:
                    continue
                along = (fx - sx) * math.cos(sh) + (fy - sy) * math.sin(sh) - paint_m
                side = -(fx - sx) * math.sin(sh) + (fy - sy) * math.cos(sh)
                heading_ok = abs((vt.rotation.yaw - math.degrees(sh) + 180) % 360 - 180) < 50
                if abs(side) > half + 0.5 or not heading_ok:
                    continue
                key = (lid, round(sx, 1), round(sy, 1))
                colour = STATE.get(lights[lid].get_state(), "?")
                if colour == "RED" and -15.0 < along:
                    near = closest_on_red.setdefault(lid, {"short_of_paint_m": None, "t": None,
                                                           "paint_from": source})
                    if near["short_of_paint_m"] is None or -along < near["short_of_paint_m"]:
                        near["short_of_paint_m"], near["t"] = round(-along, 2), row["t"]
                was = prev_along.get(key)
                prev_along[key] = along
                if was is not None and was < 0.0 <= along:
                    crossings.append(dict(row, crossed_light=lid, crossed_state=colour,
                                          paint_from=source))
                    print(f"t={row['t']:6.1f}: front bumper crossed the PAINT of light {lid}: it was "
                          f"{colour} | van thought: light {row['van_sig_id']} "
                          f"{row['van_sig_state']} | {row['behavior']}: {row['why'][:70]}", flush=True)
            rows.append(row)
            if row["mission"] in ("completed", "failed", "cancelled") or (
                    row["behavior"] == "no_mission" and time.time() - t0 > 5):
                if ended is None:
                    ended = time.time()
                elif time.time() - ended > 2.0:
                    break
            time.sleep(0.1)
    finally:
        try:
            if rows and rows[-1]["mission"] == "executing":
                post("/api/mission/stop")
        except Exception:
            pass
        if a.take_chosen:
            try:
                post("/api/test/park_cars", {"clear": True})
            except Exception:
                pass

    # ---- where it ended ---------------------------------------------------------------
    vt = van.get_transform()
    here = cmap.get_waypoint(vt.location, project_to_road=False, lane_type=carla.LaneType.Any)
    drive = cmap.get_waypoint(vt.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    dl = drive.transform
    dh = math.radians(dl.rotation.yaw)
    lat = -(vt.location.x - dl.location.x) * math.sin(dh) + (vt.location.y - dl.location.y) * math.cos(dh)
    rel = math.radians(vt.rotation.yaw) - dh
    hl = van.bounding_box.extent.x
    reach = abs(lat) - (abs(math.sin(rel)) * hl + abs(math.cos(rel)) * van.bounding_box.extent.y)
    into_lane = drive.lane_width / 2.0 - reach
    spot = (s.get("parking_spot") if isinstance(s, dict) else None) or {}
    spot_wp = None
    try:
        spot_wp = cmap.get_waypoint(vt.location, project_to_road=True,
                                    lane_type=carla.LaneType.Parking | carla.LaneType.Shoulder)
    except Exception:
        spot_wp = None
    in_strip = None
    if spot_wp is not None:
        sh = math.radians(spot_wp.transform.rotation.yaw)
        slat = -(vt.location.x - spot_wp.transform.location.x) * math.sin(sh) \
            + (vt.location.y - spot_wp.transform.location.y) * math.cos(sh)
        in_strip = abs(slat) <= spot_wp.lane_width / 2.0
    history = {}
    try:
        history = (get("/api/history") or [{}])[-1]
    except Exception:
        history = {}
    end = {"x": round(vt.location.x, 2), "y": round(vt.location.y, 2), "yaw": round(vt.rotation.yaw, 1),
           "mission_result": history.get("state"), "mission_reason": history.get("reason_ended"),
           "heading_off_lane_deg": round(abs((vt.rotation.yaw - dl.rotation.yaw + 180) % 360 - 180), 1),
           "centre_inside_parking_strip": in_strip,
           "lane_type_under_van": str(here.lane_type) if here is not None else "none",
           "offset_from_driving_lane_centre_m": round(lat, 2),
           "van_body_inside_driving_lane_m": round(max(0.0, into_lane), 2),
           "to_destination_m": round(math.hypot(vt.location.x - a.dest_x, vt.location.y - a.dest_y), 1),
           "parking_spot": spot,
           "to_spot_m": (round(math.hypot(vt.location.x - spot["x"], vt.location.y - spot["y"]), 1)
                         if spot.get("x") is not None else None)}
    with open(a.out, "w") as fh:
        json.dump({"rows": rows, "crossings": crossings, "closest_on_red": closest_on_red,
                   "end": end, "route": route, "dest": [a.dest_x, a.dest_y]}, fh)

    print("\nTRAFFIC LIGHTS (painted stop bars the front bumper crossed)")
    for c in crossings:
        knew = "knew about it" if c["van_sig_id"] == c["crossed_light"] else f"was watching light {c['van_sig_id']}"
        print(f"  light {c['crossed_light']}: {c['crossed_state']:6s} at {c['speed']:.1f} m/s; the van {knew}, "
              f"said {c['van_sig_state']}; {c['behavior']}: {c['why'][:60]}")
    reds = [c for c in crossings if c["crossed_state"] == "RED"]
    print(f"  => {len(crossings)} painted stop bars crossed, {len(reds)} on RED")
    print("\nEVERY LIGHT MET ON RED: how short of the paint the front bumper stayed (negative = over it)")
    for lid, near in sorted(closest_on_red.items()):
        print(f"  light {lid}: {near['short_of_paint_m']:+.2f} m at t={near['t']}  (paint: {near['paint_from']})")
    print("\nWHERE IT ENDED")
    for k, v in end.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
