#!/usr/bin/env python3
"""
Park a few cars in the lanes NEXT TO a mission's route, so a scored run meets traffic.

    python tools/drop_route_cars.py --at=78,139 --dest=-67.3,28 --n 4
    python tools/drop_route_cars.py --clear            # take them away again

Each car stands in a lane the route never uses (the lane beside it, either way round),
clear of junctions, facing its own lane. They never move. The point is what the van does
when something big sits one lane over: on 2026-09-11 a parked SUV in the oncoming lane held
the van behind a dead car for three minutes.

--at moves the idle van first, as record_mission_truth.py --at does, so the route this plans
against is the route the mission will take.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import carla  # noqa: E402
from check_parked_pass import get  # noqa: E402

MODELS = ["vehicle.tesla.model3", "vehicle.nissan.patrol", "vehicle.volkswagen.t2",
          "vehicle.carlamotors.carlacola", "vehicle.ford.ambulance"]
ROLE = "course"
CLEAR_OF_ROUTE_M = 2.6          # the van's half width plus room: never in its way
CLEAR_OF_JUNCTION_M = 12.0


def route_points(world, start, dest):
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    grp = GlobalRoutePlanner(world.get_map(), 2.0)
    pts, arc = [], 0.0
    hops = grp.trace_route(start, dest)
    for i, (wp, _) in enumerate(hops):
        if i:
            arc += wp.transform.location.distance(hops[i - 1][0].transform.location)
        pts.append((arc, wp))
    return pts


def neighbours(wp):
    """The lanes either side, with which way they face, nearest first."""
    out = []
    for side, nb in (("left", wp.get_left_lane()), ("right", wp.get_right_lane())):
        if nb is None or nb.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking) \
                or nb.is_junction:
            continue
        same = (nb.lane_id * wp.lane_id) > 0
        out.append((side, same, nb))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", default=None, help="move the idle van to the lane at x,y first")
    ap.add_argument("--dest", default=None, help="the mission's destination x,y")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--from-m", type=float, default=45.0, help="no nearer the start than this")
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    if a.clear:
        gone = 0
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name") == ROLE:
                v.destroy()
                gone += 1
        print(f"took away {gone} dropped cars")
        return
    if not a.dest:
        sys.exit("--dest x,y is required")

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
    dx, dy = (float(v) for v in a.dest.split(","))
    pts = route_points(world, van.get_location(), carla.Location(x=dx, y=dy, z=0.0))
    total = pts[-1][0]
    print(f"the route is {total:.0f} m long, {len(pts)} points")

    lib = world.get_blueprint_library()
    junctions = [arc for arc, wp in pts if wp.is_junction]

    used = {(wp.road_id, wp.lane_id) for _, wp in pts}      # lanes the route itself drives on

    def clear_of_other_legs(loc, arc_here):
        """Room from any OTHER part of the route (it loops round the block and comes back)."""
        return min([loc.distance(wp.transform.location) for arc, wp in pts if abs(arc - arc_here) > 20.0]
                   or [99.0])

    wanted = [a.from_m + (total - a.from_m - 60.0) * (i + 0.5) / a.n for i in range(a.n)]
    dropped = []
    for k, target in enumerate(wanted):
        best = None
        for arc, wp in pts:
            if abs(arc - target) > 45.0 or wp.is_junction:
                continue
            if any(abs(arc - j) < CLEAR_OF_JUNCTION_M for j in junctions):
                continue
            for side, same, nb in neighbours(wp):
                loc = nb.transform.location
                if (nb.road_id, nb.lane_id) in used:
                    continue                     # that lane IS part of the route
                if clear_of_other_legs(loc, arc) < CLEAR_OF_ROUTE_M + 2.6:
                    continue                     # another leg of the route runs right there
                if any(math.hypot(loc.x - d["x"], loc.y - d["y"]) < 12.0 for d in dropped):
                    continue
                score = abs(arc - target)
                if best is None or score < best[0]:
                    best = (score, arc, wp, side, same, nb)
        if best is None:
            print(f"  no room for a car near {target:.0f} m")
            continue
        _, arc, wp, side, same, nb = best
        t = nb.transform
        bp = lib.find(MODELS[k % len(MODELS)])
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", ROLE)
        actor = world.try_spawn_actor(bp, carla.Transform(
            carla.Location(t.location.x, t.location.y, t.location.z + 0.3), t.rotation))
        if actor is None:
            print(f"  could not drop a car at {arc:.0f} m -- something is there")
            continue
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass
        kind = "parking lane" if nb.lane_type == carla.LaneType.Parking else "lane"
        gap = math.hypot(t.location.x - wp.transform.location.x, t.location.y - wp.transform.location.y)
        dropped.append({"arc": round(arc, 1), "x": round(t.location.x, 2), "y": round(t.location.y, 2),
                        "model": bp.id, "side": side, "same": same})
        print(f"  {bp.id.split('.')[-1]:10s} {arc:5.0f} m along the route, in the lane {gap:.1f} m to the "
              f"{side} ({'same way' if same else 'facing back'}, {kind}), at ({t.location.x:.1f}, {t.location.y:.1f})")
    print("dropped", len(dropped), "cars")


if __name__ == "__main__":
    main()
