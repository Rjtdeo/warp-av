"""
Score the road removal against CARLA's labelled LiDAR (perception fix 2).

CARLA's semantic LiDAR stamps every point with what it hit (road, sidewalk,
car, pedestrian, prop, ...). We never drive with it; here it is the answer
key. A labelled LiDAR identical to the van's own is attached to the parked
van, a few objects are placed ahead, and every delivery is run through

    * the old rule: one flat line 35 cm above the road under the van
    * fix 2: local patches (GroundFilter)

Reported per label group: how many road points each rule deleted (want
~100 %) and how many object points each rule kept (want ~100 %), plus the
object keep-rate by distance band.

    venv\\Scripts\\python.exe tools\\ground_filter_score.py [--seconds 2] [--no-spawn]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import carla  # noqa: E402
from warp_av.perception.ground_filter import GroundFilter, flat_cut  # noqa: E402
from warp_av.adapters.lidar_sweep import LidarSweepAccumulator  # noqa: E402

# CARLA 0.9.14+ semantic tags
TAGS = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole",
        7: "traffic_light", 8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky", 12: "pedestrian",
        13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train", 18: "motorcycle", 19: "bicycle",
        20: "static", 21: "dynamic", 22: "other", 23: "water", 24: "road_line", 25: "ground", 26: "bridge",
        27: "rail_track", 28: "guard_rail"}
ROAD = {1, 24, 25}                                  # must be deleted
OBJECT = {12, 13, 14, 15, 16, 18, 19, 20, 21}       # must be kept (people, vehicles, props)
INFO = {2, 10, 6, 5, 4, 3, 9}                       # reported, no target (sidewalks, terrain, poles, ...)

SEMANTIC_DTYPE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("cos", np.float32),
                           ("idx", np.uint32), ("tag", np.uint32)])

SPAWN = [("static.prop.barrel", 8.0, 0.0), ("static.prop.plantpot04", 12.0, 0.6),
         ("static.prop.trafficcone01", 16.0, -0.4), ("vehicle.tesla.model3", 22.0, 0.0),
         ("walker.pedestrian.0001", 6.0, 1.2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--no-spawn", action="store_true", help="score the scene as it is, place nothing")
    a = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    bl = world.get_blueprint_library()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if not vans:
        sys.exit("no van in the world: start the stack first")
    van = vans[0]
    tf = van.get_transform()
    wp = cmap.get_waypoint(tf.location)
    print(f"van at ({tf.location.x:.1f}, {tf.location.y:.1f}) yaw {tf.rotation.yaw:.1f}, map {cmap.name}")

    spawned = []
    if not a.no_spawn:
        for bp_id, ahead, lateral in SPAWN:
            nxt = wp.next(ahead)
            if not nxt:
                print(f"  no lane point {ahead:.0f} m ahead, skipping {bp_id}")
                continue
            w = nxt[0].transform
            yaw = math.radians(w.rotation.yaw)
            loc = carla.Location(x=w.location.x - math.sin(yaw) * lateral, y=w.location.y + math.cos(yaw) * lateral,
                                 z=w.location.z + (1.0 if "walker" in bp_id else 0.3 if "vehicle" in bp_id else 0.05))
            act = world.try_spawn_actor(bl.find(bp_id), carla.Transform(loc, w.rotation))
            if act is None:
                loc.z += 0.5
                act = world.try_spawn_actor(bl.find(bp_id), carla.Transform(loc, w.rotation))
            if act is not None:
                try:
                    act.set_simulate_physics(False)
                except Exception:
                    pass
                spawned.append(act)
                print(f"  placed {bp_id} {ahead:.0f} m ahead, {lateral:+.1f} m")
        time.sleep(0.5)

    bp = bl.find("sensor.lidar.ray_cast_semantic")
    for k, v in (("channels", "32"), ("points_per_second", "150000"), ("range", "50.0"), ("rotation_frequency", "10"),
                 ("sensor_tick", "0.0"), ("upper_fov", "10.0"), ("lower_fov", "-30.0")):
        bp.set_attribute(k, v)
    lid = world.spawn_actor(bp, carla.Transform(carla.Location(x=0.0, z=2.5)), attach_to=van)
    gf = GroundFilter(lidar_height_m=2.5)
    # the pipeline works on FULL sweeps (fix 1): glue the labelled deliveries
    # the same way, carrying the label as a 5th column; the accumulator also
    # drops the van's own body returns (its ego box) exactly as the stack does
    acc = LidarSweepAccumulator()
    totals = {}      # tag -> [n, kept_flat, kept_patches]
    bands = {}       # (band, group) -> [n, kept_flat, kept_patches]
    stats = {"deliveries": 0, "sweeps": 0, "ms": [], "borrowed": [], "own_body": 0}

    def on_scan(m):
        arr = np.frombuffer(m.raw_data, dtype=SEMANTIC_DTYPE)
        stats["deliveries"] += 1
        raw = np.c_[arr["x"], arr["y"], arr["z"], arr["cos"], arr["tag"].astype(np.float32)].astype(np.float32)
        before = acc.dropped_self_returns
        sweep = acc.add(raw, m.transform.get_matrix(), float(m.timestamp))
        stats["own_body"] += acc.dropped_self_returns - before
        # score only once per rotation, when the sweep is whole
        if acc.span_s < 0.09 or sweep.shape[0] == 0:
            return
        pts = sweep[:, :4]
        tags = sweep[:, 4].astype(np.int64)
        old = flat_cut(pts, lidar_height_m=2.5, min_above_m=0.35).keep
        res = gf.apply(pts)
        new = res.keep
        stats["sweeps"] += 1
        stats["ms"].append(res.ms)
        stats["borrowed"].append(res.borrowed_tiles)
        dist = np.hypot(pts[:, 0], pts[:, 1])
        for t in np.unique(tags):
            m_ = tags == t
            row = totals.setdefault(int(t), [0, 0, 0])
            row[0] += int(m_.sum()); row[1] += int(old[m_].sum()); row[2] += int(new[m_].sum())
            group = "road" if t in ROAD else "object" if t in OBJECT else None
            if group:
                for lo, hi in ((0, 10), (10, 20), (20, 35), (35, 50)):
                    mb = m_ & (dist >= lo) & (dist < hi)
                    r = bands.setdefault(((lo, hi), group), [0, 0, 0])
                    r[0] += int(mb.sum()); r[1] += int(old[mb].sum()); r[2] += int(new[mb].sum())

    lid.listen(on_scan)
    time.sleep(a.seconds)
    lid.stop()
    time.sleep(0.2)
    try:
        lid.destroy()
        for s in spawned:
            s.destroy()
    except Exception:
        pass

    if not stats["ms"]:
        sys.exit(f"nothing scored ({stats['deliveries']} deliveries, no whole sweep): run longer")
    print(f"\n{stats['sweeps']} whole sweeps scored from {stats['deliveries']} deliveries; patches filter "
          f"{np.mean(stats['ms']):.2f} ms mean, {np.max(stats['ms']):.2f} ms max per sweep; borrowed tiles per sweep "
          f"{np.mean(stats['borrowed']):.1f}; van's own body points ignored: {stats['own_body']}")
    print(f"\n{'label':14s} {'points':>8s} | {'old: kept':>10s} {'%':>6s} | {'patches: kept':>14s} {'%':>6s}   target")
    for t, (n, k_old, k_new) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
        name = TAGS.get(t, f"tag{t}")
        target = "DELETE" if t in ROAD else "KEEP" if t in OBJECT else ""
        print(f"{name:14s} {n:8d} | {k_old:10d} {100.0 * k_old / n:5.1f}% | {k_new:14d} {100.0 * k_new / n:5.1f}%   {target}")
    road_n = sum(v[0] for t, v in totals.items() if t in ROAD)
    obj_n = sum(v[0] for t, v in totals.items() if t in OBJECT)
    if road_n:
        print(f"\nROAD deleted:  old {100.0 * (1 - sum(v[1] for t, v in totals.items() if t in ROAD) / road_n):5.1f}%   "
              f"patches {100.0 * (1 - sum(v[2] for t, v in totals.items() if t in ROAD) / road_n):5.1f}%   (want > 99 %)")
    if obj_n:
        print(f"OBJECT kept:   old {100.0 * sum(v[1] for t, v in totals.items() if t in OBJECT) / obj_n:5.1f}%   "
              f"patches {100.0 * sum(v[2] for t, v in totals.items() if t in OBJECT) / obj_n:5.1f}%   (want > 95 %)")
    print("\nby distance band:")
    for (lo, hi), group in sorted(bands, key=lambda k: (k[1], k[0])):
        n, k_old, k_new = bands[((lo, hi), group)]
        if n == 0:
            continue
        if group == "road":
            print(f"  {lo:2d}-{hi:2d} m road   deleted: old {100.0 * (1 - k_old / n):5.1f}%  patches {100.0 * (1 - k_new / n):5.1f}%  ({n} pts)")
        else:
            print(f"  {lo:2d}-{hi:2d} m object kept:    old {100.0 * k_old / n:5.1f}%  patches {100.0 * k_new / n:5.1f}%  ({n} pts)")


if __name__ == "__main__":
    main()
