"""
Record what static and dynamic things look like to the van's LiDAR, with CARLA's truth
attached (Planning V2 -- static vs dynamic).

Why. The camera can name a person, a vehicle and a bicycle, and nothing else. A lamp post,
a kerb, a bollard and a cone all come out as "obstacle", so whether a thing can MOVE has to
be judged from what the LiDAR measures: its shape, its size, where it sits. Before writing
any rule for that, we need to know what those things actually look like to OUR LiDAR -- not
guessed thresholds, which is how a kerb sliver ended up "moving at 4 m/s" last week.

How. Two LiDARs on the same mount, firing the same rays:

  * PLAIN    -- configured exactly like the van's own, CARLA's default 45 % drop-off. This
               is what the van really sees, and it is the only thing fed to the pipeline.
  * LABELLED -- CARLA's semantic LiDAR. No drop-off, and every point carries the class of
               the thing it hit. The answer key. Never fed to the pipeline.

Same rays, so every plain point has an exact twin in the labelled set: the drop-off only
deletes points, it never moves them. Each blob the van's own pipeline finds -- ground filter,
kerb lines, kerb-point removal, clustering, split-blob merging, the real functions -- is
labelled by a majority vote of its points' twins.

The van is parked at each viewpoint while it records, so the points of every wedge share
one sensor frame and no de-skew is involved.

Output: one row per blob: what the van MEASURED, next to what the thing REALLY IS.

    python tools/record_static_truth.py --viewpoints 120 --out static_truth.npz

Refuses to run while the stack is up (tools/scratch_world.py): it spawns its own van and
sensors and steps the simulator by hand, which would break a running stack.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import carla  # noqa: E402
from scratch_world import ScratchWorld  # noqa: E402
from warp_av.adapters.carla_sensor_adapter import decode_lidar, LIDAR_ROTATION_HZ  # noqa: E402
from warp_av.perception.ground_filter import GroundFilter, remove_road_edge_points  # noqa: E402
from warp_av.perception.road_edges import find_road_edges  # noqa: E402
from warp_av.perception.tracking import cluster_points, merge_split_clusters, MIN_POINTS_FAR  # noqa: E402

SEMANTIC_DTYPE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                           ("cos", np.float32), ("idx", np.uint32), ("tag", np.uint32)])

# CARLA 0.9.15's CityObjectLabel, sorted by the only question that matters here: can it move?
STATIC_TAGS = {2: "sidewalk/kerb", 3: "building", 4: "wall", 5: "fence", 6: "pole",
               7: "traffic light", 8: "traffic sign", 9: "vegetation", 10: "terrain",
               20: "static prop", 25: "ground", 26: "bridge", 27: "rail track", 28: "guard rail"}
DYNAMIC_TAGS = {12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train",
                18: "motorcycle", 19: "bicycle", 21: "dynamic prop"}
ROAD_TAGS = {1: "road", 24: "road line"}      # road surface that leaked past the ground filter
TAG_NAME = {**STATIC_TAGS, **DYNAMIC_TAGS, **ROAD_TAGS,
            0: "none", 11: "sky", 22: "other", 23: "water"}

MATCH_CELL_M = 0.05       # twin lookup: same ray means same point, so 5 cm is generous
CLUSTER_CELL_M = 0.8      # the van's own (camera_lidar_perception.DEFAULT_CLUSTER_CELL_M)
WEDGE_TICKS = 3           # 3 x 0.05 s at 10 Hz = 1.5 rotations: a full sweep with overlap


def category_of(tag: int) -> str:
    if tag in STATIC_TAGS:
        return "static"
    if tag in DYNAMIC_TAGS:
        return "dynamic"
    if tag in ROAD_TAGS:
        return "road"
    return "other"


def decode_semantic(m) -> np.ndarray:
    arr = np.frombuffer(m.raw_data, dtype=SEMANTIC_DTYPE)
    out = np.empty((arr.shape[0], 4), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = arr["x"], arr["y"], arr["z"]
    out[:, 3] = arr["tag"].astype(np.float32)
    return out


class TwinIndex:
    """Find each plain point's twin in the labelled sweep. A grid hash, no scipy."""

    def __init__(self, labelled: np.ndarray, cell: float = MATCH_CELL_M):
        self.cell = cell
        self.pts = labelled
        keys = np.floor(labelled[:, :3] / cell).astype(np.int64)
        self.buckets = collections.defaultdict(list)
        for i, k in enumerate(map(tuple, keys)):
            self.buckets[k].append(i)

    def tag_of(self, p) -> int:
        kx, ky, kz = (int(math.floor(v / self.cell)) for v in p[:3])
        best, best_d = -1, self.cell * 2.0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for i in self.buckets.get((kx + dx, ky + dy, kz + dz), ()):
                        q = self.pts[i]
                        d = math.sqrt((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2)
                        if d < best_d:
                            best, best_d = int(q[3]), d
        return best


def lidar_bp(bl, semantic: bool):
    bp = bl.find("sensor.lidar.ray_cast_semantic" if semantic else "sensor.lidar.ray_cast")
    # exactly the van's LiDAR (carla_sensor_adapter): 32 ch, 150k pts/s, 50 m, 10 Hz,
    # CARLA's default fov (+10 / -30) and, on the plain one, CARLA's default drop-off
    bp.set_attribute("channels", "32")
    bp.set_attribute("points_per_second", "150000")
    bp.set_attribute("range", "50.0")
    bp.set_attribute("rotation_frequency", str(int(LIDAR_ROTATION_HZ)))
    bp.set_attribute("sensor_tick", "0.0")
    return bp


def viewpoints(cmap, n: int, seed: int):
    """Spread across the whole map, driving lanes only, a few metres apart at least."""
    rng = random.Random(seed)
    wps = [w for w in cmap.generate_waypoints(6.0) if w.lane_type == carla.LaneType.Driving]
    rng.shuffle(wps)
    chosen = []
    for w in wps:
        loc = w.transform.location
        if all(loc.distance(c.transform.location) > 12.0 for c in chosen):
            chosen.append(w)
        if len(chosen) >= n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--viewpoints", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="static_truth.npz")
    a = ap.parse_args()

    rows = []
    match_hits = match_total = 0
    t_start = time.time()
    with ScratchWorld(fixed_dt=0.05) as sim:
        w = sim.world
        cmap = w.get_map()
        bl = w.get_blueprint_library()
        van_bp = sim.blueprint("vehicle.mercedes.sprinter")
        spots = viewpoints(cmap, a.viewpoints, a.seed)
        print(f"{len(spots)} viewpoints across {cmap.name}")
        mount = carla.Transform(carla.Location(x=0.0, z=2.5))
        gf = GroundFilter()

        for vi, wp in enumerate(spots):
            t = wp.transform
            van = sim.try_spawn(van_bp, carla.Transform(
                carla.Location(t.location.x, t.location.y, t.location.z + 0.3), t.rotation))
            if van is None:
                continue
            plain = sim.spawn(lidar_bp(bl, False), mount, attach_to=van)
            label = sim.spawn(lidar_bp(bl, True), mount, attach_to=van)
            got_p, got_l = [], []
            plain.listen(lambda m: got_p.append(decode_lidar(m)))
            label.listen(lambda m: got_l.append(decode_semantic(m)))
            for _ in range(4):                     # settle the suspension
                sim.tick()
            got_p.clear(); got_l.clear()
            for _ in range(WEDGE_TICKS):
                sim.tick()
            time.sleep(0.05)
            plain.stop(); label.stop()
            if not got_p or not got_l:
                for x in (plain, label, van):
                    x.destroy(); sim._mine.remove(x)
                continue
            pts = np.concatenate(got_p, axis=0)
            lab = np.concatenate(got_l, axis=0)

            # ---- the van's own pipeline, on the plain points only -------------------------
            ground = gf.apply(pts)
            mask = ground.keep
            sel = pts[mask]
            heights = ground.above[mask]
            try:
                edges = find_road_edges(pts[:, :2], ground.above)
            except Exception:
                edges = None
            sel, heights, _dropped = remove_road_edge_points(sel, heights, cluster_points, edges=edges)
            clusters = cluster_points(sel[:, :2].tolist(), heights=heights.tolist(),
                                      cell=CLUSTER_CELL_M, min_points_far=MIN_POINTS_FAR,
                                      return_members=True)
            clusters = merge_split_clusters(clusters)

            # ---- the answer key -----------------------------------------------------------
            twins = TwinIndex(lab)
            vt = van.get_transform()
            yaw = math.radians(vt.rotation.yaw)
            cy, sy = math.cos(yaw), math.sin(yaw)
            lane_w = wp.lane_width
            for c in clusters:
                members = c.get("members") or []
                tags = []
                for mi in members:
                    tg = twins.tag_of(sel[mi])
                    match_total += 1
                    if tg >= 0:
                        match_hits += 1
                        tags.append(tg)
                if not tags:
                    continue
                count = collections.Counter(tags)
                tag, votes = count.most_common(1)[0]
                # where it sits in the world, and whether that is road at all
                wx = vt.location.x + cy * c["x"] - sy * c["y"]
                wy = vt.location.y + sy * c["x"] + cy * c["y"]
                on_road = cmap.get_waypoint(carla.Location(wx, wy, vt.location.z),
                                            project_to_road=False,
                                            lane_type=carla.LaneType.Driving) is not None
                kerb_gap = None
                if edges is not None:
                    side = edges.right if c["y"] > 0 else edges.left
                    if side is not None and getattr(side, "confident", False):
                        kerb_gap = abs(c["y"] - side.lateral_at(c["x"]))
                rows.append(dict(
                    view=vi, road_id=int(wp.road_id), junction=bool(wp.is_junction),
                    x=float(c["x"]), y=float(c["y"]), distance=float(math.hypot(c["x"], c["y"])),
                    length=float(c.get("length_m") or 0.0), width=float(c.get("width_m") or 0.0),
                    height=float(c.get("height") if c.get("height") is not None else float("nan")),
                    yaw_deg=float(c.get("yaw_deg") or 0.0), n=int(c["n"]), weak=bool(c.get("weak")),
                    lane_half_width=float(lane_w / 2.0), on_road=bool(on_road),
                    kerb_gap=float("nan") if kerb_gap is None else float(kerb_gap),
                    tag=int(tag), tag_name=TAG_NAME.get(int(tag), f"tag{tag}"),
                    category=category_of(int(tag)), purity=float(votes / len(tags)),
                    tags_seen=json.dumps({TAG_NAME.get(k, str(k)): v for k, v in count.most_common(3)})))
            for x in (plain, label, van):
                x.destroy(); sim._mine.remove(x)
            sim.tick()
            if (vi + 1) % 20 == 0:
                print(f"  {vi + 1}/{len(spots)} viewpoints, {len(rows)} blobs so far")

    if not rows:
        sys.exit("nothing recorded")
    keys = list(rows[0].keys())
    np.savez_compressed(a.out, **{k: np.array([r[k] for r in rows]) for k in keys})
    cats = collections.Counter(r["category"] for r in rows)
    names = collections.Counter(r["tag_name"] for r in rows)
    print(f"\n{len(rows)} blobs from {len(set(r['view'] for r in rows))} viewpoints "
          f"in {time.time() - t_start:.0f} s -> {a.out}")
    print(f"twin match rate: {100.0 * match_hits / max(1, match_total):.1f}% of points")
    print("by category:", dict(cats))
    print("by class   :", dict(names.most_common(14)))


if __name__ == "__main__":
    main()
