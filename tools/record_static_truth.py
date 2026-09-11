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
from warp_av.perception.camera_lidar_perception import DEFAULT_KEEP_ABOVE_M  # noqa: E402
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
# anywhere a vehicle can be. Town10HD draws its parking strips as SHOULDER lanes: 153 of
# the 378 cars in the second recording sat on one, so leaving it out made them look
# 'clear of the road'.
DRIVABLE = (carla.LaneType.Driving | carla.LaneType.Parking | carla.LaneType.Bidirectional
            | carla.LaneType.Shoulder | carla.LaneType.Biking)
SETTLE_MIN_TICKS = 20     # 1 s: the drop, the bounce, and the walkers finding their feet
SETTLE_MAX_TICKS = 80
SETTLED_MPS = 0.02


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


# Town10HD's streets are empty unless something is spawned, so without this the answer key
# holds buildings, poles and parked cars and not one person -- and "never call a person
# static" is the boundary that matters most. So each viewpoint gets company.
#
# The first recording (8,687 blobs) showed where rules break: a car parked beside a lamp
# post comes out as ONE blob, and borrows the post's height -- 3.8 m tall, so a "tall and
# off the road means building" rule called 7 parked cars static. So the company now
# includes exactly those hard cases: people and vehicles standing right beside poles,
# trees and the building line, lorries and buses (tall, like a wall), and parked cars.
WALKERS_PER_VIEW = 5
VEHICLES_PER_VIEW = 3
GLUED_WALKERS_PER_VIEW = 3        # beside a pole or a tree, touching distance
TWO_WHEELERS = ("vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets",
                "vehicle.harley-davidson.low_rider", "vehicle.kawasaki.ninja",
                "vehicle.yamaha.yzf", "vehicle.vespa.zx125")
BIG_VEHICLES = ("vehicle.carlamotors.carlacola", "vehicle.carlamotors.european_hgv",
                "vehicle.carlamotors.firetruck", "vehicle.mitsubishi.fusorosa",
                "vehicle.ford.ambulance", "vehicle.volkswagen.t2", "vehicle.volkswagen.t2_2021",
                "vehicle.tesla.cybertruck")


def _sidewalk_near(wp, max_hops: int = 4):
    """Walk sideways from a driving lane until a pavement lane turns up, either side."""
    for step in ("get_right_lane", "get_left_lane"):
        cur = wp
        for _ in range(max_hops):
            cur = getattr(cur, step)()
            if cur is None:
                break
            if cur.lane_type == carla.LaneType.Sidewalk:
                return cur
    return None


def _lane_of_type_near(wp, lane_type, max_hops: int = 3):
    for step in ("get_right_lane", "get_left_lane"):
        cur = wp
        for _ in range(max_hops):
            cur = getattr(cur, step)()
            if cur is None:
                break
            if cur.lane_type == lane_type:
                return cur
    return None


def _offset(t, lat):
    """A point `lat` metres to the right of a waypoint transform (left if negative)."""
    yaw = math.radians(t.rotation.yaw)
    return (t.location.x - math.sin(yaw) * lat, t.location.y + math.cos(yaw) * lat)


def _four_wheelers(bl):
    return [b for b in bl.filter("vehicle.*")
            if int(b.get_attribute("number_of_wheels")) == 4 and "sprinter" not in b.id]


def _things_near(things, x, y, lo, hi):
    """Map objects (poles, trees) whose middle is between lo and hi metres from (x, y)."""
    if things is None or not len(things):
        return []
    d = np.hypot(things[:, 0] - x, things[:, 1] - y)
    return [things[i] for i in np.where((d > lo) & (d < hi))[0]]


def populate(sim, bl, wp, rng, cmap=None, poles=None, trees=None):
    """People on the pavement and in the road, a mix of vehicles and two-wheelers, and the
    hard cases: people glued to poles and trees, a vehicle parked beside a pole, a lorry or
    bus, a parked car -- all within about 25 m of the viewpoint. Everything is spawned
    through the sandbox, so everything is destroyed again whatever happens."""
    made = []
    walker_bps = list(bl.filter("walker.pedestrian.*"))
    side = _sidewalk_near(wp)
    spots = []
    for d in (6.0, 11.0, 16.0, 21.0):
        base = (side or wp).next(d)
        if base:
            spots.append((base[0], side is not None))
    # one person IN the road too: the crossing case is the one that must never read as static
    road_ahead = wp.next(9.0)
    if road_ahead:
        spots.append((road_ahead[0], False))
    for k, (w_, on_pavement) in enumerate(spots[:WALKERS_PER_VIEW]):
        t = w_.transform
        if on_pavement and k == 1:
            # against the building line: the far edge of the pavement
            lat = (w_.lane_width / 2.0 - 0.35) * (1 if rng.random() < 0.5 else -1)
        else:
            lat = rng.uniform(-0.8, 0.8) if on_pavement else rng.uniform(-1.2, 1.2)
        x, y = _offset(t, lat)
        a = sim.try_spawn(rng.choice(walker_bps), carla.Transform(
            carla.Location(x, y, t.location.z + 1.0), carla.Rotation(yaw=rng.uniform(0, 360))))
        if a is not None:
            made.append(a)
        if on_pavement and k == 2:
            # a second person beside the first: two people chatting make one wide blob
            x2, y2 = _offset(t, lat + 0.65)
            b = sim.try_spawn(rng.choice(walker_bps), carla.Transform(
                carla.Location(x2, y2, t.location.z + 1.0), carla.Rotation(yaw=rng.uniform(0, 360))))
            if b is not None:
                made.append(b)
    # people glued to a pole or a tree: the merge that turns a person into "a tall thing"
    vx, vy = wp.transform.location.x, wp.transform.location.y
    glued = _things_near(poles, vx, vy, 5.0, 25.0)
    rng.shuffle(glued)
    glued_trees = _things_near(trees, vx, vy, 5.0, 25.0)
    rng.shuffle(glued_trees)
    targets = glued[:2] + glued_trees[:1]
    for (px, py, pz, reach, _hz) in targets[:GLUED_WALKERS_PER_VIEW]:
        ground_z = wp.transform.location.z
        if cmap is not None:
            g = cmap.get_waypoint(carla.Location(px, py, pz), project_to_road=True,
                                  lane_type=carla.LaneType.Sidewalk)
            if g is not None:
                ground_z = g.transform.location.z
        for _try in range(3):
            ang = rng.uniform(0, 2 * math.pi)
            r = reach + rng.uniform(0.35, 0.6)
            a = sim.try_spawn(rng.choice(walker_bps), carla.Transform(
                carla.Location(px + r * math.cos(ang), py + r * math.sin(ang), ground_z + 1.0),
                carla.Rotation(yaw=rng.uniform(0, 360))))
            if a is not None:
                made.append(a)
                break
    # vehicles: in the next lane and behind/ahead in our own, some two-wheelers among them
    lanes = [wp.get_left_lane(), wp.get_right_lane(), wp]
    fours = _four_wheelers(bl)
    for k in range(VEHICLES_PER_VIEW):
        lane = lanes[k % len(lanes)]
        if lane is None or lane.lane_type != carla.LaneType.Driving:
            continue
        ahead = lane.next(rng.uniform(8.0, 24.0))
        if not ahead:
            continue
        t = ahead[0].transform
        name = rng.choice(TWO_WHEELERS) if rng.random() < 0.5 else None
        bps = list(bl.filter(name)) if name else fours
        if not bps:
            continue
        a = sim.try_spawn(rng.choice(bps), carla.Transform(
            carla.Location(t.location.x, t.location.y, t.location.z + 0.4), t.rotation))
        if a is not None:
            made.append(a)
    # a lorry or a bus: as tall as a shop front, and it still drives off
    big = [b for n in BIG_VEHICLES for b in bl.filter(n)]
    lane = rng.choice([l for l in lanes if l is not None and l.lane_type == carla.LaneType.Driving] or [wp])
    ahead = lane.next(rng.uniform(10.0, 26.0))
    if big and ahead:
        t = ahead[0].transform
        # pulled over against the kerb half the time: that is where a delivery lorry stops
        lat = (lane.lane_width / 2.0) * 0.6 * (1 if rng.random() < 0.5 else 0)
        x, y = _offset(t, lat)
        a = sim.try_spawn(rng.choice(big), carla.Transform(
            carla.Location(x, y, t.location.z + 0.5), t.rotation))
        if a is not None:
            made.append(a)
    # a parked car: in a parking lane if the street has one, else tight against the kerb
    park = _lane_of_type_near(wp, carla.LaneType.Parking)
    base = park or wp
    ahead = base.next(rng.uniform(6.0, 22.0))
    if fours and ahead:
        t = ahead[0].transform
        lat = 0.0 if park is not None else (base.lane_width / 2.0 - 1.05)
        x, y = _offset(t, lat)
        a = sim.try_spawn(rng.choice(fours), carla.Transform(
            carla.Location(x, y, t.location.z + 0.4), t.rotation))
        if a is not None:
            made.append(a)
    # a car parked right beside a lamp post: the exact blob that fooled the first rules
    if fours and cmap is not None:
        near_road = []
        for (px, py, pz, reach, _hz) in _things_near(poles, vx, vy, 6.0, 24.0):
            d = cmap.get_waypoint(carla.Location(px, py, pz), project_to_road=True,
                                  lane_type=carla.LaneType.Driving)
            if d is None:
                continue
            off = math.hypot(px - d.transform.location.x, py - d.transform.location.y)
            if off < d.lane_width / 2.0 + 2.5:
                near_road.append((d, px, py, off))
        rng.shuffle(near_road)
        for (d, px, py, off) in near_road[:1]:
            t = d.transform
            yaw = math.radians(t.rotation.yaw)
            # which side of the lane centre the pole is on
            right = (-math.sin(yaw)) * (px - t.location.x) + math.cos(yaw) * (py - t.location.y)
            sign = 1.0 if right > 0 else -1.0
            lat = sign * max(0.0, off - 1.0 - 0.45)
            x, y = _offset(t, lat)
            a = sim.try_spawn(rng.choice(fours), carla.Transform(
                carla.Location(x, y, t.location.z + 0.4), t.rotation))
            if a is not None:
                made.append(a)
    return made


def map_things(world, label):
    """(x, y, z, reach) of every map object with this label; reach = its half-size."""
    out = []
    for o in world.get_environment_objects(label):
        bb = o.bounding_box
        out.append((bb.location.x, bb.location.y, bb.location.z, max(bb.extent.x, bb.extent.y),
                    bb.extent.z))
    return np.array(out, dtype=float) if out else np.zeros((0, 5))


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
    edge_errors = collections.Counter()
    edge_sides = 0
    t_start = time.time()
    with ScratchWorld(fixed_dt=0.05) as sim:
        w = sim.world
        cmap = w.get_map()
        bl = w.get_blueprint_library()
        van_bp = sim.blueprint("vehicle.mercedes.sprinter")
        spots = viewpoints(cmap, a.viewpoints, a.seed)
        print(f"{len(spots)} viewpoints across {cmap.name}")
        mount = carla.Transform(carla.Location(x=0.0, z=2.5))
        # exactly the van's road filter: it keeps points 8 cm up, not the class default 12 cm
        gf = GroundFilter(keep_above_m=DEFAULT_KEEP_ABOVE_M)
        poles = map_things(w, carla.CityObjectLabel.Poles)
        trees = map_things(w, carla.CityObjectLabel.Vegetation)
        # trees: taller than a person, and not a hedge or a patch of grass
        trees = trees[(trees[:, 3] < 3.0) & (trees[:, 4] > 1.5)] if len(trees) else trees
        print(f"{len(poles)} poles and {len(trees)} trees on the map to stand people beside")

        for vi, wp in enumerate(spots):
            t = wp.transform
            van = sim.try_spawn(van_bp, carla.Transform(
                carla.Location(t.location.x, t.location.y, t.location.z + 0.3), t.rotation))
            if van is None:
                continue
            company = populate(sim, bl, wp, random.Random(a.seed * 1000 + vi),
                               cmap=cmap, poles=poles, trees=trees)
            plain = sim.spawn(lidar_bp(bl, False), mount, attach_to=van)
            label = sim.spawn(lidar_bp(bl, True), mount, attach_to=van)
            got_p, got_l = [], []
            plain.listen(lambda m: got_p.append(decode_lidar(m)))
            label.listen(lambda m: got_l.append(decode_semantic(m)))
            # Settle before recording. The van is dropped from 30 cm and a 30 cm fall alone
            # takes 0.25 s, so the first version's 4 ticks (0.2 s) recorded a van still in
            # the air or bouncing: the road smeared across the wedges and no kerb line was
            # ever found. Wait until it has truly stopped moving.
            for k in range(SETTLE_MAX_TICKS):
                sim.tick()
                if k >= SETTLE_MIN_TICKS:
                    v = van.get_velocity()
                    if math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) < SETTLED_MPS:
                        break
            got_p.clear(); got_l.clear()
            for _ in range(WEDGE_TICKS):
                sim.tick()
            time.sleep(0.05)
            plain.stop(); label.stop()
            if not got_p or not got_l:
                for x in [plain, label, van] + company:
                    try:
                        x.destroy(); sim._mine.remove(x)
                    except Exception:
                        pass
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
            except Exception as e:
                edges = None
                edge_errors[repr(e)[:120]] += 1
            if edges is not None:
                edge_sides += sum(1 for sd in (edges.left, edges.right)
                                  if sd is not None and sd.confident)
            sel, heights, _dropped = remove_road_edge_points(sel, heights, cluster_points, edges=edges)
            clusters = cluster_points(sel[:, :2].tolist(), heights=heights.tolist(),
                                      cell=CLUSTER_CELL_M, min_points_far=MIN_POINTS_FAR,
                                      return_members=True)
            clusters = merge_split_clusters(clusters)
            # the van drops these too: a 2-point blob, low and beside the lane, is a kerb crumb
            clusters = [c for c in clusters
                        if not (c.get("weak") and (c.get("height") is not None and c["height"] < 0.30)
                                and abs(c["y"]) > 1.2)]

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
                here = cmap.get_waypoint(carla.Location(wx, wy, vt.location.z),
                                         project_to_road=False, lane_type=carla.LaneType.Any)
                lane_here = str(here.lane_type) if here is not None else "none"
                kerb_gap = None
                kerb_beyond = None
                if edges is not None:
                    side = edges.right if c["y"] > 0 else edges.left
                    if side is not None and getattr(side, "confident", False):
                        kerb_gap = abs(c["y"] - side.lateral_at(c["x"]))
                        # signed, from the blob's innermost point: + = beyond the kerb
                        # (pavement side), - = some of it is inside the road
                        mp = sel[members]
                        edge_y = np.array([side.lateral_at(float(px)) for px in mp[:, 0]])
                        s_i = (mp[:, 1] - edge_y) if c["y"] > 0 else (edge_y - mp[:, 1])
                        kerb_beyond = float(np.min(s_i))
                # how far its nearest point is from anywhere a vehicle can be (DRIVABLE).
                # + = that far clear of it; 0 or - = on it.
                mp = sel[members]
                pick = mp if len(mp) <= 24 else mp[np.linspace(0, len(mp) - 1, 24).astype(int)]
                gaps = []
                for (px, py) in pick[:, :2]:
                    gx = vt.location.x + cy * px - sy * py
                    gy = vt.location.y + sy * px + cy * py
                    g = cmap.get_waypoint(carla.Location(float(gx), float(gy), vt.location.z),
                                          project_to_road=True, lane_type=DRIVABLE)
                    if g is not None:
                        gl = g.transform.location
                        gaps.append(math.hypot(gx - gl.x, gy - gl.y) - g.lane_width / 2.0)
                road_gap = min(gaps) if gaps else float("nan")
                # merge-proof height: how the points are spread up the blob, not just its
                # tallest one (one lamp-post point made a parked car "3.8 m tall")
                hm = heights[members]
                hm = hm[np.isfinite(hm)]
                hq = np.percentile(hm, [10, 50, 90]) if len(hm) else [float("nan")] * 3
                frac_high = float(np.mean(hm > 2.0)) if len(hm) else float("nan")
                rows.append(dict(
                    view=vi, road_id=int(wp.road_id), junction=bool(wp.is_junction),
                    x=float(c["x"]), y=float(c["y"]), distance=float(math.hypot(c["x"], c["y"])),
                    length=float(c.get("length_m") or 0.0), width=float(c.get("width_m") or 0.0),
                    height=float(c.get("height") if c.get("height") is not None else float("nan")),
                    yaw_deg=float(c.get("yaw_deg") or 0.0), n=int(c["n"]), weak=bool(c.get("weak")),
                    lane_half_width=float(lane_w / 2.0), on_road=bool(on_road),
                    kerb_gap=float("nan") if kerb_gap is None else float(kerb_gap),
                    kerb_beyond=float("nan") if kerb_beyond is None else kerb_beyond,
                    lane_here=lane_here, road_gap=float(road_gap),
                    h_q10=float(hq[0]), h_q50=float(hq[1]), h_q90=float(hq[2]),
                    frac_high=frac_high, wx=float(wx), wy=float(wy),
                    tag=int(tag), tag_name=TAG_NAME.get(int(tag), f"tag{tag}"),
                    category=category_of(int(tag)), purity=float(votes / len(tags)),
                    tags_seen=json.dumps({TAG_NAME.get(k, str(k)): v for k, v in count.most_common(3)})))
            for x in [plain, label, van] + company:
                try:
                    x.destroy(); sim._mine.remove(x)
                except Exception:
                    pass
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
    print(f"kerb lines found: {edge_sides} confident sides; errors: {dict(edge_errors) or 'none'}")


if __name__ == "__main__":
    main()
