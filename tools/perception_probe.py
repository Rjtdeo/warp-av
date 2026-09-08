"""
Perception probe: measurement only, no change to the van.

Question it answers: at what distance does the camera + LiDAR pipeline see a
small road object, and at which STAGE is the object lost?

    stage 1  raw LiDAR points on the object (before / after the height cut,
             after the 3x downsample the pipeline applies)
    stage 2  LiDAR cluster found for it (cluster_points)
    stage 3  camera: any YOLOX box on the object's image column, and which
             class the model gives it (only person / vehicle are used later)
    stage 4  tracked object reported by CameraLidarPerception.update()
             (needs MIN_HITS sightings, forgotten after DROP_AFTER_S)
    stage 5  the live stack's own view over /api/state (cross-check)

How: attach a second camera + LiDAR to the parked van, identical to the
stack's own (800x600 fov 90 at x=2.0 z=1.8; 32-ch 150k pts/s 50 m 10 Hz at
z=2.5), and run the SAME CameraLidarPerception class on them. One object at
a time is spawned on the lane centre ahead at a list of distances; at each
distance ~3 s of frames are recorded. The van does not move. Everything the
probe spawns (objects + its two sensors) is destroyed at the end.

Run on the CARLA machine, stack running and idle:
    venv\\Scripts\\python.exe tools\\perception_probe.py [--distances 25,20,15,12,10,8,6,4]
                                                       [--objects barrel,cone,barrier,planter,car,person]
                                                       [--frames 30] [--api http://localhost:5000]
Writes logs/perception_probe_<time>.csv and .json and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import carla        # noqa: E402

from warp_av.adapters.carla_sensor_adapter import CameraFrame, LidarScan          # noqa: E402
from warp_av.adapters.lidar_sweep import LidarSweepAccumulator, azimuth_coverage_bins  # noqa: E402
from warp_av.perception.camera_lidar_perception import (CameraLidarPerception,   # noqa: E402
                                                        COCO_CLASSES)
from warp_av.perception.tracking import ObjectTracker                            # noqa: E402

try:
    import requests
except Exception:      # the cross-check is optional
    requests = None

OBJECTS = {
    "barrel":  ("static.prop.barrel", 0.0),
    "cone":    ("static.prop.trafficcone01", 0.0),
    "barrier": ("static.prop.streetbarrier", 0.0),
    "planter": ("static.prop.plantpot04", 0.0),
    "box":     ("static.prop.creasedbox01", 0.0),
    "car":     ("vehicle.tesla.model3", 0.3),
    "person":  ("walker.pedestrian.0001", 1.0),
}
NEAR_M = 1.5           # a reported object this close to the truth counts as "seen"
ROW_PERIOD_S = 0.1     # sampling cadence of the measurement rows (same in every mode)


class FakeAdapter:
    """What CameraLidarPerception reads: latest_camera, latest_lidar, vehicle."""

    def __init__(self, vehicle, sweep=None):
        self.vehicle = vehicle
        self.latest_camera = None
        self.latest_lidar = None
        self.frames = 0
        self.sweep = sweep            # LidarSweepAccumulator or None (raw per-frame wedges)

    def on_camera(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.latest_camera = CameraFrame(image=arr, width=image.width, height=image.height,
                                         fov=float(image.fov), timestamp=time.time())

    def on_lidar(self, scan):
        pts = np.frombuffer(scan.raw_data, dtype=np.float32).reshape((-1, 4))
        frames, span = 1, 0.0
        if self.sweep is not None:
            pts = self.sweep.add(pts, scan.transform.get_matrix(), float(scan.timestamp))
            frames, span = self.sweep.frames_in_sweep, self.sweep.span_s
        else:
            pts = pts.copy()
        self.latest_lidar = LidarScan(points=pts, timestamp=time.time(), frames=frames, span_s=span)
        self.frames += 1


def attach_sensors(world, vehicle, adapter, sensor_tick="0.1"):
    bl = world.get_blueprint_library()
    cam_bp = bl.find("sensor.camera.rgb")
    for k, v in (("image_size_x", "800"), ("image_size_y", "600"), ("fov", "90"), ("sensor_tick", "0.1")):
        cam_bp.set_attribute(k, v)
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=2.0, z=1.8)), attach_to=vehicle)
    cam.listen(adapter.on_camera)
    lid_bp = bl.find("sensor.lidar.ray_cast")
    for k, v in (("channels", "32"), ("points_per_second", "150000"), ("range", "50.0"),
                 ("rotation_frequency", "10"), ("sensor_tick", str(sensor_tick)),
                 ("upper_fov", "10.0"), ("lower_fov", "-30.0")):
        lid_bp.set_attribute(k, v)
    lid = world.spawn_actor(lid_bp, carla.Transform(carla.Location(x=0.0, z=2.5)), attach_to=vehicle)
    lid.listen(adapter.on_lidar)
    return [cam, lid]


def ego_frame(vehicle_tf, wx, wy):
    """World -> van frame (x forward, y as CARLA has it: positive to the right).
    The inverse of the rotation CameraLidarPerception.update() applies."""
    yaw = math.radians(vehicle_tf.rotation.yaw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx, dy = wx - vehicle_tf.location.x, wy - vehicle_tf.location.y
    return dx * cy + dy * sy, -dx * sy + dy * cy


def stack_objects(api):
    if requests is None or not api:
        return None
    try:
        st = requests.get(api + "/api/state", timeout=2).json()
        return [(o["x"], o["y"], o["type"]) for o in st.get("perception", {}).get("objects", [])]
    except Exception:
        return None


def measure(perc, adapter, vehicle, actor, api, frames):
    """Record `frames` LiDAR frames with the object in place."""
    perc.tracker = ObjectTracker()                     # each distance is a fresh approach
    perc._last_inference_time = 0.0                    # force a fresh camera inference
    bb = actor.bounding_box.extent
    r_obj = math.hypot(bb.x, bb.y)
    rows = []
    last = adapter.frames
    last_row_t = 0.0
    deadline = time.time() + frames * 0.5 + 5.0
    while len(rows) < frames and time.time() < deadline:
        # one row per ROW_PERIOD_S in BOTH arms: with sweeps on the LiDAR
        # delivers ~50 times a second, with sweeps off 10 times, and the two
        # runs must watch the object for the same length of time
        if adapter.frames == last or time.time() - last_row_t < ROW_PERIOD_S:
            time.sleep(0.005)
            continue
        last = adapter.frames
        last_row_t = time.time()
        t_row = time.perf_counter()
        tf = vehicle.get_transform()
        loc = actor.get_location()
        ox, oy = ego_frame(tf, loc.x, loc.y)          # object in van frame
        true_dist = math.hypot(ox, oy)

        # stage 0: what one LiDAR scan contains (points, and how much of the
        # circle it covers in 10-degree bins: 36 = a full sweep)
        pts = adapter.latest_lidar.points
        scan_cover = azimuth_coverage_bins(pts)

        # stage 1: raw points on the object; the road cut exactly as the pipeline does it
        horiz = np.hypot(pts[:, 0] - ox, pts[:, 1] - oy) <= r_obj + 0.35
        raw_n = int(horiz.sum())
        if perc.ground_filter_mode == "patches":
            hmask = perc.ground_filter.apply(pts).keep
        else:
            hmask = (pts[:, 2] > perc.minimum_lidar_z) & (pts[:, 2] < perc.maximum_lidar_z)
        masked_n = int((horiz & hmask).sum())
        # the pipeline keeps every 3rd height-masked point: count what survives on the object
        idx = np.where(hmask)[0][::3]
        down_n = int(horiz[idx].sum()) if masked_n else 0

        out = perc.update()                            # stages 2-4, the real code

        # stage 2: cluster near the object
        clusters = getattr(perc, "last_clusters", []) or []
        near_c = [c for c in clusters if math.hypot(c["x"] - ox, c["y"] - oy) <= max(NEAR_M, r_obj + 0.5)]
        cl = min(near_c, key=lambda c: math.hypot(c["x"] - ox, c["y"] - oy)) if near_c else None

        # stage 3: camera boxes on the object's image column
        cam = adapter.latest_camera
        fx = cam.width / 2.0
        u = cam.width / 2.0 + fx * (oy / (ox - 2.0)) if ox > 3.0 else None
        boxes = []
        for det in perc._cached_camera_detections:
            # det.box is (x_left, y_top, width, height): right edge = x + w.
            # (The stack's fusion step reads box[2], the WIDTH, as the right
            # edge - see docs/PERCEPTION_BASELINE.md, finding 4.)
            bx1, bx2 = det.box[0], det.box[0] + det.box[2]
            if u is not None and bx1 - 20 <= u <= bx2 + 20:
                name = COCO_CLASSES[det.class_id] if det.class_id < len(COCO_CLASSES) else str(det.class_id)
                boxes.append((name, round(float(det.confidence), 2)))

        # stage 4: tracked object reported
        seen = [o for o in out.objects if math.hypot(o.x - ox, o.y - oy) <= max(NEAR_M, r_obj + 0.5)]
        rep = min(seen, key=lambda o: math.hypot(o.x - ox, o.y - oy)) if seen else None

        # stage 5: the live stack
        so = stack_objects(api)
        stack_hit = None
        if so is not None:
            near = [t for (x, y, t) in so if math.hypot(x - loc.x, y - loc.y) <= max(NEAR_M, r_obj + 0.5)]
            stack_hit = near[0] if near else ""

        rows.append({
            "true_dist": true_dist, "scan_pts": int(len(pts)), "scan_cover": scan_cover,
            "row_ms": (time.perf_counter() - t_row) * 1000.0,
            "raw_pts": raw_n, "masked_pts": masked_n, "down_pts": down_n,
            "cluster": cl is not None, "cluster_n": cl["n"] if cl else 0,
            "cluster_extent": cl["extent"] if cl else 0.0,
            "cluster_cls": (cl.get("cls") or "") if cl else "",
            "cam_boxes": boxes,
            "tracked": rep is not None, "tracked_type": rep.object_type.value if rep else "",
            "tracked_dist": rep.distance if rep else None,
            "stack_seen": stack_hit not in (None, ""), "stack_type": stack_hit or "",
            "stack_available": so is not None, "healthy": out.healthy, "reason": out.reason,
        })
    return rows


def summarise(name, want_d, rows):
    n = max(1, len(rows))
    frac = lambda k: sum(1 for r in rows if r[k]) / n
    mean = lambda k: sum(r[k] for r in rows) / n
    cls_votes = {}
    for r in rows:
        for (c, conf) in r["cam_boxes"]:
            cls_votes[c] = cls_votes.get(c, 0) + 1
    cam_cls = ", ".join(f"{c}x{k}" for c, k in sorted(cls_votes.items(), key=lambda kv: -kv[1])[:3])
    fused = {r["cluster_cls"] for r in rows if r["cluster_cls"]}
    ttypes = {r["tracked_type"] for r in rows if r["tracked_type"]}
    stypes = {r["stack_type"] for r in rows if r["stack_type"]}
    return {
        "object": name, "distance_m": want_d, "frames": len(rows),
        "true_dist_m": round(mean("true_dist"), 2) if rows else None,
        "scan_pts": round(mean("scan_pts")), "scan_cover_bins": round(mean("scan_cover"), 1),
        "row_ms": round(mean("row_ms")),
        "raw_pts": round(mean("raw_pts"), 1), "masked_pts": round(mean("masked_pts"), 1),
        "down_pts": round(mean("down_pts"), 1),
        "cluster_frac": round(frac("cluster"), 2), "cluster_n": round(mean("cluster_n"), 1),
        "cam_any_frac": round(sum(1 for r in rows if r["cam_boxes"]) / n, 2), "cam_classes": cam_cls,
        "fused_cls": "/".join(sorted(fused)) or "-",
        "tracked_frac": round(frac("tracked"), 2), "tracked_type": "/".join(sorted(ttypes)) or "-",
        "stack_frac": round(frac("stack_seen"), 2) if any(r["stack_available"] for r in rows) else None,
        "stack_type": "/".join(sorted(stypes)) or "-",
        "unhealthy_frames": sum(1 for r in rows if not r["healthy"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    # 127.0.0.1, not localhost: on Windows 'localhost' tries IPv6 first and
    # every request can lose a second or two
    ap.add_argument("--api", default=os.environ.get("WARP_API", "http://127.0.0.1:5000"))
    ap.add_argument("--distances", default="25,20,15,12,10,8,6,4")
    ap.add_argument("--objects", default="barrel,cone,barrier,planter,car,person")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--lateral", type=float, default=0.0, help="metres right of the lane centre")
    ap.add_argument("--sweep", choices=["on", "off"], default="on",
                    help="on: glue per-frame wedges into full sweeps like the fixed stack; "
                         "off: one raw wedge per scan like the stack before fix 1")
    ap.add_argument("--sensor-tick", default=None,
                    help="LiDAR sensor_tick override (default: 0.0 with --sweep on, 0.1 with off)")
    ap.add_argument("--ground", choices=["patches", "flat"], default="patches",
                    help="road removal: patches (fix 2) or flat (the old 35 cm line)")
    ap.add_argument("--no-stack", action="store_true", help="skip the live-stack cross-check")
    a = ap.parse_args()
    if a.no_stack:
        a.api = None
    if a.sensor_tick is None:
        a.sensor_tick = "0.0" if a.sweep == "on" else "0.1"
    if a.sweep == "on" and float(a.sensor_tick) != 0.0:
        sys.exit("--sweep on needs every simulator frame: use --sensor-tick 0.0 (the stack does)")

    client = carla.Client(a.host, a.port)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if not vans:
        sys.exit("no Sprinter in the world: start the stack first")
    van = vans[0]
    if van.get_velocity().length() > 0.1:
        sys.exit("the van is moving: the probe needs it parked (stack idle)")
    tf0 = van.get_transform()
    wp0 = cmap.get_waypoint(tf0.location)
    print(f"van at ({tf0.location.x:.1f}, {tf0.location.y:.1f}) yaw {tf0.rotation.yaw:.1f}, "
          f"map {cmap.name}, lane width {wp0.lane_width:.2f}")

    adapter = FakeAdapter(van, sweep=LidarSweepAccumulator() if a.sweep == "on" else None)
    print(f"lidar mode: {'full sweeps' if a.sweep == 'on' else 'raw per-frame wedges'} (sensor_tick {a.sensor_tick})")
    spawned = []
    sensors = []
    results = []
    bl = world.get_blueprint_library()
    try:
        sensors = attach_sensors(world, van, adapter, sensor_tick=a.sensor_tick)
        while adapter.latest_camera is None or adapter.latest_lidar is None:
            time.sleep(0.05)
        perc = CameraLidarPerception(adapter)
        perc.ground_filter_mode = a.ground
        print(f"road removal: {a.ground}")
        dists = [float(d) for d in a.distances.split(",")]
        for name in a.objects.split(","):
            bp_id, z_up = OBJECTS[name]
            bp = bl.find(bp_id)
            for d in dists:
                nxt = wp0.next(d)
                if not nxt:
                    print(f"  {name} @ {d} m: no lane point")
                    continue
                w = nxt[0].transform
                yaw = math.radians(w.rotation.yaw)
                # right of the lane centre = +90 deg from heading in CARLA's frame
                loc = carla.Location(x=w.location.x - math.sin(yaw) * a.lateral,
                                     y=w.location.y + math.cos(yaw) * a.lateral,
                                     z=w.location.z + z_up + 0.05)
                actor = world.try_spawn_actor(bp, carla.Transform(loc, w.rotation))
                if actor is None:
                    loc.z += 0.5
                    actor = world.try_spawn_actor(bp, carla.Transform(loc, w.rotation))
                if actor is None:
                    print(f"  {name} @ {d} m: spawn failed")
                    continue
                spawned.append(actor)
                try:
                    actor.set_simulate_physics(False)
                except Exception:
                    pass
                time.sleep(0.6)                                   # let both sensors see it
                rows = measure(perc, adapter, van, actor, a.api, a.frames)
                s = summarise(name, d, rows)
                results.append(s)
                print(f"  {name:8s} {d:5.1f} m | scan {s['scan_pts']:5d} pts {s['scan_cover_bins']:4.1f}/36 "
                      f"{s['row_ms']:4d} ms n{s['frames']:2d} | pts raw {s['raw_pts']:5.1f} cut {s['masked_pts']:5.1f} "
                      f"down {s['down_pts']:4.1f} | cluster {s['cluster_frac']:.2f} (n {s['cluster_n']:.1f}) "
                      f"| cam {s['cam_any_frac']:.2f} [{s['cam_classes']}] fused {s['fused_cls']} "
                      f"| tracked {s['tracked_frac']:.2f} {s['tracked_type']} "
                      f"| stack {s['stack_frac']} {s['stack_type']}", flush=True)
                actor.destroy()
                spawned.remove(actor)
                time.sleep(0.3)
    finally:
        for s in sensors:
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        for x in spawned:
            try:
                x.destroy()
            except Exception:
                pass

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "logs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"perception_probe_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0].keys()) if results else ["object"])
        wr.writeheader()
        wr.writerows(results)
    (out_dir / f"perception_probe_{stamp}.json").write_text(json.dumps(
        {"map": cmap.name, "van": [tf0.location.x, tf0.location.y, tf0.rotation.yaw],
         "lateral_m": a.lateral, "sweep": a.sweep, "sensor_tick": a.sensor_tick, "ground": a.ground,
         "results": results}, indent=1))
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
