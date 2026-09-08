"""
Record a perception fixture on the CARLA machine (Perception V2, day 2).

Saves a short stretch of real sensor data from the parked van so the
perception pipeline can be replayed and scored on the Mac / in CI, without
CARLA and without the camera model:

  * LiDAR deliveries from a PLAIN LiDAR configured exactly like the van's own
    (32 ch, 150k pts/s, 50 m, 10 Hz, z=2.5, CARLA's default 45 % drop-off):
    x, y, z, intensity, ring id, each delivery's simulation time and sensor
    pose. This is what the pipeline replays.
  * the same seconds from CARLA's LABELLED LiDAR (no drop-off, so about
    twice as dense): the answer key, saved separately and never fed to the
    pipeline;
  * a few front-camera frames as JPEG, with timestamps;
  * the true position and bounding box of every object we placed.

    venv\\Scripts\\python.exe tools\\record_fixture.py --name town03_straight --seconds 1.2

Output: tests/fixtures/perception/<name>/  (deliveries.npz, camera_*.jpg,
frames.json, objects.json, meta.json). A 1.2 s recording is about 3 MB.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import carla  # noqa: E402
import cv2    # noqa: E402
from warp_av.adapters.carla_sensor_adapter import decode_lidar, ring_ids  # noqa: E402

SEMANTIC_DTYPE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32), ("cos", np.float32),
                           ("idx", np.uint32), ("tag", np.uint32)])
PLACE = [("static.prop.barrel", 9.0, 0.0), ("static.prop.plantpot04", 13.0, 0.6),
         ("static.prop.trafficcone01", 16.0, -0.5), ("vehicle.tesla.model3", 22.0, 0.0),
         ("walker.pedestrian.0001", 6.0, 1.3), ("static.prop.bin", 12.0, -1.6)]


def decode_semantic(m):
    arr = np.frombuffer(m.raw_data, dtype=SEMANTIC_DTYPE)
    n = arr.shape[0]
    out = np.empty((n, 6), dtype=np.float32)
    out[:, 0] = arr["x"]; out[:, 1] = arr["y"]; out[:, 2] = arr["z"]; out[:, 3] = arr["cos"]
    out[:, 4] = ring_ids(m, n); out[:, 5] = arr["tag"].astype(np.float32)
    return out


def check_ring_order(points):
    """CARLA has no beam noise: within one ring the elevation is constant and
    it falls with the ring id (+10 deg .. -30 deg). Returns (ok, detail)."""
    ring = points[:, 4]
    el = np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))
    means, spreads = [], []
    for r in sorted(set(ring[ring >= 0].astype(int).tolist())):
        e = el[ring == r]
        means.append(float(e.mean())); spreads.append(float(e.max() - e.min()))
    ok = len(means) >= 2 and all(a > b for a, b in zip(means, means[1:])) and max(spreads) < 0.2
    return ok, f"rings {len(means)}, elevation {means[0]:.1f} .. {means[-1]:.1f} deg, max spread {max(spreads):.3f} deg" if means else "no rings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--no-spawn", action="store_true")
    a = ap.parse_args()
    out_dir = ROOT / "tests" / "fixtures" / "perception" / a.name

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    cmap = world.get_map()
    bl = world.get_blueprint_library()
    vans = list(world.get_actors().filter("vehicle.mercedes.sprinter"))
    if not vans:
        sys.exit("no van in the world: start the stack first")
    van = vans[0]
    if van.get_velocity().length() > 0.1:
        sys.exit("the van is moving: park it first (stack idle)")
    tf = van.get_transform()
    wp = cmap.get_waypoint(tf.location)

    out_dir.mkdir(parents=True, exist_ok=True)
    spawned = []
    sensors = []
    try:
      if not a.no_spawn:
        for bp_id, ahead, right in PLACE:
            nxt = wp.next(ahead)
            if not nxt:
                continue
            w = nxt[0].transform
            yaw = math.radians(w.rotation.yaw)
            loc = carla.Location(x=w.location.x - math.sin(yaw) * right, y=w.location.y + math.cos(yaw) * right,
                                 z=w.location.z + (1.0 if "walker" in bp_id else 0.3 if "vehicle" in bp_id else 0.05))
            act = world.try_spawn_actor(bl.find(bp_id), carla.Transform(loc, w.rotation))
            if act is None:
                loc.z += 0.5
                act = world.try_spawn_actor(bl.find(bp_id), carla.Transform(loc, w.rotation))
            if act is None:
                print(f"  {bp_id}: spawn failed")
                continue
            try:
                act.set_simulate_physics(False)
                bb = act.bounding_box
                bottom = bb.location.z - bb.extent.z
                if bottom < -0.03:
                    act.set_transform(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z - bottom), w.rotation))
            except Exception:
                pass
            spawned.append((bp_id, act))
            print(f"  placed {bp_id} {ahead:.0f} m ahead, {right:+.1f} m right")
        time.sleep(0.6)

      # the pipeline's input: a plain LiDAR exactly like the van's own (CARLA defaults keep
      # the 45 % drop-off); the answer key: the labelled LiDAR, same geometry, no drop-off
      lidar_attrs = (("channels", "32"), ("points_per_second", "150000"), ("range", "50.0"),
                     ("rotation_frequency", "10"), ("sensor_tick", "0.0"), ("upper_fov", "10.0"), ("lower_fov", "-30.0"))
      pbp = bl.find("sensor.lidar.ray_cast")
      lbp = bl.find("sensor.lidar.ray_cast_semantic")
      for k, v in lidar_attrs:
          pbp.set_attribute(k, v); lbp.set_attribute(k, v)
      cbp = bl.find("sensor.camera.rgb")
      for k, v in (("image_size_x", "800"), ("image_size_y", "600"), ("fov", "90"), ("sensor_tick", "0.1")):
          cbp.set_attribute(k, v)
      mount = carla.Transform(carla.Location(x=0.0, z=2.5))
      plain = world.spawn_actor(pbp, mount, attach_to=van); sensors.append(plain)
      labelled = world.spawn_actor(lbp, mount, attach_to=van); sensors.append(labelled)
      camera = world.spawn_actor(cbp, carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-10.0)),
                                 attach_to=van); sensors.append(camera)

      deliveries = []        # plain: (sim_time, matrix, points Nx5 x y z intensity ring)
      labels = []            # labelled: (sim_time, matrix, points Nx6 x y z cos ring tag)
      frames = []            # (sim_time, jpeg bytes)

      def on_plain(m):
          deliveries.append((float(m.timestamp), np.asarray(m.transform.get_matrix(), dtype=np.float64), decode_lidar(m)))

      def on_labelled(m):
          labels.append((float(m.timestamp), np.asarray(m.transform.get_matrix(), dtype=np.float64), decode_semantic(m)))

      def on_camera(im):
          arr = np.frombuffer(im.raw_data, dtype=np.uint8).reshape((im.height, im.width, 4))[:, :, :3].copy()
          ok, jpg = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
          if ok:
              frames.append((float(im.timestamp), jpg.tobytes()))

      plain.listen(on_plain); labelled.listen(on_labelled); camera.listen(on_camera)
      time.sleep(a.seconds)
      for s_ in sensors:
          s_.stop()
      time.sleep(0.2)

      if not deliveries:
          sys.exit("no LiDAR deliveries recorded")

      def save_stream(name, stream):
          pts = np.concatenate([d[2] for d in stream], axis=0)
          idx = np.concatenate([np.full(d[2].shape[0], i, dtype=np.int32) for i, d in enumerate(stream)])
          np.savez_compressed(out_dir / name, points=pts, delivery_index=idx,
                              delivery_time=np.array([d[0] for d in stream], dtype=np.float64),
                              delivery_matrix=np.stack([d[1] for d in stream]))
          return pts

      pts = save_stream("deliveries.npz", deliveries)             # written before anything can still go wrong
      lab = save_stream("labels.npz", labels) if labels else None
      for k, (ts, jpg) in enumerate(frames):
          (out_dir / f"camera_{k:03d}.jpg").write_bytes(jpg)
      (out_dir / "frames.json").write_text(json.dumps([{"file": f"camera_{k:03d}.jpg", "sim_time": ts}
                                                       for k, (ts, _) in enumerate(frames)], indent=1))
      ok, detail = check_ring_order(pts)
      print(f"ring order check: {'OK' if ok else 'FAILED'} ({detail})")

      objects = []
      for bp_id, act in spawned:
          try:
              t = act.get_transform(); bb = act.bounding_box
              objects.append({"blueprint": bp_id, "x": t.location.x, "y": t.location.y, "z": t.location.z,
                              "yaw_deg": t.rotation.yaw,
                              "extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                              "box_offset": [bb.location.x, bb.location.y, bb.location.z]})
          except Exception as e:
              print(f"  {bp_id}: pose unreadable ({e}), left out of objects.json")
      (out_dir / "objects.json").write_text(json.dumps(objects, indent=1))
      per_rotation = pts.shape[0] / max(1e-9, (deliveries[-1][0] - deliveries[0][0]) * 10.0)
      (out_dir / "meta.json").write_text(json.dumps({
          "map": cmap.name, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": a.seconds,
          "van": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z, "yaw_deg": tf.rotation.yaw},
          "lidar": {"channels": 32, "points_per_second": 150000, "range_m": 50.0, "rotation_hz": 10,
                    "mount": {"x": 0.0, "z": 2.5}, "upper_fov": 10.0, "lower_fov": -30.0,
                    "dropoff": "CARLA defaults (dropoff_general_rate 0.45)",
                    "columns": ["x", "y", "z", "intensity", "ring"], "points_per_rotation": round(per_rotation)},
          "labels": {"file": "labels.npz", "columns": ["x", "y", "z", "cos_incidence", "ring", "semantic_tag"],
                     "note": "labelled LiDAR has no drop-off: answer key only, never pipeline input",
                     "points": int(lab.shape[0]) if lab is not None else 0},
          "camera": {"width": 800, "height": 600, "fov": 90, "mount": {"x": 2.0, "z": 1.8, "pitch_deg": -10.0}},
          "ring_order_ok": ok, "deliveries": len(deliveries), "points": int(pts.shape[0]),
          "camera_frames": len(frames)}, indent=1))
      size_mb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1e6
      print(f"wrote {out_dir}: {len(deliveries)} plain deliveries ({pts.shape[0]} pts, ~{per_rotation:.0f} per rotation), "
            f"{len(labels)} labelled deliveries, {len(frames)} frames, {size_mb:.1f} MB")
    finally:
      for s_ in sensors:
          try:
              s_.destroy()
          except Exception:
              pass
      for _, act in spawned:
          try:
              act.destroy()
          except Exception:
              pass


if __name__ == "__main__":
    main()
