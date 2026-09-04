r"""
Put the lidar bay finder in front of the real simulator and mark it.

    python tools\probe_bay_finder.py --samples 40

At each of N spots beside a parking lane (the same rule the RL arena uses to
pick practice bays) a lidar identical to the van's is placed 2.5 m up, one
sweep is taken, and bay_finder.find_bays() is run on it. Each answer is marked
against what the map knows: where the parking lane's centre line is (side
error, heading error) and where the map's decorative parked vehicles are (a
found bay must not overlap one; a genuinely free stretch should be found).

Writes rl/exams/<date>_bay_finder_probe.csv. Run INSTEAD of the trainer or an
exam - it drives the simulator clock.
"""
import argparse
import csv
import datetime
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import carla
import numpy as np

from rl.parking_env import static_vehicle_outlines
from warp_av.perception.bay_finder import find_bays, fit_kerb, LIDAR_HEIGHT_M


def candidate_spots(cmap, rng, n):
    spots = []
    for wp in cmap.generate_waypoints(8.0):
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        r = wp.get_right_lane()
        if r is None or r.lane_type not in (carla.LaneType.Parking, carla.LaneType.Shoulder):
            continue
        if r.lane_width < 1.8:
            continue
        spots.append((wp, r))
    rng.shuffle(spots)
    return spots[:n]


def to_sensor_frame(px, py, tf):
    """World point -> sensor frame (x forward, y RIGHT) for a sensor at tf."""
    yaw = math.radians(tf.rotation.yaw)
    dx, dy = px - tf.location.x, py - tf.location.y
    c, s = math.cos(-yaw), math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    cmap = world.get_map()
    rng = random.Random(a.seed)
    statics = static_vehicle_outlines(world)

    bp = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    bp.set_attribute("channels", "32")
    bp.set_attribute("points_per_second", "150000")
    bp.set_attribute("range", "50.0")
    bp.set_attribute("rotation_frequency", "10")
    bp.set_attribute("sensor_tick", "0.1")

    rows = []
    out = os.path.join(os.path.dirname(__file__), "..", "rl", "exams",
                       f"{datetime.date.today().isoformat()}_bay_finder_probe.csv")
    try:
        for k, (wp, lane) in enumerate(candidate_spots(cmap, rng, a.samples), 1):
            tf = carla.Transform(
                carla.Location(wp.transform.location.x, wp.transform.location.y,
                               wp.transform.location.z + LIDAR_HEIGHT_M),
                carla.Rotation(yaw=wp.transform.rotation.yaw))
            latest = {}
            sensor = world.spawn_actor(bp, tf)
            sensor.listen(lambda scan: latest.__setitem__("pts", np.frombuffer(
                scan.raw_data, dtype=np.float32).reshape((-1, 4)).copy()))
            for _ in range(3):
                world.tick()
            sensor.stop()
            sensor.destroy()
            pts = latest.get("pts")
            if pts is None or len(pts) == 0:
                print(f"#{k:2d} no lidar data"); continue

            # ground truth: parking-lane centre in the sensor frame
            lx, ly = to_sensor_frame(lane.transform.location.x, lane.transform.location.y, tf)
            lane_yaw = math.radians(lane.transform.rotation.yaw - wp.transform.rotation.yaw)
            lane_yaw = (lane_yaw + math.pi) % (2 * math.pi) - math.pi
            # the map's kerb face: lane centre + half the lane width, to the right
            kerb_truth = ly + lane.lane_width / 2.0
            # decorative vehicles IN THE PARKING STRIP as along-intervals (occupied truth);
            # anything beyond the pavement is not in the bay and must not count
            occ = []
            for outline in statics:
                sx = [to_sensor_frame(px, py, tf) for px, py in outline]
                if any(-12 < x < 40 and (kerb_truth - 4.0) < y < (kerb_truth + 0.3) for x, y in sx):
                    occ.append((min(x for x, _ in sx), max(x for x, _ in sx)))

            kerb = fit_kerb(pts)
            bays = find_bays(pts)
            # Diagnostic: does the lidar even SEE each decorative vehicle in the
            # strip, and where does it sit relative to the detected kerb?
            veh_notes = []
            for outline in statics:
                sx = [to_sensor_frame(px, py, tf) for px, py in outline]
                if not any(-12 < x < 40 and (kerb_truth - 4.0) < y < (kerb_truth + 0.3) for x, y in sx):
                    continue
                x0, x1 = min(x for x, _ in sx), max(x for x, _ in sx)
                y0, y1 = min(y for _, y in sx), max(y for _, y in sx)
                zz = pts[:, 2] + LIDAR_HEIGHT_M
                inside = ((pts[:, 0] > x0 - 0.3) & (pts[:, 0] < x1 + 0.3) & (pts[:, 1] > y0 - 0.3)
                          & (pts[:, 1] < y1 + 0.3) & (zz > 0.35) & (zz < 2.2))
                lat = kerb.right_of(0.5 * (x0 + x1), 0.5 * (y0 + y1)) if kerb else float("nan")
                veh_notes.append(f"veh x {x0:5.1f}..{x1:5.1f} m, {lat:+.1f} m road-side of detected kerb, {int(inside.sum()):3d} body pts")
            nearest = bays[0] if bays else None
            side_err = head_err = float("nan")
            overlaps = False
            kerb_err = float("nan")
            if kerb is not None:
                kerb_err = (kerb.a + kerb.b * 0.0) - kerb_truth   # detected kerb face vs the map's, beside the sensor
            if nearest is not None:
                side_err = (-nearest.y) - ly           # slot centre vs lane centre, + = further right
                head_err = math.degrees(nearest.yaw - lane_yaw)
                overlaps = any(not (nearest.along_end < s or nearest.along_start > e) for s, e in occ)
            free_truth = any(True for _ in [0]) and not occ   # no decoration within view = free
            rows.append(dict(sample=k, kerb_found=kerb is not None, kerb_err_m=round(kerb_err, 2) if kerb else "",
                             bays_found=len(bays),
                             nearest_x=round(nearest.x, 2) if nearest else "",
                             nearest_len=round(nearest.length, 1) if nearest else "",
                             side_err_m=round(side_err, 2) if nearest else "",
                             heading_err_deg=round(head_err, 1) if nearest else "",
                             overlaps_parked_vehicle=overlaps, decor_vehicles_in_view=len(occ),
                             lane_width=round(lane.lane_width, 2), points=len(pts)))
            print(f"#{k:2d} kerb {'yes' if kerb else 'NO '} (err {kerb_err:+.2f} m)  bays {len(bays):2d}  "
                  + (f"nearest at {nearest.x:5.1f} m, len {nearest.length:4.1f} m, side err {side_err:+.2f} m, "
                     f"heading err {head_err:+.1f} deg, overlaps parked vehicle: {overlaps}" if nearest else "no bay")
                  + f"  | decor vehicles in view {len(occ)}")
            for note in veh_notes:
                print(f"       {note}")
    finally:
        world.apply_settings(original)

    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        found = [r for r in rows if r["bays_found"] > 0]
        se = [abs(float(r["side_err_m"])) for r in found]
        he = [abs(float(r["heading_err_deg"])) for r in found]
        print(f"\n{len(rows)} spots: kerb found at {sum(1 for r in rows if r['kerb_found'])}, "
              f"a bay found at {len(found)}, of which {sum(1 for r in found if r['overlaps_parked_vehicle'])} "
              f"overlap a decorative parked vehicle (bad)")
        ke = [abs(float(r["kerb_err_m"])) for r in rows if r["kerb_found"]]
        if ke:
            print(f"kerb face    mean {np.mean(ke):.2f} m off the map's  worst {max(ke):.2f} m")
        if found:
            print(f"side error   mean {np.mean(se):.2f} m  worst {max(se):.2f} m   (slot centre vs the map's lane centre)")
            print(f"heading err  mean {np.mean(he):.1f} deg worst {max(he):.1f} deg")
        print(f"written: {out}")


if __name__ == "__main__":
    main()
