#!/usr/bin/env python3
"""
Survey the current CARLA town for REAL parking bays (map lanes typed
Parking/Shoulder that the van's bay-parking can use).

    python tools/find_parking_bays.py

For every bay found it prints: location, length, width, and WHICH destination
spawn point (the dashboard dropdown numbers) is closest — so you can send the
van there and watch it park in the bay. Ends with a verdict on whether this
town is usable for bay-parking demos.
"""
import math
from collections import defaultdict

import carla


def main(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    cmap = world.get_map()
    town = cmap.name.split("/")[-1]

    # Sample every driving lane every 5 m and look right for Parking/Shoulder.
    segments = defaultdict(list)   # (road_id, lane_id of bay) -> [locations]
    widths = {}
    for wp in cmap.generate_waypoints(5.0):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        probe = wp
        for _ in range(3):     # slide to rightmost driving lane
            r = probe.get_right_lane()
            if r is not None and r.lane_type == carla.LaneType.Driving:
                probe = r
            else:
                break
        bay = probe.get_right_lane()
        if (bay is not None
                and bay.lane_type in (carla.LaneType.Parking, carla.LaneType.Shoulder)
                and bay.lane_width >= 1.8):
            key = (bay.road_id, bay.lane_id, str(bay.lane_type).split(".")[-1])
            segments[key].append(bay.transform.location)
            widths[key] = bay.lane_width

    sps = cmap.get_spawn_points()
    print(f"Town: {town} — scanned {len(sps)} destinations")
    if not segments:
        print("\nNO parking/shoulder bays in this town's map.")
        print("Verdict: bay-parking cannot trigger here — the van will always kerb-hug.")
        print("Options: switch town (Town10HD/Town03 usually have some) or hand-annotate slots.")
        return

    print(f"\n{len(segments)} bay segment(s) found:\n")
    print(f"{'#':>2} {'type':<9} {'width':>5} {'length':>7}  {'centre (x, y)':<20} nearest destination")
    usable = 0
    for i, (key, locs) in enumerate(sorted(segments.items(), key=lambda kv: -len(kv[1]))):
        length = 5.0 * len(locs)
        cx = sum(l.x for l in locs) / len(locs)
        cy = sum(l.y for l in locs) / len(locs)
        best_i, best_d = -1, 1e9
        for j, sp in enumerate(sps):
            d = math.hypot(sp.location.x - cx, sp.location.y - cy)
            if d < best_d:
                best_i, best_d = j, d
        van_len_slots = int(length // 7)
        if length >= 10:
            usable += 1
        print(f"{i:>2} {key[2]:<9} {widths[key]:>4.1f}m {length:>6.0f}m  ({cx:>7.1f}, {cy:>7.1f})   "
              f"point {best_i} ({best_d:.0f} m away) — fits ~{van_len_slots} van(s)")

    print(f"\nVerdict: {usable} bay(s) long enough (>=10 m) for a parking demo.")
    if usable:
        print("Pick the destination number shown above in the dashboard dropdown,")
        print("run the mission, and watch the console print  kind: 'bay'  at mission start.")
    else:
        print("Bays exist but are too short — recommend hand-annotated slots (Option 3).")


if __name__ == "__main__":
    main()
