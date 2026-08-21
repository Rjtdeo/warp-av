#!/usr/bin/env python3
"""
Measure how far CARLA's traffic-light stop waypoints sit BEFORE the actual
junction (the visual crosswalk area). Explains the 'why does it stop so far
back?' gap and gives the calibration number to close it.

    python tools/probe_stop_lines.py
"""
import statistics

import carla


def main(host="localhost", port=2000):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    gaps = []
    print(f"{'light @':<24} {'stop-wp -> junction gap (m)'}")
    for tl in world.get_actors().filter("traffic.traffic_light"):
        try:
            sws = tl.get_stop_waypoints()
        except Exception:
            continue
        for sw in sws:
            wp, dist = sw, 0.0
            for _ in range(40):
                if wp.is_junction:
                    break
                nxt = wp.next(1.0)
                if not nxt:
                    break
                wp = nxt[0]
                dist += 1.0
            else:
                continue
            if wp.is_junction:
                gaps.append(dist)
                loc = tl.get_location()
                print(f"({loc.x:7.1f},{loc.y:7.1f})        {dist:5.1f}")
    if gaps:
        print(f"\nlights sampled: {len(gaps)}")
        print(f"median gap: {statistics.median(gaps):.1f} m | min {min(gaps):.1f} | max {max(gaps):.1f}")
        print("\nThis is how far CARLA's stop line sits before the junction.")
        print("The van's bumper stops ~0.5 m before that stop line — the visual")
        print("gap you see is (median gap + ~0.5) metres before the crosswalk.")
    else:
        print("no measurable lights found")


if __name__ == "__main__":
    main()
