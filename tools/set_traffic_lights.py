#!/usr/bin/env python3
"""
Force all CARLA traffic lights to a state — makes the red-light behavior testable
on demand instead of waiting for the light cycle.

    python tools/set_traffic_lights.py red      # every light red, frozen
    python tools/set_traffic_lights.py green    # every light green, frozen
    python tools/set_traffic_lights.py auto     # back to normal cycling

Options: --host (default localhost), --port (default 2000)
"""
import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", choices=["red", "green", "yellow", "auto"])
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    a = ap.parse_args()

    client = carla.Client(a.host, a.port)
    client.set_timeout(5.0)
    world = client.get_world()
    lights = list(world.get_actors().filter("traffic.traffic_light"))
    if not lights:
        print("No traffic lights in this map.")
        return

    if a.state == "auto":
        for tl in lights:
            tl.freeze(False)
        print(f"{len(lights)} lights back to automatic cycling.")
        return

    target = {"red": carla.TrafficLightState.Red,
              "green": carla.TrafficLightState.Green,
              "yellow": carla.TrafficLightState.Yellow}[a.state]
    for tl in lights:
        tl.set_state(target)
        tl.freeze(True)
    print(f"{len(lights)} lights frozen {a.state.upper()}. Run with 'auto' to release.")


if __name__ == "__main__":
    main()
