"""
Record what the van's front camera sees while it drives, to measure the "something on the
lens" check on real pictures (Planning V2, fix 1).

Why. The check splits the picture into 192 patches and calls the lens dirty when more than
5 of them stop changing while the rest of the picture moves. On 2026-09-10 it fired on a
CLEAN lens in about half of the drives -- "11 to 23 patches are not changing" -- and each
time the safety supervisor slowed the van to walking pace and then stopped it. It had been
tuned on random-noise test pictures, where every patch changes every frame. A real picture
has patches that never change: the van's own bonnet at the bottom, plain sky at the top.

How. A Sprinter with exactly the van's front camera (800 x 600, 90 degrees, 10 Hz, 2.0 m
forward, 1.8 m up, tilted 10 degrees down -- carla_sensor_adapter), driven by CARLA's
autopilot through different streets and weathers, parked for the first few seconds of each
drive. Every frame's thumbnail -- the same every-eighth-pixel picture the check itself
uses -- is saved with the van's speed, plus one full picture per drive to look at.

    python tools/record_lens_check.py --seconds 40 --out lens_check.npz
    python tools/record_lens_check.py --score lens_check.npz      # no simulator needed

The score runs the van's own check (carla_sensor_adapter) over every frame, with the lens
clean, with the fault injector's five drops pasted on, and with one drop, and says how much
of the moving time each would have the van called "camera broken".

Recording refuses to run while the stack is up (tools/scratch_world.py).
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

sys.path.insert(0, str(ROOT / "src"))

WEATHERS = ("ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "ClearNight",
            "SoftRainNoon", "WetCloudyNoon")
PARKED_S = 4.0           # the first seconds of each drive, standing still
THUMB_STEP = 8           # the check's own thumbnail: every eighth pixel


def _with_drops(img, n):
    """The fault injector's drops (carla_sensor_adapter._on_camera, camera_drops)."""
    out = img.copy()
    h, w = out.shape[0], out.shape[1]
    for k in range(n):
        y = int(h * (0.15 + 0.22 * (k % 3)))
        x = int(w * (0.12 + 0.19 * (k % 4)))
        out[y:y + h // 5, x:x + w // 6] = 100 + 7 * k
    return out


def score(path):
    from warp_av.adapters.carla_sensor_adapter import CarlaSensorAdapter
    d = np.load(path)
    thumbs, speed, drive, names = d["thumbs"], d["speed"], d["drive"], d["names"]

    def broken_share(di, drops):
        a = object.__new__(CarlaSensorAdapter)
        a._last_thumb, a._camera_bad, a._camera_bad_why = None, 0, ""
        moving = broken = 0
        for i in np.where(drive == di)[0]:
            big = np.repeat(np.repeat(thumbs[i], 8, axis=0), 8, axis=1)
            a._check_the_picture(_with_drops(big, drops))
            if speed[i] > 0.5:
                moving += 1
                broken += bool(a.camera_fault())
        return 100.0 * broken / max(1, moving)

    print(f"{'drive':22s} {'clean lens':>11s} {'5 drops':>8s} {'1 drop':>7s}   (% of moving time called 'camera broken')")
    for di, name in enumerate(names):
        print(f"{str(name):22s} {broken_share(di, 0):10.0f}% {broken_share(di, 5):7.0f}% {broken_share(di, 1):6.0f}%")


def front_camera(sim, van):
    import carla
    bp = sim.blueprint("sensor.camera.rgb")
    bp.set_attribute("image_size_x", "800")
    bp.set_attribute("image_size_y", "600")
    bp.set_attribute("fov", "90")
    bp.set_attribute("sensor_tick", "0.1")          # 10 Hz, as on the van
    tf = carla.Transform(carla.Location(x=2.0, z=1.8), carla.Rotation(pitch=-10))
    return sim.spawn(bp, tf, attach_to=van)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0, help="per drive")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="lens_check.npz")
    ap.add_argument("--score", default=None, help="score a recording instead of making one")
    a = ap.parse_args()
    if a.score:
        return score(a.score)
    import carla
    sys.path.insert(0, str(ROOT / "tools"))
    from scratch_world import ScratchWorld
    rng = random.Random(a.seed)

    thumbs, speeds, drives, frames, full = [], [], [], [], []
    names = []
    with ScratchWorld(fixed_dt=0.05) as sim:
        w = sim.world
        tm = sim.client.get_trafficmanager(8000)
        tm.set_synchronous_mode(True)
        try:
            spots = w.get_map().get_spawn_points()
            rng.shuffle(spots)
            van_bp = sim.blueprint("vehicle.mercedes.sprinter")
            for di, weather in enumerate(WEATHERS):
                preset = getattr(carla.WeatherParameters, weather, None)
                if preset is None:
                    continue
                w.set_weather(preset)
                van = None
                while van is None and spots:
                    van = sim.try_spawn(van_bp, spots.pop())
                if van is None:
                    break
                cam = front_camera(sim, van)
                got = []
                cam.listen(lambda img: got.append((img.frame, np.frombuffer(img.raw_data, dtype=np.uint8)
                                                   .reshape((img.height, img.width, 4)).copy())))
                slow = di % 2 == 1
                started = False
                t_end = PARKED_S + a.seconds
                seen = 0
                for k in range(int(t_end / 0.05)):
                    if not started and k * 0.05 >= PARKED_S:
                        van.set_autopilot(True, tm.get_port())
                        tm.ignore_lights_percentage(van, 100)      # keep it moving
                        tm.vehicle_percentage_speed_difference(van, 60 if slow else 0)
                        started = True
                    sim.tick()
                    v = van.get_velocity()
                    speed = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
                    while seen < len(got):
                        fno, img = got[seen]
                        seen += 1
                        thumbs.append(img[::THUMB_STEP, ::THUMB_STEP, :3].copy())
                        speeds.append(speed)
                        drives.append(di)
                        frames.append(fno)
                        if len(full) <= di and speed > 2.0:
                            full.append(img[:, :, :3].copy())
                cam.stop()
                names.append(f"{weather}{' slow' if slow else ''}")
                print(f"  drive {di}: {names[-1]}, {sum(1 for d in drives if d == di)} frames, "
                      f"top speed {max(s for s, d in zip(speeds, drives) if d == di):.1f} m/s", flush=True)
                for x in (cam, van):
                    try:
                        if x is van:
                            van.set_autopilot(False, tm.get_port())
                        x.destroy()
                        sim._mine.remove(x)
                    except Exception:
                        pass
                sim.tick()
        finally:
            tm.set_synchronous_mode(False)

    np.savez_compressed(a.out, thumbs=np.array(thumbs, dtype=np.uint8), speed=np.array(speeds),
                        drive=np.array(drives), frame=np.array(frames), names=np.array(names),
                        full=np.array(full, dtype=np.uint8))
    print(f"{len(thumbs)} frames from {len(names)} drives -> {a.out}")


if __name__ == "__main__":
    main()
