#!/usr/bin/env python3
"""
Verify Troy fix #5 (smooth driving) from a mission log.

Usage:
    python tools/check_smoothness.py               # newest logs/*.jsonl
    python tools/check_smoothness.py logs/mission_0003.jsonl

Reads the per-tick telemetry a mission writes and prints the three numbers the
fix is about, each with a target:

  1. steering flips/sec while moving   (weave)          target <= 2.0
  2. brake ticks while at speed        (random taps)    target <= 5
  3. speed wobble at cruise (stddev)   (sawtooth speed) target <= 0.5 m/s

Compare an old log vs a new log to see the before/after.
"""
import glob
import json
import math
import os
import sys


def load(path):
    ticks = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "command" in d and "pose" in d:
                ticks.append(d)
    return ticks


def analyze(path):
    ticks = load(path)
    if len(ticks) < 30:
        print(f"{path}: only {len(ticks)} ticks — run a longer mission")
        return

    t0, t1 = ticks[0]["t"], ticks[-1]["t"]
    speeds = [t["pose"]["speed"] for t in ticks]
    steers = [t["command"]["steer"] for t in ticks]
    brakes = [t["command"]["brake"] for t in ticks]
    vmax = max(speeds)

    # --- 1. steering weave: sign flips per second while actually moving ---
    moving = [(t, s) for t, s in zip(ticks, steers) if t["pose"]["speed"] > 2.0]
    flips = 0
    prev_sign = 0
    for _, s in moving:
        sign = 1 if s > 0.02 else -1 if s < -0.02 else 0
        if sign != 0:
            if prev_sign != 0 and sign != prev_sign:
                flips += 1
            prev_sign = sign
    moving_time = len(moving) * (t1 - t0) / max(1, len(ticks))
    flips_per_s = flips / moving_time if moving_time > 0 else 0.0

    # --- 2. brake taps at speed. Braking is JUSTIFIED when an object is within
    # 25 m in our path, or we are stopping/arriving — only the rest are "taps".
    brake_ticks = 0
    justified_ticks = 0
    for t, b in zip(ticks, brakes):
        if t["pose"]["speed"] <= 5.0 or b <= 0.01:
            continue
        closest = t.get("perception", {}).get("closest", 999)
        behavior = t.get("behavior", "")
        if closest < 25.0 or behavior.startswith("stopped") or behavior in ("approaching_destination", "mission_complete"):
            justified_ticks += 1
        else:
            brake_ticks += 1

    # --- 3. speed stability at cruise (top-25% speed portion of the drive) ---
    cruise = [v for v in speeds if v > 0.75 * vmax] if vmax > 3 else []
    if cruise:
        mean = sum(cruise) / len(cruise)
        std = math.sqrt(sum((v - mean) ** 2 for v in cruise) / len(cruise))
    else:
        mean = std = 0.0

    def verdict(ok):
        return "OK " if ok else "BAD"

    print(f"\n=== {path} ===")
    print(f"duration {t1 - t0:6.1f} s | ticks {len(ticks)} | max speed {vmax:.1f} m/s")
    print(f"[{verdict(flips_per_s <= 2.0)}] steering flips/sec while moving : {flips_per_s:5.2f}   (target <= 2.0)")
    print(f"[{verdict(brake_ticks <= 5)}] UNJUSTIFIED brake ticks at speed : {brake_ticks:5d}   (target <= 5; {justified_ticks} justified ticks for objects/arrival not counted)")
    print(f"[{verdict(std <= 0.5)}] speed wobble at cruise (stddev)  : {std:5.2f} m/s (target <= 0.5, cruise mean {mean:.1f})")
    print("max |steer| command:", round(max(abs(s) for s in steers), 3))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        logs = glob.glob(os.path.join("logs", "*.jsonl"))
        if not logs:
            sys.exit("no logs/*.jsonl found — run a mission first")
        paths = [max(logs, key=os.path.getmtime)]
    for p in paths:
        analyze(p)
