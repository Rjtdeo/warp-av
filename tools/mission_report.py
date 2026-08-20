#!/usr/bin/env python3
"""
Full mission report + verification from a Warp AV telemetry log.

    python tools/mission_report.py                     # newest logs/*.jsonl
    python tools/mission_report.py logs/mission_0010.jsonl

Prints: run summary, verification checklist ([OK]/[BAD]), behavior timeline,
all events, every stop with its reason, and the parking result.
"""
import glob
import json
import math
import os
import re
import sys


def load(path):
    ticks, events = [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            (ticks if "pose" in d else events if "event" in d else []).append(d)
    return ticks, events


def fmt_t(t0, t):
    return f"{t - t0:7.1f}s"


def main(path):
    ticks, events = load(path)
    if len(ticks) < 10:
        sys.exit(f"{path}: only {len(ticks)} ticks — not a real run")
    t0 = min(ticks[0]["t"], events[0]["t"] if events else ticks[0]["t"])
    t1 = ticks[-1]["t"]
    speeds = [t["pose"]["speed"] for t in ticks]
    xs = [t["pose"]["x"] for t in ticks]
    ys = [t["pose"]["y"] for t in ticks]
    dist = sum(math.hypot(a - b, c - d) for a, b, c, d in zip(xs[1:], xs[:-1], ys[1:], ys[:-1]))

    print("=" * 72)
    print(f"MISSION REPORT — {os.path.basename(path)}")
    print("=" * 72)
    print(f"duration {t1 - t0:6.1f} s | ticks {len(ticks)} | distance {dist:6.0f} m | "
          f"max {max(speeds):.1f} m/s | moving avg {sum(v for v in speeds if v > 0.5) / max(1, sum(1 for v in speeds if v > 0.5)):.1f} m/s")
    print(f"start ({xs[0]:.1f}, {ys[0]:.1f})  ->  end ({xs[-1]:.1f}, {ys[-1]:.1f})")

    # ---------------- events ----------------
    print("\n--- EVENTS " + "-" * 60)
    spot_kind, parked_line = None, None
    for e in events:
        print(f"{fmt_t(t0, e['t'])}  {e['event']:<22} {e.get('description', '')}")
        if e["event"] == "parking_spot":
            spot_kind = "bay" if "BAY" in e.get("description", "") else "kerb"
        if e["event"] == "mission_completed":
            parked_line = e.get("description", "")

    # ---------------- behavior timeline ----------------
    print("\n--- BEHAVIOR TIMELINE " + "-" * 49)
    last = None
    seen_behaviors = []
    for t in ticks:
        key = (t.get("behavior"), t.get("safety", {}).get("state"))
        if key != last:
            print(f"{fmt_t(t0, t['t'])}  {t.get('behavior', ''):<24} v={t['pose']['speed']:4.1f}  {t.get('behavior_reason', '')[:70]}")
            last = key
            seen_behaviors.append(t.get("behavior"))

    # ---------------- stops ----------------
    print("\n--- STOPS (speed < 0.2 for >= 1 s) " + "-" * 36)
    in_stop, s_start, n_stops = False, 0, 0
    reasons = {}
    def top_reason():
        return max(reasons.items(), key=lambda kv: kv[1])[0] if reasons else ""
    for t in ticks:
        v = t["pose"]["speed"]
        if v < 0.2 and not in_stop:
            in_stop, s_start, reasons = True, t["t"], {}
        if v < 0.2 and in_stop:
            r = t.get("behavior_reason", "")
            reasons[r] = reasons.get(r, 0) + 1
        elif v >= 0.2 and in_stop:
            in_stop = False
            if t["t"] - s_start >= 1.0:
                n_stops += 1
                print(f"{fmt_t(t0, s_start)}  {t['t'] - s_start:5.1f} s   {top_reason()[:66]}")
    if in_stop and t1 - s_start >= 1.0:
        n_stops += 1
        print(f"{fmt_t(t0, s_start)}  {t1 - s_start:5.1f} s   {top_reason()[:66]}  (final)")
    if n_stops == 0:
        print("  none")

    # ---------------- smoothness (same rules as check_smoothness) ----------------
    moving = [(t, t["command"]["steer"]) for t in ticks if t["pose"]["speed"] > 2.0]
    flips, prev = 0, 0
    for _, s in moving:
        sign = 1 if s > 0.02 else -1 if s < -0.02 else 0
        if sign:
            if prev and sign != prev:
                flips += 1
            prev = sign
    mt = len(moving) * (t1 - t0) / max(1, len(ticks))
    flips_s = flips / mt if mt else 0.0
    bad_brakes = justified = 0
    for t in ticks:
        if t["pose"]["speed"] <= 5.0 or t["command"]["brake"] <= 0.01:
            continue
        if (t.get("perception", {}).get("closest", 999) < 25.0
                or str(t.get("behavior", "")).startswith(("stopped", "waiting", "parking"))
                or t.get("behavior") in ("approaching_destination", "mission_complete")):
            justified += 1
        else:
            bad_brakes += 1
    vmax = max(speeds)
    # wobble only on SETTLED cruise: exclude accelerating/braking ramps
    cruise = []
    for prev, t in zip(ticks, ticks[1:]):
        v = t["pose"]["speed"]
        if (v > 0.75 * vmax
                and abs(v - prev["pose"]["speed"]) <= 0.05
                and "curve ahead" not in (t.get("behavior_reason") or "")
                and t.get("behavior") not in ("following_vehicle", "approaching_destination", "parking")):
            cruise.append(v)
    if len(cruise) >= 20:
        m = sum(cruise) / len(cruise)
        wob = math.sqrt(sum((v - m) ** 2 for v in cruise) / len(cruise))
    else:
        wob = 0.0

    # parking precision from the completion line
    p_dist = p_head = None
    if parked_line:
        m1 = re.search(r"Parked ([\d.]+) m .*?(\d+) deg", parked_line)
        if m1:
            p_dist, p_head = float(m1.group(1)), float(m1.group(2))

    # ---------------- verification ----------------
    def row(ok, label, val):
        print(f"[{'OK ' if ok else 'BAD'}] {label:<38} {val}")

    print("\n--- VERIFICATION " + "-" * 54)
    completed = any(e["event"] == "mission_completed" for e in events)
    row(completed, "mission completed", parked_line or "no")
    if spot_kind:
        row(True, "parking spot type", "BAY (off the driving lane)" if spot_kind == "bay" else "kerb-hug (no bay on street)")
    if p_dist is not None:
        row(p_dist <= 1.5, "parked distance from spot", f"{p_dist} m (target <= 1.5)")
        row(p_head <= 15, "parked heading vs road", f"{p_head:.0f} deg (target <= 15)")
    row(flips_s <= 2.0, "steering flips/sec while moving", f"{flips_s:.2f} (target <= 2.0)")
    row(bad_brakes <= 5, "unjustified brake ticks at speed", f"{bad_brakes} ({justified} justified excluded)")
    row(wob <= 0.5, "speed wobble at clean cruise", f"{wob:.2f} m/s (target <= 0.5)")
    unsafe = sorted({t.get("safety", {}).get("state") for t in ticks} - {"ok", None, ""})
    row(not unsafe, "safety supervisor", "ok the whole run" if not unsafe else f"non-ok states: {unsafe}")
    interesting = [b for b in dict.fromkeys(seen_behaviors)
                   if b not in (None, "no_mission", "idle")]
    print("\nbehaviors this run: " + " -> ".join(interesting))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        logs = glob.glob(os.path.join("logs", "*.jsonl"))
        if not logs:
            sys.exit("no logs/*.jsonl found")
        main(max(logs, key=os.path.getmtime))
