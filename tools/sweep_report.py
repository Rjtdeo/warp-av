#!/usr/bin/env python3
"""
Aggregate a mission-sweep results directory into the data tables of the
final report. The narrative (findings, fixes, recommendations) is written
by a human/agent around this output.

    python3 tools/sweep_report.py sweep_out/full > sweep_out/full/DATA.md
"""
import json
import glob
import math
import os
import statistics
import sys


def load(dirpath):
    runs = []
    for f in sorted(glob.glob(os.path.join(dirpath, "run_*.json"))):
        try:
            runs.append(json.load(open(f)))
        except Exception:
            pass
    return runs


def pct(a, b):
    return f"{100.0 * a / b:.0f}%" if b else "—"


def fmt_stats(vals, unit=""):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.2f}{unit}"
    return (f"mean {statistics.mean(vals):.2f}{unit} · "
            f"min {min(vals):.2f} · max {max(vals):.2f}")


def main():
    dirpath = sys.argv[1] if len(sys.argv) > 1 else "sweep_out/full"
    runs = load(dirpath)
    executed = [r for r in runs if r.get("verdict") not in ("TRIMMED",)]
    trimmed = [r for r in runs if r.get("verdict") == "TRIMMED"]

    out = []
    w = out.append

    verdicts = {}
    for r in executed:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    w(f"## Overall")
    w("")
    w(f"- Runs executed: **{len(executed)}** (planned 200, {len(trimmed)} trimmed "
      f"by operator decision to finish sooner — trimmed runs are NOT counted anywhere)")
    w(f"- Verdicts: " + " · ".join(f"**{k} {v}**" for k, v in sorted(verdicts.items())))
    w(f"- Collisions (ground-truth contact sensor): "
      f"**{sum(r.get('collisions', 0) for r in executed)}**")
    dists = [r.get("route_m") for r in executed if r.get("route_m")]
    w(f"- Distance driven (sum of planned routes): **{sum(dists)/1000:.1f} km** "
      f"across {len(dists)} missions")
    w("")

    # ---- by kind ----
    w("## By test kind")
    w("")
    w("| kind | runs | PASS | FAIL | GAP | notes |")
    w("|---|---|---|---|---|---|")
    for kind in ("baseline", "red_light", "jaywalker", "cutin", "fault"):
        ks = [r for r in executed if r.get("kind") == kind]
        if not ks:
            continue
        p = sum(1 for r in ks if r["verdict"] == "PASS")
        fl = sum(1 for r in ks if r["verdict"] == "FAIL")
        g = sum(1 for r in ks if r["verdict"] == "GAP")
        note = ""
        if kind == "red_light":
            holds = [(r.get("hazard") or {}).get("hold_junction_m") for r in ks]
            holds = [h for h in holds if isinstance(h, (int, float))]
            if holds:
                note = f"hold at junction edge: {fmt_stats(holds, ' m')}"
        if kind == "jaywalker":
            stopped = sum(1 for r in ks if (r.get("hazard") or {}).get("stopped_for_ped"))
            fired = sum(1 for r in ks if r.get("hazard") and not (r["hazard"] or {}).get("not_triggered"))
            note = f"stopped for the pedestrian in {stopped}/{fired} triggered runs"
        if kind == "cutin":
            reacted = sum(1 for r in ks if (r.get("hazard") or {}).get("reacted"))
            fired = sum(1 for r in ks if r.get("hazard") and not (r["hazard"] or {}).get("not_triggered"))
            note = f"reacted (followed/held) in {reacted}/{fired} triggered runs"
        if kind == "fault":
            times = [(r.get("hazard") or {}).get("safety_reacted_s") for r in ks]
            times = [t for t in times if isinstance(t, (int, float))]
            if times:
                note = f"safety reacted in {fmt_stats(times, ' s')}"
        w(f"| {kind} | {len(ks)} | {p} | {fl} | {g} | {note} |")
    w("")

    # ---- by weather ----
    w("## By weather")
    w("")
    w("| weather | runs | PASS | FAIL | GAP |")
    w("|---|---|---|---|---|")
    seen_w = []
    for r in executed:
        if r.get("weather") not in seen_w:
            seen_w.append(r.get("weather"))
    for wx in seen_w:
        ks = [r for r in executed if r.get("weather") == wx]
        w(f"| {wx} | {len(ks)} | "
          f"{sum(1 for r in ks if r['verdict']=='PASS')} | "
          f"{sum(1 for r in ks if r['verdict']=='FAIL')} | "
          f"{sum(1 for r in ks if r['verdict']=='GAP')} |")
    w("")

    # ---- parking ----
    w("## Parking (the flagship)")
    w("")
    offered = sum(1 for r in executed if r.get("slots_found", 0) > 0)
    chosen = sum(1 for r in executed if r.get("slot_chosen"))
    w(f"- Slot boxes found & offered: **{offered}/{len(executed)}** runs "
      f"(chosen slot in {chosen}) — the exact data feed the dashboard draws")
    slot_runs = [r for r in executed if r.get("park_target_kind") == "slot" and r.get("completed")]
    inside = [r for r in slot_runs if r.get("parked_inside_box")]
    w(f"- Completed slot-parkings: **{len(slot_runs)}**, fully inside the box: "
      f"**{len(inside)}** ({pct(len(inside), len(slot_runs))})")
    lens = [r.get("park_margin_len_m") for r in slot_runs]
    wids = [r.get("park_margin_wid_m") for r in slot_runs]
    heads = [r.get("park_heading_off_deg") for r in slot_runs]
    w(f"- Margins front/back: {fmt_stats(lens, ' m')}; side: {fmt_stats(wids, ' m')}; "
      f"heading offset: {fmt_stats(heads, '°')}")
    # split by stack version (before/after the precision fix)
    by_ver = {}
    for r in slot_runs:
        by_ver.setdefault(r.get("version", "?"), []).append(r)
    if len(by_ver) > 1:
        w(f"- By stack version (fixes landed mid-sweep):")
        for ver, rs in by_ver.items():
            ins = sum(1 for r in rs if r.get("parked_inside_box"))
            w(f"    - `{ver}`: {ins}/{len(rs)} inside "
              f"(side margin {fmt_stats([r.get('park_margin_wid_m') for r in rs], ' m')})")
    kerb = sum(1 for r in executed
               if r.get("completed") and r.get("park_target_kind") in ("bay", "kerb"))
    w(f"- Routes with no usable slot: parked at the kerb/bay instead in {kerb} runs (honest fallback)")
    resc = sum(1 for r in executed if r.get("rescan_retargeted"))
    w(f"- Occupied-slot re-targeting observed: {resc} runs")
    w("")

    # ---- smoothness ----
    w("## Driving quality across all executed runs")
    w("")
    flips = [r.get("steer_flips_per_s") for r in executed if r.get("steer_flips_per_s") is not None]
    wob = [r.get("cruise_wobble") for r in executed if r.get("cruise_wobble") is not None]
    brakes = [r.get("brake_anomaly_ticks", 0) for r in executed]
    w(f"- Steering flips/sec: {fmt_stats(flips)} (alarm threshold 2.0)")
    w(f"- Settled-cruise speed wobble: {fmt_stats(wob, ' m/s')}")
    w(f"- Unjustified-brake ticks: total {sum(brakes)} across all runs")
    clr = [r.get("min_object_m") for r in executed if r.get("min_object_m") is not None]
    w(f"- Closest approach to any object while moving: {fmt_stats(clr, ' m')}")
    w("")

    # ---- failure ledger ----
    w("## Every FAIL and ERROR, with cause")
    w("")
    w("| run | kind | weather | verdict | what happened |")
    w("|---|---|---|---|---|")
    for r in executed:
        if r["verdict"] in ("FAIL", "ERROR"):
            why = str(r.get("why", "")).replace("|", "\\|")[:160]
            w(f"| {r['run']} | {r.get('kind')} | {r.get('weather')} | {r['verdict']} | {why} |")
    w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
