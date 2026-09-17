"""
V1 evidence layer: one flat, honest row per scenario run, and the per-run summary table.

Nothing here drives the van. It reads what the runner recorded (the trace and the metrics) and
writes it down in a shape a person can scan and a spreadsheet can load:

    scenarios/results/runs/<run_id>/<WAV-id>.json          verdict, metrics, meta, evidence row
    scenarios/results/runs/<run_id>/<WAV-id>.trace.jsonl   every poll: /api/state + CARLA truth
    scenarios/results/runs/<run_id>/run_manifest.json      who ran what, on which code, when
    scenarios/results/runs/<run_id>/SUMMARY.md + summary.csv   one row per scenario

Also home to the box-gap geometry: the true clearance between two rectangles seen from above,
which is what "minimum clearance" has to mean for a van that is 2 m wide.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

V1_FIELDS = [
    "scenario_id", "run_id", "verdict", "reason", "family", "capability_status",
    "git_sha_runner", "git_sha_stack", "git_dirty", "seed", "map", "map_expected", "carla_server", "started_at",
    "start_xy", "goal_xy", "duration_s", "distance_m", "completed", "mission_record_state", "mission_reason_ended",
    "final_goal_distance_m", "final_mission_state", "stuck_at_end", "stuck_at_end_s",
    "collision_count", "stack_collision_count",
    "min_gap_to_actor_m", "min_distance_to_actor_m", "min_distance_any_actor_m", "min_perception_closest_m",
    "behavior_final", "behaviors_seen", "safety_final", "safety_states_seen", "planner_reason_final",
    "loop_hz_mean", "loop_hz_min", "loop_hz_p05", "tick_ms_max", "poll_hz_actual", "poll_api_ms_median", "poll_truth_ms_median",
    "moved_at_red_m", "start_clean", "ego_reset_off_m", "warnings", "result_file", "trace_file",
]

Box = Sequence[float]   # [x, y, yaw_rad, half_length, half_width]


# ---------------------------------------------------------------- geometry
def obb_corners(box: Box) -> List[Tuple[float, float]]:
    x, y, yaw, hl, hw = box[0], box[1], box[2], box[3], box[4]
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        out.append((x + c * dx - s * dy, y + s * dx + c * dy))
    return out


def _seg_dist(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _separated(P, Q) -> bool:
    """Separating-axis test for two convex polygons: True when some edge normal keeps them apart."""
    for poly in (P, Q):
        for i in range(len(poly)):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % len(poly)]
            nx, ny = -(by - ay), bx - ax
            p_proj = [nx * x + ny * y for x, y in P]
            q_proj = [nx * x + ny * y for x, y in Q]
            if max(p_proj) < min(q_proj) or max(q_proj) < min(p_proj):
                return True
    return False


def box_gap(a: Box, b: Box) -> float:
    """Smallest distance between two rectangles seen from above; 0.0 when they overlap."""
    P, Q = obb_corners(a), obb_corners(b)
    if not _separated(P, Q):
        return 0.0
    best = float("inf")
    for poly, other in ((P, Q), (Q, P)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _seg_dist(p, other[i], other[(i + 1) % len(other)]))
    return best


# ---------------------------------------------------------------- the flat row
def _r(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def build_evidence(result: dict) -> dict:
    """One flat row (V1 result format) from a runner result dict. Missing things stay None: a
    blank cell is honest, a guessed number is not."""
    m = result.get("metrics", {}) or {}
    meta = result.get("meta", {}) or {}
    stack = meta.get("stack") or {}
    row = {
        "scenario_id": result.get("scenario_id"),
        "run_id": result.get("run_id"),
        "verdict": result.get("verdict"),
        "reason": result.get("reason"),
        "family": result.get("family"),
        "capability_status": result.get("capability_status"),
        "git_sha_runner": meta.get("git_sha_runner"),
        "git_sha_stack": stack.get("git_sha"),
        "git_dirty": meta.get("git_dirty"),
        "seed": meta.get("seed"),
        "map": meta.get("map"),
        "map_expected": meta.get("map_expected"),
        "carla_server": (meta.get("carla_version") or {}).get("server"),
        "started_at": meta.get("started_at"),
        "start_xy": meta.get("start_xy"),
        "goal_xy": meta.get("goal_xy"),
        "duration_s": _r(m.get("elapsed_s")),
        "distance_m": _r(m.get("distance_m"), 1),
        "completed": m.get("mission_completed"),
        "mission_record_state": m.get("mission_record_state"),
        "mission_reason_ended": m.get("mission_reason_ended"),
        "final_goal_distance_m": _r(m.get("final_goal_distance_m"), 1),
        "final_mission_state": m.get("final_mission_state"),
        "stuck_at_end": m.get("stuck_at_end"),
        "stuck_at_end_s": m.get("stuck_at_end_s"),
        "collision_count": m.get("collision_count"),
        "stack_collision_count": m.get("stack_collision_count"),
        "min_gap_to_actor_m": _r(m.get("min_gap_to_actor_m")),
        "min_distance_to_actor_m": _r(m.get("min_distance_to_actor_m")),
        "min_distance_any_actor_m": _r(m.get("min_distance_any_actor_m")),
        "min_perception_closest_m": _r(m.get("min_perception_closest_m"), 1),
        "behavior_final": m.get("behavior_final"),
        "behaviors_seen": m.get("behaviors_seen"),
        "safety_final": m.get("safety_final"),
        "safety_states_seen": m.get("safety_states_seen"),
        "planner_reason_final": m.get("planner_reason_final"),
        "loop_hz_mean": _r(m.get("loop_hz_mean"), 1),
        "loop_hz_min": _r(m.get("loop_hz_min"), 1),
        "loop_hz_p05": _r(m.get("loop_hz_p05"), 1),
        "tick_ms_max": _r(m.get("tick_ms_max"), 1),
        "poll_hz_actual": _r(m.get("poll_hz_actual"), 1),
        "poll_api_ms_median": _r(m.get("poll_api_ms_median"), 1),
        "poll_truth_ms_median": _r(m.get("poll_truth_ms_median"), 1),
        "moved_at_red_m": _r(m.get("moved_at_red_m"), 1),
        "start_clean": (meta.get("start_check") or {}).get("clean"),
        "ego_reset_off_m": (meta.get("ego_reset") or {}).get("settled_off_m"),
        "warnings": result.get("warnings"),
        "result_file": result.get("result_file"),
        "trace_file": result.get("trace_file"),
    }
    return row


# ---------------------------------------------------------------- per-run summary
def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v).replace("|", "/").replace("\n", " ")


SUMMARY_COLUMNS = [
    ("scenario_id", "ID"), ("verdict", "Verdict"), ("family", "Family"), ("completed", "Done"), ("mission_record_state", "Mission record"),
    ("final_goal_distance_m", "To goal m"), ("stuck_at_end_s", "Stuck at end s"),
    ("collision_count", "Coll (sensor)"), ("stack_collision_count", "Coll (stack)"),
    ("min_gap_to_actor_m", "Min gap m"), ("min_distance_any_actor_m", "Nearest actor m"),
    ("distance_m", "Dist m"), ("duration_s", "Time s"), ("loop_hz_mean", "Hz mean"), ("loop_hz_min", "Hz min"),
    ("behavior_final", "Final behaviour"), ("safety_final", "Safety"), ("planner_reason_final", "Planner says"),
    ("start_clean", "Clean start"), ("reason", "Verdict reason"),
]


def load_run_results(run_dir: Path) -> List[dict]:
    out = []
    for p in sorted(Path(run_dir).glob("WAV-*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    out.sort(key=lambda r: (r.get("started_at") or 0, r.get("scenario_id") or ""))
    return out


def write_run_summary(run_dir: Path) -> Path:
    """Rebuild SUMMARY.md and summary.csv from every WAV-*.json in the run folder."""
    run_dir = Path(run_dir)
    results = load_run_results(run_dir)
    rows = [r.get("evidence") or build_evidence(r) for r in results]
    manifest = {}
    mp = run_dir / "run_manifest.json"
    if mp.exists():
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    tally: Dict[str, int] = {}
    for r in rows:
        tally[r.get("verdict") or "?"] = tally.get(r.get("verdict") or "?", 0) + 1
    collisions = sum(1 for r in rows if (r.get("collision_count") or 0) > 0)

    lines = [f"# Run {run_dir.name}", ""]
    if manifest:
        lines.append(f"Runner code `{manifest.get('git_sha_runner')}`" + (" (dirty tree)" if manifest.get("git_dirty") else "") +
                     f" · stack `{manifest.get('git_sha_stack')}` · map {manifest.get('map')} · CARLA {manifest.get('carla_server')}"
                     f" · seed {manifest.get('seed')} · ego reset {manifest.get('reset_ego')} · started {manifest.get('started_at')}")
        lines.append("")
    lines.append("Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())) +
                 f" · scenarios with a collision: {collisions} of {len(rows)}")
    lines.append("")
    lines.append("| " + " | ".join(h for _, h in SUMMARY_COLUMNS) + " |")
    lines.append("|" + "|".join("---" for _ in SUMMARY_COLUMNS) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(k)) for k, _ in SUMMARY_COLUMNS) + " |")
    lines.append("")
    lines.append("Columns: *Coll (sensor)* is the runner's own collision sensor on the ego; *Coll (stack)* is the van's "
                 "collision counter over the same window. *Min gap* is the true box-to-box clearance to a scenario actor "
                 "(0 = touching). *Nearest actor* is centre-to-centre to any vehicle, walker or prop actor in the world. "
                 "*Hz* is the van's decision rate as it reports it (smoothed). Map-mesh objects are invisible to both distances.")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=V1_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(r.get(k)) for k in V1_FIELDS})
    return run_dir / "SUMMARY.md"
