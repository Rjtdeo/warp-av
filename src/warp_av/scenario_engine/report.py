"""Aggregate scenarios/results/*.json into a pass-rate report (Markdown + JSON)."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

from .runner import RESULTS_DIR

VERDICTS = ["PASS", "FAIL", "GAP", "ERROR"]


def load_results(results_dir: Path = RESULTS_DIR) -> List[dict]:
    out = []
    for p in sorted(Path(results_dir).glob("WAV-*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def build_report(results: List[dict], catalog_index: dict | None = None) -> str:
    total_cat = len(catalog_index["scenarios"]) if catalog_index else None
    by_cat = defaultdict(Counter)
    by_fam = defaultdict(Counter)
    by_status = defaultdict(Counter)
    overall = Counter()
    for r in results:
        v = r["verdict"]
        overall[v] += 1
        by_cat[r["category"]][v] += 1
        by_fam[r["family"]][v] += 1
        by_status[r["capability_status"]][v] += 1

    def row(name, c):
        n = sum(c.values())
        pr = (100.0 * c["PASS"] / max(1, n - c["GAP"] - c["ERROR"]))
        return f"| {name} | {n} | {c['PASS']} | {c['FAIL']} | {c['GAP']} | {c['ERROR']} | {pr:.0f}% |"

    lines = ["# Scenario Run Report", ""]
    lines.append(f"Results: **{len(results)}** scenarios run" + (f" of {total_cat} in catalog" if total_cat else "") + ".")
    lines.append("")
    lines.append("Pass-rate = PASS / (PASS + FAIL): GAP (documented not-yet-implemented capability) and ERROR (runner could not execute) are excluded.")
    lines.append("")
    lines += ["| Scope | Run | PASS | FAIL | GAP | ERROR | Pass-rate |", "|---|---:|---:|---:|---:|---:|---:|", row("**overall**", overall)]
    lines += ["", "## By capability status", "", "| Status | Run | PASS | FAIL | GAP | ERROR | Pass-rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in ("implemented", "partial", "not_implemented"):
        if s in by_status:
            lines.append(row(s, by_status[s]))
    lines += ["", "## By category", "", "| Category | Run | PASS | FAIL | GAP | ERROR | Pass-rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(by_cat):
        lines.append(row(c, by_cat[c]))
    lines += ["", "## By family", "", "| Family | Run | PASS | FAIL | GAP | ERROR | Pass-rate |", "|---|---:|---:|---:|---:|---:|---:|"]
    for f in sorted(by_fam):
        lines.append(row(f, by_fam[f]))
    fails = [r for r in results if r["verdict"] in ("FAIL", "ERROR")]
    if fails:
        lines += ["", "## Failures & errors", "", "| ID | Name | Verdict | Reason | Collisions | Min dist (m) |", "|---|---|---|---|---:|---:|"]
        for r in fails:
            m = r.get("metrics", {})
            lines.append(f"| {r['scenario_id']} | {r['name']} | {r['verdict']} | {r['reason']} | {m.get('collision_count', '?')} | {m.get('min_distance_to_actor_m', '')} |")
    # safety KPIs
    coll = sum(1 for r in results if r.get("metrics", {}).get("collision_count", 0))
    rt = [r["metrics"]["safety_reaction_time_s"] for r in results if r.get("metrics", {}).get("safety_reaction_time_s") is not None]
    st = [r["metrics"]["stopped_within_s_of_trigger"] for r in results if r.get("metrics", {}).get("stopped_within_s_of_trigger") is not None]
    lines += ["", "## Safety KPIs", "",
              f"- Scenarios with ≥1 collision: **{coll}** / {len(results)}",
              f"- Safety-supervisor reaction time (fault→non-OK state): n={len(rt)}, max={max(rt) if rt else '-'} s, mean={round(sum(rt)/len(rt),2) if rt else '-'} s",
              f"- Stop time after stimulus: n={len(st)}, max={max(st) if st else '-'} s, mean={round(sum(st)/len(st),2) if st else '-'} s", ""]
    return "\n".join(lines)
