#!/usr/bin/env python3
"""Aggregate scenarios/results/ into scenarios/results/REPORT.md (+ summary.json)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pathlib import Path
from warp_av.scenario_engine.report import load_results, build_report
from warp_av.scenario_engine.runner import RESULTS_DIR

if __name__ == "__main__":
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR
    results = load_results(out_dir)
    idx_path = Path(__file__).parent / "catalog" / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else None
    md = build_report(results, idx)
    (out_dir / "REPORT.md").write_text(md)
    (out_dir / "summary.json").write_text(json.dumps(
        [{k: r[k] for k in ("scenario_id", "category", "family", "capability_status", "verdict", "reason")} | {"metrics": r.get("metrics", {})}
         for r in results], indent=1))
    print(md)
