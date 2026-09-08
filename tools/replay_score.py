"""
Replay every recorded fixture through the real perception pipeline and print
the scorecard (Perception V2, day 3). Runs on any machine, no CARLA.

    python3 tools/replay_score.py                 # all fixtures, current pipeline
    python3 tools/replay_score.py --ground flat   # A/B: the old road cut
    python3 tools/replay_score.py --thin 3        # A/B: the old thinning
    python3 tools/replay_score.py --json out.json
    python3 tools/replay_score.py --set-baseline  # write expected.json next to each fixture
                                                  # (current numbers minus a margin) for the tests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from warp_av.perception.replay import WARMUP_UPDATES, all_fixtures, load_fixture, replay, format_report  # noqa: E402

RECALL_MARGIN = 0.10
GROUND_MARGIN = 0.02
PHANTOM_MARGIN = 1.0      # phantoms per update allowed above today's count
MERGE_MARGIN = 0.15       # blobs covering two objects at once, allowed above today's count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground", choices=["patches", "flat"], default="patches")
    ap.add_argument("--thin", type=int, default=1)
    ap.add_argument("--cell", type=float, default=None, help="clustering grid in metres (default: the van's)")
    ap.add_argument("--keep-above", type=float, default=None, help="ground cut in metres (default: the van's)")
    ap.add_argument("--fixture", default=None, help="one fixture name; default all")
    ap.add_argument("--json", default=None)
    ap.add_argument("--set-baseline", action="store_true")
    a = ap.parse_args()

    paths = all_fixtures() if a.fixture is None else [ROOT / "tests" / "fixtures" / "perception" / a.fixture]
    results = []
    for p in paths:
        fx = load_fixture(p)
        results.append(replay(fx, ground_mode=a.ground, thin=a.thin,
                              cluster_cell_m=a.cell, keep_above_m=a.keep_above))
    print(format_report(results))
    if a.json:
        Path(a.json).write_text(json.dumps([{"fixture": r.fixture, "updates": r.updates, "rows": r.as_rows(),
                                             "unhealthy": r.unhealthy, "solid_reports": r.solid_reports, "kerb_reports": r.kerb_reports,
                                             "phantoms": r.phantoms, "phantoms_near": r.phantoms_near, "phantoms_in_lane": r.phantoms_in_lane,
                                             "merged": r.merged, "footprints": r.footprint_rows(),
                                             "ground": r.ground, "update_ms": float(sum(r.update_ms) / max(1, len(r.update_ms)))}
                                            for r in results], indent=1))
    if a.set_baseline:
        for p, r in zip(paths, results):
            expected = {
                "note": "thresholds = the numbers measured when this file was written, minus a margin; a change "
                        "that drops below them fails tests/test_replay_harness.py",
                "objects": {o.name: {"min_recall": round(max(0.0, o.recall - RECALL_MARGIN), 2) if o.visible else 0.0,
                                     "distance_m": round(o.distance, 1), "visible": o.visible,
                                     "labelled_points": o.labelled_points} for o in r.objects},
                "max_phantoms_in_lane": r.phantoms_in_lane,
                "max_merged_per_update": round(r.merged / max(1, r.updates - WARMUP_UPDATES - r.unhealthy) + MERGE_MARGIN, 2),
                "max_phantoms_per_update": round(r.phantoms / max(1, r.updates - WARMUP_UPDATES - r.unhealthy) + PHANTOM_MARGIN, 2),
                "min_road_deleted": round(max(0.0, r.ground.get("road_deleted", 0.99) - GROUND_MARGIN), 3),
                "min_object_kept": round(max(0.0, r.ground.get("object_kept", 0.8) - GROUND_MARGIN * 2), 3),
            }
            (p / "expected.json").write_text(json.dumps(expected, indent=1))
            print(f"wrote {p / 'expected.json'}")


if __name__ == "__main__":
    main()
