#!/usr/bin/env python3
"""
Run catalog scenarios against a running CARLA + Warp AV stack.

  python3 scenarios/run_scenario.py WAV-0001                 # one scenario
  python3 scenarios/run_scenario.py --category pedestrian    # a category
  python3 scenarios/run_scenario.py --tag estop --limit 5
  python3 scenarios/run_scenario.py --status implemented     # only what should pass today
  python3 scenarios/run_scenario.py --all                    # everything (hours)
  python3 scenarios/run_scenario.py WAV-0200 --dry-run       # print the execution plan, no CARLA needed

Results: scenarios/results/runs/<run_id>/<ID>.json (+ .trace.jsonl, run_manifest.json, SUMMARY.md,
summary.csv). One folder per run, nothing overwritten. Then: python3 scenarios/report.py <run folder>

V1 evidence options:
  --ids-file scratch/v1/subset.txt     scenario ids, one per line (# comments allowed); repeats allowed
  --run-id v1_first10                  name the run folder (default: time + git sha)
  --reset-ego spawn:12 | x,y,yaw       teleport the ego to the same start before every scenario
  --seed 7                             pin CARLA's traffic manager (autopilot actors) to a seed
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from warp_av.scenario_engine.catalog import Catalog
from warp_av.scenario_engine.runner import ScenarioRunner, RESULTS_DIR, parse_reset_spec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="scenario ids (WAV-0001) or yaml paths")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--category"); ap.add_argument("--family"); ap.add_argument("--tag")
    ap.add_argument("--status", choices=["implemented", "partial", "not_implemented"])
    ap.add_argument("--town"); ap.add_argument("--limit", type=int)
    ap.add_argument("--api", default=os.environ.get("WARP_API", "http://localhost:5000"))
    ap.add_argument("--carla-host", default=os.environ.get("CARLA_HOST", "localhost"))
    ap.add_argument("--carla-port", type=int, default=int(os.environ.get("CARLA_PORT", "2000")))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-fail", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--ids-file", help="file of scenario ids, one per line")
    ap.add_argument("--run-id", help="name of the run folder under scenarios/results/runs/")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--reset-ego", help="'spawn:N' or 'x,y,yaw_deg': teleport the ego there before every scenario")
    ap.add_argument("--seed", type=int, help="traffic-manager seed for autopilot actors (recorded either way)")
    a = ap.parse_args()

    cat = Catalog()
    if a.ids_file:
        ids = [ln.split("#")[0].strip() for ln in open(a.ids_file, encoding="utf-8")]
        ids = [i for i in ids if i]
    elif a.ids:
        ids = a.ids
    elif a.all or a.category or a.family or a.tag or a.status or a.town:
        ids = cat.select(category=a.category, family=a.family, tag=a.tag, status=a.status, town=a.town, limit=a.limit)
    else:
        ap.error("give scenario ids or a filter (--all/--category/--family/--tag/--status/--town)")
    print(f"{len(ids)} scenario(s) selected")
    runner = ScenarioRunner(api_url=a.api, carla_host=a.carla_host, carla_port=a.carla_port, dry_run=a.dry_run,
                            verbose=not a.quiet, results_dir=a.results_dir, run_id=a.run_id, seed=a.seed,
                            reset_ego=parse_reset_spec(a.reset_ego))
    runner.planned_ids = list(ids)
    if not a.dry_run:
        print(f"run folder: {runner.run_dir}")
    tally = {}
    for sid in ids:
        res = runner.run(cat.load(sid))
        tally[res["verdict"]] = tally.get(res["verdict"], 0) + 1
        if a.stop_on_fail and res["verdict"] == "FAIL":
            break
    print("\nSummary:", tally)
    if not a.dry_run:
        print(f"evidence: {runner.run_dir / 'SUMMARY.md'}")

if __name__ == "__main__":
    main()
