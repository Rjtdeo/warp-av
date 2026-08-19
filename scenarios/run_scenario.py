#!/usr/bin/env python3
"""
Run catalog scenarios against a running CARLA + Warp AV stack.

  python3 scenarios/run_scenario.py WAV-0001                 # one scenario
  python3 scenarios/run_scenario.py --category pedestrian    # a category
  python3 scenarios/run_scenario.py --tag estop --limit 5
  python3 scenarios/run_scenario.py --status implemented     # only what should pass today
  python3 scenarios/run_scenario.py --all                    # everything (hours)
  python3 scenarios/run_scenario.py WAV-0200 --dry-run       # print the execution plan, no CARLA needed

Results: scenarios/results/<ID>.json (+ .trace.jsonl). Then: python3 scenarios/report.py
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from warp_av.scenario_engine.catalog import Catalog
from warp_av.scenario_engine.runner import ScenarioRunner

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
    a = ap.parse_args()

    cat = Catalog()
    if a.ids:
        ids = a.ids
    elif a.all or a.category or a.family or a.tag or a.status or a.town:
        ids = cat.select(category=a.category, family=a.family, tag=a.tag, status=a.status, town=a.town, limit=a.limit)
    else:
        ap.error("give scenario ids or a filter (--all/--category/--family/--tag/--status/--town)")
    print(f"{len(ids)} scenario(s) selected")
    runner = ScenarioRunner(api_url=a.api, carla_host=a.carla_host, carla_port=a.carla_port, dry_run=a.dry_run, verbose=not a.quiet)
    tally = {}
    for sid in ids:
        res = runner.run(cat.load(sid))
        tally[res["verdict"]] = tally.get(res["verdict"], 0) + 1
        if a.stop_on_fail and res["verdict"] == "FAIL":
            break
    print("\nSummary:", tally)

if __name__ == "__main__":
    main()
