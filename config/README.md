# config/

| File | Used by |
|---|---|
| `scenario_runner.yaml` | documents runner defaults + the stack thresholds the catalog brackets (`scenarios/run_scenario.py` reads CLI/env; values here are the reference) |

Runtime CARLA host/port and vehicle blueprint are still constructor defaults in `src/warp_av/main.py` / `adapters/` (see KNOWN_ISSUES: "Hardcoded CARLA settings").
