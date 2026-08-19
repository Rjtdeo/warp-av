# tests/

Run: `python3 -m pytest tests/` (no CARLA needed; a stub `carla` module is injected by `conftest.py`).

| File | Covers |
|---|---|
| `test_scenario_catalog.py` | generator produces exactly 1000 valid, unique, deterministic scenarios; required categories present; every scenario has a collision guard; on-disk catalog == generator |
| `test_evaluator.py` | metrics from synthetic traces (stop time, reaction time, min distance, route deviation, resume) and PASS/FAIL/GAP/ERROR verdict logic |
| `test_safety_and_faults.py` | safety supervisor per-fault behaviour + e-stop latching; behaviour priority order & reasons; controller NaN injection; vehicle-adapter command validation (NaN/stale → brake); fault-injector dispatch |

Scenario *execution* tests need a live simulator: `python3 scenarios/run_scenario.py --status implemented` then `python3 scenarios/report.py`. Results are documented in `scenarios/results/REPORT.md` when run.
