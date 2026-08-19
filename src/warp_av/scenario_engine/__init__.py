"""
Scenario Engine — declarative scenarios for the Warp AV stack.

    schema.py     : the scenario data model + validation (no CARLA needed)
    generator.py  : deterministic generator for the 1000-scenario catalog
    catalog.py    : load / query the catalog on disk
    runner.py     : execute a scenario against CARLA + the running autonomy API
    evaluator.py  : turn a telemetry trace into pass / fail / gap verdicts
    report.py     : aggregate results into a pass-rate table

Only runner.py imports `carla`. Everything else runs anywhere.
"""
