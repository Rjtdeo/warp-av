#!/usr/bin/env python3
"""Regenerate scenarios/catalog/ (1000 scenarios). Deterministic — safe to re-run."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pathlib import Path
from warp_av.scenario_engine.generator import generate, write_catalog

if __name__ == "__main__":
    out = Path(__file__).parent / "catalog"
    scenarios = generate()
    per_cat = write_catalog(scenarios, out)
    print(f"Wrote {len(scenarios)} scenarios to {out}")
    for k, v in per_cat.items():
        print(f"  {k:26s} {v}")
