#!/usr/bin/env python3
"""Build the static site deployed to Vercel: web/public/ = catalog browser + operator console + docs.
Run: python3 scripts/build_site.py   (then `vercel --prod` from web/ or repo root)"""
import json, shutil, sys, os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "public"
cat = ROOT / "scenarios" / "catalog"

(OUT / "catalog").mkdir(parents=True, exist_ok=True)
idx = json.loads((cat / "index.json").read_text())
full = []
for e in idx["scenarios"]:
    full.append(yaml.safe_load((cat / e["path"]).read_text()))
(OUT / "catalog" / "scenarios.json").write_text(json.dumps(full, separators=(",", ":")))
(OUT / "catalog" / "index.json").write_text(json.dumps(idx, separators=(",", ":")))
shutil.copy(ROOT / "src/warp_av/console/index.html", OUT / "console.html")
(OUT / "docs").mkdir(exist_ok=True)
for md in ["README.md", "KNOWN_ISSUES.md", "OPEN_SOURCE.md", "docs/SCENARIO_STRATEGY.md", "scenarios/README.md", "architecture/README.md", "scenarios/catalog/CATALOG.md"]:
    p = ROOT / md
    if p.exists():
        shutil.copy(p, OUT / "docs" / p.name.replace("README.md", (p.parent.name if p.parent != ROOT else "README") + ".md"))
rep = ROOT / "scenarios/results/REPORT.md"
if rep.exists():
    shutil.copy(rep, OUT / "docs" / "REPORT.md")
print(f"built {OUT}: {len(full)} scenarios")
