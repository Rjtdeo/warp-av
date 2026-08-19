"""Load / query the on-disk scenario catalog."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import yaml

from .schema import validate_scenario

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "scenarios" / "catalog"


class Catalog:
    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root)
        idx_path = self.root / "index.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"{idx_path} missing — run scenarios/generate_catalog.py")
        self.index = json.loads(idx_path.read_text())
        self._by_id = {s["id"]: s for s in self.index["scenarios"]}

    def __len__(self):
        return len(self._by_id)

    def ids(self) -> List[str]:
        return list(self._by_id)

    def load(self, sid: str) -> dict:
        entry = self._by_id.get(sid)
        if entry is None:
            # allow loading by path too
            p = Path(sid)
            if p.exists():
                s = yaml.safe_load(p.read_text())
                validate_scenario(s)
                return s
            raise KeyError(f"unknown scenario {sid}")
        s = yaml.safe_load((self.root / entry["path"]).read_text())
        validate_scenario(s)
        return s

    def select(self, ids: Optional[Iterable[str]] = None, category: Optional[str] = None,
               family: Optional[str] = None, tag: Optional[str] = None, status: Optional[str] = None,
               town: Optional[str] = None, limit: Optional[int] = None) -> List[str]:
        out = []
        for e in self.index["scenarios"]:
            if ids and e["id"] not in set(ids):
                continue
            if category and e["category"] != category:
                continue
            if family and e["family"] != family:
                continue
            if tag and tag not in e["tags"]:
                continue
            if status and e["capability_status"] != status:
                continue
            if town and e["town"] != town:
                continue
            out.append(e["id"])
        return out[:limit] if limit else out
