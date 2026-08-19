"""
Deterministic catalog generator.

    python3 scenarios/generate_catalog.py            # writes scenarios/catalog/

The catalog is 1000 scenarios = sum of the family quotas below.  Same seed ⇒
byte-identical output, so the catalog can be regenerated and diffed in review.
"""
from __future__ import annotations

import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import yaml

from . import gen_required_a as A
from . import gen_required_b as B
from . import gen_extended as E
from .schema import CATEGORIES, validate_scenario

SEED = 20260819
CATALOG_VERSION = "1.0.0"

# (family function, quota).  Order == id order.
FAMILIES: List[Tuple[Callable, int]] = [
    # --- normal_mission (60)
    (A.nm_basic, 24), (A.nm_spawn_pairs, 14), (A.nm_adverse_weather, 12), (A.nm_speed_sweep, 10),
    # --- vehicle_ahead (110)
    (A.va_stopped_lead, 20), (A.va_slow_lead, 14), (A.va_lead_brake_hard, 12), (A.va_cut_in, 14),
    (A.va_cut_out_reveal, 8), (A.va_oncoming, 10), (A.va_reversing, 6), (A.va_traffic_autopilot, 10),
    (A.va_queue, 6), (A.va_intersection_crossing, 10),
    # --- pedestrian (100)
    (A.ped_cross, 22), (A.ped_occluded_dart, 12), (A.ped_standing_edge, 10), (A.ped_walk_along, 8),
    (A.ped_cross_and_stop, 9), (A.ped_group, 8), (A.ped_child, 8), (A.ped_low_visibility, 12),
    (A.ped_at_start_and_destination, 5), (A.ped_crosswalk, 6),
    # --- static_obstacle (90)
    (B.so_single_prop, 36), (B.so_multi_stagger, 6), (B.so_sudden_appear, 14), (B.so_after_curve, 8),
    (B.so_low_and_large, 10), (B.so_at_destination, 6), (B.so_low_visibility, 10),
    # --- blocked_route (70)
    (B.br_full_block_persistent, 10), (B.br_block_clears, 14), (B.br_block_at_intersection, 8),
    (B.br_block_near_destination, 8), (B.br_construction_zone, 10), (B.br_traffic_jam, 8),
    (B.br_unreachable, 4), (B.br_block_immediately, 8),
    # --- component_failure (110)
    (B.cf_disable, 27), (B.cf_stale_and_freeze, 12), (B.cf_low_confidence, 8), (B.cf_recover, 14),
    (B.cf_double_failure, 10), (B.cf_failure_with_hazard, 10), (B.cf_failure_during_estop_and_arrival, 6),
    (B.cf_crash_and_invalid, 10), (B.cf_tick_latency, 8), (B.cf_flapping, 5),
    # --- emergency_stop (60)
    (B.es_at_speed, 21), (B.es_in_turn_and_hazard, 10), (B.es_clear_and_resume, 12), (B.es_repeated, 7),
    (B.es_during_pause_and_before_mission, 10),
    # --- operator_action (60)
    (E.oa_pause_resume, 14), (E.oa_stop_mission, 6), (E.oa_change_destination, 8), (E.oa_speed_limit_change, 10),
    (E.oa_pause_with_hazard, 6), (E.oa_back_to_back, 6), (E.oa_degenerate, 10),
    # --- sensor_degradation (50)
    (E.sd_dropout, 16), (E.sd_noise_latency, 24), (E.sd_multi_and_flap, 10),
    # --- localization_degradation (40)
    (E.ld_drift_jump, 16), (E.ld_stale_and_loss, 14), (E.ld_confidence_ramp, 10),
    # --- odd_boundary (40)
    (E.ob_geofence, 20), (E.ob_speed_and_weather, 20),
    # --- traffic_control (30)
    (E.tc_traffic_light, 18), (E.tc_signs, 12),
    # --- road_geometry (30)
    (E.rg_geometry, 30),
    # --- timing_latency (30)
    (E.tl_timing, 30),
    # --- vulnerable_road_user (40)
    (E.vru, 28), (E.vru_special, 12),
    # --- compound (30)
    (E.compound, 30),
    # --- edge_case (30)
    (E.edge, 30),
    # --- endurance (20)
    (E.endurance, 20),
]

EXPECTED_TOTAL = 1000


def generate(seed: int = SEED) -> List[dict]:
    scenarios: List[dict] = []
    for i, (fn, quota) in enumerate(FAMILIES):
        rng = random.Random(f"{seed}:{fn.__name__}")
        batch = fn(rng, quota)
        if len(batch) != quota:
            raise RuntimeError(f"{fn.__name__} returned {len(batch)} != quota {quota}")
        scenarios.extend(batch)

    if len(scenarios) != EXPECTED_TOTAL:
        raise RuntimeError(f"catalog has {len(scenarios)} scenarios, expected {EXPECTED_TOTAL}")

    # Assign ids in catalog order (stable) and de-duplicate names.
    seen_names = Counter()
    for idx, s in enumerate(scenarios, start=1):
        s["id"] = f"WAV-{idx:04d}"
        seen_names[s["name"]] += 1
        if seen_names[s["name"]] > 1:
            s["name"] = f"{s['name']} [v{seen_names[s['name']]}]"
        s["catalog_version"] = CATALOG_VERSION
        validate_scenario(s)
    return scenarios


def write_catalog(scenarios: List[dict], root: Path) -> Dict[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    # wipe stale generated files (only ours)
    for p in root.rglob("WAV-*.yaml"):
        p.unlink()
    per_cat = Counter()
    index = []
    for s in scenarios:
        cat_dir = root / s["category"]
        cat_dir.mkdir(exist_ok=True)
        path = cat_dir / f"{s['id']}.yaml"
        with path.open("w") as f:
            yaml.safe_dump(_ordered(s), f, sort_keys=False, width=110, allow_unicode=True)
        per_cat[s["category"]] += 1
        index.append({
            "id": s["id"], "name": s["name"], "category": s["category"], "family": s["family"],
            "capability_status": s["capability_status"], "tags": s["tags"],
            "town": s["odd"]["town"], "weather": s["odd"]["weather"], "light": s["odd"]["light"],
            "timeout_s": s["timeout_s"], "path": str(path.relative_to(root)),
        })
    (root / "index.json").write_text(json.dumps({
        "catalog_version": CATALOG_VERSION, "seed": SEED, "count": len(scenarios),
        "per_category": dict(per_cat), "scenarios": index}, indent=1))
    (root / "CATALOG.md").write_text(_catalog_md(scenarios, per_cat))
    return dict(per_cat)


_KEY_ORDER = ["id", "name", "category", "family", "capability_status", "catalog_version", "description", "tags",
              "required_capabilities", "odd", "mission", "actors", "events", "expected", "pass_criteria",
              "fail_criteria", "safety_boundary", "data_to_collect", "timeout_s"]


def _ordered(s: dict) -> dict:
    return {k: s[k] for k in _KEY_ORDER if k in s}


def _catalog_md(scenarios: List[dict], per_cat: Counter) -> str:
    lines = ["# Warp AV Scenario Catalog", "",
             f"Version {CATALOG_VERSION} · seed {SEED} · **{len(scenarios)} scenarios** · generated by `scenarios/generate_catalog.py`", "",
             "| Category | Count | implemented | partial | not_implemented |", "|---|---:|---:|---:|---:|"]
    for cat in CATEGORIES:
        rows = [s for s in scenarios if s["category"] == cat]
        st = Counter(s["capability_status"] for s in rows)
        lines.append(f"| {cat} | {len(rows)} | {st['implemented']} | {st['partial']} | {st['not_implemented']} |")
    st = Counter(s["capability_status"] for s in scenarios)
    lines.append(f"| **total** | **{len(scenarios)}** | {st['implemented']} | {st['partial']} | {st['not_implemented']} |")
    lines += ["", "`capability_status` is an honest flag: *implemented* = the current stack is expected to pass; "
              "*partial* = the stack reacts but not with the target behaviour; *not_implemented* = the scenario defines "
              "a contract the stack does not meet yet (it is still runnable — the result is recorded as a gap, not a failure).", ""]
    # family table
    lines += ["## Families", "", "| Family | Category | Count | Status |", "|---|---|---:|---|"]
    fam = OrderedDict()
    for s in scenarios:
        fam.setdefault(s["family"], [s["category"], 0, s["capability_status"]])[1] += 1
    for f, (c, n, stt) in fam.items():
        lines.append(f"| {f} | {c} | {n} | {stt} |")
    lines += ["", "## All scenarios", "", "| ID | Name | Category | Status | Town | Weather |", "|---|---|---|---|---|---|"]
    for s in scenarios:
        lines.append(f"| {s['id']} | {s['name']} | {s['category']} | {s['capability_status']} | {s['odd']['town']} | {s['odd']['weather']} |")
    return "\n".join(lines) + "\n"
