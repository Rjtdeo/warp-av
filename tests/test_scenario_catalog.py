"""Catalog integrity: 1000 scenarios, valid schema, deterministic, every required category present."""
import json
from collections import Counter
from pathlib import Path

import pytest

from warp_av.scenario_engine.generator import generate, FAMILIES, EXPECTED_TOTAL
from warp_av.scenario_engine.schema import validate_scenario, CATEGORIES, ScenarioValidationError
from warp_av.scenario_engine.catalog import Catalog, DEFAULT_ROOT

REQUIRED = ["normal_mission", "vehicle_ahead", "pedestrian", "static_obstacle", "blocked_route", "component_failure", "emergency_stop"]


@pytest.fixture(scope="module")
def scenarios():
    return generate()


def test_exactly_1000(scenarios):
    assert len(scenarios) == EXPECTED_TOTAL == 1000
    assert sum(q for _, q in FAMILIES) == 1000


def test_ids_unique_and_sequential(scenarios):
    ids = [s["id"] for s in scenarios]
    assert len(set(ids)) == 1000
    assert ids[0] == "WAV-0001" and ids[-1] == "WAV-1000"


def test_every_scenario_validates(scenarios):
    for s in scenarios:
        validate_scenario(s)


def test_required_categories_present(scenarios):
    c = Counter(s["category"] for s in scenarios)
    for r in REQUIRED:
        assert c[r] >= 50, f"{r} has only {c[r]}"
    assert set(c) <= set(CATEGORIES)


def test_deterministic():
    a = json.dumps(generate(), sort_keys=True)
    b = json.dumps(generate(), sort_keys=True)
    assert a == b


def test_every_scenario_has_collision_guard(scenarios):
    for s in scenarios:
        metrics = [c["metric"] for c in s["pass_criteria"] + s["fail_criteria"]]
        assert "collision_count" in metrics, s["id"]


def test_implemented_scenarios_have_checkable_pass_criteria(scenarios):
    for s in scenarios:
        if s["capability_status"] == "implemented":
            assert len(s["pass_criteria"]) >= 1, s["id"]


def test_status_mix_is_honest(scenarios):
    c = Counter(s["capability_status"] for s in scenarios)
    assert c["implemented"] > 300
    assert c["not_implemented"] > 50   # the catalog must keep documenting gaps


def test_validator_rejects_bad_scenario(scenarios):
    bad = dict(scenarios[0]); bad["category"] = "nope"
    with pytest.raises(ScenarioValidationError):
        validate_scenario(bad)
    bad = json.loads(json.dumps(scenarios[0])); bad["events"] = [{"trigger": {"at_s": 1}, "action": "inject", "params": {"component": "x", "action": "disable"}}]
    with pytest.raises(ScenarioValidationError):
        validate_scenario(bad)


@pytest.mark.skipif(not (DEFAULT_ROOT / "index.json").exists(), reason="catalog not generated")
def test_on_disk_catalog_matches_generator(scenarios):
    cat = Catalog()
    assert len(cat) == 1000
    on_disk = cat.load("WAV-0500")
    gen = next(s for s in scenarios if s["id"] == "WAV-0500")
    assert on_disk == gen
    assert len(cat.select(category="pedestrian")) == 100
    assert cat.select(tag="estop", limit=3)
