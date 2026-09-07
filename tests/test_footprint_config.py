"""
Planning V2, phase 1C: the live wiring of swept-path blocking. Flag off by
default; real van size from the bounding box with a fallback; configurable
margin; telemetry; main.py hands the footprint to the planner only when ON.
"""
import os
import types

from warp_av.planning.footprint import VehicleFootprint
from warp_av.planning.footprint_config import (FootprintBlockingConfig, footprint_from_vehicle,
                                                vehicle_half_dimensions, FALLBACK_HALF_LENGTH_M,
                                                FALLBACK_HALF_WIDTH_M, DEFAULT_SAFETY_MARGIN_M)

MAIN = os.path.join(os.path.dirname(__file__), "..", "src", "warp_av", "main.py")


def fake_vehicle(hl=3.10, hw=1.05):
    ext = types.SimpleNamespace(x=hl, y=hw, z=1.2)
    return types.SimpleNamespace(bounding_box=types.SimpleNamespace(extent=ext))


class BrokenVehicle:
    @property
    def bounding_box(self):
        raise RuntimeError("actor destroyed")


def test_flag_defaults_off_and_planner_gets_no_footprint():
    cfg = FootprintBlockingConfig.from_vehicle(fake_vehicle())
    assert cfg.enabled is False
    assert cfg.active_footprint() is None            # the planner sees footprint=None -> old rules
    assert cfg.state()["footprint_blocking_enabled"] is False


def test_when_on_the_footprint_is_handed_to_the_planner():
    cfg = FootprintBlockingConfig.from_vehicle(fake_vehicle())
    assert cfg.set(enabled=True)["success"] is True
    fp = cfg.active_footprint()
    assert isinstance(fp, VehicleFootprint)
    assert fp.half_length == 3.10 and fp.half_width == 1.05 and fp.safety_margin == DEFAULT_SAFETY_MARGIN_M
    assert cfg.set(enabled="false")["success"] is True and cfg.active_footprint() is None


def test_carla_dimensions_are_used_when_available():
    hl, hw, source = vehicle_half_dimensions(fake_vehicle(2.96, 0.99))
    assert (hl, hw, source) == (2.96, 0.99, "carla")
    fp = footprint_from_vehicle(fake_vehicle(3.4, 1.1))
    assert (fp.half_length, fp.half_width) == (3.4, 1.1)


def test_fallback_dimensions_when_the_box_is_unreadable_or_absurd():
    for bad in (BrokenVehicle(), None, object(), fake_vehicle(0.0, 0.0), fake_vehicle(float("nan"), 1.0)):
        hl, hw, source = vehicle_half_dimensions(bad)
        assert (hl, hw, source) == (FALLBACK_HALF_LENGTH_M, FALLBACK_HALF_WIDTH_M, "fallback")
    cfg = FootprintBlockingConfig.from_vehicle(BrokenVehicle())
    assert cfg.state()["vehicle_half_length_m"] == 2.96 and cfg.state()["dimensions_source"] == "fallback"


def test_safety_margin_is_configurable_and_validated():
    cfg = FootprintBlockingConfig.from_vehicle(fake_vehicle())
    assert cfg.footprint.safety_margin == 0.30
    r = cfg.set(safety_margin_m=0.5)
    assert r["success"] and cfg.footprint.safety_margin == 0.5 and cfg.footprint.swept_half_width == 1.55
    assert cfg.set(safety_margin_m=-0.1)["success"] is False
    assert cfg.set(safety_margin_m=9.0)["success"] is False
    assert cfg.set(safety_margin_m="lots")["success"] is False
    assert cfg.footprint.safety_margin == 0.5            # rejected values leave it untouched
    assert cfg.set(enabled="maybe")["success"] is False and cfg.enabled is False


def test_state_reports_the_values_in_use():
    cfg = FootprintBlockingConfig.from_vehicle(fake_vehicle(3.10, 1.05))
    cfg.set(enabled=True, safety_margin_m=0.4)
    assert cfg.state() == {"footprint_blocking_enabled": True, "vehicle_half_length_m": 3.1,
                           "vehicle_half_width_m": 1.05, "safety_margin_m": 0.4, "dimensions_source": "carla"}


def test_main_wires_the_footprint_flag_and_telemetry():
    src = open(MAIN).read()
    # the live filter call hands over active_footprint() (None while OFF)
    assert "footprint=self.footprint_blocking.active_footprint()" in src
    # exactly one live call site, so the flag cannot be bypassed by a second path
    assert src.count("self.planner.filter_to_route_corridor(") == 1
    # built from the real vehicle, not hard-coded numbers
    assert "FootprintBlockingConfig.from_vehicle(self.vehicle_adapter.vehicle)" in src
    assert "2.96" not in src.split("FootprintBlockingConfig.from_vehicle")[1][:400]
    # telemetry block and runtime route
    assert '"planning": {**self.footprint_blocking.state()' in src
    assert "/api/planning/footprint_blocking" in src
