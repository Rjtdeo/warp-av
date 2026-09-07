"""
Live wiring for swept-path blocking (Planning V2, phase 1C).

The planner's filter_to_route_corridor() takes an optional VehicleFootprint
(phase 1B). This module decides WHAT footprint the live stack hands it:

  * the van's real half-length / half-width, read once from the CARLA
    bounding box at start-up, with the verified defaults as the fallback;
  * a configurable safety margin (default 0.30 m);
  * a runtime flag, default OFF. Off, active_footprint() is None and the
    planner behaves exactly as before. On, the footprint is passed through.

No CARLA import: the vehicle is duck-typed (anything with
.bounding_box.extent.x / .y), so this is testable offline.
"""
from __future__ import annotations

from typing import Optional

from .footprint import VehicleFootprint

FALLBACK_HALF_LENGTH_M = 2.96      # the CARLA Sprinter, measured 2026-09-05
FALLBACK_HALF_WIDTH_M = 0.99
DEFAULT_SAFETY_MARGIN_M = 0.30
MIN_SAFETY_MARGIN_M = 0.0
MAX_SAFETY_MARGIN_M = 1.50
MIN_PLAUSIBLE_HALF_M = 0.3         # smaller than this and the box is not a van


def vehicle_half_dimensions(vehicle):
    """(half_length_m, half_width_m, source). source is 'carla' when the
    bounding box was readable and plausible, else 'fallback'."""
    try:
        ext = vehicle.bounding_box.extent
        hl, hw = float(ext.x), float(ext.y)
        if hl >= MIN_PLAUSIBLE_HALF_M and hw >= MIN_PLAUSIBLE_HALF_M and hl == hl and hw == hw:
            return hl, hw, "carla"
    except Exception:
        pass
    return FALLBACK_HALF_LENGTH_M, FALLBACK_HALF_WIDTH_M, "fallback"


def footprint_from_vehicle(vehicle, safety_margin: float = DEFAULT_SAFETY_MARGIN_M) -> VehicleFootprint:
    hl, hw, _ = vehicle_half_dimensions(vehicle)
    return VehicleFootprint(half_length=hl, half_width=hw, safety_margin=float(safety_margin))


class FootprintBlockingConfig:
    """The runtime switch and the footprint the live planner call uses."""

    def __init__(self, half_length: float = FALLBACK_HALF_LENGTH_M, half_width: float = FALLBACK_HALF_WIDTH_M,
                 safety_margin: float = DEFAULT_SAFETY_MARGIN_M, enabled: bool = False,
                 dimensions_source: str = "fallback"):
        self.enabled = bool(enabled)                 # DEFAULT OFF
        self.dimensions_source = dimensions_source
        self.footprint = VehicleFootprint(half_length=float(half_length), half_width=float(half_width),
                                          safety_margin=float(safety_margin))

    @classmethod
    def from_vehicle(cls, vehicle, safety_margin: float = DEFAULT_SAFETY_MARGIN_M,
                     enabled: bool = False) -> "FootprintBlockingConfig":
        hl, hw, source = vehicle_half_dimensions(vehicle)
        return cls(hl, hw, safety_margin, enabled=enabled, dimensions_source=source)

    def active_footprint(self) -> Optional[VehicleFootprint]:
        """What to pass to filter_to_route_corridor(footprint=...)."""
        return self.footprint if self.enabled else None

    def set(self, enabled=None, safety_margin_m=None):
        """Runtime change from the operator API. Returns a result dict."""
        if safety_margin_m is not None:
            try:
                m = float(safety_margin_m)
            except (TypeError, ValueError):
                return {"success": False, "reason": "safety_margin_m must be a number", **self.state()}
            if not (MIN_SAFETY_MARGIN_M <= m <= MAX_SAFETY_MARGIN_M) or m != m:
                return {"success": False,
                        "reason": f"safety_margin_m must be {MIN_SAFETY_MARGIN_M}-{MAX_SAFETY_MARGIN_M} m",
                        **self.state()}
            self.footprint = VehicleFootprint(half_length=self.footprint.half_length,
                                              half_width=self.footprint.half_width, safety_margin=m)
        if enabled is not None:
            if isinstance(enabled, str):
                if enabled.strip().lower() in ("true", "1", "on", "yes"):
                    enabled = True
                elif enabled.strip().lower() in ("false", "0", "off", "no"):
                    enabled = False
                else:
                    return {"success": False, "reason": "enabled must be true or false", **self.state()}
            self.enabled = bool(enabled)
        return {"success": True, **self.state()}

    def state(self) -> dict:
        """The 'planning' block of /api/state."""
        return {
            "footprint_blocking_enabled": self.enabled,
            "vehicle_half_length_m": round(self.footprint.half_length, 3),
            "vehicle_half_width_m": round(self.footprint.half_width, 3),
            "safety_margin_m": round(self.footprint.safety_margin, 3),
            "dimensions_source": self.dimensions_source,
        }
