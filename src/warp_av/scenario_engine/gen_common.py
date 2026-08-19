"""
Shared building blocks for the scenario generator.

Everything here is deterministic: no wall clock, no global random.
"""

from __future__ import annotations

import itertools
import random
from typing import Any, Dict, List, Sequence

# ------------------------------------------------------------------
# ODD vocab
# ------------------------------------------------------------------

# preset -> (light, extra tags)
WEATHERS: Dict[str, tuple] = {
    "ClearNoon":      ("day",   []),
    "CloudyNoon":     ("day",   ["overcast"]),
    "WetNoon":        ("day",   ["wet_road"]),
    "WetCloudyNoon":  ("day",   ["wet_road", "overcast"]),
    "SoftRainNoon":   ("day",   ["rain", "wet_road"]),
    "MidRainyNoon":   ("day",   ["rain", "wet_road"]),
    "HardRainNoon":   ("day",   ["heavy_rain", "wet_road", "reduced_visibility"]),
    "ClearSunset":    ("dusk",  ["low_sun_glare"]),
    "WetSunset":      ("dusk",  ["wet_road", "low_sun_glare"]),
    "HardRainSunset": ("dusk",  ["heavy_rain", "wet_road", "reduced_visibility"]),
    "ClearNight":     ("night", ["night"]),
    "WetNight":       ("night", ["night", "wet_road"]),
    "SoftRainNight":  ("night", ["night", "rain", "wet_road"]),
    "HardRainNight":  ("night", ["night", "heavy_rain", "wet_road", "reduced_visibility"]),
    "DustStorm":      ("day",   ["dust", "reduced_visibility"]),
}
DAY_WEATHERS = [w for w, (l, _) in WEATHERS.items() if l == "day" and w != "DustStorm"]
CLEAR_DAY = ["ClearNoon", "CloudyNoon"]
ADVERSE = ["HardRainNoon", "HardRainSunset", "HardRainNight", "WetNight", "DustStorm"]

# CARLA towns and what they are good for
TOWNS = {
    "Town01": "small town, simple T-junctions, 2-lane roads",
    "Town02": "small town, compact grid",
    "Town03": "downtown, roundabout, tunnel, 5-lane junction, hills",
    "Town04": "highway loop with ramps, surrounded by mountains",
    "Town05": "grid city, multi-lane, intersections",
    "Town10HD_Opt": "urban, high detail, crosswalks, parked cars",
}
SIMPLE_TOWNS = ["Town01", "Town02", "Town03", "Town05"]

VEHICLE_BPS = {
    "car":        "vehicle.tesla.model3",
    "car2":       "vehicle.audi.tt",
    "suv":        "vehicle.nissan.patrol",
    "van":        "vehicle.volkswagen.t2",
    "ambulance":  "vehicle.ford.ambulance",
    "truck":      "vehicle.carlamotors.carlacola",
    "firetruck":  "vehicle.carlamotors.firetruck",
    "bus":        "vehicle.mitsubishi.fusorosa",
    "motorcycle": "vehicle.kawasaki.ninja",
    "scooter":    "vehicle.vespa.zx125",
    "bicycle":    "vehicle.bh.crossbike",
    "police":     "vehicle.dodge.charger_police",
}
PED_BPS = {
    "adult":  "walker.pedestrian.0001",
    "adult2": "walker.pedestrian.0004",
    "child":  "walker.pedestrian.0011",   # CARLA child model
    "child2": "walker.pedestrian.0012",
}
PROP_BPS = {
    "cone":        "static.prop.trafficcone01",
    "barrier":     "static.prop.streetbarrier",
    "barrel":      "static.prop.barrel",
    "box":         "static.prop.box03",
    "pallet":      "static.prop.plantpot06",     # nearest pallet-sized low prop
    "trashcan":    "static.prop.trashcan02",
    "bin":         "static.prop.bin",
    "container":   "static.prop.container",
    "debris":      "static.prop.dirtdebris01",
    "tire":        "static.prop.wheelbarrow",    # no tire prop; low wide object stand-in
    "shoppingcart":"static.prop.shoppingcart",
    "constructioncone": "static.prop.constructioncone",
    "warning":     "static.prop.warningconstruction",
    "bench":       "static.prop.bench01",
    "glass":       "static.prop.glasscontainer",
}

# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def grid(*lists: Sequence) -> List[tuple]:
    return list(itertools.product(*lists))


def take(rng: random.Random, items: List, n: int) -> List:
    """Deterministic shuffle then take n (cycling if grid is too small)."""
    items = list(items)
    rng.shuffle(items)
    if not items:
        return []
    out = []
    while len(out) < n:
        out.extend(items)
    return out[:n]


def odd(town: str, weather: str, road_type: str, speed_limit: float, **extra) -> dict:
    light, _ = WEATHERS[weather]
    d = {"town": town, "weather": weather, "light": light,
         "road_type": road_type, "speed_limit_mps": speed_limit}
    d.update(extra)
    return d


def weather_tags(weather: str) -> List[str]:
    light, tags = WEATHERS[weather]
    return [light] + tags


def mission(dest: dict, cruise: float = 8.0, origin: dict | None = None, start_at_s: float = 0.0) -> dict:
    return {
        "origin": origin or {"mode": "ego_current"},
        "destination": dest,
        "cruise_speed_mps": cruise,
        "start_at_s": start_at_s,
    }


def dest_ahead(distance_m: float) -> dict:
    return {"mode": "route_ahead", "distance_m": distance_m}


def dest_sp(index: int) -> dict:
    return {"mode": "spawn_point", "index": index}


def actor(name: str, type_: str, blueprint: str, spawn: dict, behavior: dict,
          trigger: dict | None = None) -> dict:
    a = {"name": name, "type": type_, "blueprint": blueprint, "spawn": spawn, "behavior": behavior}
    if trigger:
        a["trigger"] = trigger
    return a


def route_ahead(distance_m: float, lateral_m: float = 0.0, yaw_offset_deg: float = 0.0) -> dict:
    """Spawn on the planned route `distance_m` along it; lateral_m >0 = right of lane centre."""
    return {"mode": "route_ahead", "distance_m": distance_m, "lateral_m": lateral_m,
            "yaw_offset_deg": yaw_offset_deg}


def event(trigger: dict, action: str, **params) -> dict:
    e = {"trigger": trigger, "action": action}
    if params:
        e["params"] = params
    return e


def event_p(trigger: dict, action: str, params: dict) -> dict:
    e = {"trigger": trigger, "action": action}
    if params:
        e["params"] = params
    return e


def inject(trigger: dict, component: str, action: str, **extra) -> dict:
    p = {"component": component, "action": action}
    p.update(extra)
    return event_p(trigger, "inject", p)


def crit(metric: str, op: str, value) -> dict:
    return {"metric": metric, "op": op, "value": value}


NO_COLLISION = crit("collision_count", "==", 0)
COLLIDED = crit("collision_count", ">", 0)


def scenario(*, name, category, family, description, tags, caps, status, odd_, mission_,
             actors=None, events=None, expected=None, pass_c=None, fail_c=None,
             boundary, data=None, timeout=90) -> dict:
    return {
        "id": None,  # assigned by generator
        "name": name,
        "category": category,
        "family": family,
        "description": description,
        "tags": sorted(set(tags)),
        "required_capabilities": caps,
        "capability_status": status,
        "odd": odd_,
        "mission": mission_,
        "actors": actors or [],
        "events": events or [],
        "expected": expected or {},
        "pass_criteria": pass_c or [],
        "fail_criteria": fail_c if fail_c is not None else [COLLIDED],
        "safety_boundary": boundary,
        "data_to_collect": data or DEFAULT_DATA,
        "timeout_s": timeout,
    }


DEFAULT_DATA = [
    "pose", "speed", "behavior", "behavior_reason", "safety_state", "safety_reason",
    "command", "perception.objects", "perception.closest_distance", "mission.state",
    "warnings", "errors", "actor_positions", "collisions",
]

# Reaction-time budget: at cruise speed v, stopping from v with ~4 m/s^2 plus 0.5 s latency
def stop_budget_s(speed_mps: float, margin_s: float = 1.5) -> float:
    return round(0.5 + speed_mps / 4.0 + margin_s, 1)
