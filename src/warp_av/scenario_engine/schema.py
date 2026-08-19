"""
Scenario schema.

A scenario is a plain dict (serialised as YAML) with a fixed shape.  We validate
with hand-written checks rather than jsonschema so the catalog tooling has zero
extra dependencies beyond PyYAML.

Top-level keys
--------------
id                  "WAV-0001" .. unique
name                short human name
category            one of CATEGORIES
family              template key the scenario was generated from
description         what happens, in prose
tags                list of strings (weather, light, actor kind, ...)
required_capabilities   what the autonomy stack must be able to do
capability_status   implemented | partial | not_implemented   (honest gap flag)
odd                 operating design domain: town, weather, light, road, speed limit
mission             origin / destination / cruise speed
actors              list of actor specs (vehicle / pedestrian / prop)
events              list of timed or triggered injections (faults, operator actions)
expected            what *should* happen (behaviour, safety state, final mission state)
pass_criteria       machine-checkable list: {metric, op, value}
fail_criteria       any of these => FAIL regardless of pass criteria
safety_boundary     prose: what must never happen
data_to_collect     list of metric / log names to keep
timeout_s           runner gives up after this
"""

from __future__ import annotations

from typing import Any, Dict, List

CATEGORIES = [
    # the 7 required by the assignment
    "normal_mission",
    "vehicle_ahead",
    "pedestrian",
    "static_obstacle",
    "blocked_route",
    "component_failure",
    "emergency_stop",
    # road-readiness extensions
    "operator_action",
    "sensor_degradation",
    "localization_degradation",
    "odd_boundary",
    "traffic_control",
    "road_geometry",
    "timing_latency",
    "vulnerable_road_user",
    "compound",
    "edge_case",
    "endurance",
]

CAPABILITY_STATUS = ["implemented", "partial", "not_implemented"]

ACTOR_TYPES = ["vehicle", "pedestrian", "prop"]

# Behaviours the runner knows how to drive.
ACTOR_BEHAVIORS = [
    "stopped",            # spawn and do nothing
    "constant_speed",     # vehicle: move forward at speed_mps
    "brake_hard",         # vehicle: constant_speed then full brake at trigger
    "cut_in",             # vehicle: adjacent lane, constant speed, steer into ego lane at trigger
    "cut_out",            # vehicle: in ego lane ahead, steers out of lane at trigger (reveals what is behind)
    "autopilot",          # vehicle: CARLA traffic manager drives it
    "oncoming",           # vehicle: spawned facing ego in opposite lane, constant speed
    "reverse",            # vehicle: reverses toward ego at trigger
    "cross_road",         # pedestrian: walks across the lane at walk_speed_mps at trigger
    "walk_along",         # pedestrian: walks along the road edge (direction: toward|away)
    "stand",              # pedestrian: stands still
    "cross_and_stop",     # pedestrian: walks into lane then stops in it
    "dart_out",           # pedestrian: appears from occlusion at run speed
    "appear",             # prop: spawned only when trigger fires (sudden obstacle)
    "remove_after",       # prop: removed N seconds after trigger (clears blocked route)
]

# Events the runner can fire via the autonomy API.
EVENT_ACTIONS = [
    "estop",
    "estop_clear",
    "pause",
    "resume",
    "stop_mission",
    "start_mission",          # params: destination spec
    "change_destination",     # params: destination spec
    "set_speed_limit",        # params: cruise_speed_mps
    "inject",                 # params: component, action (disable|enable|freeze|stale|drop|latency|noise|low_confidence|crash)
    "set_weather",            # params: preset or overrides
    "wait",                   # no-op marker, useful in sequences
]

INJECT_COMPONENTS = [
    "perception", "localization", "camera", "lidar", "gnss", "imu",
    "controller", "planner", "vehicle_connection", "tick_latency", "api",
]
INJECT_ACTIONS = ["disable", "enable", "freeze", "stale", "drop", "latency",
                  "noise", "low_confidence", "crash", "nan_command", "clock_jump"]

TRIGGER_KEYS = ["at_s", "ego_within_m", "ego_speed_gt", "after_event", "on_behavior", "on_mission_state"]

METRIC_OPS = ["==", "!=", ">=", "<=", ">", "<", "in", "not_in", "contains"]

# Metrics the evaluator can compute from a trace.
METRICS = [
    "collision_count",
    "min_distance_to_actor_m",      # closest ego got to any scenario actor
    "final_mission_state",
    "mission_completed",
    "behaviors_seen",               # list
    "safety_states_seen",           # list
    "behavior_reasons_seen",        # list of reason strings
    "stopped_within_s_of_trigger",  # seconds from trigger to speed<0.2 (None if never)
    "time_to_first_brake_s",
    "max_speed_mps",
    "mean_speed_mps",
    "speed_at_trigger_mps",
    "max_abs_steer",
    "steer_oscillation_index",      # sign changes per second of steer cmd
    "route_deviation_max_m",        # max distance of ego from planned route
    "stop_duration_s",              # longest continuous stop
    "resumed_after_clear",          # bool: speed > 1 m/s after an estop_clear/resume/enable event
    "errors_seen",                  # list of error strings from safety
    "warnings_seen",
    "final_speed_mps",
    "elapsed_s",
    "safety_reaction_time_s",       # seconds from fault inject to safety_state != ok
    "tick_gap_max_s",               # largest gap between API state timestamps
    "out_of_route_time_s",
]

# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

class ScenarioValidationError(ValueError):
    pass


def _req(d: dict, key: str, typ, where: str):
    if key not in d:
        raise ScenarioValidationError(f"{where}: missing '{key}'")
    if typ is not None and not isinstance(d[key], typ):
        raise ScenarioValidationError(f"{where}: '{key}' must be {typ}, got {type(d[key]).__name__}")
    return d[key]


def validate_scenario(s: Dict[str, Any]) -> None:
    """Raise ScenarioValidationError if `s` is not a well-formed scenario."""
    sid = _req(s, "id", str, "scenario")
    where = f"scenario {sid}"
    _req(s, "name", str, where)
    cat = _req(s, "category", str, where)
    if cat not in CATEGORIES:
        raise ScenarioValidationError(f"{where}: unknown category '{cat}'")
    _req(s, "family", str, where)
    _req(s, "description", str, where)
    tags = _req(s, "tags", list, where)
    if not all(isinstance(t, str) for t in tags):
        raise ScenarioValidationError(f"{where}: tags must be strings")
    _req(s, "required_capabilities", list, where)
    cs = _req(s, "capability_status", str, where)
    if cs not in CAPABILITY_STATUS:
        raise ScenarioValidationError(f"{where}: bad capability_status '{cs}'")

    odd = _req(s, "odd", dict, where)
    for k in ("town", "weather", "light", "road_type", "speed_limit_mps"):
        _req(odd, k, None, f"{where}.odd")

    mission = _req(s, "mission", dict, where)
    _validate_location(_req(mission, "origin", dict, f"{where}.mission"), f"{where}.mission.origin")
    dest = _req(mission, "destination", dict, f"{where}.mission")
    _validate_location(dest, f"{where}.mission.destination")
    if "cruise_speed_mps" in mission and not isinstance(mission["cruise_speed_mps"], (int, float)):
        raise ScenarioValidationError(f"{where}: cruise_speed_mps must be numeric")

    actors = _req(s, "actors", list, where)
    names = set()
    for i, a in enumerate(actors):
        aw = f"{where}.actors[{i}]"
        nm = _req(a, "name", str, aw)
        if nm in names:
            raise ScenarioValidationError(f"{aw}: duplicate actor name {nm}")
        names.add(nm)
        t = _req(a, "type", str, aw)
        if t not in ACTOR_TYPES:
            raise ScenarioValidationError(f"{aw}: bad actor type {t}")
        _req(a, "blueprint", str, aw)
        sp = _req(a, "spawn", dict, aw)
        if sp.get("mode") not in ("route_ahead", "relative_to_ego", "absolute", "at_destination"):
            raise ScenarioValidationError(f"{aw}: bad spawn.mode {sp.get('mode')}")
        if sp["mode"] == "route_ahead" and "distance_m" not in sp:
            raise ScenarioValidationError(f"{aw}: route_ahead needs distance_m")
        beh = _req(a, "behavior", dict, aw)
        if beh.get("kind") not in ACTOR_BEHAVIORS:
            raise ScenarioValidationError(f"{aw}: bad behavior.kind {beh.get('kind')}")
        if "trigger" in a:
            _validate_trigger(a["trigger"], aw)

    events = _req(s, "events", list, where)
    for i, e in enumerate(events):
        ew = f"{where}.events[{i}]"
        _validate_trigger(_req(e, "trigger", dict, ew), ew)
        act = _req(e, "action", str, ew)
        if act not in EVENT_ACTIONS:
            raise ScenarioValidationError(f"{ew}: bad action {act}")
        params = e.get("params", {})
        if act == "inject":
            if params.get("component") not in INJECT_COMPONENTS:
                raise ScenarioValidationError(f"{ew}: bad inject component {params.get('component')}")
            if params.get("action") not in INJECT_ACTIONS:
                raise ScenarioValidationError(f"{ew}: bad inject action {params.get('action')}")
        if act in ("start_mission", "change_destination"):
            _validate_location(_req(params, "destination", dict, ew), f"{ew}.params.destination")

    _req(s, "expected", dict, where)
    for key in ("pass_criteria", "fail_criteria"):
        crits = _req(s, key, list, where)
        for i, c in enumerate(crits):
            cw = f"{where}.{key}[{i}]"
            m = _req(c, "metric", str, cw)
            if m not in METRICS:
                raise ScenarioValidationError(f"{cw}: unknown metric {m}")
            op = _req(c, "op", str, cw)
            if op not in METRIC_OPS:
                raise ScenarioValidationError(f"{cw}: bad op {op}")
            if "value" not in c:
                raise ScenarioValidationError(f"{cw}: missing value")
    _req(s, "safety_boundary", str, where)
    _req(s, "data_to_collect", list, where)
    to = _req(s, "timeout_s", (int, float), where)
    if to <= 0:
        raise ScenarioValidationError(f"{where}: timeout_s must be > 0")


def _validate_location(loc: dict, where: str):
    mode = loc.get("mode")
    if mode == "spawn_point":
        if not isinstance(loc.get("index"), int):
            raise ScenarioValidationError(f"{where}: spawn_point needs int index")
    elif mode == "ego_current":
        pass
    elif mode == "route_ahead":
        if not isinstance(loc.get("distance_m"), (int, float)):
            raise ScenarioValidationError(f"{where}: route_ahead needs distance_m")
    elif mode == "xy":
        for k in ("x", "y"):
            if not isinstance(loc.get(k), (int, float)):
                raise ScenarioValidationError(f"{where}: xy needs numeric {k}")
    elif mode == "off_map":
        pass
    else:
        raise ScenarioValidationError(f"{where}: bad location mode {mode}")


def _validate_trigger(tr: dict, where: str):
    if not isinstance(tr, dict) or not tr:
        raise ScenarioValidationError(f"{where}: trigger must be a non-empty dict")
    for k in tr:
        if k not in TRIGGER_KEYS:
            raise ScenarioValidationError(f"{where}: bad trigger key {k}")
