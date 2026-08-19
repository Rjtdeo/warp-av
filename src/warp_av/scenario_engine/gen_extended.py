"""
Families for the road-readiness extension categories:
operator_action, sensor_degradation, localization_degradation, odd_boundary, traffic_control,
road_geometry, timing_latency, vulnerable_road_user, compound, edge_case, endurance.
"""
from __future__ import annotations

import random
from typing import List

from .gen_common import (
    WEATHERS, DAY_WEATHERS, CLEAR_DAY, ADVERSE, SIMPLE_TOWNS,
    VEHICLE_BPS, PED_BPS, PROP_BPS,
    grid, take, odd, weather_tags, mission, dest_ahead, dest_sp, actor, route_ahead,
    event, inject, crit, NO_COLLISION, COLLIDED, scenario, stop_budget_s,
)

# =====================================================================
# OPERATOR ACTION
# =====================================================================

def oa_pause_resume(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2.0, 5.0, 8.0], [1, 5, 20], ["Town03", "Town05"]), n)
    for v, hold, town in combos:
        out.append(scenario(
            name=f"Pause at {v} m/s, resume after {hold} s ({town})",
            category="operator_action", family="oa_pause_resume",
            description=(f"Operator pauses while ego is at {v} m/s, holds {hold} s, resumes. Pause must disengage (brake to 0), mission state → paused; "
                         "resume re-engages and the mission completes. Route must NOT be re-planned from scratch on resume."),
            tags=["pause", "resume", f"hold_{hold}", "mission_state_machine", "day"],
            caps=["pause_resume", "engage_disengage", "mission_completion"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[event({"ego_speed_gt": v - 0.5}, "pause"), event({"after_event": 0, "at_s": hold}, "resume")],
            expected={"mission_state_sequence": ["executing", "paused", "executing", "completed"]},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True), crit("stop_duration_s", ">=", hold * 0.8)],
            boundary="No motion while paused.", timeout=90 + hold,
        ))
    return out


def oa_stop_mission(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([3.0, 8.0], ["idle_after", "new_mission_after"], ["Town03"]), n)
    for v, follow, town in combos:
        events = [event({"ego_speed_gt": v - 0.5}, "stop_mission")]
        if follow == "new_mission_after":
            events.append(event({"after_event": 0, "at_s": 5.0}, "start_mission", destination=dest_ahead(80)))
        out.append(scenario(
            name=f"Stop mission at {v} m/s → {follow.replace('_', ' ')}",
            category="operator_action", family="oa_stop_mission",
            description=f"Operator stops (cancels) the mission at {v} m/s. Vehicle brakes to zero, mission → cancelled, logged. " + ("A new mission 5 s later must work normally." if follow == "new_mission_after" else "Vehicle stays idle."),
            tags=["stop_mission", follow, "mission_state_machine", "day"],
            caps=["mission_cancel", "engage_disengage"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=events,
            expected={"final_mission_state": "cancelled" if follow == "idle_after" else "completed"},
            pass_c=[NO_COLLISION] + ([crit("final_mission_state", "==", "cancelled"), crit("final_speed_mps", "<=", 0.1)] if follow == "idle_after"
                                     else [crit("mission_completed", "==", True)]),
            boundary="Cancel brings vehicle to rest.", timeout=80,
        ))
    return out


def oa_change_destination(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["further", "nearer", "behind", "same"], [3.0, 8.0]), n)
    for kind, v in combos:
        newdest = {"further": dest_ahead(300), "nearer": dest_ahead(40), "behind": {"mode": "route_ahead", "distance_m": -60}, "same": dest_ahead(200)}[kind]
        out.append(scenario(
            name=f"Change destination mid-mission → {kind} (at {v} m/s)",
            category="operator_action", family="oa_change_destination",
            description=(f"While driving at {v} m/s the operator sets a new destination ({kind}). Current implementation re-plans from the current pose and starts a new mission record. "
                         + ("'Behind' requires a U-turn/loop — route planner must find a legal one, not reverse." if kind == "behind" else
                            "'Same' must be a no-op-ish re-plan without a stop." if kind == "same" else "")),
            tags=["change_destination", kind, "replanning", "day"],
            caps=["replanning", "mission_state_machine"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[event({"ego_speed_gt": v - 0.5}, "change_destination", destination=newdest)],
            expected={"final_mission_state": "completed"},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True)],
            boundary="No reversing; no stop longer than 3 s during re-plan.", timeout=180,
        ))
    return out


def oa_speed_limit_change(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([(8.0, 3.0), (8.0, 12.0), (5.0, 0.0), (3.0, 8.0), (8.0, 8.0)], ["Town04", "Town05"]), n)
    for (v0, v1), town in combos:
        out.append(scenario(
            name=f"Speed limit {v0} → {v1} m/s mid-mission ({town})",
            category="operator_action", family="oa_speed_limit_change",
            description=(f"Operator changes cruise/speed limit from {v0} to {v1} m/s while driving. "
                         + ("Zero means 'creep-stop': vehicle must stop but mission stays executing." if v1 == 0 else
                            "Speed tracking must converge within 5 s without overshoot > 1 m/s.")),
            tags=["speed_limit", f"{v0}_to_{v1}", "live_change", "day"],
            caps=["runtime_config", "speed_control"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "arterial", max(v0, v1) + 1),
            mission_=mission(dest_ahead(300), cruise=v0),
            events=[event({"ego_speed_gt": v0 - 0.5 if v0 > 0 else 0.0}, "set_speed_limit", cruise_speed_mps=v1)],
            expected={"max_speed_after_change": v1 + 1.0},
            pass_c=[NO_COLLISION, crit("max_speed_mps", "<=", max(v0, v1) + 1.0)] + ([crit("final_speed_mps", "<=", 0.3)] if v1 == 0 else []),
            boundary="Never exceed the newer limit by >1 m/s after 5 s.", timeout=120,
        ))
    return out


def oa_pause_with_hazard(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["pedestrian_arrives_while_paused", "obstacle_placed_while_paused", "resume_into_stopped_vehicle"], [5, 10]), n)
    for kind, hold in combos:
        if kind == "pedestrian_arrives_while_paused":
            actors = [actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(6, lateral_m=4.0), {"kind": "cross_and_stop", "speed_mps": 1.2, "dwell_s": 60}, trigger={"on_mission_state": "paused"})]
        elif kind == "obstacle_placed_while_paused":
            actors = [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(5), {"kind": "appear"}, trigger={"on_mission_state": "paused"})]
        else:
            actors = [actor("lead", "vehicle", VEHICLE_BPS["car"], route_ahead(4), {"kind": "appear"}, trigger={"on_mission_state": "paused"})]
        out.append(scenario(
            name=f"Pause; {kind.replace('_', ' ')}; resume after {hold} s",
            category="operator_action", family="oa_pause_with_hazard",
            description=f"Vehicle paused; a hazard appears 4–6 m ahead during the pause; operator resumes after {hold} s. First tick after resume must already see the hazard and refuse to move.",
            tags=["pause", "resume", "hazard_during_pause", "priority_ordering", "day"],
            caps=["pause_resume", "in_path_check", "stop_for_obstacle"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            actors=actors,
            events=[event({"ego_speed_gt": 5.0}, "pause"), event({"after_event": 0, "at_s": hold}, "resume")],
            expected={"behavior_after_resume": "stopped_*"},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No motion into a hazard after resume.", timeout=90,
        ))
    return out


def oa_back_to_back(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2, 3, 5], [0.0, 2.0]), n)
    for count, gap in combos:
        events = []
        for i in range(1, count):
            events.append(event({"on_mission_state": "completed", "at_s": gap}, "start_mission", destination=dest_ahead(80 + 20 * i)))
        out.append(scenario(
            name=f"{count} back-to-back missions, {gap} s gap",
            category="operator_action", family="oa_back_to_back",
            description=f"{count} missions issued sequentially, each {gap} s after the previous completes. Checks mission history grows, logger rolls files, behaviour resets (has_mission/mission_complete flags).",
            tags=["back_to_back", "mission_history", "logging", "day"],
            caps=["mission_state_machine", "logging_rollover"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(80), cruise=8.0),
            events=events,
            expected={"missions_completed": count},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True)],
            boundary="No stale state leaking between missions.", timeout=60 * count,
        ))
    return out


def oa_degenerate(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["to_current_location", "to_3m_ahead", "double_start", "resume_without_pause", "pause_when_idle"], ["Town03"]), n)
    for kind, town in combos:
        mission_ = mission(dest_ahead(200), cruise=8.0)
        events = []
        if kind == "to_current_location":
            mission_ = mission({"mode": "ego_current"}, cruise=8.0)
            desc = "Destination == current position. Must complete immediately (distance < 5 m) with no motion."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 0.3), crit("mission_completed", "==", True)]
        elif kind == "to_3m_ahead":
            mission_ = mission(dest_ahead(3), cruise=8.0)
            desc = "Destination 3 m ahead (inside the 5 m arrival threshold). Same as above — completes instantly."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 0.5), crit("mission_completed", "==", True)]
        elif kind == "double_start":
            events = [event({"at_s": 0.2}, "start_mission", destination=dest_ahead(200))]
            desc = "Start pressed twice within 200 ms. Second must be rejected or idempotent; only one mission record."
            pass_c = [NO_COLLISION, crit("mission_completed", "==", True)]
        elif kind == "resume_without_pause":
            events = [event({"ego_speed_gt": 5.0}, "resume")]
            desc = "Resume pressed while executing (never paused). Must be a no-op."
            pass_c = [NO_COLLISION, crit("mission_completed", "==", True)]
        else:
            mission_ = mission(dest_ahead(200), cruise=8.0, start_at_s=5.0)
            events = [event({"at_s": 1.0}, "pause"), event({"at_s": 2.0}, "resume")]
            desc = "Pause/resume pressed with no mission. Must not crash the API or engage autonomy."
            pass_c = [NO_COLLISION, crit("mission_completed", "==", True)]
        out.append(scenario(
            name=f"Degenerate operator input: {kind.replace('_', ' ')}",
            category="operator_action", family="oa_degenerate",
            description=desc,
            tags=["degenerate_input", kind, "api_robustness", "day"],
            caps=["api_input_validation", "mission_state_machine"],
            status="partial",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission_, events=events,
            expected={"no_crash": True},
            pass_c=pass_c,
            boundary="API never crashes; vehicle never moves without a valid mission.", timeout=80,
        ))
    return out


# =====================================================================
# SENSOR DEGRADATION  (camera / lidar / gnss / imu)
# =====================================================================

SENSORS = ["camera", "lidar", "gnss", "imu"]


def sd_dropout(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(SENSORS, [0.3, 1.0, 2.5, 6.0]), n)
    for s, dur in combos:
        out.append(scenario(
            name=f"{s} dropout {dur} s",
            category="sensor_degradation", family="sd_dropout",
            description=(f"{s} stream stops for {dur} s at speed then returns. Sensor-adapter health thresholds: camera/lidar/imu 1 s, gnss 2 s. "
                         "TODAY: sensor health is not consumed by the supervisor (perception/localization read ground truth) so the vehicle keeps driving. "
                         "TARGET: dropout longer than threshold → warning (gnss/imu, if odometry still OK) or intervention (camera/lidar)."),
            tags=["sensor", s, "dropout", f"dur_{dur}", "day"],
            caps=[f"{s}_health", "sensor_health_to_supervisor"],
            status="not_implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, s, "drop", duration_s=dur)],
            expected={"target_safety_state": "warning" if s in ("gnss", "imu") else "intervention", "current": "ok"},
            pass_c=[NO_COLLISION],
            boundary="Documented gap.", timeout=60,
        ))
    return out


def sd_noise_latency(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(SENSORS, ["noise", "latency"], ["low", "mid", "high"]), n)
    lv = {"low": 1, "mid": 2, "high": 3}
    for s, act, level in combos:
        params = {"noise": {"sigma_scale": lv[level]}, "latency": {"latency_s": [0.1, 0.3, 0.8][lv[level] - 1]}}[act]
        out.append(scenario(
            name=f"{s} {act} {level}",
            category="sensor_degradation", family="sd_noise_latency",
            description=(f"{s} data corrupted by {act} ({level}). With ground-truth perception/localization this is inert; "
                         "once real perception/fusion exists this family measures detection range/accuracy degradation and localization drift."),
            tags=["sensor", s, act, level, "future_perception", "day"],
            caps=[f"{s}_{act}_tolerance"],
            status="not_implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, s, act, **params)],
            expected={"target": "graceful degradation, metrics recorded"},
            pass_c=[NO_COLLISION],
            boundary="Documented gap.", timeout=60,
        ))
    return out


def sd_multi_and_flap(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([("camera", "lidar"), ("gnss", "imu"), ("camera", "gnss"), ("lidar", "imu")], ["both_drop", "flap"]), n)
    for (a, b), kind in combos:
        if kind == "both_drop":
            events = [inject({"ego_speed_gt": 5.0}, a, "drop", duration_s=3.0), inject({"ego_speed_gt": 5.0}, b, "drop", duration_s=3.0)]
            desc = f"{a} and {b} drop simultaneously for 3 s — likely a shared bus/power/network fault. Target: intervention + root-cause hint ('multiple sensors lost at once')."
        else:
            events = [inject({"at_s": 6 + i}, a, "drop", duration_s=0.4) for i in range(8)]
            desc = f"{a} drops 0.4 s every second (flaky connector). Each gap is under threshold; target is a rate-based health metric, not just age."
        out.append(scenario(
            name=f"{a}+{b} {kind.replace('_', ' ')}",
            category="sensor_degradation", family="sd_multi_and_flap",
            description=desc,
            tags=["sensor", a, b, kind, "day"],
            caps=["sensor_health_to_supervisor", "rate_based_health"],
            status="not_implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=events,
            expected={"target_safety_state": "intervention"},
            pass_c=[NO_COLLISION],
            boundary="Documented gap.", timeout=60,
        ))
    return out


# =====================================================================
# LOCALIZATION DEGRADATION
# =====================================================================

def ld_drift_jump(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["drift", "jump"], [0.5, 1.5, 4.0, 10.0], [True, False]), n)
    for kind, mag, with_conf in combos:
        out.append(scenario(
            name=f"Localization {kind} {mag} m ({'confidence reflects it' if with_conf else 'confidence stays 1.0'})",
            category="localization_degradation", family="ld_drift_jump",
            description=(f"Pose estimate {'drifts linearly by' if kind == 'drift' else 'jumps by'} {mag} m laterally at speed. "
                         + ("Confidence is lowered accordingly → supervisor should stop at <0.3." if with_conf else
                            "Confidence stays 1.0 — the dangerous case: vehicle steers toward a phantom route. Target: plausibility check (pose vs odometry/wheel speed) detects it.")),
            tags=["localization", kind, f"mag_{mag}", "confidence_truthful" if with_conf else "silent_error", "day"],
            caps=["localization_plausibility", "localization_confidence"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", 8.0),
            mission_=mission(dest_ahead(250), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, "localization", "noise", offset_m=mag, mode=kind, confidence=(max(0.0, 1.0 - mag / 5.0) if with_conf else 1.0))],
            expected={"route_deviation": f"~{mag} m unless caught"},
            pass_c=[NO_COLLISION, crit("route_deviation_max_m", "<=", max(2.0, mag + 1.0))],
            boundary="Never leave the road due to a localization error.", timeout=90,
        ))
    return out


def ld_stale_and_loss(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([0.5, 1.2, 3.0], ["urban_canyon", "tunnel", "open"], [True, False]), n)
    for dur, env, recovers in combos:
        town = {"urban_canyon": "Town10HD_Opt", "tunnel": "Town03", "open": "Town04"}[env]
        events = [inject({"ego_speed_gt": 5.0}, "localization", "stale", age_s=dur)]
        if recovers:
            events.append(inject({"after_event": 0, "at_s": dur + 2}, "localization", "enable"))
        out.append(scenario(
            name=f"Localization stale {dur} s in {env}{' then recovers' if recovers else ''}",
            category="localization_degradation", family="ld_stale_and_loss",
            description=(f"Localization stops updating for {dur} s ({env}; models GNSS loss without dead-reckoning). "
                         f"Threshold 1 s → {'intervention' if dur >= 1 else 'no trip'}. "
                         + ("After recovery: resume policy applies." if recovers else "Never recovers: hold stopped, operator must take over.")),
            tags=["localization", "stale", env, f"dur_{dur}", "day"],
            caps=["staleness_detection", "dead_reckoning"],
            status="implemented",
            odd_=odd(town, "ClearNoon", env, 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=events,
            expected={"safety_state": "intervention" if dur >= 1 else "ok"},
            pass_c=[NO_COLLISION] + ([crit("safety_states_seen", "contains", "intervention")] if dur >= 1 else [crit("safety_states_seen", "not_in", ["intervention"])]),
            boundary="No driving on a pose older than 1 s.", timeout=60,
        ))
    return out


def ld_confidence_ramp(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2.0, 10.0, 30.0], [0.25, 0.0]), n)
    for ramp, floor in combos:
        out.append(scenario(
            name=f"Confidence ramps to {floor} over {ramp} s",
            category="localization_degradation", family="ld_confidence_ramp",
            description=f"Confidence decays linearly from 1.0 to {floor} over {ramp} s (e.g. GNSS HDOP growing). Supervisor must intervene exactly when it crosses 0.3, and the slowdown before that is a target behaviour (degrade speed with confidence).",
            tags=["localization", "confidence", "ramp", "day"],
            caps=["localization_confidence", "graceful_degradation"],
            status="implemented",
            odd_=odd("Town04", "ClearNoon", "arterial", 8.0),
            mission_=mission(dest_ahead(300), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, "localization", "low_confidence", value=floor, ramp_s=ramp)],
            expected={"safety_state": "intervention"},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "intervention")],
            boundary="Intervention at confidence < 0.3.", timeout=60 + ramp,
        ))
    return out


# =====================================================================
# ODD BOUNDARY / GEOFENCE
# =====================================================================

def ob_geofence(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["destination_outside", "route_exits", "ego_pushed_outside"], [50, 150], ["Town03", "Town05"]), n)
    for kind, r, town in combos:
        out.append(scenario(
            name=f"Geofence radius {r} m: {kind.replace('_', ' ')} ({town})",
            category="odd_boundary", family="ob_geofence",
            description=(f"Allowed operating area = {r} m radius around start. "
                         + {"destination_outside": "Destination is outside → mission must be rejected at planning time.",
                            "route_exits": "Destination inside but the only route leaves the circle → planner must reject or supervisor must stop at the boundary.",
                            "ego_pushed_outside": "Ego is teleported just outside the fence mid-mission → supervisor intervention 'outside operating area'."}[kind]
                         + " Geofencing is listed in KNOWN_ISSUES as not implemented — these scenarios define the contract."),
            tags=["geofence", kind, f"radius_{r}", "day"],
            caps=["geofence", "odd_enforcement"],
            status="not_implemented",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0, geofence_radius_m=r),
            mission_=mission(dest_ahead(r + 100) if kind != "ego_pushed_outside" else dest_ahead(r - 20), cruise=8.0),
            events=([inject({"ego_speed_gt": 5.0}, "localization", "noise", offset_m=r, mode="jump")] if kind == "ego_pushed_outside" else []),
            expected={"target": "mission rejected / intervention with reason 'outside allowed operating area'"},
            pass_c=[NO_COLLISION],
            boundary="Never operate outside the fence.", timeout=90,
        ))
    return out


def ob_speed_and_weather(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["speed_limit_above_odd", "fog_beyond_odd", "night_no_lights", "weather_turns_mid_mission", "storm"], ["Town03", "Town10HD_Opt"]), n)
    for kind, town in combos:
        w = {"speed_limit_above_odd": "ClearNoon", "fog_beyond_odd": "ClearNoon", "night_no_lights": "ClearNight", "weather_turns_mid_mission": "ClearNoon", "storm": "HardRainNight"}[kind]
        events, cruise, extra = [], 8.0, {}
        if kind == "speed_limit_above_odd":
            cruise = 20.0
            desc = "Operator requests cruise 20 m/s but ODD max is 12 m/s. System must clamp to the ODD cap and warn, not obey."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 13.0)]
            extra = {"odd_max_speed_mps": 12.0}
        elif kind == "fog_beyond_odd":
            events = [event({"at_s": 0.5}, "set_weather", fog_density=90, fog_distance=5)]
            desc = "Dense fog (visibility ~10 m) set before start. ODD says min visibility 50 m. Target: refuse to start or cap speed to what the visibility allows. Ground truth is unaffected today."
            pass_c = [NO_COLLISION]
        elif kind == "night_no_lights":
            desc = "Night, ego lights off (CARLA light state). Target: lights on automatically at night; camera perception degrades without them."
            pass_c = [NO_COLLISION]
        elif kind == "weather_turns_mid_mission":
            events = [event({"ego_speed_gt": 5.0}, "set_weather", preset="HardRainNight")]
            desc = "Weather switches from clear to heavy rain at night mid-mission. Target: reduce speed automatically; raise warning 'ODD: weather degraded'."
            pass_c = [NO_COLLISION]
        else:
            desc = "Full storm from the start. Baseline record for the most adverse CARLA preset."
            pass_c = [NO_COLLISION]
        out.append(scenario(
            name=f"ODD boundary: {kind.replace('_', ' ')} ({town})",
            category="odd_boundary", family="ob_speed_and_weather",
            description=desc,
            tags=["odd", kind] + weather_tags(w),
            caps=["odd_enforcement", "speed_cap", "weather_awareness"],
            status="not_implemented",
            odd_=odd(town, w, "urban_2lane", 12.0, **extra),
            mission_=mission(dest_ahead(250), cruise=cruise),
            events=events,
            expected={"target": "ODD enforcement"},
            pass_c=pass_c,
            boundary="Never exceed ODD speed cap.", timeout=120,
        ))
    return out


# =====================================================================
# TRAFFIC CONTROL (all not_implemented — defines the contract)
# =====================================================================

def tc_traffic_light(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["red", "green", "yellow", "red_then_green", "green_then_red_on_approach", "all_red_flash"], ["Town03", "Town05", "Town10HD_Opt"]), n)
    for state, town in combos:
        out.append(scenario(
            name=f"Traffic light {state.replace('_', ' ')} ({town})",
            category="traffic_control", family="tc_traffic_light",
            description=(f"Next signalised junction on the route is forced to '{state}'. Current stack ignores traffic lights entirely (will run a red). "
                         "Target: stop at red/yellow-if-stoppable, proceed on green, handle light change on approach (dilemma zone). "
                         "Runner sets the light state via CARLA TrafficLight API; ego's reaction is recorded against 'stop line crossed while red'."),
            tags=["traffic_light", state, "intersection", "day"],
            caps=["traffic_light_detection", "stop_line_map", "signal_compliance"],
            status="not_implemented",
            odd_=odd(town, "ClearNoon", "signalised_intersection", 8.0),
            mission_=mission(dest_sp(5), cruise=7.0),
            events=[event({"at_s": 0.5}, "wait", traffic_light=state)],
            expected={"target": "no red-light violation", "current": "violation expected"},
            pass_c=[NO_COLLISION],
            boundary="Never enter junction on red.", timeout=120,
        ))
    return out


def tc_signs(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["stop_sign", "yield", "speed_limit_30", "no_entry", "one_way_wrong_way", "pedestrian_crossing_sign"], ["Town01", "Town02"]), n)
    for sign, town in combos:
        out.append(scenario(
            name=f"Sign: {sign.replace('_', ' ')} ({town})",
            category="traffic_control", family="tc_signs",
            description=f"Route passes a {sign.replace('_', ' ')}. Stack has no sign detection or map-based regulatory elements. Target behaviour documented; today the scenario records whether ego stops/slows at all.",
            tags=["sign", sign, "day"],
            caps=["sign_detection", "map_regulatory_elements"],
            status="not_implemented",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_sp(7), cruise=7.0),
            expected={"target": {"stop_sign": "full stop ≥1 s", "yield": "slow, yield if traffic", "speed_limit_30": "≤8.3 m/s", "no_entry": "route must avoid", "one_way_wrong_way": "route must avoid", "pedestrian_crossing_sign": "slow"}[sign]},
            pass_c=[NO_COLLISION],
            boundary="Regulatory compliance (target).", timeout=120,
        ))
    return out


# =====================================================================
# ROAD GEOMETRY
# =====================================================================

def rg_geometry(rng: random.Random, n: int) -> List[dict]:
    out = []
    feats = [("sharp_turn", "Town01", 5.0), ("sharp_turn", "Town02", 8.0), ("roundabout", "Town03", 6.0), ("hill", "Town03", 8.0),
             ("tunnel", "Town03", 8.0), ("t_junction_left", "Town01", 6.0), ("t_junction_right", "Town01", 6.0), ("unprotected_left", "Town05", 6.0),
             ("lane_merge", "Town04", 10.0), ("highway_ramp", "Town04", 12.0), ("narrow_street", "Town10HD_Opt", 5.0), ("u_turn", "Town03", 5.0),
             ("multilane_keep_lane", "Town05", 10.0), ("long_straight", "Town04", 12.0), ("s_curve", "Town02", 6.0)]
    combos = take(rng, grid(feats, CLEAR_DAY), n)
    for (feat, town, v), w in combos:
        out.append(scenario(
            name=f"Geometry: {feat.replace('_', ' ')} at {v} m/s ({town})",
            category="road_geometry", family="rg_geometry",
            description=(f"Route selected to include a {feat.replace('_', ' ')} in {town}; cruise {v} m/s. "
                         "Pure-pursuit with a 5 m lookahead and gain 1.5: measure cross-track error, steer oscillation and corner-cutting. "
                         "Speed is not yet curvature-aware — taking a sharp turn at 8 m/s will overshoot; that is the data we want."),
            tags=["geometry", feat, f"cruise_{v}", "control_tuning"] + weather_tags(w),
            caps=["path_following", "curvature_aware_speed", "lateral_stability"],
            status="partial",
            odd_=odd(town, w, feat, v + 1),
            mission_=mission({"mode": "route_ahead", "distance_m": 250, "feature": feat}, cruise=v),
            expected={"route_deviation_max_m": 1.5},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True), crit("route_deviation_max_m", "<=", 2.0), crit("steer_oscillation_index", "<=", 4.0)],
            boundary="Stay in lane through the feature.", timeout=150,
        ))
    return out


# =====================================================================
# TIMING / LATENCY (API + loop)
# =====================================================================

def tl_timing(rng: random.Random, n: int) -> List[dict]:
    out = []
    kinds = [("api_poll_storm", "Console/API polled at 50 Hz by 5 clients: Flask thread contends with the loop; measure tick_gap_max."),
             ("api_unresponsive", "API thread blocked 5 s; autonomy loop must keep running and the supervisor must not trip (API is not safety-critical)."),
             ("command_age", "Vehicle adapter receives a command older than max_command_age (0.5 s) — target: adapter rejects stale commands & brakes."),
             ("tick_jitter", "Tick period randomly 50–300 ms for 20 s. Measure control quality; staleness must not trip under 1 s."),
             ("slow_perception", "Perception update takes 400 ms per tick: loop rate drops to ~2 Hz. Under threshold but reaction distance at 8 m/s grows by 3.2 m."),
             ("logger_disk_full", "Logger write fails (disk full). Must not crash the loop; should surface a warning."),
             ("carla_tick_stall", "CARLA server stalls 2 s (e.g. map streaming). vehicle_alive check + staleness should catch it.")]
    combos = take(rng, grid(kinds, [8.0, 5.0]), n)
    for (kind, desc), v in combos:
        comp = {"api_poll_storm": "api", "api_unresponsive": "api", "command_age": "controller", "tick_jitter": "tick_latency",
                "slow_perception": "perception", "logger_disk_full": "api", "carla_tick_stall": "vehicle_connection"}[kind]
        act = {"api_poll_storm": "latency", "api_unresponsive": "freeze", "command_age": "stale", "tick_jitter": "latency",
               "slow_perception": "latency", "logger_disk_full": "crash", "carla_tick_stall": "freeze"}[kind]
        out.append(scenario(
            name=f"Timing: {kind.replace('_', ' ')} at {v} m/s",
            category="timing_latency", family="tl_timing",
            description=desc,
            tags=["timing", kind, "day"],
            caps=["watchdog", "latency_tolerance", "thread_isolation"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", v + 1),
            mission_=mission(dest_ahead(250), cruise=v),
            events=[inject({"ego_speed_gt": v - 1}, comp, act, kind=kind)],
            expected={"loop_keeps_running": True},
            pass_c=[NO_COLLISION, crit("tick_gap_max_s", "<=", 2.5)],
            boundary="Non-safety threads must never stall the autonomy loop.", timeout=90,
        ))
    return out


# =====================================================================
# VULNERABLE ROAD USERS (cyclist, motorcycle, scooter, wheelchair, animal stand-in)
# =====================================================================

def vru(rng: random.Random, n: int) -> List[dict]:
    out = []
    kinds = [("bicycle", "cyclist"), ("motorcycle", "motorcyclist"), ("scooter", "scooter rider")]
    behaviours = [("ahead_slow", "rides ahead in lane at 4 m/s"), ("filtering", "filters past from behind on the left at 8 m/s"),
                  ("cross", "crosses the lane at 5 m/s"), ("wobble", "rides at lane edge (1.4 m) and weaves ±0.6 m")]
    combos = take(rng, grid(kinds, behaviours, [25, 40]), n)
    for (bp, who), (beh, bdesc), d in combos:
        if beh == "ahead_slow":
            a = actor("vru", "vehicle", VEHICLE_BPS[bp], route_ahead(d), {"kind": "constant_speed", "speed_mps": 4.0})
        elif beh == "filtering":
            a = actor("vru", "vehicle", VEHICLE_BPS[bp], route_ahead(-15, lateral_m=-1.8), {"kind": "constant_speed", "speed_mps": 8.0}, trigger={"ego_speed_gt": 4.0})
        elif beh == "cross":
            a = actor("vru", "vehicle", VEHICLE_BPS[bp], route_ahead(d, lateral_m=8.0, yaw_offset_deg=-90), {"kind": "constant_speed", "speed_mps": 5.0}, trigger={"ego_within_m": d + 15})
        else:
            a = actor("vru", "vehicle", VEHICLE_BPS[bp], route_ahead(d, lateral_m=1.4), {"kind": "constant_speed", "speed_mps": 4.0, "weave_m": 0.6})
        out.append(scenario(
            name=f"{who.capitalize()} {beh.replace('_', ' ')} ({d} m)",
            category="vulnerable_road_user", family="vru",
            description=(f"A {who} {bdesc}. Classified as VEHICLE by the ground-truth adapter (blueprint vehicle.*) — so it gets vehicle treatment, not the always-stop pedestrian rule. "
                         "Target: VRU class with larger clearance (1.5 m) and no overtaking in-lane."),
            tags=["vru", bp, beh, "day"],
            caps=["vru_classification", "lateral_clearance", "adaptive_following"],
            status="partial",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 150), cruise=8.0),
            actors=[a],
            expected={"no_collision": True, "min_clearance": 1.5},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.5)],
            boundary="≥1.5 m from any VRU.", timeout=90,
        ))
    return out


def vru_special(rng: random.Random, n: int) -> List[dict]:
    out = []
    kinds = [("wheelchair_slow_cross", "pedestrian", PED_BPS["adult"], {"kind": "cross_road", "speed_mps": 0.5, "direction": "toward_lane"}, 5.0, "A very slow (0.5 m/s) crossing — wheelchair user proxy. Long dwell in the lane; ego must wait 14+ s without creeping."),
             ("animal_small", "prop", PROP_BPS["bin"], {"kind": "appear"}, 0.0, "Small animal proxy (CARLA has no animals): a bin-sized object appears 12 m ahead. Low-height small object detection."),
             ("ped_with_stroller", "pedestrian", PED_BPS["adult2"], {"kind": "cross_road", "speed_mps": 0.9, "direction": "toward_lane"}, 5.0, "Slow pedestrian crossing proxy for stroller; long lane occupancy."),
             ("skateboarder", "pedestrian", PED_BPS["adult"], {"kind": "dart_out", "speed_mps": 4.5}, 5.0, "Fast pedestrian (4.5 m/s) — skateboard/e-scooter rider on foot model. Highest closing speed of the pedestrian class.")]
    combos = take(rng, grid(kinds, [18, 30]), n)
    for (kind, typ, bp, beh, lat, desc), d in combos:
        out.append(scenario(
            name=f"VRU special: {kind.replace('_', ' ')} ({d} m)",
            category="vulnerable_road_user", family="vru_special",
            description=desc,
            tags=["vru", kind, "day"],
            caps=["pedestrian_detection", "small_object_detection", "resume_when_clear"],
            status="partial" if kind == "animal_small" else "implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=6.0),
            actors=[actor("vru", typ, bp, route_ahead(d if kind != "animal_small" else 12, lateral_m=lat), beh, trigger={"ego_within_m": d + 8})],
            expected={"no_collision": True},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact.", timeout=90,
        ))
    return out


# =====================================================================
# COMPOUND
# =====================================================================

def compound(rng: random.Random, n: int) -> List[dict]:
    out = []
    recipes = [
        ("ped_rain_night", "HardRainNight", [actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(25, lateral_m=5.0), {"kind": "cross_road", "speed_mps": 1.3, "direction": "toward_lane"}, trigger={"ego_within_m": 35})], [],
         "Pedestrian crossing in heavy rain at night.", ["stopped_pedestrian"]),
        ("vehicle_ahead_then_perception_fail", "ClearNoon", [actor("lead", "vehicle", VEHICLE_BPS["car"], route_ahead(30), {"kind": "stopped"})], [inject({"ego_within_m": 18}, "perception", "disable")],
         "Stopped car ahead; perception dies at 18 m. Supervisor must stop the vehicle blind, in time.", ["stopped_safety"]),
        ("obstacle_plus_localization_degrade", "ClearNoon", [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(35), {"kind": "stopped"})], [inject({"ego_speed_gt": 5.0}, "localization", "low_confidence", value=0.2, ramp_s=3.0)],
         "Barrier ahead while localization confidence decays. Two independent reasons to stop; reason string should name the first that fires.", ["stopped_obstacle", "stopped_safety"]),
        ("cut_in_then_brake", "WetNoon", [actor("cutter", "vehicle", VEHICLE_BPS["car"], route_ahead(12, lateral_m=-3.5), {"kind": "cut_in", "speed_mps": 8.0, "side": "left", "then_brake": True}, trigger={"ego_within_m": 30})], [],
         "Car cuts in from the left at 12 m then brakes hard. Classic rear-end setup on a wet road.", ["stopped_vehicle"]),
        ("jam_plus_estop_plus_clear", "ClearNoon", [actor(f"jam{i}", "vehicle", VEHICLE_BPS["car"], route_ahead(25 + 7 * i), {"kind": "stopped"}) for i in range(3)], [event({"on_behavior": "stopped_vehicle"}, "estop"), event({"after_event": 0, "at_s": 3}, "estop_clear"), event({"after_event": 1, "at_s": 1}, "resume")],
         "Stopped behind a jam; e-stop; clear; resume. Resume must not creep into the jam.", ["stopped_vehicle"]),
        ("fog_plus_pedestrian_plus_slow_lead", "ClearNoon", [actor("lead", "vehicle", VEHICLE_BPS["van"], route_ahead(20), {"kind": "constant_speed", "speed_mps": 3.0}), actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(60, lateral_m=5.0), {"kind": "cross_road", "speed_mps": 1.2, "direction": "toward_lane"}, trigger={"ego_within_m": 70})], [event({"at_s": 0.5}, "set_weather", fog_density=70)],
         "Fog, slow van ahead, pedestrian crosses in front of the van. Ego must handle lead + VRU together.", ["stopped_pedestrian", "stopped_vehicle"]),
        ("ped_plus_change_destination", "ClearNoon", [actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(25, lateral_m=5.0), {"kind": "cross_and_stop", "speed_mps": 1.2, "dwell_s": 10}, trigger={"ego_within_m": 35})], [event({"on_behavior": "stopped_pedestrian"}, "change_destination", destination=dest_ahead(60))],
         "Stopped for a pedestrian; operator changes destination. New plan must not cause motion while the pedestrian is still in the lane.", ["stopped_pedestrian"]),
        ("double_fault_with_obstacle", "ClearNoon", [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(40), {"kind": "stopped"})], [inject({"ego_speed_gt": 5.0}, "gnss", "drop", duration_s=5), inject({"ego_speed_gt": 5.0}, "camera", "drop", duration_s=5)],
         "GNSS + camera drop while approaching a barrier. Ground truth still stops for the barrier; records the gap in sensor-health wiring.", ["stopped_obstacle"]),
        ("night_cyclist_plus_oncoming", "ClearNight", [actor("cyc", "vehicle", VEHICLE_BPS["bicycle"], route_ahead(25, lateral_m=1.2), {"kind": "constant_speed", "speed_mps": 4.0}), actor("onc", "vehicle", VEHICLE_BPS["truck"], route_ahead(60, lateral_m=-3.5, yaw_offset_deg=180), {"kind": "oncoming", "speed_mps": 8.0})],
         [], "Cyclist at lane edge with an oncoming truck: no room to pass. Must follow the cyclist, not squeeze.", []),
        ("construction_plus_pause", "ClearNoon", [actor(f"cone{i}", "prop", PROP_BPS["constructioncone"], route_ahead(30 + 3 * i, lateral_m=1.2), {"kind": "stopped"}) for i in range(4)], [event({"ego_within_m": 20}, "pause"), event({"after_event": 0, "at_s": 5}, "resume")],
         "Paused right before a cone taper; resumed. Cones at the path edge.", []),
    ]
    combos = take(rng, grid(recipes, ["Town10HD_Opt", "Town03", "Town05"]), n)
    for (key, w, actors, events, desc, behs), town in combos:
        out.append(scenario(
            name=f"Compound: {key.replace('_', ' ')} ({town})",
            category="compound", family="compound",
            description=desc + " Compound scenarios exist to catch priority-ordering bugs between independent subsystems.",
            tags=["compound", key, "priority_ordering"] + weather_tags(w),
            caps=["priority_ordering", "multi_hazard"],
            status="partial",
            odd_=odd(town, w, "urban_mixed", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            actors=[dict(a) for a in actors], events=[dict(e) for e in events],
            expected={"behaviors_any": behs},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact with anything.", timeout=120,
        ))
    return out


# =====================================================================
# EDGE CASES
# =====================================================================

def edge(rng: random.Random, n: int) -> List[dict]:
    out = []
    cases = [
        ("actor_spawned_overlapping_ego", [actor("obs", "prop", PROP_BPS["box"], route_ahead(0.5), {"kind": "stopped"})], [], "A prop is spawned 0.5 m ahead (inside ego's bumper zone) — distance < danger distance from tick 1. No motion at all.", [crit("max_speed_mps", "<=", 0.2)]),
        ("actor_deleted_mid_stop", [actor("lead", "vehicle", VEHICLE_BPS["car"], route_ahead(30), {"kind": "remove_after", "delay_s": 4}, trigger={"on_behavior": "stopped_vehicle"})], [], "Lead vehicle is destroyed (despawned) while ego is stopped for it. Perception must drop it cleanly; ego resumes.", [crit("mission_completed", "==", True)]),
        ("object_behind_ego", [actor("tail", "vehicle", VEHICLE_BPS["car"], route_ahead(-8), {"kind": "stopped"})], [], "A car sits 8 m BEHIND ego. obj.x < 0 → must be ignored by the in-path check; mission completes.", [crit("mission_completed", "==", True), crit("behaviors_seen", "not_in", ["stopped_vehicle"])]),
        ("object_exactly_at_path_edge", [actor("obs", "prop", PROP_BPS["cone"], route_ahead(25, lateral_m=1.75), {"kind": "stopped"})], [], "Cone exactly at |y| = path_width/2 = 1.75 m. Boundary condition of `abs(obj.y) < path_width/2` — strict inequality means 'not blocked'. Records which way it falls.", []),
        ("object_exactly_at_danger_distance", [actor("obs", "prop", PROP_BPS["cone"], route_ahead(5.0), {"kind": "stopped"})], [], "Cone at exactly 5.0 m = danger_distance at start. Strict `<` means not blocked → ego would creep. Boundary test.", [crit("min_distance_to_actor_m", ">=", 1.0)]),
        ("object_at_detection_range", [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(49.5), {"kind": "stopped"})], [], "Barrier at 49.5 m ≈ detection_range 50 m. Should be detected on first tick; checks range boundary.", [crit("behaviors_seen", "contains", "stopped_obstacle")]),
        ("two_objects_same_distance", [actor("a", "prop", PROP_BPS["cone"], route_ahead(25, lateral_m=-0.8), {"kind": "stopped"}), actor("b", "pedestrian", PED_BPS["adult"], route_ahead(25, lateral_m=0.8), {"kind": "stand"})], [], "A cone and a pedestrian at identical range. closest_type must resolve to PEDESTRIAN (higher priority) — today it is whichever is iterated first (vehicles→walkers→props): pedestrian wins by iteration order, which is luck, not design.", [crit("behaviors_seen", "contains", "stopped_pedestrian")]),
        ("mission_while_moving_manual", [], [event({"at_s": 0.5}, "start_mission", destination=dest_ahead(150))], "Mission started while the vehicle still rolls from a previous disengage (non-zero speed at engage). Controller PID integral must be reset.", []),
        ("very_long_route", [], [], "Destination ~2 km away (spawn point far side of town). Route has 1000+ waypoints; get_next_waypoint is O(n) per tick — watch tick time.", [crit("tick_gap_max_s", "<=", 0.5)]),
        ("zero_cruise_speed", [], [], "Cruise speed configured 0. Behaviour asks for 0 → controller brakes → mission never completes. Should be rejected at config time.", [crit("max_speed_mps", "<=", 0.3)]),
        ("negative_destination_distance", [], [], "Destination parameter negative / malformed JSON to /api/mission/start. API must 400, not 500, and no mission record.", []),
        ("rapid_pause_resume_spam", [], [event({"at_s": 5 + 0.3 * i}, ("pause" if i % 2 == 0 else "resume")) for i in range(16)], "Pause/resume toggled 16 times in 5 s. State machine must settle; no exception in the loop.", [crit("mission_completed", "==", True)]),
        ("estop_during_route_planning", [], [event({"at_s": 0.05}, "estop")], "E-stop fires in the same tick the route is being planned. Planning should finish/abort cleanly; no motion.", [crit("final_speed_mps", "<=", 0.1)]),
        ("spawn_point_index_out_of_range", [], [], "Destination spawn point index 999 (API returns only 20). Must be rejected gracefully.", []),
        ("all_props_at_once", [actor(f"p{i}", "prop", PROP_BPS[k], route_ahead(30 + 2 * i, lateral_m=(i % 3 - 1) * 1.2), {"kind": "stopped"}) for i, k in enumerate(["cone", "barrel", "box", "trashcan", "bin", "bench", "shoppingcart", "debris"])], [], "Eight props in 16 m: perception object list is large; closest-selection must still pick the nearest in-path one.", [crit("behaviors_seen", "contains", "stopped_obstacle")]),
    ]
    combos = take(rng, grid(cases, ["Town03", "Town05"]), n)
    for (key, actors, events, desc, extra_pass), town in combos:
        mission_ = mission(dest_ahead(150), cruise=8.0)
        if key == "very_long_route":
            mission_ = mission(dest_sp(19), cruise=10.0)
        if key == "zero_cruise_speed":
            mission_ = mission(dest_ahead(150), cruise=0.0)
        if key == "negative_destination_distance":
            mission_ = mission({"mode": "route_ahead", "distance_m": -1e9}, cruise=8.0)
        if key == "spawn_point_index_out_of_range":
            mission_ = mission(dest_sp(999), cruise=8.0)
        if key == "mission_while_moving_manual":
            mission_ = mission(dest_ahead(150), cruise=8.0, start_at_s=4.0)
        out.append(scenario(
            name=f"Edge: {key.replace('_', ' ')} ({town})",
            category="edge_case", family="edge",
            description=desc,
            tags=["edge", key, "boundary_condition", "day"],
            caps=["robustness", "boundary_conditions"],
            status="partial",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission_, actors=[dict(a) for a in actors], events=[dict(e) for e in events],
            expected={"no_crash": True},
            pass_c=[NO_COLLISION] + extra_pass,
            boundary="No exception escapes the loop; no contact.", timeout=120 if key != "very_long_route" else 600,
        ))
    return out


# =====================================================================
# ENDURANCE
# =====================================================================

def endurance(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([10, 20, 30], [0, 15, 40], ["Town03", "Town05", "Town10HD_Opt"]), n)
    for minutes, traffic, town in combos:
        actors = [actor(f"tm_{i}", "vehicle", VEHICLE_BPS["car" if i % 2 else "suv"], {"mode": "absolute", "spawn_point_index": 3 + i}, {"kind": "autopilot"}) for i in range(traffic)]
        actors += [actor(f"w_{i}", "pedestrian", PED_BPS["adult" if i % 2 else "adult2"], {"mode": "absolute", "spawn_point_index": 30 + i, "sidewalk": True}, {"kind": "walk_along", "speed_mps": 1.2, "direction": "random"}) for i in range(traffic // 2)]
        events = [event({"on_mission_state": "completed", "at_s": 1.0}, "start_mission", destination=dest_sp(3 + k % 15)) for k in range(minutes)]
        out.append(scenario(
            name=f"Endurance {minutes} min, {traffic} traffic vehicles ({town})",
            category="endurance", family="endurance",
            description=(f"Chain missions for ~{minutes} minutes among {traffic} autopilot vehicles and {traffic // 2} wandering pedestrians. "
                         "Looks for: memory growth, log file size, tick-time creep, mission-history growth, any unhandled exception, disengagement count."),
            tags=["endurance", f"min_{minutes}", f"traffic_{traffic}", "soak", "day"],
            caps=["stability", "logging_rollover", "multi_object_tracking"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "urban_mixed", 10.0),
            mission_=mission(dest_sp(2), cruise=8.0),
            actors=actors, events=events,
            expected={"no_crash": True, "disengagements": 0},
            pass_c=[NO_COLLISION, crit("tick_gap_max_s", "<=", 1.0), crit("safety_states_seen", "not_in", ["intervention"])],
            boundary="No collision, no unhandled exception over the soak.", timeout=minutes * 60 + 120,
        ))
    return out
