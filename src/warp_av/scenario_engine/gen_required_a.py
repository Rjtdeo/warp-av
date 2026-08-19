"""
Families for: normal_mission, vehicle_ahead, pedestrian.

Each family function takes (rng, n) and returns exactly n scenario dicts (ids unassigned).
"""
from __future__ import annotations

import random
from typing import List

from .gen_common import (
    WEATHERS, DAY_WEATHERS, CLEAR_DAY, ADVERSE, SIMPLE_TOWNS, TOWNS,
    VEHICLE_BPS, PED_BPS, PROP_BPS,
    grid, take, odd, weather_tags, mission, dest_ahead, dest_sp, actor, route_ahead,
    event, inject, crit, NO_COLLISION, COLLIDED, scenario, stop_budget_s,
)

# =====================================================================
# NORMAL MISSION
# =====================================================================

def nm_basic(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([80, 150, 250, 400], SIMPLE_TOWNS, DAY_WEATHERS, [5.0, 8.0]), n)
    for dist, town, w, v in combos:
        out.append(scenario(
            name=f"Normal mission {dist} m, {town}, {w}, cruise {v} m/s",
            category="normal_mission", family="nm_basic",
            description=(f"Ego receives a destination {dist} m along the road network in {town} "
                         f"under {w}. No actors. Must follow the route, hold ~{v} m/s, slow near the "
                         "destination and complete the mission."),
            tags=["baseline", f"dist_{dist}"] + weather_tags(w),
            caps=["route_planning", "path_following", "speed_control", "mission_completion"],
            status="implemented",
            odd_=odd(town, w, "urban_2lane", v + 2),
            mission_=mission(dest_ahead(dist), cruise=v),
            expected={"final_mission_state": "completed",
                      "behaviors": ["following_route", "approaching_destination", "mission_complete"]},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True),
                    crit("max_speed_mps", "<=", v + 1.5),
                    crit("route_deviation_max_m", "<=", 2.0),
                    crit("steer_oscillation_index", "<=", 3.0)],
            boundary="Never leave the drivable lane; never exceed speed limit by >1.5 m/s.",
            timeout=60 + dist / 2,
        ))
    return out


def nm_spawn_pairs(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(range(1, 20), SIMPLE_TOWNS, CLEAR_DAY), n)
    for idx, town, w in combos:
        out.append(scenario(
            name=f"Normal mission to spawn point {idx} ({town})",
            category="normal_mission", family="nm_spawn_pairs",
            description=f"Mission from ego spawn to map spawn point #{idx} in {town}. Exercises longer, multi-turn routes.",
            tags=["baseline", "multi_turn"] + weather_tags(w),
            caps=["route_planning", "path_following", "turning", "mission_completion"],
            status="implemented",
            odd_=odd(town, w, "urban_mixed", 10.0),
            mission_=mission(dest_sp(idx), cruise=8.0),
            expected={"final_mission_state": "completed"},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True),
                    crit("route_deviation_max_m", "<=", 2.5)],
            boundary="Stay on road; complete without safety intervention.",
            timeout=240,
        ))
    return out


def nm_adverse_weather(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(list(WEATHERS.keys()), ["Town03", "Town10HD_Opt"]), n)
    for w, town in combos:
        out.append(scenario(
            name=f"Normal mission in {w} ({town})",
            category="normal_mission", family="nm_adverse_weather",
            description=(f"200 m mission in {w}. Ground-truth perception is weather-invariant today; "
                         "this family becomes meaningful once camera/lidar perception is real. "
                         "Records baseline control behaviour on wet/low-visibility surfaces."),
            tags=["weather_sweep"] + weather_tags(w),
            caps=["route_planning", "path_following", "weather_robustness"],
            status="partial",
            odd_=odd(town, w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=6.0 if w in ADVERSE else 8.0),
            expected={"final_mission_state": "completed"},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True),
                    crit("route_deviation_max_m", "<=", 2.5)],
            boundary="No lane departure even on wet road.",
            timeout=150,
        ))
    return out


def nm_speed_sweep(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0], ["Town04", "Town05"]), n)
    for v, town in combos:
        out.append(scenario(
            name=f"Speed sweep cruise {v} m/s ({town})",
            category="normal_mission", family="nm_speed_sweep",
            description=(f"300 m straight-ish mission at cruise {v} m/s. Checks speed tracking error, "
                         "overshoot and steering stability as speed rises (pure-pursuit oscillation is a known issue)."),
            tags=["speed_sweep", f"cruise_{v}", "control_tuning", "day"],
            caps=["speed_control", "lateral_stability"],
            status="implemented" if v <= 10 else "partial",
            odd_=odd(town, "ClearNoon", "arterial", v + 1),
            mission_=mission(dest_ahead(300), cruise=v),
            expected={"final_mission_state": "completed", "max_speed_mps": v + 1.0},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True),
                    crit("max_speed_mps", "<=", v + 1.0),
                    crit("steer_oscillation_index", "<=", 2.0 if v <= 8 else 4.0),
                    crit("route_deviation_max_m", "<=", 1.5)],
            boundary="Speed overshoot <1 m/s; no oscillatory steering growth.",
            timeout=120,
        ))
    return out


# =====================================================================
# VEHICLE AHEAD
# =====================================================================

def _stop_for_vehicle_pass(speed: float, reaction_extra: float = 0.0):
    return [NO_COLLISION,
            crit("behaviors_seen", "contains", "stopped_vehicle"),
            crit("min_distance_to_actor_m", ">=", 1.5),
            crit("stopped_within_s_of_trigger", "<=", stop_budget_s(speed) + reaction_extra)]


def va_stopped_lead(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([12, 20, 30, 45, 60, 80], ["car", "van", "truck"], DAY_WEATHERS + ["ClearNight"], [5.0, 8.0]), n)
    for gap, kind, w, v in combos:
        out.append(scenario(
            name=f"Stopped {kind} {gap} m ahead, ego {v} m/s, {w}",
            category="vehicle_ahead", family="va_stopped_lead",
            description=(f"A stationary {kind} occupies the ego lane {gap} m along the route when the mission starts. "
                         f"Ego cruises at {v} m/s, must detect, slow (<15 m) and stop (<5 m) with a reason string naming the vehicle."),
            tags=["stationary_lead", kind, f"gap_{gap}"] + weather_tags(w),
            caps=["vehicle_detection", "in_path_check", "stop_for_vehicle"],
            status="implemented",
            odd_=odd("Town03", w, "urban_2lane", v + 2),
            mission_=mission(dest_ahead(gap + 120), cruise=v),
            actors=[actor("lead", "vehicle", VEHICLE_BPS[kind], route_ahead(gap), {"kind": "stopped"})],
            expected={"behavior": "stopped_vehicle", "reason_contains": "VEHICLE"},
            pass_c=_stop_for_vehicle_pass(v) + [crit("behavior_reasons_seen", "contains", "VEHICLE")],
            boundary="Stop ≥1.5 m behind lead; no contact.",
            timeout=60,
        ))
    return out


def va_slow_lead(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [20, 35], ["car", "truck", "bus"]), n)
    for lv, gap, kind in combos:
        out.append(scenario(
            name=f"Slow lead {kind} at {lv} m/s, gap {gap} m",
            category="vehicle_ahead", family="va_slow_lead",
            description=(f"Lead {kind} drives the ego lane at {lv} m/s starting {gap} m ahead. "
                         "Ego (8 m/s) closes in. Current stack has no car-following: it will slow at <15 m then stop at <5 m, "
                         "then re-accelerate when the gap opens — a stop-and-go oscillation. Target capability is "
                         "adaptive following at a time-gap."),
            tags=["moving_lead", "car_following", kind, "day"],
            caps=["vehicle_detection", "relative_speed_estimation", "adaptive_following"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", 10.0),
            mission_=mission(dest_ahead(400), cruise=8.0),
            actors=[actor("lead", "vehicle", VEHICLE_BPS[kind], route_ahead(gap),
                          {"kind": "constant_speed", "speed_mps": lv})],
            expected={"no_collision": True, "behavior_any": ["following_route", "stopped_vehicle"],
                      "target_capability": "follow at >=2 s gap without full stops"},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 2.0)],
            boundary="Never closer than 2 m to lead.",
            timeout=120,
        ))
    return out


def va_lead_brake_hard(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([15, 25, 35], [5.0, 8.0, 10.0], ["car", "suv"]), n)
    for gap, v, kind in combos:
        out.append(scenario(
            name=f"Lead {kind} hard-brakes, gap {gap} m, ego {v} m/s",
            category="vehicle_ahead", family="va_lead_brake_hard",
            description=(f"Lead {kind} drives at {v} m/s {gap} m ahead, then brakes fully once ego reaches speed. "
                         "Tests reaction latency of the perceive→decide→brake chain at closing speed."),
            tags=["moving_lead", "hard_brake", kind, f"gap_{gap}", "day"],
            caps=["vehicle_detection", "ttc_estimation", "emergency_braking"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", v + 2),
            mission_=mission(dest_ahead(350), cruise=v),
            actors=[actor("lead", "vehicle", VEHICLE_BPS[kind], route_ahead(gap),
                          {"kind": "brake_hard", "speed_mps": v}, trigger={"ego_speed_gt": v - 1.0})],
            expected={"behavior": "stopped_vehicle"},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0),
                    crit("behaviors_seen", "contains", "stopped_vehicle")],
            boundary="No rear-end contact.",
            timeout=90,
        ))
    return out


def va_cut_in(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["left", "right"], [8, 12, 18, 25], [4.0, 8.0], ["car", "motorcycle"]), n)
    for side, gap, lv, kind in combos:
        lat = -3.5 if side == "left" else 3.5
        out.append(scenario(
            name=f"{kind} cuts in from {side}, gap {gap} m at {lv} m/s",
            category="vehicle_ahead", family="va_cut_in",
            description=(f"A {kind} in the adjacent {side} lane, {gap} m ahead, moving at {lv} m/s, steers into the ego lane "
                         "when ego is within 30 m. Ego must treat it as in-path as soon as it crosses the lane boundary."),
            tags=["cut_in", side, kind, f"gap_{gap}", "day"],
            caps=["vehicle_detection", "lateral_position_tracking", "in_path_check"],
            status="partial",
            odd_=odd("Town05", "ClearNoon", "multilane", 10.0),
            mission_=mission(dest_ahead(300), cruise=8.0),
            actors=[actor("cutter", "vehicle", VEHICLE_BPS[kind], route_ahead(gap, lateral_m=lat),
                          {"kind": "cut_in", "speed_mps": lv, "side": side}, trigger={"ego_within_m": 30})],
            expected={"behavior_any": ["stopped_vehicle", "following_route"], "no_collision": True},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.5)],
            boundary="No side or rear contact with cutting vehicle.",
            timeout=90,
        ))
    return out


def va_cut_out_reveal(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([20, 30], ["car", "van"], ["car", "truck"]), n)
    for gap, lead_kind, hidden_kind in combos:
        out.append(scenario(
            name=f"Lead {lead_kind} cuts out revealing stopped {hidden_kind} ({gap} m)",
            category="vehicle_ahead", family="va_cut_out_reveal",
            description=(f"Lead {lead_kind} drives ahead at 6 m/s; a stopped {hidden_kind} sits {gap + 25} m along the route. "
                         "When ego is within 25 m of the lead, the lead swerves out of lane, exposing the stopped vehicle with little time. "
                         "Ground-truth perception sees both from the start — with real sensors the hidden one is occluded."),
            tags=["cut_out", "occlusion", lead_kind, hidden_kind, "day"],
            caps=["vehicle_detection", "occlusion_reasoning", "emergency_braking"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", 10.0),
            mission_=mission(dest_ahead(300), cruise=8.0),
            actors=[actor("lead", "vehicle", VEHICLE_BPS[lead_kind], route_ahead(gap),
                          {"kind": "cut_out", "speed_mps": 6.0, "side": "left"}, trigger={"ego_within_m": 25}),
                    actor("hidden", "vehicle", VEHICLE_BPS[hidden_kind], route_ahead(gap + 25), {"kind": "stopped"})],
            expected={"behavior": "stopped_vehicle"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_vehicle"),
                    crit("min_distance_to_actor_m", ">=", 1.5)],
            boundary="No contact with revealed vehicle.",
            timeout=90,
        ))
    return out


def va_oncoming(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([40, 70], [5.0, 10.0], ["car", "truck"], ["ClearNoon", "ClearNight"]), n)
    for gap, lv, kind, w in combos:
        out.append(scenario(
            name=f"Oncoming {kind} at {lv} m/s, {w}",
            category="vehicle_ahead", family="va_oncoming",
            description=(f"A {kind} approaches in the opposite lane at {lv} m/s from {gap} m. It is NOT in ego's path. "
                         "Ego must not stop for it (false positive check); lane width heuristic = 3.5 m."),
            tags=["oncoming", "false_positive_check", kind] + weather_tags(w),
            caps=["vehicle_detection", "lane_assignment", "in_path_check"],
            status="implemented",
            odd_=odd("Town01", w, "rural_2lane", 10.0),
            mission_=mission(dest_ahead(250), cruise=8.0),
            actors=[actor("oncoming", "vehicle", VEHICLE_BPS[kind], route_ahead(gap, lateral_m=-3.5, yaw_offset_deg=180),
                          {"kind": "oncoming", "speed_mps": lv})],
            expected={"behaviors_not": ["stopped_vehicle"], "final_mission_state": "completed"},
            pass_c=[NO_COLLISION, crit("mission_completed", "==", True),
                    crit("behaviors_seen", "not_in", ["stopped_vehicle"])],
            boundary="No unnecessary stop for oncoming traffic; no lane departure.",
            timeout=90,
        ))
    return out


def va_reversing(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([15, 25, 35], ["car", "van"]), n)
    for gap, kind in combos:
        out.append(scenario(
            name=f"{kind} reverses toward ego from {gap} m",
            category="vehicle_ahead", family="va_reversing",
            description=f"A {kind} {gap} m ahead starts reversing toward ego at 2 m/s once ego is within 30 m. Closing speed is higher than it looks.",
            tags=["reversing", kind, "day"],
            caps=["vehicle_detection", "relative_speed_estimation", "stop_for_vehicle"],
            status="partial",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=6.0),
            actors=[actor("rev", "vehicle", VEHICLE_BPS[kind], route_ahead(gap), {"kind": "reverse", "speed_mps": 2.0},
                          trigger={"ego_within_m": 30})],
            expected={"behavior": "stopped_vehicle"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_vehicle"),
                    crit("min_distance_to_actor_m", ">=", 1.5)],
            boundary="No contact.", timeout=60,
        ))
    return out


def va_traffic_autopilot(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([5, 10, 20, 40], ["Town03", "Town05", "Town10HD_Opt"], ["ClearNoon", "WetNoon"]), n)
    for count, town, w in combos:
        actors = [actor(f"tm_{i}", "vehicle", VEHICLE_BPS["car" if i % 3 else "suv"],
                        {"mode": "absolute", "spawn_point_index": 2 + i}, {"kind": "autopilot"})
                  for i in range(count)]
        out.append(scenario(
            name=f"Ambient traffic: {count} autopilot vehicles, {town}, {w}",
            category="vehicle_ahead", family="va_traffic_autopilot",
            description=(f"{count} CARLA traffic-manager vehicles roam {town}. Ego runs a 300 m mission amid them. "
                         "Tests in-path filtering with many detections and behaviour when traffic crosses the route."),
            tags=["ambient_traffic", f"density_{count}", "multi_actor"] + weather_tags(w),
            caps=["vehicle_detection", "in_path_check", "multi_object_tracking"],
            status="implemented",
            odd_=odd(town, w, "urban_mixed", 10.0),
            mission_=mission(dest_ahead(300), cruise=8.0),
            actors=actors,
            expected={"no_collision": True},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No collision with any traffic participant.",
            timeout=180,
        ))
    return out


def va_queue(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2, 3, 4], [20, 35]), n)
    for count, gap in combos:
        actors = [actor(f"q{i}", "vehicle", VEHICLE_BPS["car"], route_ahead(gap + 8 * i), {"kind": "stopped"}) for i in range(count)]
        out.append(scenario(
            name=f"Queue of {count} stopped vehicles from {gap} m",
            category="vehicle_ahead", family="va_queue",
            description=f"{count} stopped cars queue nose-to-tail starting {gap} m ahead. Ego must stop behind the LAST (closest) one, not an intermediate.",
            tags=["queue", "multi_actor", "day"],
            caps=["vehicle_detection", "closest_in_path_selection"],
            status="implemented",
            odd_=odd("Town05", "CloudyNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(gap + 150), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_vehicle"},
            pass_c=_stop_for_vehicle_pass(8.0),
            boundary="Stop behind nearest vehicle.", timeout=60,
        ))
    return out


def va_intersection_crossing(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["left", "right"], [6.0, 10.0], ["Town03", "Town05"]), n)
    for side, lv, town in combos:
        lat = -12 if side == "left" else 12
        out.append(scenario(
            name=f"Cross traffic from {side} at {lv} m/s ({town})",
            category="vehicle_ahead", family="va_intersection_crossing",
            description=(f"A car approaches the next intersection from the {side} at {lv} m/s, timed to cross as ego arrives. "
                         "Current in-path box (3.5 m wide, forward only) sees it late. Target: intersection right-of-way reasoning."),
            tags=["intersection", "cross_traffic", side, "day"],
            caps=["vehicle_detection", "intersection_handling", "right_of_way"],
            status="not_implemented",
            odd_=odd(town, "ClearNoon", "intersection", 8.0),
            mission_=mission(dest_ahead(250), cruise=7.0),
            actors=[actor("crosser", "vehicle", VEHICLE_BPS["car"], route_ahead(60, lateral_m=lat, yaw_offset_deg=(90 if side == "left" else -90)),
                          {"kind": "constant_speed", "speed_mps": lv}, trigger={"ego_within_m": 45})],
            expected={"no_collision": True, "target_capability": "yield to crossing traffic"},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 2.0)],
            boundary="No collision in intersection.", timeout=90,
        ))
    return out


# =====================================================================
# PEDESTRIAN
# =====================================================================

def _ped_pass(speed: float):
    return [NO_COLLISION, crit("behaviors_seen", "contains", "stopped_pedestrian"),
            crit("min_distance_to_actor_m", ">=", 1.5),
            crit("behavior_reasons_seen", "contains", "PEDESTRIAN")]


def ped_cross(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["left", "right"], [15, 25, 35, 50], [1.0, 1.5, 2.8], ["adult", "adult2"], CLEAR_DAY + ["WetNoon"]), n)
    for side, d, ws, bp, w in combos:
        lat = -5.0 if side == "left" else 5.0
        kind = "runs" if ws > 2 else "walks"
        out.append(scenario(
            name=f"Pedestrian {kind} across from {side}, {d} m, {ws} m/s, {w}",
            category="pedestrian", family="ped_cross",
            description=(f"An adult pedestrian waits on the {side} kerb {d} m ahead and {kind} across the lane at {ws} m/s "
                         f"when ego is within {d + 10} m. Ego must stop with reason naming PEDESTRIAN and never resume until the lane is clear."),
            tags=["crossing", side, f"walk_{ws}", f"dist_{d}"] + weather_tags(w),
            caps=["pedestrian_detection", "in_path_check", "stop_for_pedestrian"],
            status="implemented",
            odd_=odd("Town10HD_Opt", w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 120), cruise=8.0),
            actors=[actor("ped", "pedestrian", PED_BPS[bp], route_ahead(d, lateral_m=lat),
                          {"kind": "cross_road", "speed_mps": ws, "direction": "toward_lane"}, trigger={"ego_within_m": d + 10})],
            expected={"behavior": "stopped_pedestrian", "reason_contains": "PEDESTRIAN"},
            pass_c=_ped_pass(8.0),
            boundary="Absolutely no contact; stop ≥1.5 m.", timeout=60,
        ))
    return out


def ped_occluded_dart(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([15, 22, 30], ["car", "van", "truck"], ["adult", "child"]), n)
    for d, blocker, bp in combos:
        out.append(scenario(
            name=f"{bp.capitalize()} darts from behind parked {blocker} at {d} m",
            category="pedestrian", family="ped_occluded_dart",
            description=(f"A {blocker} is parked on the right shoulder {d} m ahead. A {bp} hidden behind it runs into the lane at 3 m/s "
                         f"when ego is within {d + 5} m. Minimum reaction distance case. Ground truth sees through the occluder; "
                         "real sensors will not — this is the scenario that motivates occlusion-aware speed caps."),
            tags=["occlusion", "dart_out", bp, blocker, "high_severity", "day"],
            caps=["pedestrian_detection", "occlusion_reasoning", "emergency_braking"],
            status="partial",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_parked_cars", 6.0),
            mission_=mission(dest_ahead(d + 100), cruise=6.0),
            actors=[actor("parked", "vehicle", VEHICLE_BPS[blocker], route_ahead(d, lateral_m=3.2), {"kind": "stopped"}),
                    actor("ped", "pedestrian", PED_BPS[bp], route_ahead(d + 1.5, lateral_m=5.5),
                          {"kind": "dart_out", "speed_mps": 3.0}, trigger={"ego_within_m": d + 5})],
            expected={"behavior": "stopped_pedestrian"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_pedestrian"),
                    crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact with pedestrian.", timeout=60,
        ))
    return out


def ped_standing_edge(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([1.2, 1.9, 2.6, 3.5], [20, 40], ["left", "right"]), n)
    for lat, d, side in combos:
        sl = lat if side == "right" else -lat
        inside = lat < 1.75
        out.append(scenario(
            name=f"Pedestrian standing {lat} m {side} of lane centre at {d} m",
            category="pedestrian", family="ped_standing_edge",
            description=(f"A stationary pedestrian stands {lat} m to the {side} of the lane centre line, {d} m ahead. "
                         + ("Inside the 3.5 m path box: ego must stop." if inside else
                            "Outside the path box: ego should slow (<15 m rule) and pass, not stop. Boundary test of the in-path heuristic.")),
            tags=["standing", side, "path_box_boundary", "day"],
            caps=["pedestrian_detection", "in_path_check", "lateral_clearance"],
            status="implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=8.0),
            actors=[actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(d, lateral_m=sl), {"kind": "stand"})],
            expected={"behavior": "stopped_pedestrian" if inside else "following_route",
                      "final_mission_state": None if inside else "completed"},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)] +
                   ([crit("behaviors_seen", "contains", "stopped_pedestrian")] if inside else
                    [crit("mission_completed", "==", True)]),
            boundary="≥1.0 m lateral clearance when passing; stop if inside lane.", timeout=60,
        ))
    return out


def ped_walk_along(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["toward", "away"], [1.0, 1.6], [25, 40]), n)
    for direction, lat, d in combos:
        out.append(scenario(
            name=f"Pedestrian walking {direction} ego along lane edge ({lat} m)",
            category="pedestrian", family="ped_walk_along",
            description=(f"Pedestrian walks {direction} ego at 1.2 m/s, {lat} m right of lane centre, starting {d} m ahead. "
                         "Sits on the path-box boundary: ego should slow and stop only if the walker drifts inside."),
            tags=["walk_along", direction, "path_box_boundary", "day"],
            caps=["pedestrian_detection", "in_path_check", "pedestrian_motion_prediction"],
            status="partial",
            odd_=odd("Town01", "ClearNoon", "rural_no_sidewalk", 8.0),
            mission_=mission(dest_ahead(d + 120), cruise=6.0),
            actors=[actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(d, lateral_m=lat, yaw_offset_deg=(180 if direction == "toward" else 0)),
                          {"kind": "walk_along", "speed_mps": 1.2, "direction": direction})],
            expected={"no_collision": True},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="≥1 m clearance.", timeout=80,
        ))
    return out


def ped_cross_and_stop(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([20, 30, 45], ["left", "right"], [5, 15, 40]), n)
    for d, side, dwell in combos:
        lat = -5.0 if side == "left" else 5.0
        out.append(scenario(
            name=f"Pedestrian stops in lane for {dwell} s ({d} m, from {side})",
            category="pedestrian", family="ped_cross_and_stop",
            description=(f"Pedestrian crosses from the {side} and halts in the middle of the ego lane for {dwell} s, then finishes crossing. "
                         "Ego must stop, wait (no creeping), and resume only after the lane is clear; mission should still complete."),
            tags=["crossing", "dwell", f"dwell_{dwell}", side, "resume_after_clear", "day"],
            caps=["pedestrian_detection", "stop_for_pedestrian", "resume_when_clear"],
            status="implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=8.0),
            actors=[actor("ped", "pedestrian", PED_BPS["adult2"], route_ahead(d, lateral_m=lat),
                          {"kind": "cross_and_stop", "speed_mps": 1.4, "dwell_s": dwell}, trigger={"ego_within_m": d + 10})],
            expected={"behavior": "stopped_pedestrian", "final_mission_state": "completed"},
            pass_c=_ped_pass(8.0) + [crit("mission_completed", "==", True), crit("stop_duration_s", ">=", dwell * 0.8)],
            boundary="No creeping toward pedestrian while stopped.", timeout=90 + dwell,
        ))
    return out


def ped_group(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([3, 5, 8], [20, 35], ["left", "right"]), n)
    for count, d, side in combos:
        lat = -5.0 if side == "left" else 5.0
        actors = [actor(f"ped{i}", "pedestrian", PED_BPS["adult" if i % 2 else "adult2"],
                        route_ahead(d + rng.uniform(-2, 2), lateral_m=lat + (0.8 * i if side == "right" else -0.8 * i)),
                        {"kind": "cross_road", "speed_mps": round(rng.uniform(0.9, 1.6), 1), "direction": "toward_lane"},
                        trigger={"ego_within_m": d + 10}) for i in range(count)]
        out.append(scenario(
            name=f"Group of {count} pedestrians crossing from {side} at {d} m",
            category="pedestrian", family="ped_group",
            description=f"{count} pedestrians cross in a loose group at different speeds. Ego must wait for the LAST one, not resume after the first clears.",
            tags=["crossing", "group", "multi_actor", side, "day"],
            caps=["pedestrian_detection", "multi_object_tracking", "resume_when_clear"],
            status="implemented",
            odd_=odd("Town10HD_Opt", "CloudyNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_pedestrian", "final_mission_state": "completed"},
            pass_c=_ped_pass(8.0) + [crit("mission_completed", "==", True)],
            boundary="No contact with any group member.", timeout=120,
        ))
    return out


def ped_child(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([12, 20, 30], ["child", "child2"], ["ClearNoon", "ClearSunset"]), n)
    for d, bp, w in combos:
        out.append(scenario(
            name=f"Child runs into lane at {d} m, {w}",
            category="pedestrian", family="ped_child",
            description=(f"A child-sized walker runs (2.5 m/s) into the lane from the right at {d} m. Small object, short notice. "
                         "With ground truth this is detected; with cameras, small-object recall at range is the risk."),
            tags=["child", "small_object", "dart_out", "high_severity"] + weather_tags(w),
            caps=["pedestrian_detection", "small_object_detection", "emergency_braking"],
            status="partial",
            odd_=odd("Town10HD_Opt", w, "residential", 6.0),
            mission_=mission(dest_ahead(d + 100), cruise=6.0),
            actors=[actor("child", "pedestrian", PED_BPS[bp], route_ahead(d, lateral_m=4.5),
                          {"kind": "dart_out", "speed_mps": 2.5}, trigger={"ego_within_m": d + 5})],
            expected={"behavior": "stopped_pedestrian"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_pedestrian"), crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact.", timeout=60,
        ))
    return out


def ped_low_visibility(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["ClearNight", "WetNight", "HardRainNight", "HardRainNoon", "DustStorm", "ClearSunset"], [20, 35], ["left", "right"]), n)
    for w, d, side in combos:
        lat = -5.0 if side == "left" else 5.0
        out.append(scenario(
            name=f"Pedestrian crossing in {w} from {side} ({d} m)",
            category="pedestrian", family="ped_low_visibility",
            description=(f"Standard crossing pedestrian under {w}. Identical to ped_cross for ground-truth perception; "
                         "kept as a regression pair so the delta vs clear weather is measurable once camera perception lands."),
            tags=["crossing", "low_visibility", side] + weather_tags(w),
            caps=["pedestrian_detection", "low_light_perception"],
            status="partial",
            odd_=odd("Town10HD_Opt", w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=6.0),
            actors=[actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(d, lateral_m=lat),
                          {"kind": "cross_road", "speed_mps": 1.3, "direction": "toward_lane"}, trigger={"ego_within_m": d + 10})],
            expected={"behavior": "stopped_pedestrian"},
            pass_c=_ped_pass(6.0),
            boundary="No contact.", timeout=60,
        ))
    return out


def ped_at_start_and_destination(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["start", "destination"], [2.5, 4.0, 6.0]), n)
    for where, d in combos:
        if where == "start":
            actors = [actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(d), {"kind": "stand"})]
            desc = f"A pedestrian is standing {d} m directly ahead when the mission is started. Ego must not move off at all."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 0.5), crit("behaviors_seen", "contains", "stopped_pedestrian")]
        else:
            actors = [actor("ped", "pedestrian", PED_BPS["adult"], {"mode": "at_destination", "offset_m": -d}, {"kind": "stand"})]
            desc = f"A pedestrian stands {d} m before the destination point. Ego must stop for them even though it is in approaching_destination mode."
            pass_c = [NO_COLLISION, crit("behaviors_seen", "contains", "stopped_pedestrian"), crit("min_distance_to_actor_m", ">=", 1.5)]
        out.append(scenario(
            name=f"Pedestrian at mission {where} ({d} m)",
            category="pedestrian", family="ped_at_start_and_destination",
            description=desc,
            tags=["edge", where, "day"],
            caps=["pedestrian_detection", "stop_for_pedestrian", "priority_ordering"],
            status="implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(120), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_pedestrian"},
            pass_c=pass_c,
            boundary="No contact; no motion if pedestrian already ahead at start.", timeout=45,
        ))
    return out


def ped_crosswalk(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([1.0, 1.5], ["left", "right"], ["waiting", "crossing"]), n)
    for ws, side, state in combos:
        lat = -4.5 if side == "left" else 4.5
        beh = {"kind": "stand"} if state == "waiting" else {"kind": "cross_road", "speed_mps": ws, "direction": "toward_lane"}
        out.append(scenario(
            name=f"Pedestrian {state} at marked crosswalk ({side})",
            category="pedestrian", family="ped_crosswalk",
            description=(f"At a marked crosswalk in Town10 a pedestrian is {state} on the {side}. "
                         + ("Target behaviour: yield to a waiting pedestrian at a crosswalk (not implemented — stack only reacts to in-path objects)."
                            if state == "waiting" else "Ego must stop for the crossing pedestrian.")),
            tags=["crosswalk", state, side, "day"],
            caps=["pedestrian_detection", "crosswalk_map_awareness", "yield_at_crosswalk"],
            status="not_implemented" if state == "waiting" else "implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "crosswalk", 6.0),
            mission_=mission(dest_ahead(150), cruise=6.0),
            actors=[actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(35, lateral_m=lat), beh, trigger={"ego_within_m": 45})],
            expected={"target_capability": "yield at crosswalk", "no_collision": True},
            pass_c=[NO_COLLISION] + ([] if state == "waiting" else [crit("behaviors_seen", "contains", "stopped_pedestrian")]),
            boundary="No contact.", timeout=60,
        ))
    return out
