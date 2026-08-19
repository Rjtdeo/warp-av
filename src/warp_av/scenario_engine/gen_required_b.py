"""
Families for: static_obstacle, blocked_route, component_failure, emergency_stop.
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
# STATIC OBSTACLE
# =====================================================================

def _obs_pass(speed):
    return [NO_COLLISION, crit("behaviors_seen", "contains", "stopped_obstacle"),
            crit("min_distance_to_actor_m", ">=", 1.0),
            crit("stopped_within_s_of_trigger", "<=", stop_budget_s(speed) + 2)]


def so_single_prop(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["cone", "barrier", "barrel", "box", "trashcan", "pallet", "shoppingcart", "bench"],
                            [0.0, 1.0, 2.4], [20, 40], CLEAR_DAY), n)
    for prop, lat, d, w in combos:
        in_path = abs(lat) < 1.75
        out.append(scenario(
            name=f"{prop} at {d} m, lateral {lat} m ({'in path' if in_path else 'shoulder'})",
            category="static_obstacle", family="so_single_prop",
            description=(f"A {prop} is placed {d} m ahead offset {lat} m right of the lane centre. "
                         + ("It is inside the path box → ego must stop with reason OBSTACLE." if in_path else
                            "It is on the shoulder/outside the path box → ego should slow but pass and complete the mission.")),
            tags=["prop", prop, "in_path" if in_path else "shoulder", f"lat_{lat}"] + weather_tags(w),
            caps=["obstacle_detection", "in_path_check", "stop_for_obstacle"],
            status="implemented",
            odd_=odd("Town03", w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=8.0),
            actors=[actor("obs", "prop", PROP_BPS[prop], route_ahead(d, lateral_m=lat), {"kind": "stopped"})],
            expected={"behavior": "stopped_obstacle" if in_path else "following_route"},
            pass_c=_obs_pass(8.0) if in_path else [NO_COLLISION, crit("mission_completed", "==", True)],
            boundary="No contact; no unnecessary stop for shoulder objects.", timeout=60,
        ))
    return out


def so_multi_stagger(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2, 3, 4], [25, 40]), n)
    for count, d in combos:
        actors = [actor(f"obs{i}", "prop", PROP_BPS["cone" if i % 2 else "barrel"],
                        route_ahead(d + 6 * i, lateral_m=(-1.0 if i % 2 else 1.0)), {"kind": "stopped"}) for i in range(count)]
        out.append(scenario(
            name=f"{count} staggered props from {d} m (slalom-like)",
            category="static_obstacle", family="so_multi_stagger",
            description=f"{count} props alternate ±1 m off centre every 6 m. All inside the path box; ego must stop at the first. A lateral-avoidance planner would thread them — not implemented.",
            tags=["prop", "multi_actor", "stagger", "day"],
            caps=["obstacle_detection", "stop_for_obstacle", "lateral_avoidance"],
            status="partial",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 120), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_obstacle", "target_capability": "avoid if clearance allows"},
            pass_c=_obs_pass(8.0),
            boundary="No contact.", timeout=60,
        ))
    return out


def so_sudden_appear(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([10, 15, 22, 30], [6.0, 8.0, 10.0], ["box", "barrel", "debris"]), n)
    for trig, v, prop in combos:
        out.append(scenario(
            name=f"{prop} appears {trig} m ahead at {v} m/s",
            category="static_obstacle", family="so_sudden_appear",
            description=(f"A {prop} is spawned in the lane when ego is {trig} m away while cruising at {v} m/s "
                         "(falls off a truck / round a blind corner). Tests raw stopping-distance margin: "
                         f"at {v} m/s with ~0.5 s latency and 4 m/s² braking ego needs ~{round(0.5*v + v*v/8, 1)} m."),
            tags=["prop", prop, "sudden", f"trig_{trig}", f"cruise_{v}", "high_severity", "day"],
            caps=["obstacle_detection", "emergency_braking", "latency_budget"],
            status="implemented",
            odd_=odd("Town04", "ClearNoon", "arterial", v + 1),
            mission_=mission(dest_ahead(250), cruise=v),
            actors=[actor("obs", "prop", PROP_BPS[prop], route_ahead(120), {"kind": "appear"}, trigger={"ego_within_m": trig})],
            expected={"behavior": "stopped_obstacle", "may_collide_if": f"trig < stopping distance at {v} m/s"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_obstacle")],
            boundary="Document the speed/distance pairs that collide — that defines the max safe cruise speed.", timeout=60,
        ))
    return out


def so_after_curve(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["Town03", "Town01", "Town02"], ["barrier", "cone"], [5.0, 8.0]), n)
    for town, prop, v in combos:
        out.append(scenario(
            name=f"{prop} just after a curve ({town}, {v} m/s)",
            category="static_obstacle", family="so_after_curve",
            description=(f"Route chosen to include a bend; {prop} placed 8 m past the bend apex. "
                         "Because the path box is a straight 3.5 m-wide rectangle in ego frame, an obstacle around a bend "
                         "may be judged 'out of path' until very late. Route-aware in-path check is the target."),
            tags=["prop", prop, "curve", "path_box_limitation", "day"],
            caps=["obstacle_detection", "route_aware_in_path_check"],
            status="partial",
            odd_=odd(town, "ClearNoon", "curve", v + 1),
            mission_=mission(dest_sp(4), cruise=v),
            actors=[actor("obs", "prop", PROP_BPS[prop], {"mode": "route_ahead", "distance_m": 60, "lateral_m": 0.0, "after_curve": True}, {"kind": "stopped"})],
            expected={"behavior": "stopped_obstacle"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_obstacle")],
            boundary="No contact.", timeout=90,
        ))
    return out


def so_low_and_large(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["debris", "tire", "glass", "container", "warning", "constructioncone"], [25, 40], [6.0, 8.0]), n)
    for prop, d, v in combos:
        low = prop in ("debris", "tire", "glass")
        out.append(scenario(
            name=f"{'Low' if low else 'Large'} obstacle {prop} at {d} m",
            category="static_obstacle", family="so_low_and_large",
            description=(f"A {'low-profile' if low else 'large'} {prop} in the lane at {d} m. "
                         + ("Low objects are the hardest class for lidar ground-segmentation and camera detectors; the ground-truth adapter finds it trivially — "
                            "tag kept so the real-perception delta is visible." if low else "Large object; trivial for any sensor — sanity case.")),
            tags=["prop", prop, "low_profile" if low else "large", "day"],
            caps=["obstacle_detection", "small_object_detection" if low else "obstacle_detection"],
            status="partial" if low else "implemented",
            odd_=odd("Town05", "ClearNoon", "urban_2lane", v + 1),
            mission_=mission(dest_ahead(d + 100), cruise=v),
            actors=[actor("obs", "prop", PROP_BPS[prop], route_ahead(d), {"kind": "stopped"})],
            expected={"behavior": "stopped_obstacle"},
            pass_c=_obs_pass(v),
            boundary="No contact.", timeout=60,
        ))
    return out


def so_at_destination(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2.0, 4.0, 8.0], ["cone", "box"]), n)
    for d, prop in combos:
        out.append(scenario(
            name=f"{prop} {d} m before destination",
            category="static_obstacle", family="so_at_destination",
            description=(f"A {prop} sits {d} m before the destination. "
                         + ("Within the 5 m arrival threshold: the stack may declare MISSION_COMPLETE before/while stopping — ordering bug check." if d < 5 else
                            "Ego should stop for it and NOT complete the mission.")),
            tags=["prop", prop, "destination", "priority_ordering", "day"],
            caps=["obstacle_detection", "priority_ordering", "mission_completion"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(120), cruise=8.0),
            actors=[actor("obs", "prop", PROP_BPS[prop], {"mode": "at_destination", "offset_m": -d}, {"kind": "stopped"})],
            expected={"behavior_any": ["stopped_obstacle", "mission_complete"]},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact even in the arrival zone.", timeout=60,
        ))
    return out


def so_low_visibility(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(ADVERSE + ["ClearNight", "SoftRainNight"], ["barrier", "box", "cone"], [25]), n)
    for w, prop, d in combos:
        out.append(scenario(
            name=f"{prop} at {d} m in {w}",
            category="static_obstacle", family="so_low_visibility",
            description=f"Regression pair of so_single_prop under {w}. Ground-truth invariant today; real perception will not be.",
            tags=["prop", prop, "low_visibility"] + weather_tags(w),
            caps=["obstacle_detection", "low_light_perception"],
            status="partial",
            odd_=odd("Town03", w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=6.0),
            actors=[actor("obs", "prop", PROP_BPS[prop], route_ahead(d), {"kind": "stopped"})],
            expected={"behavior": "stopped_obstacle"},
            pass_c=_obs_pass(6.0),
            boundary="No contact.", timeout=60,
        ))
    return out


# =====================================================================
# BLOCKED ROUTE
# =====================================================================

def _full_block_actors(d: float, prefix="blk"):
    return [actor(f"{prefix}{i}", "prop", PROP_BPS["barrier"], route_ahead(d, lateral_m=lat), {"kind": "stopped"})
            for i, lat in enumerate([-3.5, -1.2, 1.2, 3.5])]


def br_full_block_persistent(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([25, 45, 70], ["Town03", "Town05"], [30, 60]), n)
    for d, town, wait in combos:
        out.append(scenario(
            name=f"Full road block at {d} m, persists ({town})",
            category="blocked_route", family="br_full_block_persistent",
            description=(f"Barriers span both lanes {d} m ahead and never clear. Ego must stop and hold. Expected current behaviour: "
                         f"stopped_obstacle indefinitely (no re-plan). Target: after {wait} s declare route blocked, attempt re-route, "
                         "else fail the mission with reason and notify operator."),
            tags=["full_block", "persistent", "no_replan", "day"],
            caps=["obstacle_detection", "blocked_route_detection", "replanning", "mission_failure_reporting"],
            status="partial",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 150), cruise=8.0),
            actors=_full_block_actors(d),
            expected={"behavior_any": ["stopped_obstacle", "stopped_blocked"], "final_mission_state_any": ["executing", "failed"],
                      "target_capability": "stopped_blocked + mission failed with reason"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_obstacle"), crit("stop_duration_s", ">=", wait * 0.9)],
            boundary="Never attempt to drive around onto sidewalk/oncoming lane.", timeout=wait + 40,
        ))
    return out


def br_block_clears(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([5, 10, 20, 40], [25, 45], CLEAR_DAY), n)
    for clear_after, d, w in combos:
        actors = _full_block_actors(d)
        for a in actors:
            a["behavior"] = {"kind": "remove_after", "delay_s": clear_after}
            a["trigger"] = {"on_behavior": "stopped_obstacle"}
        out.append(scenario(
            name=f"Full block clears after {clear_after} s ({d} m)",
            category="blocked_route", family="br_block_clears",
            description=f"Both lanes blocked at {d} m; barriers are removed {clear_after} s after ego stops. Ego must resume automatically and complete the mission.",
            tags=["full_block", "clears", f"clear_{clear_after}", "resume_after_clear"] + weather_tags(w),
            caps=["obstacle_detection", "stop_for_obstacle", "resume_when_clear"],
            status="implemented",
            odd_=odd("Town03", w, "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 120), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_obstacle", "final_mission_state": "completed"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_obstacle"), crit("mission_completed", "==", True),
                    crit("stop_duration_s", ">=", clear_after * 0.8)],
            boundary="No contact; no premature resume.", timeout=90 + clear_after,
        ))
    return out


def br_block_at_intersection(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["Town03", "Town05"], ["barrier", "truck"], [8.0, 6.0]), n)
    for town, what, v in combos:
        if what == "truck":
            actors = [actor("blk", "vehicle", VEHICLE_BPS["truck"], {"mode": "route_ahead", "distance_m": 60, "lateral_m": 0.0, "at_next_junction": True, "yaw_offset_deg": 90}, {"kind": "stopped"})]
        else:
            actors = [actor(f"blk{i}", "prop", PROP_BPS["barrier"], {"mode": "route_ahead", "distance_m": 60, "lateral_m": lat, "at_next_junction": True}, {"kind": "stopped"}) for i, lat in enumerate([-2.5, 0, 2.5])]
        out.append(scenario(
            name=f"Junction blocked by {what} ({town})",
            category="blocked_route", family="br_block_at_intersection",
            description=f"The next junction on the route is blocked by a {what}. Ego must stop before the junction box (not inside it). Target: re-route via another arm.",
            tags=["intersection", "full_block", what, "day"],
            caps=["obstacle_detection", "junction_awareness", "replanning"],
            status="partial",
            odd_=odd(town, "ClearNoon", "intersection", v + 1),
            mission_=mission(dest_sp(6), cruise=v),
            actors=actors,
            expected={"behavior_any": ["stopped_obstacle", "stopped_vehicle"]},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="Do not stop inside the junction box.", timeout=120,
        ))
    return out


def br_block_near_destination(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([8, 15, 25], [True, False]), n)
    for d, clears in combos:
        actors = _full_block_actors(0)
        for a in actors:
            a["spawn"] = {"mode": "at_destination", "offset_m": -d, "lateral_m": a["spawn"]["lateral_m"]}
            if clears:
                a["behavior"] = {"kind": "remove_after", "delay_s": 10}
                a["trigger"] = {"on_behavior": "stopped_obstacle"}
        out.append(scenario(
            name=f"Block {d} m before destination ({'clears' if clears else 'persists'})",
            category="blocked_route", family="br_block_near_destination",
            description=(f"Full block {d} m short of the goal. "
                         + ("Clears after 10 s → must complete." if clears else
                            "Persists → must NOT report completion even though it is close; should end as blocked/failed with reason.")),
            tags=["full_block", "destination", "clears" if clears else "persistent", "day"],
            caps=["obstacle_detection", "mission_completion", "mission_failure_reporting"],
            status="implemented" if clears else "partial",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(150), cruise=8.0),
            actors=actors,
            expected={"final_mission_state": "completed" if clears else "executing|failed"},
            pass_c=[NO_COLLISION] + ([crit("mission_completed", "==", True)] if clears else [crit("mission_completed", "==", False)]),
            boundary="No false completion.", timeout=90,
        ))
    return out


def br_construction_zone(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([30, 50], ["partial", "full"], ["ClearNoon", "WetNoon", "ClearNight"]), n)
    for d, kind, w in combos:
        lats = [-1.0, -0.3, 0.4, 1.1] if kind == "full" else [1.0, 1.4, 1.8]
        actors = [actor(f"cone{i}", "prop", PROP_BPS["constructioncone"], route_ahead(d + 3 * i, lateral_m=lat), {"kind": "stopped"}) for i, lat in enumerate(lats)]
        actors.append(actor("sign", "prop", PROP_BPS["warning"], route_ahead(d - 10, lateral_m=3.0), {"kind": "stopped"}))
        out.append(scenario(
            name=f"Construction zone ({kind} lane) at {d} m, {w}",
            category="blocked_route", family="br_construction_zone",
            description=(f"Taper of construction cones {'closes the lane' if kind == 'full' else 'narrows the lane to ~2 m'} at {d} m with a warning sign before it. "
                         + ("Full closure: stop & hold; target is lane change / re-route." if kind == "full" else
                            "Partial: cones sit at the path-box edge; ego should slow and pass if the remaining width is sufficient — currently it will likely stop (no width reasoning).")),
            tags=["construction", kind, "cones"] + weather_tags(w),
            caps=["obstacle_detection", "drivable_width_estimation", "lane_change"],
            status="partial",
            odd_=odd("Town05", w, "construction", 6.0),
            mission_=mission(dest_ahead(d + 120), cruise=6.0),
            actors=actors,
            expected={"behavior_any": ["stopped_obstacle", "following_route"]},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 0.8)],
            boundary="No cone contact.", timeout=90,
        ))
    return out


def br_traffic_jam(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([4, 8], [20, 35], [True, False]), n)
    for count, d, moves in combos:
        actors = [actor(f"jam{i}", "vehicle", VEHICLE_BPS["car" if i % 2 else "suv"], route_ahead(d + 7 * i),
                        ({"kind": "constant_speed", "speed_mps": 1.0} if moves else {"kind": "stopped"}),
                        trigger={"at_s": 25} if moves else None) for i in range(count)]
        out.append(scenario(
            name=f"Traffic jam of {count} cars, {'creeping' if moves else 'static'}",
            category="blocked_route", family="br_traffic_jam",
            description=(f"{count} cars queue from {d} m. "
                         + ("After 25 s they creep at 1 m/s; ego should creep behind the last one (stop-and-go)." if moves else
                            "They never move; ego holds behind the queue.")),
            tags=["jam", "multi_actor", "creeping" if moves else "static", "day"],
            caps=["vehicle_detection", "adaptive_following", "stop_and_go"],
            status="partial",
            odd_=odd("Town05", "CloudyNoon", "arterial", 8.0),
            mission_=mission(dest_ahead(d + 200), cruise=8.0),
            actors=actors,
            expected={"behavior": "stopped_vehicle"},
            pass_c=[NO_COLLISION, crit("behaviors_seen", "contains", "stopped_vehicle"), crit("min_distance_to_actor_m", ">=", 1.5)],
            boundary="No rear-end in stop-and-go.", timeout=120,
        ))
    return out


def br_unreachable(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["off_map", "opposite_side_blocked"], ["Town01", "Town03"]), n)
    for kind, town in combos:
        if kind == "off_map":
            dest = {"mode": "off_map"}
            desc = "Destination is off the road network. Planner must fail cleanly; mission state → failed with reason 'Route planning failed'; vehicle must not move."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 0.3), crit("final_mission_state", "in", ["failed", "idle"])]
            status = "implemented"
            actors = []
        else:
            dest = dest_ahead(200)
            desc = "Route fully blocked immediately (5 m) AND again at 60 m. No alternative. Ego holds at the first block; target is mission failure with 'route blocked' after a timeout."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 2.0)]
            status = "partial"
            actors = _full_block_actors(6, "a") + _full_block_actors(60, "b")
        out.append(scenario(
            name=f"Unreachable destination: {kind} ({town})",
            category="blocked_route", family="br_unreachable",
            description=desc,
            tags=["unreachable", kind, "planner_failure", "day"],
            caps=["route_planning", "mission_failure_reporting"],
            status=status,
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest, cruise=8.0),
            actors=actors,
            expected={"final_mission_state": "failed"},
            pass_c=pass_c,
            boundary="No motion toward an unplannable goal.", timeout=60,
        ))
    return out


def br_block_immediately(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([3, 5, 8], ["barrier", "truck"]), n)
    for d, what in combos:
        actors = (_full_block_actors(d) if what == "barrier" else
                  [actor("blk", "vehicle", VEHICLE_BPS["truck"], route_ahead(d), {"kind": "stopped"})])
        out.append(scenario(
            name=f"Blocked {d} m from start by {what}",
            category="blocked_route", family="br_block_immediately",
            description=f"Route is blocked {d} m from the start position before the mission begins. Ego must not move off (or creep <0.5 m/s) and must report why.",
            tags=["full_block", "at_start", what, "day"],
            caps=["obstacle_detection", "stop_for_obstacle"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(150), cruise=8.0),
            actors=actors,
            expected={"behavior_any": ["stopped_obstacle", "stopped_vehicle"], "max_speed_mps": 0.5},
            pass_c=[NO_COLLISION, crit("max_speed_mps", "<=", 0.8)],
            boundary="No contact with an obstacle already inside the danger distance.", timeout=40,
        ))
    return out


# =====================================================================
# COMPONENT FAILURE
# =====================================================================

COMPONENTS = ["perception", "localization", "controller", "planner", "camera", "lidar", "gnss", "imu", "vehicle_connection"]

# what the safety supervisor / behaviour is expected to do today
COMPONENT_EXPECT = {
    "perception":         ("intervention", "Perception system unhealthy", "implemented"),
    "localization":       ("intervention", "Localization unhealthy", "implemented"),
    "controller":         ("intervention", "Controller unhealthy", "implemented"),
    "planner":            ("intervention", "Route planning failed", "partial"),   # only matters at (re)plan time
    "camera":             ("intervention", "camera stale", "not_implemented"),      # sensor health not wired to supervisor yet
    "lidar":              ("intervention", "lidar stale", "not_implemented"),
    "gnss":               ("warning", "gnss stale", "not_implemented"),
    "imu":                ("warning", "imu stale", "not_implemented"),
    "vehicle_connection": ("intervention", "Vehicle connection lost", "implemented"),
}
TIMINGS = {"before_start": -1.0, "at_speed": 6.0, "while_stopped_for_obstacle": None}


def _fail_events(component, action, timing, **extra):
    if timing == "before_start":
        trig = {"at_s": 0.5}
    elif timing == "at_speed":
        trig = {"ego_speed_gt": 5.0}
    else:
        trig = {"on_behavior": "stopped_obstacle"}
    return [inject(trig, component, action, **extra)]


def cf_disable(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(COMPONENTS, list(TIMINGS.keys())), n)
    for comp, timing in combos:
        ss, reason, status = COMPONENT_EXPECT[comp]
        actors = [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(30), {"kind": "stopped"})] if timing == "while_stopped_for_obstacle" else []
        pass_c = [NO_COLLISION]
        if status == "implemented":
            pass_c += [crit("safety_states_seen", "contains", ss), crit("errors_seen", "contains", reason),
                       crit("safety_reaction_time_s", "<=", 0.5)]
            if timing == "at_speed":
                pass_c.append(crit("stopped_within_s_of_trigger", "<=", stop_budget_s(6.0)))
            if timing == "before_start":
                pass_c.append(crit("max_speed_mps", "<=", 0.3))
        out.append(scenario(
            name=f"Disable {comp} {timing.replace('_', ' ')}",
            category="component_failure", family="cf_disable",
            description=(f"{comp} is disabled {timing.replace('_', ' ')} (mission starts at t=2 s). "
                         f"Expected: safety supervisor → {ss} with '{reason}', vehicle brought to a stop and held; behaviour reports stopped_safety. "
                         + ("" if status == "implemented" else
                            "NOTE: sensor-adapter health (camera/lidar/gnss/imu) is tracked in CarlaSensorAdapter but is NOT yet fed into SafetySupervisor — "
                            "this scenario documents the gap; with ground-truth perception the vehicle will keep driving.")),
            tags=["fault_injection", comp, timing, "day"],
            caps=["component_health_monitoring", "safe_stop", f"{comp}_health"],
            status=status,
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0, start_at_s=2.0),
            actors=actors,
            events=_fail_events(comp, "disable", timing),
            expected={"safety_state": ss, "error_contains": reason, "behavior": "stopped_safety"},
            pass_c=pass_c,
            boundary="Vehicle must reach and hold zero speed after detection; no autonomous motion while unhealthy.",
            timeout=60,
        ))
    return out


def cf_stale_and_freeze(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["perception", "localization"], ["stale", "freeze"], [0.6, 1.2, 3.0]), n)
    for comp, act, age in combos:
        out.append(scenario(
            name=f"{comp} {act} ({age} s old data)",
            category="component_failure", family="cf_stale_and_freeze",
            description=(f"{comp} keeps returning data but its timestamp {'stops advancing' if act == 'freeze' else f'is held {age} s in the past'}. "
                         f"Supervisor threshold is 1.0 s: {'must NOT trip (under threshold)' if age < 1.0 else 'must trip staleness check'}. "
                         "Freeze is the nastier case: values look plausible but are old — exactly what a hung sensor driver produces."),
            tags=["fault_injection", comp, act, f"age_{age}", "staleness", "day"],
            caps=["staleness_detection", "safe_stop"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, comp, act, age_s=age)],
            expected={"safety_state": "intervention" if age >= 1.0 else "ok", "reason_contains": "stale"},
            pass_c=[NO_COLLISION] + ([crit("safety_states_seen", "contains", "intervention"), crit("safety_reaction_time_s", "<=", age + 0.5)]
                                     if age >= 1.0 else [crit("safety_states_seen", "not_in", ["intervention"]), crit("mission_completed", "==", True)]),
            boundary="Stale data must never drive the vehicle for more than max_age.", timeout=60,
        ))
    return out


def cf_low_confidence(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([0.6, 0.31, 0.29, 0.1, 0.0], [True, False]), n)
    for conf, ramp in combos:
        trips = conf < 0.3
        out.append(scenario(
            name=f"Localization confidence {'ramps to' if ramp else 'drops to'} {conf}",
            category="component_failure", family="cf_low_confidence",
            description=(f"Localization confidence is {'ramped over 5 s' if ramp else 'stepped'} to {conf} at speed. Threshold 0.3. "
                         + ("Must trip with 'Localization confidence too low'." if trips else "Must NOT trip; hysteresis check near threshold.")),
            tags=["fault_injection", "localization", "confidence", f"conf_{conf}", "day"],
            caps=["localization_confidence", "safe_stop"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, "localization", "low_confidence", value=conf, ramp_s=5.0 if ramp else 0.0)],
            expected={"safety_state": "intervention" if trips else "ok"},
            pass_c=[NO_COLLISION] + ([crit("safety_states_seen", "contains", "intervention")] if trips else
                                     [crit("safety_states_seen", "not_in", ["intervention"])]),
            boundary="Stop when confidence < 0.3.", timeout=60,
        ))
    return out


def cf_recover(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["perception", "localization", "controller", "vehicle_connection", "camera", "lidar"], [2, 5, 15]), n)
    for comp, after in combos:
        ss, reason, status = COMPONENT_EXPECT[comp]
        out.append(scenario(
            name=f"{comp} fails then recovers after {after} s",
            category="component_failure", family="cf_recover",
            description=(f"{comp} disabled at speed, re-enabled {after} s later. Ego must stop on failure and — policy question — "
                         "either resume autonomously (current behaviour: yes, autonomy is still engaged and behaviour resumes) or require operator re-engage. "
                         "Scenario records what happens; the safety doc should state the chosen policy."),
            tags=["fault_injection", comp, "recovery", f"after_{after}", "policy_question", "day"],
            caps=["component_health_monitoring", "recovery_policy"],
            status=status if status != "not_implemented" else "not_implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, comp, "disable"),
                    inject({"after_event": 0, "at_s": after}, comp, "enable")],
            expected={"safety_state": ss, "then": "ok", "resumed_after_clear": "policy-dependent"},
            pass_c=[NO_COLLISION] + ([crit("safety_states_seen", "contains", ss)] if status == "implemented" else []),
            boundary="No motion while unhealthy.", timeout=90,
        ))
    return out


def cf_double_failure(rng: random.Random, n: int) -> List[dict]:
    out = []
    pairs = [("perception", "localization"), ("localization", "controller"), ("perception", "vehicle_connection"),
             ("camera", "lidar"), ("gnss", "imu"), ("controller", "planner")]
    combos = take(rng, grid(pairs, [0.0, 1.0]), n)
    for (a, b), gap in combos:
        out.append(scenario(
            name=f"Double failure: {a} + {b} ({'simultaneous' if gap == 0 else f'{gap}s apart'})",
            category="component_failure", family="cf_double_failure",
            description=f"{a} then {b} fail {gap} s apart at speed. Supervisor must report BOTH errors (not just the first it hits) and stop once.",
            tags=["fault_injection", a, b, "double", "day"],
            caps=["component_health_monitoring", "multi_fault_reporting"],
            status="partial",   # supervisor returns on first failed check → second error not listed
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, a, "disable"), inject({"after_event": 0, "at_s": gap}, b, "disable")],
            expected={"safety_state": "intervention", "errors_count": 2},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "intervention")],
            boundary="Stop and hold.", timeout=60,
        ))
    return out


def cf_failure_with_hazard(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["perception", "localization", "controller"], ["pedestrian", "vehicle"], [20, 30]), n)
    for comp, hazard, d in combos:
        act = (actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(d, lateral_m=5.0), {"kind": "cross_road", "speed_mps": 1.3, "direction": "toward_lane"}, trigger={"ego_within_m": d + 10})
               if hazard == "pedestrian" else actor("lead", "vehicle", VEHICLE_BPS["car"], route_ahead(d), {"kind": "stopped"}))
        out.append(scenario(
            name=f"{comp} fails while {hazard} is ahead ({d} m)",
            category="component_failure", family="cf_failure_with_hazard",
            description=(f"{comp} is disabled exactly when ego is 15 m from a {hazard} in its lane. The stop must still happen — "
                         "either the hazard logic (if perception still works) or the supervisor (if it does not). Worst-case: perception dies with a pedestrian 15 m ahead at 8 m/s."),
            tags=["fault_injection", comp, hazard, "high_severity", "day"],
            caps=["component_health_monitoring", "safe_stop", "fail_safe_braking"],
            status="implemented",
            odd_=odd("Town10HD_Opt", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(d + 100), cruise=8.0),
            actors=[act],
            events=[inject({"ego_within_m": 15}, comp, "disable")],
            expected={"safety_state": "intervention", "no_collision": True},
            pass_c=[NO_COLLISION, crit("min_distance_to_actor_m", ">=", 1.0)],
            boundary="No contact regardless of which layer stops the vehicle.", timeout=60,
        ))
    return out


def cf_failure_during_estop_and_arrival(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["during_estop", "at_arrival"], ["perception", "localization", "vehicle_connection"]), n)
    for when, comp in combos:
        if when == "during_estop":
            events = [event({"ego_speed_gt": 5.0}, "estop"), inject({"after_event": 0, "at_s": 2.0}, comp, "disable"),
                      event({"after_event": 0, "at_s": 6.0}, "estop_clear")]
            desc = f"E-stop is latched, then {comp} fails while stopped, then the operator clears the e-stop. After clearing, the supervisor must still report {comp} unhealthy and refuse motion."
            pass_c = [NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop"), crit("safety_states_seen", "contains", "intervention"),
                      crit("final_speed_mps", "<=", 0.3)]
        else:
            events = [inject({"on_behavior": "approaching_destination"}, comp, "disable")]
            desc = f"{comp} fails within 20 m of the goal. Mission must NOT be marked completed by coasting in; expect stopped_safety before arrival."
            pass_c = [NO_COLLISION, crit("safety_states_seen", "contains", "intervention")]
        out.append(scenario(
            name=f"{comp} fails {when.replace('_', ' ')}",
            category="component_failure", family="cf_failure_during_estop_and_arrival",
            description=desc,
            tags=["fault_injection", comp, when, "priority_ordering", "day"],
            caps=["component_health_monitoring", "estop_latching", "priority_ordering"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(150), cruise=8.0),
            events=events,
            expected={"safety_state_sequence": ["emergency_stop", "intervention"] if when == "during_estop" else ["intervention"]},
            pass_c=pass_c,
            boundary="No motion while any fault is active.", timeout=60,
        ))
    return out


def cf_crash_and_invalid(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["crash", "nan_command", "clock_jump"], ["perception", "controller", "localization", "api"]), n)
    for act, comp in combos:
        desc = {
            "crash": f"{comp}.update() raises an exception mid-tick. main loop catches per-tick errors and sleeps — meaning the LAST command keeps being applied by CARLA. Target: a tick exception must trigger a brake command and supervisor error.",
            "nan_command": f"{comp} emits NaN/inf (steering=NaN). Vehicle adapter clamps with max/min which does NOT reject NaN. Target: command validation rejects NaN → brake.",
            "clock_jump": f"System clock jumps +30 s (NTP step). All staleness checks use time.time() → everything looks stale → spurious intervention. Target: monotonic clocks.",
        }[act]
        out.append(scenario(
            name=f"{act} in {comp}",
            category="component_failure", family="cf_crash_and_invalid",
            description=desc,
            tags=["fault_injection", comp, act, "software_fault", "day"],
            caps=["exception_containment", "command_validation", "monotonic_time"],
            status="not_implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, comp, act)],
            expected={"safety_state": "intervention", "target_capability": "fail-safe brake on software fault"},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "intervention")],
            boundary="A software fault must never leave the last throttle command applied.", timeout=60,
        ))
    return out


def cf_tick_latency(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([0.05, 0.15, 0.3, 0.6, 1.5], ["constant", "spike"]), n)
    for lat, mode in combos:
        out.append(scenario(
            name=f"Tick latency {lat} s ({mode})",
            category="component_failure", family="cf_tick_latency",
            description=(f"Main loop tick is delayed by {lat} s ({'every tick' if mode == 'constant' else 'single spike'}). "
                         f"Nominal period 0.1 s; staleness threshold 1.0 s. {'Should trip staleness.' if lat >= 1.0 else 'Under threshold: must keep driving but control quality degrades — measure oscillation & deviation.'}"),
            tags=["fault_injection", "tick_latency", f"lat_{lat}", mode, "day"],
            caps=["watchdog", "latency_tolerance"],
            status="partial",
            odd_=odd("Town04", "ClearNoon", "arterial", 8.0),
            mission_=mission(dest_ahead(250), cruise=8.0),
            events=[inject({"ego_speed_gt": 5.0}, "tick_latency", "latency", latency_s=lat, mode=mode)],
            expected={"safety_state": "intervention" if lat >= 1.0 else "ok"},
            pass_c=[NO_COLLISION, crit("route_deviation_max_m", "<=", 3.0)] +
                   ([crit("safety_states_seen", "contains", "intervention")] if lat >= 1.0 else []),
            boundary="Loop stalls ≥1 s must stop the vehicle.", timeout=90,
        ))
    return out


def cf_flapping(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["perception", "localization", "vehicle_connection"], [0.5, 2.0]), n)
    for comp, period in combos:
        events = []
        for i in range(6):
            events.append(inject({"at_s": 8 + i * period * 2}, comp, "disable"))
            events.append(inject({"at_s": 8 + i * period * 2 + period}, comp, "enable"))
        out.append(scenario(
            name=f"{comp} flaps every {period} s",
            category="component_failure", family="cf_flapping",
            description=f"{comp} toggles healthy/unhealthy 6 times with period {period} s. Without hysteresis the vehicle lurches stop/go. Target: debounce + require N s healthy before resuming.",
            tags=["fault_injection", comp, "flapping", "hysteresis", "day"],
            caps=["component_health_monitoring", "hysteresis"],
            status="partial",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=events,
            expected={"no_lurching": True},
            pass_c=[NO_COLLISION],
            boundary="No stop/go oscillation faster than 1 Hz.", timeout=90,
        ))
    return out


# =====================================================================
# EMERGENCY STOP
# =====================================================================

def es_at_speed(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0], ["ClearNoon", "WetNoon", "HardRainNoon"]), n)
    for v, w in combos:
        trig = {"ego_speed_gt": max(v - 0.5, 0.0)} if v > 0 else {"at_s": 1.0}
        out.append(scenario(
            name=f"E-stop at {v} m/s, {w}",
            category="emergency_stop", family="es_at_speed",
            description=(f"Operator e-stop while ego travels at {v} m/s on {w}. Vehicle adapter must apply full brake + handbrake on the same tick, "
                         "safety state → emergency_stop (latched), behaviour → stopped_safety, reason names the operator. "
                         f"Stopping distance recorded; wet-road friction only matters once CARLA friction is varied."),
            tags=["estop", f"speed_{v}", "latching"] + weather_tags(w),
            caps=["estop", "estop_latching", "immediate_brake"],
            status="implemented",
            odd_=odd("Town04", w, "arterial", 12.0),
            mission_=mission(dest_ahead(300), cruise=max(v, 1.0)),
            events=[event(trig, "estop")],
            expected={"safety_state": "emergency_stop", "reason_contains": "EMERGENCY STOP"},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop"),
                    crit("stopped_within_s_of_trigger", "<=", stop_budget_s(v, margin_s=0.8)),
                    crit("safety_reaction_time_s", "<=", 0.3), crit("final_speed_mps", "<=", 0.1)],
            boundary="Brake applied within one tick; remains stopped until cleared.", timeout=40,
        ))
    return out


def es_in_turn_and_hazard(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["in_turn", "with_pedestrian_ahead", "at_obstacle_stop", "at_destination"], ["Town03", "Town10HD_Opt"]), n)
    for where, town in combos:
        actors, events, mission_ = [], [], mission(dest_sp(3), cruise=8.0)
        if where == "in_turn":
            events = [event({"on_behavior": "following_route", "at_s": 6.0}, "estop")]
            desc = "E-stop fired mid-turn (steer command non-zero). Handbrake on a turning van: check it does not spin; steering must freeze, not centre."
        elif where == "with_pedestrian_ahead":
            mission_ = mission(dest_ahead(150), cruise=8.0)
            actors = [actor("ped", "pedestrian", PED_BPS["adult"], route_ahead(30, lateral_m=5.0), {"kind": "cross_road", "speed_mps": 1.3, "direction": "toward_lane"}, trigger={"ego_within_m": 40})]
            events = [event({"ego_within_m": 25}, "estop")]
            desc = "E-stop fired as a pedestrian starts crossing 25 m ahead. E-stop must win over everything; reason must be e-stop, not pedestrian."
        elif where == "at_obstacle_stop":
            mission_ = mission(dest_ahead(150), cruise=8.0)
            actors = [actor("obs", "prop", PROP_BPS["barrier"], route_ahead(30), {"kind": "stopped"})]
            events = [event({"on_behavior": "stopped_obstacle"}, "estop")]
            desc = "Ego already stopped for an obstacle; e-stop pressed. State must change to emergency_stop even though speed is 0; clearing must NOT resume into the obstacle."
        else:
            mission_ = mission(dest_ahead(120), cruise=8.0)
            events = [event({"on_behavior": "approaching_destination"}, "estop")]
            desc = "E-stop within 20 m of the goal. Mission must not be marked completed by the arrival logic while e-stopped."
        out.append(scenario(
            name=f"E-stop {where.replace('_', ' ')} ({town})",
            category="emergency_stop", family="es_in_turn_and_hazard",
            description=desc,
            tags=["estop", where, "priority_ordering", "day"],
            caps=["estop", "priority_ordering", "estop_latching"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "urban_mixed", 8.0),
            mission_=mission_,
            actors=actors, events=events,
            expected={"safety_state": "emergency_stop"},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop"), crit("final_speed_mps", "<=", 0.1)],
            boundary="E-stop overrides all other logic.", timeout=60,
        ))
    return out


def es_clear_and_resume(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([3, 10], ["resume", "no_resume", "new_mission", "start_while_latched"]), n)
    for hold, follow in combos:
        events = [event({"ego_speed_gt": 5.0}, "estop"), event({"after_event": 0, "at_s": hold}, "estop_clear")]
        if follow == "resume":
            events.append(event({"after_event": 1, "at_s": 1.0}, "resume"))
            desc = f"E-stop, clear after {hold} s, operator resumes. Clearing alone must leave vehicle MANUAL (no motion); explicit resume re-engages and mission completes."
            pass_c = [NO_COLLISION, crit("resumed_after_clear", "==", True), crit("mission_completed", "==", True)]
        elif follow == "no_resume":
            desc = f"E-stop, clear after {hold} s, NO resume. Vehicle must stay stationary: clearing e-stop must not auto-re-engage autonomy."
            pass_c = [NO_COLLISION, crit("resumed_after_clear", "==", False), crit("final_speed_mps", "<=", 0.1)]
        elif follow == "new_mission":
            events.append(event({"after_event": 1, "at_s": 1.0}, "start_mission", destination=dest_ahead(100)))
            desc = f"E-stop, clear, then a NEW mission is started. Must engage and drive."
            pass_c = [NO_COLLISION, crit("resumed_after_clear", "==", True)]
        else:
            events = [event({"ego_speed_gt": 5.0}, "estop"), event({"after_event": 0, "at_s": 2.0}, "start_mission", destination=dest_ahead(100))]
            desc = "While e-stop is LATCHED an operator tries to start a new mission. engage_autonomy must be refused; vehicle stays stopped."
            pass_c = [NO_COLLISION, crit("final_speed_mps", "<=", 0.1), crit("safety_states_seen", "contains", "emergency_stop")]
        out.append(scenario(
            name=f"E-stop clear → {follow.replace('_', ' ')} (hold {hold} s)",
            category="emergency_stop", family="es_clear_and_resume",
            description=desc,
            tags=["estop", "clear", follow, "state_machine", "day"],
            caps=["estop", "estop_clear_policy", "engage_disengage"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(200), cruise=8.0),
            events=events,
            expected={"after_clear": "manual, no motion", "after_resume": "autonomous"},
            pass_c=pass_c,
            boundary="Clearing e-stop never produces motion by itself.", timeout=90,
        ))
    return out


def es_repeated(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid([2, 5, 10], [0.2, 1.0, 4.0]), n)
    for count, period in combos:
        events = []
        t = 5.0
        for i in range(count):
            events.append(event({"at_s": t}, "estop"))
            events.append(event({"at_s": t + period}, "estop_clear"))
            events.append(event({"at_s": t + period + 0.3}, "resume"))
            t += period + 2.0
        out.append(scenario(
            name=f"{count} e-stops, {period} s apart",
            category="emergency_stop", family="es_repeated",
            description=f"E-stop/clear/resume cycled {count} times with {period} s hold. Checks the state machine has no stuck state and the latch is idempotent.",
            tags=["estop", "repeated", "state_machine", "day"],
            caps=["estop", "state_machine_robustness"],
            status="implemented",
            odd_=odd("Town03", "ClearNoon", "urban_2lane", 8.0),
            mission_=mission(dest_ahead(250), cruise=8.0),
            events=events,
            expected={"no_stuck_state": True},
            pass_c=[NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop")],
            boundary="Every e-stop must brake; every clear must not move the vehicle.", timeout=60 + count * 5,
        ))
    return out


def es_during_pause_and_before_mission(rng: random.Random, n: int) -> List[dict]:
    out = []
    combos = take(rng, grid(["during_pause", "before_mission", "double_press"], ["Town03", "Town05"]), n)
    for kind, town in combos:
        if kind == "during_pause":
            events = [event({"ego_speed_gt": 5.0}, "pause"), event({"after_event": 0, "at_s": 2.0}, "estop"),
                      event({"after_event": 1, "at_s": 3.0}, "resume")]
            desc = "Mission paused (disengaged), then e-stop, then operator presses resume while e-stop still latched. Resume must be refused."
            pass_c = [NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop"), crit("final_speed_mps", "<=", 0.1)]
            mission_ = mission(dest_ahead(200), cruise=8.0)
        elif kind == "before_mission":
            events = [event({"at_s": 0.5}, "estop")]
            mission_ = mission(dest_ahead(200), cruise=8.0, start_at_s=3.0)
            desc = "E-stop pressed before any mission; mission start attempted 3 s later. Must not move."
            pass_c = [NO_COLLISION, crit("max_speed_mps", "<=", 0.2)]
        else:
            events = [event({"ego_speed_gt": 5.0}, "estop"), event({"after_event": 0, "at_s": 0.1}, "estop")]
            mission_ = mission(dest_ahead(200), cruise=8.0)
            desc = "E-stop pressed twice in 100 ms (double-click). Second press must be idempotent."
            pass_c = [NO_COLLISION, crit("safety_states_seen", "contains", "emergency_stop"), crit("final_speed_mps", "<=", 0.1)]
        out.append(scenario(
            name=f"E-stop {kind.replace('_', ' ')} ({town})",
            category="emergency_stop", family="es_during_pause_and_before_mission",
            description=desc,
            tags=["estop", kind, "state_machine", "day"],
            caps=["estop", "state_machine_robustness"],
            status="implemented",
            odd_=odd(town, "ClearNoon", "urban_2lane", 8.0),
            mission_=mission_, events=events,
            expected={"safety_state": "emergency_stop"},
            pass_c=pass_c,
            boundary="No motion while latched.", timeout=40,
        ))
    return out
