"""
Evaluator: telemetry trace → metrics → verdict.

A trace is a list of samples:
    {"t": float, "state": <dict from GET /api/state>, "actors": {name: (x, y)}, "ego": (x, y)}
plus runner metadata: trigger_time, event_times, collisions, route_xy, elapsed.

Verdicts
    PASS   all pass_criteria hold and no fail_criteria holds
    FAIL   a fail_criterion holds, or a pass criterion fails on an `implemented` scenario
    GAP    pass criteria fail but capability_status != implemented (documented gap, not a regression)
    ERROR  the runner could not execute the scenario (CARLA down, spawn failed, API unreachable)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .evidence import box_gap


def _op(op: str, actual, expected) -> bool:
    try:
        if op == "==":
            return actual == expected
        if op == "!=":
            return actual != expected
        if actual is None:
            return False
        if op == ">=":
            return actual >= expected
        if op == "<=":
            return actual <= expected
        if op == ">":
            return actual > expected
        if op == "<":
            return actual < expected
        if op == "in":
            return actual in expected
        if op == "not_in":
            # for list-valued metrics: none of the expected items appear
            if isinstance(actual, (list, tuple, set)):
                return not any(e in actual for e in expected)
            return actual not in expected
        if op == "contains":
            if isinstance(actual, (list, tuple, set)):
                if isinstance(expected, str):
                    return any(expected in str(a) for a in actual) if any(isinstance(a, str) for a in actual) else expected in actual
                return expected in actual
            return expected in str(actual)
    except TypeError:
        return False
    return False


def _dist_point_to_polyline(px, py, pts) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best if best != float("inf") else 0.0


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    v = sorted(vals)
    k = max(0, min(len(v) - 1, int(round(q * (len(v) - 1)))))
    return v[k]


def _v1_metrics(trace: List[dict], states: List[dict], ts: List[float], speeds: List[float],
                behaviors: List[str], safety_states: List[str]) -> Dict[str, Any]:
    """V1 evidence: numbers taken from CARLA truth and the van's own timing, all optional.
    Every field is None when the trace does not carry what it needs."""
    m: Dict[str, Any] = {}
    egos = [s.get("ego") for s in trace]
    m["distance_m"] = round(sum(math.dist(a, b) for a, b in zip(egos, egos[1:])
                                if a and b and a[0] is not None and b[0] is not None), 1)
    truth_speeds = [float(s["ego_speed"]) for s in trace if s.get("ego_speed") is not None]
    m["max_speed_truth_mps"] = round(max(truth_speeds), 2) if truth_speeds else None
    for part in ("api", "truth", "actors"):
        vals = [float(s["poll_ms"][part]) for s in trace if isinstance(s.get("poll_ms"), dict) and s["poll_ms"].get(part) is not None]
        m[f"poll_{part}_ms_median"] = round(_percentile(vals, 0.5), 1) if vals else None
        m[f"poll_{part}_ms_max"] = round(max(vals), 1) if vals else None

    hz = [float(s["loop_hz"]) for s in states if s.get("loop_hz")]
    m["loop_hz_mean"] = round(sum(hz) / len(hz), 2) if hz else None
    m["loop_hz_min"] = round(min(hz), 2) if hz else None
    m["loop_hz_p05"] = round(_percentile(hz, 0.05), 2) if hz else None
    tick = [float(s["tick_ms"]) for s in states if s.get("tick_ms") is not None]
    m["tick_ms_mean"] = round(sum(tick) / len(tick), 1) if tick else None
    m["tick_ms_max"] = round(max(tick), 1) if tick else None
    m["poll_hz_actual"] = round((len(ts) - 1) / (ts[-1] - ts[0]), 2) if len(ts) > 1 and ts[-1] > ts[0] else None

    counts = [s["collision"]["count"] for s in states if isinstance(s.get("collision"), dict) and s["collision"].get("count") is not None]
    m["stack_collision_count"] = (counts[-1] - counts[0]) if counts else None

    near = [s["nearest_any"] for s in trace if isinstance(s.get("nearest_any"), dict) and s["nearest_any"].get("m") is not None]
    if near:
        best = min(near, key=lambda n: n["m"])
        m["min_distance_any_actor_m"] = round(best["m"], 2)
        m["nearest_any_actor_type"] = best.get("type")
    else:
        m["min_distance_any_actor_m"] = None
        m["nearest_any_actor_type"] = None

    closest = []
    for s in states:
        c = (s.get("perception") or {}).get("closest_distance")
        if isinstance(c, (int, float)) and 0 <= c < 900:
            closest.append(float(c))
    m["min_perception_closest_m"] = round(min(closest), 1) if closest else None

    gap = None
    for s in trace:
        eb, boxes = s.get("ego_box"), s.get("boxes")
        if not eb or not boxes:
            continue
        for b in boxes.values():
            g = box_gap(eb, b)
            gap = g if gap is None else min(gap, g)
    m["min_gap_to_actor_m"] = round(gap, 2) if gap is not None else None

    preasons = [((s.get("planner") or {}).get("reason") or "") for s in states]
    m["planner_reasons_seen"] = sorted(set(r for r in preasons if r))
    m["planner_reason_final"] = next((r for r in reversed(preasons) if r), None)
    m["behavior_final"] = behaviors[-1] if behaviors else None
    m["safety_final"] = safety_states[-1] if safety_states else None
    m["safety_reason_final"] = (states[-1].get("safety") or {}).get("reason") if states else None

    # movement while the light affecting the ego was red (CARLA's own colour, not the camera's)
    moved, run_len, seen = 0.0, 0.0, set()
    prev = None
    for s in trace:
        light = s.get("light") or {}
        st_ = light.get("affected_state") or light.get("forced_state")
        if st_:
            seen.add(st_)
        at_red = light.get("affected_state") == "Red"
        if at_red and prev is not None and prev.get("light") and prev["light"].get("affected_state") == "Red":
            a, b = prev.get("ego"), s.get("ego")
            if a and b and a[0] is not None and b[0] is not None:
                run_len += math.dist(a, b)
        elif not at_red:
            moved = max(moved, run_len)
            run_len = 0.0
        prev = s
    m["moved_at_red_m"] = round(max(moved, run_len), 1)
    m["light_states_seen"] = sorted(seen)
    stops = [s["light"]["forced_stop_m"] for s in trace if isinstance(s.get("light"), dict) and s["light"].get("forced_stop_m") is not None]
    m["forced_light_min_stop_m"] = round(min(stops), 1) if stops else None
    return m


def compute_metrics(trace: List[dict], meta: dict) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    if not trace:
        records = meta.get("mission_records") or []
        return {"elapsed_s": meta.get("elapsed_s", 0.0), "collision_count": len(meta.get("collisions", [])),
                "mission_completed": any(r.get("state") == "completed" for r in records),
                "mission_record_state": records[-1].get("state") if records else None,
                "mission_reason_ended": records[-1].get("reason_ended") if records else None}
    t0 = trace[0]["t"]
    states = [s["state"] for s in trace]
    speeds = [float(s.get("pose", {}).get("speed", 0.0)) for s in states]
    behaviors = [s.get("behavior", "") for s in states]
    safety_states = [s.get("safety", {}).get("state", "") for s in states]
    reasons = [s.get("behavior_reason", "") for s in states]
    steers = [float(s.get("command", {}).get("steer", 0.0)) for s in states]
    ts = [s["t"] for s in trace]

    m["elapsed_s"] = round(meta.get("elapsed_s", ts[-1] - t0), 2)
    m["collision_count"] = len(meta.get("collisions", []))
    m["final_mission_state"] = states[-1].get("mission", {}).get("state", "idle")
    records = meta.get("mission_records") or []
    m["mission_completed"] = bool(meta.get("mission_completed", False) or
                                  any(s.get("mission", {}).get("state") == "completed" for s in states) or
                                  "mission_complete" in behaviors or
                                  any(r.get("state") == "completed" for r in records))
    m["mission_reason_ended"] = records[-1].get("reason_ended") if records else None
    m["mission_record_state"] = records[-1].get("state") if records else None
    m["behaviors_seen"] = sorted(set(b for b in behaviors if b))
    m["safety_states_seen"] = sorted(set(x for x in safety_states if x))
    m["behavior_reasons_seen"] = sorted(set(r for r in reasons if r))
    m["max_speed_mps"] = round(max(speeds), 2)
    m["mean_speed_mps"] = round(sum(speeds) / len(speeds), 2)
    m["final_speed_mps"] = round(speeds[-1], 2)
    m["max_abs_steer"] = round(max(abs(x) for x in steers), 3)
    sign_changes = sum(1 for a, b in zip(steers, steers[1:]) if a * b < 0 and abs(a - b) > 0.05)
    m["steer_oscillation_index"] = round(sign_changes / max(1e-6, ts[-1] - t0), 2)
    errs, warns = set(), set()
    for s in states:
        errs.update(s.get("errors", []) or [])
        warns.update(s.get("warnings", []) or [])
    m["errors_seen"] = sorted(errs)
    m["warnings_seen"] = sorted(warns)
    m["tick_gap_max_s"] = round(max([b - a for a, b in zip(ts, ts[1:])] or [0.0]), 3)

    # longest continuous stop
    longest, cur, prev_t = 0.0, 0.0, None
    for t, v in zip(ts, speeds):
        if v < 0.2:
            cur += (t - prev_t) if prev_t is not None else 0.0
            longest = max(longest, cur)
        else:
            cur = 0.0
        prev_t = t
    m["stop_duration_s"] = round(longest, 2)

    # distance to scenario actors
    md = float("inf")
    for s in trace:
        ex, ey = s.get("ego", (None, None))
        if ex is None:
            continue
        for name, (ax, ay) in s.get("actors", {}).items():
            md = min(md, math.hypot(ex - ax, ey - ay))
    m["min_distance_to_actor_m"] = round(md, 2) if md != float("inf") else None

    # route deviation
    route = meta.get("route_xy") or []
    if len(route) >= 2:
        devs = [_dist_point_to_polyline(s["ego"][0], s["ego"][1], route) for s in trace if s.get("ego") and s["ego"][0] is not None]
        m["route_deviation_max_m"] = round(max(devs), 2) if devs else None
        m["out_of_route_time_s"] = round(sum((b - a) for a, b, d in zip(ts, ts[1:], devs[1:]) if d > 2.5), 2) if devs else 0.0
    else:
        m["route_deviation_max_m"] = None
        m["out_of_route_time_s"] = 0.0

    # trigger-relative timings
    trig = meta.get("trigger_time")
    m["speed_at_trigger_mps"] = None
    m["stopped_within_s_of_trigger"] = None
    m["time_to_first_brake_s"] = None
    if trig is not None:
        after = [(t, v, s) for t, v, s in zip(ts, speeds, states) if t >= trig]
        before = [v for t, v in zip(ts, speeds) if t <= trig]
        m["speed_at_trigger_mps"] = round(before[-1], 2) if before else None
        for t, v, s in after:
            if v < 0.2:
                m["stopped_within_s_of_trigger"] = round(t - trig, 2)
                break
        for t, v, s in after:
            if float(s.get("command", {}).get("brake", 0.0)) > 0.3:
                m["time_to_first_brake_s"] = round(t - trig, 2)
                break

    # fault → safety reaction
    fault_t = meta.get("first_fault_time")
    m["safety_reaction_time_s"] = None
    if fault_t is not None:
        for t, ss in zip(ts, safety_states):
            if t >= fault_t and ss not in ("ok", ""):
                m["safety_reaction_time_s"] = round(t - fault_t, 2)
                break

    # resumed after clear/resume/enable
    clear_t = meta.get("clear_time")
    m["resumed_after_clear"] = False
    if clear_t is not None:
        m["resumed_after_clear"] = any(v > 1.0 for t, v in zip(ts, speeds) if t > clear_t + 0.5)

    # V1 evidence: CARLA truth and stack timing, all optional
    m.update(_v1_metrics(trace, states, ts, speeds, behaviors, safety_states))
    goal = meta.get("goal_xy")
    last = trace[-1].get("ego")
    m["final_goal_distance_m"] = (round(math.dist(last, goal), 1) if goal and last and last[0] is not None else None)
    # V1.5: a scenario can PASS its own criteria and still end with the van frozen short of the
    # goal (WAV-0281/0289 sat "stopped_blocked" for the last 9 s of a 60 s timeout). Say so.
    tail = [s for s in trace if s["t"] >= ts[-1] - 3.0]
    still = tail and all(float(s["state"].get("pose", {}).get("speed") or 0.0) < 0.3 for s in tail)
    stopped_kinds = ("stopped_blocked", "stopped_obstacle", "stopped_vehicle", "stopped_pedestrian", "stopped_safety")
    m["stuck_at_end"] = bool(still and m["final_mission_state"] in ("executing", "paused")
                             and (m.get("behavior_final") in stopped_kinds)
                             and (m["final_goal_distance_m"] is None or m["final_goal_distance_m"] > 10.0))
    m["stuck_at_end_s"] = None
    if m["stuck_at_end"]:
        k = len(trace) - 1
        while k > 0 and float(trace[k]["state"].get("pose", {}).get("speed") or 0.0) < 0.3:
            k -= 1
        m["stuck_at_end_s"] = round(ts[-1] - ts[k], 1)
    return m


def evaluate(scenario: dict, metrics: Dict[str, Any], runner_error: Optional[str] = None) -> dict:
    if runner_error:
        return {"verdict": "ERROR", "reason": runner_error, "checks": []}
    checks = []
    failed_fail = False
    for c in scenario.get("fail_criteria", []):
        ok = _op(c["op"], metrics.get(c["metric"]), c["value"])
        checks.append({"kind": "fail", "metric": c["metric"], "op": c["op"], "expected": c["value"],
                       "actual": metrics.get(c["metric"]), "triggered": ok})
        failed_fail = failed_fail or ok
    all_pass = True
    for c in scenario.get("pass_criteria", []):
        ok = _op(c["op"], metrics.get(c["metric"]), c["value"])
        checks.append({"kind": "pass", "metric": c["metric"], "op": c["op"], "expected": c["value"],
                       "actual": metrics.get(c["metric"]), "passed": ok})
        all_pass = all_pass and ok
    if failed_fail:
        verdict, reason = "FAIL", "fail criterion triggered: " + ", ".join(c["metric"] for c in checks if c.get("triggered"))
    elif all_pass:
        verdict, reason = "PASS", "all pass criteria met"
    elif scenario.get("capability_status") != "implemented":
        verdict, reason = "GAP", "pass criteria unmet on a %s capability: %s" % (
            scenario.get("capability_status"), ", ".join(c["metric"] for c in checks if c.get("kind") == "pass" and not c.get("passed")))
    else:
        verdict, reason = "FAIL", "pass criteria unmet: " + ", ".join(c["metric"] for c in checks if c.get("kind") == "pass" and not c.get("passed"))
    return {"verdict": verdict, "reason": reason, "checks": checks}
