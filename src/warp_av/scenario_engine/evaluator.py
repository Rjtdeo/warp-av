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


def compute_metrics(trace: List[dict], meta: dict) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    if not trace:
        return {"elapsed_s": meta.get("elapsed_s", 0.0), "collision_count": len(meta.get("collisions", []))}
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
    m["mission_completed"] = bool(meta.get("mission_completed", False) or
                                  any(s.get("mission", {}).get("state") == "completed" for s in states) or
                                  "mission_complete" in behaviors)
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
