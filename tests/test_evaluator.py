"""Evaluator: synthetic traces → metrics → verdicts."""
from warp_av.scenario_engine.evaluator import compute_metrics, evaluate


def _state(speed, behavior="following_route", safety="ok", mission="executing", brake=0.0, steer=0.0, errors=None):
    return {"pose": {"speed": speed, "x": 0, "y": 0}, "behavior": behavior, "behavior_reason": f"r:{behavior}",
            "safety": {"state": safety}, "mission": {"state": mission}, "command": {"brake": brake, "steer": steer},
            "errors": errors or [], "warnings": []}


def _trace(speeds, **kw):
    return [{"t": 100.0 + i * 0.1, "state": _state(v, **kw), "actors": {"a": (10.0, 0.0)}, "ego": (i * 0.1, 0.0)} for i, v in enumerate(speeds)]


def test_stop_after_trigger_metrics():
    tr = _trace([8, 8, 8, 4, 1, 0.1, 0.0, 0.0], behavior="stopped_pedestrian", brake=1.0)
    m = compute_metrics(tr, {"trigger_time": 100.2, "collisions": []})
    assert m["stopped_within_s_of_trigger"] == 0.3
    assert m["time_to_first_brake_s"] == 0.0
    assert "stopped_pedestrian" in m["behaviors_seen"]
    assert m["min_distance_to_actor_m"] == 9.3
    assert m["collision_count"] == 0


def test_verdicts():
    sc = {"capability_status": "implemented",
          "pass_criteria": [{"metric": "collision_count", "op": "==", "value": 0},
                            {"metric": "behaviors_seen", "op": "contains", "value": "stopped_pedestrian"}],
          "fail_criteria": [{"metric": "collision_count", "op": ">", "value": 0}]}
    good = compute_metrics(_trace([5, 0, 0], behavior="stopped_pedestrian"), {"collisions": []})
    assert evaluate(sc, good)["verdict"] == "PASS"
    bad = compute_metrics(_trace([5, 5, 5]), {"collisions": []})
    assert evaluate(sc, bad)["verdict"] == "FAIL"
    sc2 = dict(sc, capability_status="not_implemented")
    assert evaluate(sc2, bad)["verdict"] == "GAP"
    crash = compute_metrics(_trace([5, 0]), {"collisions": [{"other": "walker"}]})
    assert evaluate(sc, crash)["verdict"] == "FAIL"
    assert evaluate(sc, good, runner_error="carla down")["verdict"] == "ERROR"


def test_not_in_on_list_metric():
    sc = {"capability_status": "implemented", "fail_criteria": [],
          "pass_criteria": [{"metric": "behaviors_seen", "op": "not_in", "value": ["stopped_vehicle"]}]}
    m = compute_metrics(_trace([5, 5], behavior="following_route"), {"collisions": []})
    assert evaluate(sc, m)["verdict"] == "PASS"
    m = compute_metrics(_trace([5, 0], behavior="stopped_vehicle"), {"collisions": []})
    assert evaluate(sc, m)["verdict"] == "FAIL"


def test_safety_reaction_and_resume():
    tr = _trace([8, 8, 8, 2, 0, 0, 0, 3, 6], safety="ok")
    for s in tr[3:7]:
        s["state"]["safety"]["state"] = "intervention"
    m = compute_metrics(tr, {"first_fault_time": 100.25, "clear_time": 100.2, "collisions": []})
    assert m["safety_reaction_time_s"] == 0.05
    assert m["resumed_after_clear"] is True
    assert "intervention" in m["safety_states_seen"]


def test_route_deviation():
    tr = _trace([5] * 5)
    m = compute_metrics(tr, {"route_xy": [(0, 1.0), (10, 1.0)], "collisions": []})
    assert m["route_deviation_max_m"] == 1.0
