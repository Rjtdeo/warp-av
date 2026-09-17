"""V1 evidence layer: run folders that are never overwritten, an honest flat row per scenario,
true box-to-box clearance, and the truth/timing metrics the evaluator adds. No CARLA needed."""
import json
import math
from pathlib import Path

import pytest

from warp_av.scenario_engine.evidence import box_gap, obb_corners, build_evidence, write_run_summary, V1_FIELDS
from warp_av.scenario_engine.evaluator import compute_metrics
from warp_av.scenario_engine.runner import ScenarioRunner, parse_reset_spec, Api


# ---------------------------------------------------------------- geometry
def test_box_gap_touching_and_apart():
    a = [0.0, 0.0, 0.0, 2.0, 1.0]            # 4 m long, 2 m wide van at the origin, facing +x
    assert box_gap(a, [6.0, 0.0, 0.0, 2.0, 1.0]) == pytest.approx(2.0)       # nose to tail: 6 - 2 - 2
    assert box_gap(a, [3.0, 0.0, 0.0, 2.0, 1.0]) == 0.0                      # overlapping
    assert box_gap(a, [0.0, 4.0, 0.0, 2.0, 1.0]) == pytest.approx(2.0)       # side by side: 4 - 1 - 1
    # a cone (0.3 x 0.3) 1 m beyond the front bumper, off to the side by 0.5 m: still 1 m ahead
    assert box_gap(a, [3.15, 0.5, 0.0, 0.15, 0.15]) == pytest.approx(1.0)


def test_box_gap_rotated():
    a = [0.0, 0.0, 0.0, 2.0, 1.0]
    # same car turned 90 degrees, centre 5 m ahead: its half-WIDTH (1 m) now faces us -> gap 5 - 2 - 1
    assert box_gap(a, [5.0, 0.0, math.pi / 2, 2.0, 1.0]) == pytest.approx(2.0)
    # corners of a 45-degree box land where trigonometry says
    c = obb_corners([0.0, 0.0, math.pi / 4, 1.0, 1.0])
    assert max(abs(x) for x, _ in c) == pytest.approx(math.sqrt(2))


# ---------------------------------------------------------------- metrics
def _state(speed, hz=5.0, tick_ms=150.0, coll=0, reason="following", closest=12.3):
    return {"pose": {"speed": speed, "x": 0, "y": 0}, "behavior": "following_route", "behavior_reason": "r",
            "safety": {"state": "ok", "reason": "fine"}, "mission": {"state": "executing"}, "command": {"brake": 0, "steer": 0},
            "errors": [], "warnings": [], "loop_hz": hz, "tick_ms": tick_ms, "collision": {"count": coll},
            "planner": {"reason": reason}, "perception": {"closest_distance": closest}}


def test_v1_metrics_from_truth():
    trace = []
    for i in range(6):
        x = 2.0 * i
        trace.append({"t": 100.0 + 0.1 * i, "state": _state(5.0, hz=4.0 + i, tick_ms=100 + 10 * i, coll=1 if i >= 4 else 0,
                                                              reason="following" if i < 5 else "blocked by prop"),
                      "actors": {"obs": (20.0, 0.0)}, "ego": (x, 0.0), "ego_speed": 5.0,
                      "ego_box": [x, 0.0, 0.0, 2.0, 1.0], "boxes": {"obs": [20.0, 0.0, 0.0, 0.5, 0.5]},
                      "nearest_any": {"m": 20.0 - x, "type": "static.prop.barrel", "id": 9},
                      "light": {"affected_state": "Red", "forced_state": "Red", "forced_stop_m": 30.0 - x} if i >= 2 else None})
    for i, s in enumerate(trace):
        s["poll_ms"] = {"api": 40.0 + i, "truth": 2000.0, "actors": 1.0}
    m = compute_metrics(trace, {"collisions": [], "goal_xy": (13.0, 4.0),
                                "mission_records": [{"mission_id": "mission_0001", "state": "completed", "reason_ended": "Arrived at destination"}]})
    assert m["distance_m"] == 10.0                       # five steps of 2 m along CARLA truth
    assert m["mission_completed"] is True                # the stack's own record says so, even though /api/state never showed it
    assert m["mission_record_state"] == "completed" and m["mission_reason_ended"] == "Arrived at destination"
    assert m["final_goal_distance_m"] == 5.0             # last truth pose (10, 0) to the goal (13, 4)
    assert m["poll_api_ms_median"] == 42.0 and m["poll_truth_ms_max"] == 2000.0 and m["poll_actors_ms_median"] == 1.0
    assert m["loop_hz_mean"] == pytest.approx(6.5) and m["loop_hz_min"] == 4.0 and m["loop_hz_p05"] == 4.0
    assert m["tick_ms_max"] == 150.0
    assert m["stack_collision_count"] == 1                # the van's own counter went 0 -> 1
    assert m["collision_count"] == 0                      # the runner's sensor saw nothing: the two disagree, on purpose
    assert m["min_distance_any_actor_m"] == 10.0 and m["nearest_any_actor_type"] == "static.prop.barrel"
    assert m["min_gap_to_actor_m"] == pytest.approx(20.0 - 0.5 - (10.0 + 2.0))   # true clearance, not centre distance
    assert m["min_distance_to_actor_m"] == 10.0
    assert m["min_perception_closest_m"] == 12.3
    assert m["planner_reason_final"] == "blocked by prop" and m["planner_reasons_seen"] == ["blocked by prop", "following"]
    assert m["moved_at_red_m"] == pytest.approx(6.0)     # moved 3 steps while CARLA said the light was red
    assert m["light_states_seen"] == ["Red"] and m["forced_light_min_stop_m"] == 20.0
    assert m["poll_hz_actual"] == 10.0


def test_v1_metrics_are_optional():
    trace = [{"t": 1.0, "state": {"pose": {"speed": 0}}, "actors": {}, "ego": (0.0, 0.0)},
             {"t": 1.1, "state": {"pose": {"speed": 0}}, "actors": {}, "ego": (0.0, 0.0)}]
    m = compute_metrics(trace, {"collisions": []})
    for k in ("loop_hz_mean", "stack_collision_count", "min_distance_any_actor_m", "min_gap_to_actor_m",
              "min_perception_closest_m", "planner_reason_final", "forced_light_min_stop_m", "final_goal_distance_m",
              "poll_api_ms_median", "mission_record_state"):
        assert m[k] is None, k
    assert m["mission_completed"] is False
    # an empty trace still reports what the mission record says
    m0 = compute_metrics([], {"collisions": [], "mission_records": [{"state": "failed", "reason_ended": "no route"}]})
    assert m0["mission_completed"] is False and m0["mission_record_state"] == "failed"
    assert m["distance_m"] == 0.0 and m["moved_at_red_m"] == 0.0


# ---------------------------------------------------------------- the flat row + persistence
def _result(sid="WAV-0001", verdict="PASS"):
    return {"scenario_id": sid, "name": "n", "category": "normal_mission", "family": "nm_basic", "capability_status": "implemented",
            "verdict": verdict, "reason": "all pass criteria met", "checks": [], "warnings": ["town mismatch"], "event_log": [],
            "collisions": [], "started_at": 1700000000.0, "elapsed_s": 12.3, "trace_len": 2, "behaviors_timeline": [],
            "run_id": "r", "runner_error": None,
            "metrics": {"elapsed_s": 12.3, "distance_m": 80.4, "mission_completed": True, "final_mission_state": "completed",
                        "collision_count": 0, "stack_collision_count": 0, "min_gap_to_actor_m": 1.234, "loop_hz_mean": 5.21,
                        "loop_hz_min": 3.9, "behavior_final": "mission_complete", "safety_final": "ok", "behaviors_seen": ["a", "b"],
                        "planner_reason_final": "clear road"},
            "meta": {"run_id": "r", "git_sha_runner": "abc1234def", "git_dirty": False, "seed": None, "map": "Town10HD",
                     "map_expected": "Town03", "carla_version": {"server": "0.9.15", "client": "0.9.15"},
                     "started_at": "2026-09-17T10:00:00", "start_xy": (1.0, 2.0), "goal_xy": (3.0, 4.0),
                     "stack": {"git_sha": "abc1234", "perception_mode": "camera_lidar"},
                     "start_check": {"clean": True, "problems": []}, "ego_reset": {"settled_off_m": 0.12}}}


def test_build_evidence_has_every_v1_field():
    row = build_evidence(_result())
    assert list(row) == V1_FIELDS
    assert row["git_sha_stack"] == "abc1234" and row["map"] == "Town10HD" and row["map_expected"] == "Town03"
    assert row["min_gap_to_actor_m"] == 1.23 and row["completed"] is True and row["start_clean"] is True
    assert "mission_record_state" in row and "final_goal_distance_m" in row and "poll_api_ms_median" in row
    assert row["seed"] is None                            # no seed handling in the engine: say so, do not invent one
    assert build_evidence({"scenario_id": "x"})["loop_hz_mean"] is None


def test_dry_run_touches_no_disk(tmp_path):
    root = tmp_path / "results"
    r = ScenarioRunner(results_dir=root, dry_run=True, verbose=False)
    out = r.run({"id": "WAV-0001", "name": "n", "category": "c", "capability_status": "implemented",
                 "odd": {"weather": "ClearNoon", "town": "Town03"}, "mission": {}, "actors": [], "events": [],
                 "pass_criteria": [], "timeout_s": 1})
    assert out["verdict"] == "DRY_RUN"
    assert not root.exists()


def test_persist_never_overwrites_and_summarises(tmp_path):
    root = tmp_path / "results"
    r = ScenarioRunner(results_dir=root, dry_run=True, verbose=False, run_id="v1_test", seed=None)
    r.planned_ids = ["WAV-0001", "WAV-0001"]
    trace = [{"t": 1.0, "state": {"pose": {"speed": 0}}, "actors": {}, "ego": (0.0, 0.0)}]
    p1 = r._persist("WAV-0001", _result(), trace)
    p2 = r._persist("WAV-0001", _result(verdict="FAIL"), trace)
    run_dir = root / "runs" / "v1_test"
    assert p1["result"] == run_dir / "WAV-0001.json" and p2["result"] == run_dir / "WAV-0001__2.json"
    assert p2["trace"] == run_dir / "WAV-0001__2.trace.jsonl"
    assert json.loads(p1["result"].read_text(encoding="utf-8"))["verdict"] == "PASS"          # the first file survived the second write
    assert json.loads(p2["result"].read_text(encoding="utf-8"))["evidence"]["result_file"].endswith("WAV-0001__2.json")
    man = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert man["run_id"] == "v1_test" and man["planned_ids"] == ["WAV-0001", "WAV-0001"] and man["git_sha_stack"] == "abc1234"
    summary = (run_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert summary.count("| WAV-0001 |") == 2 and "PASS 1" in summary and "FAIL 1" in summary
    csv_lines = (run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].split(",") == V1_FIELDS and len(csv_lines) == 3


def test_start_check_reports_what_is_not_clean():
    r = ScenarioRunner(results_dir=Path("/nonexistent"), dry_run=True, verbose=False)
    ok = r._start_check({"active_faults": {}, "safety": {"state": "ok"}, "mission": {"state": "idle"}, "pose": {"speed": 0.0},
                         "collision": {"count": 3}})
    assert ok["clean"] and ok["stack_collisions_before"] == 3
    bad = r._start_check({"active_faults": {"camera": "disable"}, "safety": {"state": "intervention", "reason": "x"},
                          "mission": {"state": "executing"}, "pose": {"speed": 2.0}, "last_tick_error": "boom"})
    assert not bad["clean"] and len(bad["problems"]) == 5


def test_parse_reset_spec():
    assert parse_reset_spec(None) is None and parse_reset_spec("") is None
    assert parse_reset_spec("spawn:12") == {"mode": "spawn_point", "index": 12}
    assert parse_reset_spec("lane:-18,140.3") == {"mode": "lane", "x": -18.0, "y": 140.3}
    assert parse_reset_spec("1.5,-2,90") == {"mode": "absolute", "x": 1.5, "y": -2.0, "yaw_deg": 90.0}
    with pytest.raises(ValueError):
        parse_reset_spec("1,2,3,4")


def test_api_never_uses_the_name_localhost():
    # Windows tries ::1 first and pays ~2 s per call when the stack listens on IPv4 only
    assert Api("http://localhost:5000/").base == "http://127.0.0.1:5000"
    assert Api("http://192.168.1.102:5000").base == "http://192.168.1.102:5000"
