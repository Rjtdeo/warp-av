# Scenarios

Two layers live here:

| Layer | What | Where |
|---|---|---|
| **Scenario catalog** | 1000 declarative, reproducible scenarios (YAML) across 18 categories, generated deterministically from ~80 parameterised families | `catalog/` (+ `CATALOG.md`, `index.json`) |
| **Scenario runner** | Executes any catalog scenario against the running CARLA + Warp AV stack through the *same operator API the console uses*, spawns actors along the planned route, injects faults, records a trace, evaluates pass/fail, writes results | `run_scenario.py`, `report.py`, engine in `src/warp_av/scenario_engine/` |

The three original hand-written scripts (`scenario_1_normal.py`, `scenario_2_vehicle.py`, `scenario_7_estop.py`) are kept as minimal, dependency-free examples; the catalog supersedes them.

---

## Quick start

```bash
# 0. (once) regenerate the catalog — deterministic, byte-identical on every run
python3 scenarios/generate_catalog.py

# 1. CARLA + autonomy stack must be running (see ../README.md)

# 2. run scenarios
python3 scenarios/run_scenario.py WAV-0001                   # one
python3 scenarios/run_scenario.py --category pedestrian      # 100 pedestrian scenarios
python3 scenarios/run_scenario.py --status implemented       # only what the stack should pass today (456)
python3 scenarios/run_scenario.py --tag estop --limit 5
python3 scenarios/run_scenario.py --all                      # full 1000 (several hours)

# 3. aggregate
python3 scenarios/report.py                                  # -> scenarios/results/REPORT.md
```

No CARLA on this machine? `--dry-run` prints the execution plan for any scenario, and `pytest tests/` validates the whole catalog + evaluator offline.

Results land in `scenarios/results/<ID>.json` (verdict, metrics, per-criterion checks, behaviour timeline, event log, collisions) and `<ID>.trace.jsonl` (the raw 10 Hz `/api/state` samples + actor positions — enough to replay the run).

---

## The 7 required scenarios → catalog

| Assignment scenario | Catalog category | Count | Canonical example |
|---|---|---:|---|
| 1 Normal mission | `normal_mission` | 60 | WAV-0001 |
| 2 Vehicle ahead | `vehicle_ahead` | 110 | `va_stopped_lead` family (WAV-0061…) |
| 3 Pedestrian | `pedestrian` | 100 | `ped_cross` family (WAV-0171…) |
| 4 Static obstacle | `static_obstacle` | 90 | `so_single_prop` (WAV-0271…) |
| 5 Blocked route | `blocked_route` | 70 | `br_full_block_persistent` (WAV-0361…) |
| 6 Component failure | `component_failure` | 110 | `cf_disable` (WAV-0431…) |
| 7 Emergency stop | `emergency_stop` | 60 | `es_at_speed` (WAV-0541…) |

Plus the road-readiness extensions: `operator_action` 60, `sensor_degradation` 50, `localization_degradation` 40, `odd_boundary` 40, `traffic_control` 30, `road_geometry` 30, `timing_latency` 30, `vulnerable_road_user` 40, `compound` 30, `edge_case` 30, `endurance` 20. Exact IDs per family: `catalog/CATALOG.md`.

---

## Honesty flag: `capability_status`

Every scenario says whether the **current** stack is expected to handle it:

| status | count | meaning | runner verdict when criteria unmet |
|---|---:|---|---|
| `implemented` | 456 | stack should pass today | **FAIL** (regression) |
| `partial` | 383 | stack reacts, but not with the target behaviour (e.g. stops instead of following, no re-plan) | **GAP** |
| `not_implemented` | 161 | defines a contract the stack does not meet yet (traffic lights, geofence, sensor-health wiring, VRU class, ODD enforcement) | **GAP** |

A collision is always a FAIL regardless of status. See `docs/SCENARIO_STRATEGY.md` for *why* each gap matters for road readiness and which ones block the next milestone.

---

## Scenario format

```yaml
id: WAV-0200
name: Child darts from behind parked van at 30 m
category: pedestrian            # one of 18
family: ped_occluded_dart       # generator template
capability_status: partial
description: |
  prose: what happens, what the stack does today, what the target is
tags: [child, dart_out, occlusion, high_severity, day]
required_capabilities: [pedestrian_detection, occlusion_reasoning, emergency_braking]
odd: {town: Town10HD_Opt, weather: ClearNoon, light: day, road_type: urban_parked_cars, speed_limit_mps: 6.0}
mission:
  origin: {mode: ego_current}
  destination: {mode: route_ahead, distance_m: 130}   # or spawn_point/index, xy, off_map
  cruise_speed_mps: 6.0
  start_at_s: 0.0              # >0 lets events fire BEFORE the mission starts
actors:                         # vehicle | pedestrian | prop
  - name: parked
    type: vehicle
    blueprint: vehicle.volkswagen.t2
    spawn: {mode: route_ahead, distance_m: 30, lateral_m: 3.2}   # along the PLANNED ROUTE, +lateral = right
    behavior: {kind: stopped}
  - name: ped
    type: pedestrian
    blueprint: walker.pedestrian.0011
    spawn: {mode: route_ahead, distance_m: 31.5, lateral_m: 5.5}
    behavior: {kind: dart_out, speed_mps: 3.0}
    trigger: {ego_within_m: 35}                                   # at_s | ego_within_m | ego_speed_gt | on_behavior | on_mission_state | after_event
events:                         # things the runner does through the operator API
  - trigger: {ego_speed_gt: 5.0}
    action: inject              # estop | estop_clear | pause | resume | stop_mission | start_mission | change_destination | set_speed_limit | inject | set_weather | wait
    params: {component: perception, action: disable}
expected: {behavior: stopped_pedestrian}
pass_criteria:                  # machine-checked metrics
  - {metric: collision_count, op: "==", value: 0}
  - {metric: behaviors_seen, op: contains, value: stopped_pedestrian}
  - {metric: min_distance_to_actor_m, op: ">=", value: 1.0}
fail_criteria:
  - {metric: collision_count, op: ">", value: 0}
safety_boundary: No contact with pedestrian.
data_to_collect: [pose, speed, behavior, ...]
timeout_s: 60
```

Actor behaviours the runner drives: `stopped, constant_speed, brake_hard, cut_in, cut_out, autopilot, oncoming, reverse, cross_road, walk_along, stand, cross_and_stop, dart_out, appear, remove_after`.
Fault injections (`/api/test/inject`): `perception|localization: disable enable freeze stale latency low_confidence noise crash`, `camera|lidar|gnss|imu: disable enable drop`, `controller: disable enable nan_command stale`, `planner: disable enable`, `vehicle_connection: disable enable freeze`, `tick_latency: latency`.

Metrics the evaluator computes: `collision_count, min_distance_to_actor_m, final_mission_state, mission_completed, behaviors_seen, safety_states_seen, behavior_reasons_seen, stopped_within_s_of_trigger, time_to_first_brake_s, max/mean/final_speed_mps, speed_at_trigger_mps, max_abs_steer, steer_oscillation_index, route_deviation_max_m, stop_duration_s, resumed_after_clear, errors_seen, warnings_seen, elapsed_s, safety_reaction_time_s, tick_gap_max_s, out_of_route_time_s`.

Full schema & validation: `src/warp_av/scenario_engine/schema.py`.

---

## How the runner works

```
run_scenario.py ─▶ Catalog.load(id) ─▶ ScenarioRunner.run()
                                          │  POST /api/estop/clear, /api/mission/stop, inject enable*   (clean slate)
                                          │  CARLA: set weather
                                          │  CARLA: spawn untriggered actors (relative to ego lane)
                                          │  POST /api/mission/start {x,y}   (destination resolved: route_ahead / spawn_point / xy)
                                          │  GET  /api/route  → anchor route-relative spawns along the ACTUAL planned route
                                          │  loop @10 Hz: GET /api/state, record ego+actor positions,
                                          │       step actor behaviours, evaluate triggers, fire events via API,
                                          │       stop at timeout or 2 s after terminal mission state
                                          │  CARLA: collision sensor on ego → collisions[]
                                          │  cleanup actors, re-enable all components
                                          ▼
                               evaluator: trace → metrics → PASS / FAIL / GAP / ERROR  → results/<ID>.json
```

Known runner limitations (also in `KNOWN_ISSUES.md`): town is not switched automatically (mismatch is recorded as a warning — restart the stack in the scenario's town, or filter `--town`); cut-in/cut-out are open-loop steering pulses, not lane-accurate; CARLA has no animals (small prop stands in); `noise`/`latency` on raw sensors are recorded but inert because perception/localization read ground truth today.
