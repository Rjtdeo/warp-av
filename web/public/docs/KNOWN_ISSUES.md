# Known Issues

## Perception Uses Ground Truth

Perception currently reads CARLA's actor list (world.get_actors()) instead of processing camera/lidar data with ML models. This means:
- Detection is perfect in simulation but wouldn't work with real sensors
- No camera-based object detection (YOLO etc.) is integrated yet
- This is an intentional shortcut to get the full pipeline working first
- Next step: run YOLOv8 on camera images for real perception

## No Lane Detection

The vehicle follows CARLA's route waypoints but does not detect lane markings visually. It relies on the map graph for staying in lane.

## Simple Steering Controller

Pure pursuit steering works but oscillates at higher speeds. A Stanley controller or MPC would be more stable. PID speed control also needs tuning.

## No Re-planning

If the route is blocked, the vehicle stops but does not re-plan around the obstacle. It waits until the path is clear.

## Single Camera Only

Only a front-facing camera is used. No rear, side, or surround view. Blind spots exist.

## No Sensor Fusion

Camera, lidar, GNSS, and IMU data are not fused. Each is used independently. Real system would need an EKF or similar.

## Console Is Polling-Based

Console polls the API every 200ms instead of using WebSocket push. Works but adds latency.

## Localization Is Perfect

In simulation, localization reads CARLA's ground truth position. With real sensors, this would need GNSS+IMU+odometry fusion with uncertainty estimation.

## Tests
`tests/` has 22 pytest cases (catalog integrity, evaluator, safety supervisor, behaviour priority, command validation, fault injector) that run without CARLA. Scenario execution (`scenarios/run_scenario.py`) needs a live CARLA + stack and is not in CI.

## Hardcoded CARLA Settings

CARLA host/port, vehicle type, sensor configurations are partially hardcoded. Should be fully configurable.

## No Map Boundaries

Vehicle doesn't know its allowed operating area. Safety supervisor should check geofencing.

## Technical Debt

- Error handling is minimal in some components
- No graceful recovery from component crashes (just stops)
- Logger doesn't handle disk-full scenarios
- No unit tests for individual components

## Gaps surfaced by the 1000-scenario catalog (see docs/SCENARIO_STRATEGY.md §3)

- **Sensor health not wired to the safety supervisor.** `CarlaSensorAdapter` tracks camera/lidar/gnss/imu staleness but `SafetySupervisor.update()` never receives it; in ground-truth perception mode a dead camera does not stop the vehicle. 50 `sensor_degradation` + 9 `cf_disable` scenarios are `not_implemented` for this reason.
- **No geofence / ODD enforcement** (40 `odd_boundary` scenarios define the contract).
- **Traffic lights / signs ignored** (30 `traffic_control` scenarios; will run a red).
- **No car-following** → stop/go oscillation behind a slow lead (`va_slow_lead`, `br_traffic_jam`).
- **No re-plan / mission-failure timeout on a persistent block.**
- **Straight ego-frame in-path box** → late detection around curves (`so_after_curve`).
- **Supervisor reports only the first failed check** → second simultaneous fault invisible (`cf_double_failure`).
- **Recovery policy undefined**: a component coming back auto-resumes motion (`cf_recover`, `cf_flapping`). Needs a decision + hysteresis.
- **Wall-clock time (`time.time()`) everywhere** → an NTP/GNSS clock step trips every staleness check.
- **Tie-breaking between a pedestrian and an obstacle at equal range depends on iteration order.**
- **`get_next_waypoint` is O(n) per tick** — fine at 200 waypoints, noticeable at 2 km routes.
- **Fault hooks exist only on ground-truth perception**; `camera_lidar_perception` has no `inject_fault` yet (injector returns `success: false`).

## Scenario runner limitations
- Does not switch CARLA towns; town mismatch is recorded as a warning. Run per town or restart the stack in the right town.
- Cut-in / cut-out are open-loop steering pulses (not lane-accurate); weave is ignored.
- CARLA has no animals; a bin prop stands in. No pallet prop; a plant pot stands in.
- `noise`/`latency` on raw sensors are accepted and logged but inert until perception/localization consume raw sensors.
- Ego vehicle is found as "the vehicle nearest the API pose" because the adapter does not set a role_name.
- The full 1000 have **not** been executed end-to-end yet; the catalog is validated structurally and by dry-run, the runner by unit tests of its evaluator. Expect first-run issues around actor spawn collisions on narrow roads.

## Fixed while building the catalog
- A tick exception used to leave the last throttle command applied; now → brake command + `tick_error` log event + surfaced in `/api/state.last_tick_error`.
- Vehicle adapter clamped NaN/inf and accepted stale commands; now rejects non-finite or >0.5 s-old commands and brakes (this is what a physical DBW gateway must do).
- `/api/mission/start` validated (400 on malformed / out-of-range input).
