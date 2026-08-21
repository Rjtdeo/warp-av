# Known Issues

## Perception Uses Ground Truth

Perception currently reads CARLA's actor list (world.get_actors()) instead of processing camera/lidar data with ML models. This means:
- Detection is perfect in simulation but wouldn't work with real sensors
- No camera-based object detection (YOLO etc.) is integrated yet
- This is an intentional shortcut to get the full pipeline working first
- Next step: run YOLOv8 on camera images for real perception

## No Lane Detection

The vehicle follows CARLA's route waypoints but does not detect lane markings visually. It relies on the map graph for staying in lane.

## Safety Buffers (Troy #4, applied)

Stop trigger (perception.danger_distance) raised 5 → 8 m in BOTH perception implementations; slow-down zone (behavior.slow_distance) 15 → 20 m. Lateral path width intentionally left at 3.5 m — widening it makes the van stop for shoulder objects (parked cars, cones) and would flip the expected result of the path-box boundary scenarios; revisit together with route-aware corridor checking.

## Steering Controller (improved, needs CARLA validation)

Fix for the observed weave/brake-taps (Troy #5): speed-scaled lookahead (1.6 s of travel, 5–13 m, was fixed 5 m), speed-scheduled steering gain (1.5 at ≤3 m/s → 0.55 at ≥10 m/s, was fixed 1.5), low-pass + rate limit on steering, and a coast band so small speed overshoot lifts off instead of tapping the brakes. Covered by tests/test_controller_stability.py (kinematic bicycle model). NOTE: the original oscillation could not be reproduced in the lag-free offline model — the tuning is validated for stability offline, but the before/after weave comparison must be done in CARLA (steer_oscillation_index in nm_speed_sweep / rg_geometry scenarios). A Stanley controller or MPC is still the longer-term answer.

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

## Parking (Troy #7, implemented)

Missions now finish PULLED OVER at the right kerb: at planning time the route's last ~15 m are bent to a kerbside spot (real Parking/Shoulder lane if the map has one, else the right edge of the rightmost driving lane); final approach tapers to walking pace (`parking` behavior) and completion requires being within 1.5 m of the spot, nearly stopped, AND within 6 deg of the lane direction (visibly parallel) (was: 5 m anywhere on the road at any speed). Precision is logged ("Parked 0.4 m from the kerbside spot, heading off 3 deg"). **FIND PARKING (slot parking):** the operator button slices the bays near the destination into van-sized 7 m slots, checks each for occupancy (any other vehicle inside = taken), retargets the mission to the best FREE slot, draws all slots on the dashboard map (green = chosen, red = occupied, grey = free) and the completion line reports "INSIDE slot #N: YES/NO (margins)" using the van's real bounding box. Slots are only created on STRAIGHT bay sections (heading spread <8 deg across the slot) at least ~6 m from any junction — no tilted corner parking, and most building-entrance cuts are excluded since they cluster at junctions. Limitations: occupancy is checked once at button-press (not re-checked on approach); the map cannot see painted curb markings or mid-block driveways, so a software slot can still land on one — hand-annotated no-parking zones per street are the clean fix if a specific spot misbehaves.

The automatic (no-button) behaviour PREFERS a real stopping bay: it scans the last 40 m of the route for a Parking/Shoulder strip beyond the lane line and parks fully inside it, off the driving lane ("kind": "bay"). No bay on that street -> kerb-hug inside the rightmost lane (~0.55 m right on a 3.5 m lane — visually subtle but that is all the room a lane has). If the pin sits in a bend/junction it parks on the nearest straight stretch before it. Limitations: forward pull-over only (no reverse/parallel manoeuvres); if there is no straight stretch within 40 m it parks on the lane as before; box tolerance 1.5 m — tighten after CARLA validation.

## Gaps surfaced by the 1000-scenario catalog (see docs/SCENARIO_STRATEGY.md §3)

- **Sensor health not wired to the safety supervisor.** `CarlaSensorAdapter` tracks camera/lidar/gnss/imu staleness but `SafetySupervisor.update()` never receives it; in ground-truth perception mode a dead camera does not stop the vehicle. 50 `sensor_degradation` + 9 `cf_disable` scenarios are `not_implemented` for this reason.
- **No geofence / ODD enforcement** (40 `odd_boundary` scenarios define the contract).
- ~~Traffic lights ignored~~ **Implemented (ground-truth mode)**: perception reports the light governing our lane (red/yellow/green), behavior rolls up to the light's actual STOP LINE (distance from CARLA's stop waypoints, cached per light) and holds ~3 m from it; if the stop-line lookup fails it stops immediately where the light takes effect. Still stops for yellow always (dilemma-zone "proceed if too close" is now possible since we know the distance — not yet implemented). Camera mode reports "none" until a light classifier exists → unchanged behavior there. Signs (stop/yield/speed) still ignored. CARLA validation of `tc_traffic_light` pending.
- ~~No car-following~~ **Implemented (ground-truth mode)**: time-gap following (1.5 s + 8 m) behind moving vehicles — `behavior.FOLLOWING_VEHICLE`. In camera+LiDAR mode detections carry no speed (no tracking yet) so the van falls back to the old slow/stop behaviour; needs object tracking to enable following there. CARLA validation of `va_slow_lead` pending.
- **No re-plan / mission-failure timeout on a persistent block.**
- ~~Straight ego-frame in-path box~~ **Fixed when a route exists**: objects are judged against the ROUTE CORRIDOR (within 1.75 m of the planned polyline, ahead along the route) — a tilted mid-turn van no longer false-stops for vehicles in neighbouring lanes, and obstacles around a bend are seen at their true along-route distance (`so_after_curve` gap closed). The ego-frame box remains the fallback when no route is active. Caveat: trusts localization — fine in sim, needs a sanity bound with real sensors.
- **Junction give-way implemented (heuristic)**: before any turn at a junction the van pauses 1.5 s and waits while a moving vehicle is within 25 m (own-lane lead and parked cars excluded); after 12 s of blocked waiting it creeps at 2 m/s rather than deadlocking; the van first ROLLS UP to ~3 m from the crossing before starting its look-and-wait (it used to freeze 8-10 m early). Limitations: radius-based conflict check, no lane-level right-of-way (a vehicle driving AWAY on the crossing road still counts as a conflict → occasional over-cautious waits), creep-on-timeout is a pragmatic policy that needs review before real roads (`va_intersection_crossing` scenarios upgrade from not_implemented to partial).
- ~~Speed not curvature-aware~~ **Implemented**: `planner.curve_speed_cap` slows the van before/through bends (1.3 m/s² lateral comfort, gradual approach); appended to the behavior reason string when active.
- ~~Corner cutting (kerb/divider clipping)~~ **Improved**: aim point measured along the route arc (not straight-line), shorter aim distance in bends, a centreline-correction steering term (mini-Stanley, gain 0.12 capped ±0.3), and slew-limited pull-away after slow points. Offline model: worst lane-centre deviation through a tight 90° corner 1.25 m → 0.78 m (van body stays in lane). CARLA validation pending — if it still clips a specific corner, raise `VehicleController.CT_GAIN` slightly (0.12 → 0.15).
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
