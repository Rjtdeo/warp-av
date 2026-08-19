# Scenario Strategy — from 7 demo scenarios to road readiness

This document explains **why the 1000-scenario catalog is shaped the way it is**, what it already
tells us about the stack, and how to use it as the regression/"training" loop on the way to
closed-lot and then road testing.

The catalog is generated (`scenarios/generate_catalog.py`), not hand-written: ~80 parameterised
*families* × a deterministic parameter sweep. That matters for three reasons:

1. **Reproducible** — same seed ⇒ byte-identical YAML; a reviewer can diff a catalog change.
2. **Auditable** — every scenario carries `capability_status`, so "the stack fails 40 % of the
   catalog" is not a scary number, it is *"fails 0 % of implemented, and here are the 544 gaps we
   already know about, ranked"*.
3. **Extensible** — adding a family is ~30 lines; re-running the generator re-IDs nothing that
   already exists (families are appended).

---

## 1. Taxonomy and quotas

| # | Category | Qty | What it measures | Road-readiness question it answers |
|--:|---|--:|---|---|
| 1 | normal_mission | 60 | route following, speed tracking, arrival logic across towns/weather/speeds | Can it drive A→B at all, repeatably, and how does control quality degrade with speed? |
| 2 | vehicle_ahead | 110 | stationary / slow / hard-braking lead, cut-in, cut-out reveal, oncoming (false-positive), reversing, ambient traffic, queues, cross-traffic | Reaction latency & headway vs closing speed; does the "in-path box" heuristic hold up? |
| 3 | pedestrian | 100 | crossing speed/side/distance, occluded dart-out, standing at lane edge (path-box boundary), walk-along, stop-in-lane dwell, groups, child model, low-light regression pairs, crosswalk | Stop distance & reaction time for the most severe class; does it resume only when *all* are clear? |
| 4 | static_obstacle | 90 | prop class × lateral offset × distance, staggered multi, sudden appear (stopping-distance sweep), after-curve, low-profile vs large, at destination, low-vis pairs | What is the max safe cruise speed for our latency + braking? Where does the straight path box fail (curves)? |
| 5 | blocked_route | 70 | persistent full block, block that clears after N s, junction block, near-destination block, construction taper (partial vs full), jam (static/creeping), unreachable dest, blocked at start | Hold vs re-plan vs fail-the-mission policy; false completion near goal; drivable-width reasoning |
| 6 | component_failure | 110 | disable each of 9 components × 3 timings; stale vs frozen data (sub/over threshold); confidence step vs ramp; fail→recover; double faults; fault with a hazard 15 m ahead; fault during e-stop / arrival; crash / NaN / clock jump; tick latency; flapping | Does the supervisor catch *every* fault class, how fast, does it hold, and does recovery require a human? |
| 7 | emergency_stop | 60 | speed sweep 0–12 m/s × weather; in a turn / with pedestrian / already stopped / at arrival; clear→resume / no-resume / new mission / start-while-latched; repeated; during pause; before mission; double press | Latency to full brake, latching, "clear never moves the vehicle", state-machine idempotence |
| 8 | operator_action | 60 | pause/resume timings, stop, change destination (further/nearer/behind/same), speed-limit change incl. 0, hazard appearing during pause, back-to-back missions, degenerate inputs | Console/API is a safety-relevant interface: every button in every state |
| 9 | sensor_degradation | 50 | camera/lidar/gnss/imu dropout durations, noise, latency, correlated multi-sensor loss, flaky connector | Contract for sensor-health → supervisor (currently **not wired** — ground-truth perception hides it) |
| 10 | localization_degradation | 40 | drift vs jump, truthful vs silent confidence, stale in canyon/tunnel/open, recovery, confidence ramps | Plausibility checks, dead-reckoning window, speed vs confidence |
| 11 | odd_boundary | 40 | geofence (dest outside / route exits / ego pushed out), speed above ODD, fog beyond ODD, night, weather change mid-mission | ODD enforcement is what makes a closed-lot test *bounded* |
| 12 | traffic_control | 30 | lights (red/green/yellow/change on approach), stop/yield/speed/no-entry/one-way signs | Not needed for closed lot; mandatory before any road |
| 13 | road_geometry | 30 | sharp turn, roundabout, hill, tunnel, T-junctions, unprotected left, merge, ramp, narrow, U-turn, multilane keep, long straight, S-curve | Pure-pursuit tuning envelope; curvature-aware speed |
| 14 | timing_latency | 30 | API poll storms, API hang, command age, jitter, slow perception, disk full, CARLA stall | Thread isolation & watchdog — the bugs that only show up on real compute |
| 15 | vulnerable_road_user | 40 | cyclist/motorcycle/scooter ahead-slow/filtering/cross/wobble; wheelchair-speed cross, small animal proxy, stroller-speed, skateboard-speed | VRU class (today they are just "vehicle" or "pedestrian") |
| 16 | compound | 30 | pedestrian+rain+night, vehicle ahead+perception dies, obstacle+localization decay, cut-in+brake, jam+estop+resume, fog+lead+pedestrian, pedestrian+destination change, cyclist+oncoming truck, construction+pause | Priority-ordering bugs between independent subsystems |
| 17 | edge_case | 30 | overlapping spawn, actor deleted mid-stop, object behind, *exact* boundary values (1.75 m, 5.0 m, 50 m), tie-breaking, moving at engage, 2 km route (O(n) lookup), zero cruise, malformed API inputs, pause/resume spam, e-stop during planning, index out of range | Boundary conditions that bite in the field |
| 18 | endurance | 20 | 10–30 min soaks with 0–40 traffic + walkers | Memory/log growth, tick creep, unhandled exceptions |

---

## 2. "Minute details" the catalog deliberately pins down

These are the concrete numbers and ordering rules the scenarios test. Each one is a place where the
stack either already has a hard-coded value (and the scenario brackets it) or where we have to choose
a value before closed-lot testing.

**Geometry of the in-path check** (`perception.py`: `path_width 3.5`, `danger_distance 5.0`, `detection_range 50`)
- Objects at lateral 1.2 / 1.9 / 2.6 / 3.5 m (`ped_standing_edge`) and exactly 1.75 m (`edge`).
- Objects at exactly 5.0 m, 49.5 m; `<` vs `<=` semantics.
- Object *behind* ego must be ignored (`object_behind_ego`); oncoming in the other lane must not stop us (`va_oncoming`).
- The box is a straight rectangle in ego frame → obstacles after a curve are seen late (`so_after_curve`); this motivates a route-aware corridor check.

**Stopping-distance budget** — at speed *v*: 0.5 s latency + v²/(2·4 m/s²). `so_sudden_appear` sweeps trigger distance 10–30 m × 6–10 m/s: the pairs that collide define the **max cruise speed we may allow**. Pass budgets (`stop_budget_s`) are derived from the same formula so they are not arbitrary.

**Staleness thresholds** (`safety_supervisor.py`: 1.0 s perception / localization, 0.5 s command) — bracketed at 0.6 / 1.2 / 3.0 s, and distinguished from **frozen** data (timestamp stops advancing) which is what a hung driver really produces.

**Confidence threshold 0.3** — tested at 0.6 / 0.31 / 0.29 / 0.1 / 0.0, step vs 5 s ramp, and the *silent* case (pose drifts, confidence still 1.0).

**Priority ordering** (`behavior.py`): safety > no-mission > localization > perception > arrival > pedestrian > vehicle > obstacle > slow-zone > approach > cruise. Scenarios deliberately create collisions between these: obstacle 2 m before destination (arrival may win and declare completion — a bug), e-stop at arrival, perception failing with a pedestrian 15 m ahead, pedestrian and cone at the same range (tie resolved by iteration order today).

**E-stop contract**: full brake + handbrake same tick; latched; *clearing never produces motion*; a new mission while latched is refused; idempotent on double press; wins over pause/resume.

**Operator API state machine**: every control in every state (pause when idle, resume without pause, start twice, stop then new mission, speed-limit 0, destination == current).

**Recovery policy** (open question made explicit by `cf_recover`): today a component coming back *auto-resumes* because autonomy stays engaged. The scenarios record it; the safety doc must decide "auto-resume after N s healthy" vs "operator re-engage". For closed-lot I recommend operator re-engage.

**Reporting faults**: `cf_double_failure` shows the supervisor returns on the first failed check, so a second simultaneous fault is invisible in `errors`. Minor today, important for diagnosis on real hardware.

---

## 3. What the catalog already exposes (before running a single one)

Writing the families against the code surfaced these gaps; they are tagged in the YAML and listed in `KNOWN_ISSUES.md`:

| Gap | Families that show it | Severity for closed lot | Fix size |
|---|---|---|---|
| Sensor-adapter health (camera/lidar/gnss/imu staleness) is computed but **never fed to the supervisor** — ground-truth perception masks it | `sd_*`, `cf_disable(camera…)` | **High** — on real hardware this is the #1 fault class | small: pass the 4 booleans into `SafetySupervisor.update` |
| No geofence / ODD enforcement | `ob_*` | **High** — a closed-lot test must be bounded | small-medium |
| Exceptions in `tick()` were caught and the *last command kept applying* | `cf_crash_and_invalid` | **High** | **fixed in this change**: tick error → brake + logged event |
| NaN / stale commands were clamped, not rejected | `cf_crash_and_invalid`, `tl_timing(command_age)` | High | **fixed**: adapter validates finiteness + age, brakes on reject |
| No car-following — stop/go oscillation behind a slow lead | `va_slow_lead`, `br_traffic_jam` | Medium | medium (time-gap controller) |
| No re-plan on persistent block / no mission-failure timeout | `br_full_block_persistent`, `br_unreachable` | Medium | medium |
| Straight in-path box → late detection around curves | `so_after_curve` | Medium | medium (corridor along route) |
| Speed not curvature-aware; pure pursuit oscillates > 8 m/s | `rg_geometry`, `nm_speed_sweep` | Medium | small (speed profile from route curvature) |
| VRU class absent (cyclists = vehicles) | `vru` | Medium | small |
| Traffic lights / signs ignored | `tc_*` | Low for closed lot, **blocking** for road | large |
| Recovery policy undefined (auto-resume) | `cf_recover`, `cf_flapping` | Medium | small + decision |
| Time uses `time.time()` (clock jumps trip staleness) | `cf_crash_and_invalid(clock_jump)` | Low now, real on vehicles with NTP/GNSS time | small (monotonic) |
| Route lookup O(n) per tick | `edge(very_long_route)` | Low | small |

---

## 4. Using the catalog as the regression / training loop

**Per change** — run the *implemented* subset (456 scenarios) — this is the suite that must stay at 0 FAIL:
```bash
python3 scenarios/run_scenario.py --status implemented --quiet && python3 scenarios/report.py
```
**Per feature** — when you implement e.g. car-following, flip `va_slow_lead` from `partial` to `implemented` in `gen_required_a.py`, regenerate, and the family becomes a FAIL-able regression gate. The report's *By capability status* table is the burndown.

**Per milestone** — the `REPORT.md` numbers we track:
- collision count (must be 0 across all runs, including GAP scenarios),
- supervisor reaction time (fault → non-OK), target ≤ 0.2 s at 10 Hz,
- stop time after stimulus vs the `stop_budget_s` formula,
- route deviation max, steer oscillation index per speed bucket,
- `tick_gap_max_s` (watchdog health).

**Towns** — scenarios name a town but the runner does not swap maps (the stack must be restarted in a town). Run by town: `--town Town03`. Restarting CARLA per town is the simplest reliable approach and is documented in the README.

**Fuzzing** — `take()` shuffles deterministically from the seed; change `SEED` in `generator.py` to get a different but equally structured 1000 (useful once all families pass and you want variance).

---

## 5. What this catalog does *not* do (yet)

- It does not vary CARLA tyre friction / vehicle mass — wet-road tags are labels, not physics, until perception is real.
- Pedestrian/vehicle behaviours are simple kinematics (constant velocity, open-loop steering pulses), not CARLA ScenarioRunner/OpenSCENARIO. I chose this on purpose: it keeps the runner at ~500 lines we fully control, and it talks to the stack through the *operator API*, which is exactly the boundary a physical test range would use. Moving the 1000 definitions to OpenSCENARIO later is a translation job, not a redesign.
- It does not yet grade *comfort* (jerk, lateral accel). Add `max_decel_mps2`/`jerk` metrics when physical testing starts.
- Expected-value metrics from *real* perception (detection range, recall by class, localization RMS) have placeholders (families tagged `future_perception`) but nothing to measure until Milestone 8/9.
