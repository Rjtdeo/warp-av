# Progress Report — Road-Readiness Sprint (Aug 19, 2026)

22 commits in one working session. Every driving change was implemented on the Mac, pushed to GitHub,
pulled on the Windows CARLA machine, and **verified live in CARLA** by the operator before moving on.
73 automated tests (no CARLA needed) guard everything against regressions. All 7 requests are now CARLA-verified end-to-end — see the reference run below.

---

## 1. Troy's requests — status

| # | Request | Status | Evidence |
|---|---|---|---|
| 5 | Fix weird steering / braking / oscillation | ✅ verified | steering flips **0.09–0.19/sec** (limit 2.0), **0 unjustified brake ticks**, speed wobble 0.13–0.21 m/s — measured from real mission logs with `tools/check_smoothness.py` |
| 4 | Bigger safety buffer to obstacles | ✅ verified | stop trigger **5 → 8 m**, slow-down zone **15 → 20 m**, in both ground-truth AND camera+LiDAR perception |
| 6 | Proper following behind vehicles | ✅ | time-gap following (1.5 s + 8 m) replaces stop-and-go; closed-loop test settles at lead speed, never halts |
| 2 | Correct left turns | ✅ verified | curve-aware speed (slows before bends), arc-following aim point, corner-cut deviation 1.25 → **0.78 m** in the reference 90° corner |
| 3 | Correct right turns | ✅ verified | same mechanism, left/right symmetric by construction (tested) |
| 1 | Observe traffic lights | ✅ verified | operator watched the van stop at red and release on green; red/yellow stop, green auto-release; hazards outrank the light |
| 7 | Park inside a parking box | ✅ **CARLA-verified** | pulls over to a kerbside spot / real parking bay, arrives parallel: **"Parked 1.45 m from the kerbside spot, heading off 8°"** (mission_0011); completion requires ≤1.5 m AND ≤15° AND nearly stopped |

## 2. Operator-feedback fixes (found by driving it, fixed same day)

| Observation from the dashboard | Fix | Commit |
|---|---|---|
| Van froze 8–10+ m before crossings at lights/turns | Rolls up to the actual **stop line** (or the junction edge when the light has no stop-line data) and holds ~3 m from it | `7a70699`, `d238222` |
| Turns took no care about other cars | **Junction give-way**: roll to the crossing, mandatory 1.5 s look, hold while a moving vehicle is within 25 m, creep after 12 s instead of deadlocking | `983f489` |
| Van rode the divider between two lanes; drifted into the left lane before right turns | Stronger centreline pull (gain 0.12→0.20) + short aim distance when >1 m off-lane; simulated 3.5 m lane change settles within **6 cm** | `47e39d2` |
| Kerb/footpath clipped in tight corners | Aim point measured **along the route arc**, shorter aim in bends, centreline correction, slew-limited corner-exit acceleration | `4026bda` |
| False "VEHICLE AHEAD" stop mid-turn (car was in the neighbouring lane, van tilted) | **Route-corridor in-path check**: objects count only if within 1.75 m of the planned route and ahead along it — also fixes late detection of obstacles around bends | `d5927c1` |

## 3. The 1000-scenario validation framework

- **1000 deterministic YAML scenarios** in 18 categories (`scenarios/catalog/`), each with machine-checkable
  pass/fail criteria and an honest `implemented / partial / not_implemented` capability label
- **Runner + evaluator + report**: executes scenarios against live CARLA through the operator API,
  produces PASS / FAIL / GAP verdicts and a pass-rate scoreboard (`scenarios/run_scenario.py`, `report.py`)
- **Fault-injection API** (`POST /api/test/inject`): disable/freeze/delay perception, localization,
  camera, LiDAR, GNSS, IMU, controller, vehicle connection — the basis of the component-failure scenarios
- Safety hardening that came out of writing it: a software crash now commands **brake** (previously the
  last throttle kept applying), the vehicle adapter **rejects NaN/stale commands**, API inputs validated

## 4. Operator console & infrastructure

- Dashboard live at **https://warp-av.vercel.app/console** (auto-deploys from GitHub `master`), able to
  operate the Windows CARLA machine remotely (`?api=http://<ip>:5000`; CORS + asset fixes)
- Scenario catalog browser at **https://warp-av.vercel.app**
- **RED LIGHT** button added to the Road Scenarios panel (freeze all lights red; CLEAR releases)
- `/api/state` now exposes: active faults, traffic-light state + stop-line distance, upcoming-junction
  detection, cruise speed, tick counter — the van's "thinking" is inspectable live

## 5. Tools built for testing & demos

| Tool | Purpose |
|---|---|
| `tools/check_smoothness.py` | grades any mission log: weave, unjustified brake ticks, speed wobble — the [OK]/[BAD] report used for every verification above |
| `tools/demo_red_light.py` | hands-free demo: finds the route with the most lights, starts the mission, freezes red on approach, verifies the stop, releases green, verifies pull-away |
| `tools/set_traffic_lights.py` | force all lights red/green/auto on demand |

## 6. Reference run — mission_0011 (the whole system in one drive)

457 m, 237 s, Town10 — full report in [logs/mission_0011_report.txt](logs/mission_0011_report.txt):

- Right-turn give-way roll-up interrupted correctly by a YELLOW light → held at the stop line 20.1 s
- Mandatory 2.1 s junction look before the right turn, then proceeded
- Operator spawned a vehicle mid-run: brief car-following (gap 21.7 m), then stopped at **7.9 m** behind it (the 8 m buffer), resumed on clear
- Two more lights (RED 32.5 s, RED 6.8 s, YELLOW 35.5 s) each held AT the stop line; left-turn give-way looks after each
- Finish: pull-over and **"Parked 1.45 m from the kerbside spot, heading off 8 deg"**
- Whole-run verification: steering flips **0.08/sec**, **0** unjustified brake ticks, speed wobble 0.43 m/s, safety supervisor `ok` throughout — every check [OK]

## 6b. What's left

1. The honest gap list lives in [KNOWN_ISSUES.md](KNOWN_ISSUES.md) — notable: camera-mode perception has
   no object tracking yet (so following/light detection run on ground truth), give-way uses a radius
   heuristic rather than true right-of-way, corridor check trusts localization (fine in sim)
3. Run the full 1000-scenario catalog on the CARLA machine and commit the first `REPORT.md` scoreboard
