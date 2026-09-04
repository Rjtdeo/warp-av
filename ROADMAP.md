# Warp AV — Software Roadmap (while hardware is in the pipeline)

The stack drives, parks, and obeys lights today — verified by a 122-mission automated
sweep ([SWEEP_REPORT.md](SWEEP_REPORT.md)). This roadmap is ordered by what closes the
gap to a real road fastest.

## Now (highest value)

1. **Camera + LiDAR perception for real.** Today the van reads object positions from the
   simulator ("ground truth") — a real van never gets that. Switch detection + tracking to
   the on-board sensors, then re-run the same 200-mission sweep in camera mode and publish
   the honest before/after pass-rate. This one number is how perception readiness is
   measured everywhere.
2. **Right-of-way reasoning at junctions.** The van looks once, then commits blindly; both
   real collisions in the sweep were crossing traffic during junction exit. It must keep
   watching while crossing and yield by rule, not by radius.
3. **Nightly regression sweep.** The overnight test rig exists (`tools/mission_sweep.py`).
   Run it nightly; every failure becomes a permanent scenario. Simulation-first testing is
   80% of how serious AV companies earn trust.

## Next

4. **Prediction layer — SHIPPED (v1).** Every moving object's position is guessed 3.5 s
   ahead; the van yields to crossers/cut-ins BEFORE they enter its path (behavior
   `yielding_predicted`). First dense-traffic live test: 11 predictive yields, mission
   completed, zero contact. Next refinement: formal who-goes-first rules on top.
5. **Swept-path collision checking.** Check the van's BODY through bends, not just its
   route line (a parked SUV in a curve was clipped this way).
6. **Motion health alarm.** Commanded speed > 0 with no movement and no obstacle for N
   seconds must raise an alarm (the "wedged but says cruising" class, seen 3×).
7. **Reverse gear + parking fallbacks.** A stolen slot must never strand the van.

## Later

8. **ODD definition + enforcement** — write down where/when the van may drive (roads,
   speeds, weather) and have the stack refuse outside it.
9. **Safety case document** — the argument + evidence file (SWEEP_REPORT.md is the seed).
10. **Scenario library growth** — keep converting every incident, sim or real, into a
    permanent catalog scenario (1000-scenario framework already in `scenarios/`).

## Learning track — Reinforcement Learning (parallel, low-risk)

RL = training a small neural network by trial and points in simulation instead of writing
rules. Industry reality: RL is used for narrow skills inside a rule-based frame — never
for the whole vehicle, and never above the safety supervisor.

- **Step 1 (done in one evening):** Stable-Baselines3 CartPole demo — watch `ep_rew_mean`
  climb. That is the whole concept, live.
- **Step 2:** wrap the parking pull-in as a gym environment (observations: distance/angle/
  speed to slot; actions: steer/throttle; reward: inside-box + centred − contact − jerk).
- **Step 3:** train PPO overnight in fast-sim; deploy as ONE module the behavior layer
  calls only for the final 10 m; safety supervisor keeps veto.
- **Step 4:** grade RL-parker vs the hand-tuned parker with the sweep (50 missions each).
  The comparison is the deliverable, win or lose.
- Watch-outs: reward design is the hard part (students cheat), and volume matters — sim
  speed-up work pays for itself here. Imitation learning (copy good recorded drives,
  then RL-polish) is the industry's usual first rung.

### Round history (parking student)

| Round | Change | Result |
|---|---|---|
| 1-2 | Flat random spawns 3-12 m out | 2,709 attempts, **0 parks** — short lane-parallel starts were geometrically unsolvable (a car cannot move sideways) |
| 3 | Hindsight curriculum: spawn along the ideal pull-in, including inside the box | 269 training parks; graded 7/30 — but see below |
| 4 | Honest exam + distance-aware limits + a ladder that moves | pending the CARLA machine |

**Round 3's 7/30 was not an exam.** `eval_parking.py` reused the training spawn mix, so
roughly one attempt in six began with the van already inside the box. Round 4 makes the
exam a fixed full distance (`--p 0.0`, the default) and writes `rl/REPORT_CARD.md` plus
`rl/eval_runs.csv` into the repo, so the number is reproducible and comparable.

**Round 4's three changes:**
1. *Room to stray is measured from where the attempt started.* A full-distance start is
   already ~16.3 m from the slot and the old flat 18 m write-off left under 2 m of slack —
   one bad second of steering ended the attempt before it had driven anywhere.
2. *Time allowed scales with the distance to cover* (12 s + 2 s/m, capped at 60 s) instead
   of a flat 30 s for every attempt.
3. *The difficulty ladder moves.* Rounds 1-3 drew all six difficulties with equal chance
   for ever, so a third of every run re-proved mastered lessons. A `Curriculum` now holds a
   focus rung, steps out when the recent success rate passes 55%, eases back below 15%, and
   mixes in 25% easy revision so old skills are not forgotten. A student that keeps winning
   walks from "already in the box" to the full exam in ~250 attempts; one that plateaus
   around 45% takes 500-2,000 and is carried up by variance rather than held for ever.


## Sensed bays — finding the parking space with the lidar (started 4 Sep)

Town10's parking lanes are unmarked strips along the kerb (the planner slices them
into 7 m slots itself), and a real test course has no map, so the bay must be seen.
`src/warp_av/perception/bay_finder.py` (pure numpy, 10 synthetic tests) fits the kerb
face as a line through the nearest low lidar points in each half-metre strip, bins
the higher points along it, and calls a free stretch of 7 m plus margins a bay,
returning the slot pose in the vehicle frame. `tools/probe_bay_finder.py` marks it on
the simulator at 40 spots beside parking lanes against the map's lane centre and the
map's decorative parked vehicles (`rl/exams/2026-09-04_bay_finder_probe.csv`).

Fourth probe (after three fixes: kerb face not pavement middle; sparse far cars
count; the slot itself is what is checked):

| measure | result |
|---|---|
| kerb line found | 37 of 40 spots |
| a bay found | 36 of 40 |
| slot overlapping a parked vehicle | **0** of 36 |
| slot centre vs the map's lane centre | median -0.01 m, 27 of 36 within 0.5 m, worst 1.34 m |
| heading vs the lane | median 0.9 deg, 33 of 36 within 3 deg, worst 6.3 deg |

The detected kerb sits a consistent ~0.6 m beyond the map's lane edge; the slot centre
lands on the map's lane centre in the median, so the map's edge is not the physical
kerb and the lidar is the one to trust. The worst side errors (about 1 m) and the
two 6-degree headings are bad fits at spots with a 2.0 m map lane; still open.

Next: plug it into `planner.find_parking_slots` as the slot source when a "sensed bays"
mode is on (map slots as fallback), re-scan on approach as the stack already does, and
exam the whole park-from-the-lidar loop with the round-6 parker.

### First whole-loop exam: park from lidar-found bays (4 Sep, 11:02-11:15)

The bay finder is wired into the stack behind `parking_source` (`/api/parking/source`,
default `map`); with `lidar` selected, FIND PARKING takes its slots from the live sweep
at the current pose and falls back to the map if no kerb is seen. `mission_sweep.py
--park-check --parking-source lidar` drove three real missions (camera+lidar
perception mode, stack 544b1f3), all three using lidar slots:

| run | what | result |
|---|---|---|
| 1 | plain slot parking | **FAIL** - hovered 0.9-1.0 m short of the spot until the clock ran out |
| 2 | chosen slot stolen by a parked car, re-scan and retarget | **FAIL** - retargeted correctly, finished 0.29 m over the slot's side line, 7.4 deg off |
| 3 | plain slot parking | **PASS** - inside, side margin -0.06 m (within tolerance), 2.8 deg |

No collisions. The loop works end to end - see the kerb, find the free stretch,
choose, drive in - and the failures are the lidar slot sitting too far towards the
kerb at some spots (the probe's worst spots were 1.0-1.3 m out at the same kind of
place), so the van cannot reach the slot centre and stops short. The map-based
park-check on 21 Aug measured +0.17 / -0.02 m at the same job. Next: bound the slot's
lateral position against the lane the van is driving in, and re-examine.
