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

### First whole-loop exam (4 Sep, 11:02-11:15): the lidar never got asked at the right moment

The bay finder is wired into the stack behind `parking_source` (`/api/parking/source`,
default `map`); with `lidar` selected, FIND PARKING runs the bay finder on the live
sweep at the current pose and falls back to the map if it sees no kerb.
`mission_sweep.py --park-check --parking-source lidar` drove three real missions
(camera+lidar perception mode, stack 544b1f3). The stack's own log for all three says
**"lidar saw no bay - falling back to the map"**: FIND PARKING runs at mission START,
hundreds of metres from the destination, where the lidar (30 m reach) cannot see the
parking lane the map is slicing. So these three runs are a **map-slot baseline in
camera mode**, not a lidar result:

| run | what | result (map slots) |
|---|---|---|
| 1 | plain slot parking | FAIL - hovered 0.9-1.0 m short of the spot until the clock ran out |
| 2 | chosen slot stolen, re-scan and retarget | FAIL - retargeted correctly, finished 0.41 m over the slot's side line, 7 deg off |
| 3 | plain slot parking | PASS - inside within tolerance (side -0.15 m), 3 deg |

No collisions. Note the baseline itself is worse than the 21 Aug park-check
(+0.17 / -0.02 m side margins, ground-truth mode) - the camera-mode stutter near bays
is still open. The lidar belongs at the moment the van reaches the bay: the next step
is to run the bay finder in the approach re-scan (`_recheck_parking_on_approach`),
replace the map slot with the nearest free lidar slot, and re-examine.

### Second whole-loop exam (4 Sep, 11:19-11:26): first park in a lidar-found bay

Stack 90fdc82 adds the approach re-scan: with `parking_source = lidar`, once the
destination is within 22 m the bay finder runs on the live sweep and the target moves
to the nearest free lidar slot. Same three-mission park-check, `--parking-source lidar`:

| run | what | slot source | result |
|---|---|---|---|
| 1 | plain slot parking | map (lidar saw no bay at start; approach re-scan at 22 m also saw none) | PASS - inside, 0.11 m side, 3 deg |
| 2 | chosen slot stolen | map; stuck 180 s behind the thief, "no free slot ahead" (the known thief-with-no-slot-ahead gap); the lidar re-scan never got its turn | FAIL |
| 3 | plain slot parking | **lidar - "2 slots from the LIDAR"** (the start was already beside the bay) | **PASS - inside a lidar-found slot, margins 0.01 m front/back, 0.12 m side, 2 deg** |

2 of 3 against the map baseline's 1 of 3, no collisions, and the first end-to-end park
in a bay the van found for itself. Two things to fix next: the approach re-scan is a
single shot at 22 m and saw nothing in run 1 - it should keep trying as the van closes
in, and also when it is blocked near the destination (run 2 is exactly the case where
real occupancy from the lidar would have found the way out); and the finder should say
why it found nothing (points in the kerb band, edge strips) so a "no bay" is
diagnosable.

### Third and fourth exams (4 Sep, 11:28-11:43): the re-scan works; the stolen-slot run now collides

Stack ab76c1e added a retrying approach re-scan (every 2 s from 22 m down to 6 m) with
a logged reason for every miss; 57b547a added the consistency gate after run 1 of the
third exam, watched live, showed the lidar locking onto a **bus-stop platform kerb**
(slot 1.1 m right of the van; the real bay was set back 6.5 m) and the van stopping for
the shelter as a "vehicle".

| exam | run 1 | run 2 (slot stolen, 2 parked cars) | run 3 |
|---|---|---|---|
| third (ab76c1e) | GAP - bus-stop slot, stuck | **FAIL - collision with a street prop**, stack restarted by the rig | PASS - re-scan moved the target 2.38 m onto a lidar slot, inside, 0.01 m side |
| fourth (57b547a) | PASS - map slot; re-scan saw "no straight line fits (curve or clutter)" | **FAIL - collision with the parked Tesla**, stack restarted by the rig | PASS |

The lidar path has now parked the van twice in bays it found itself (0.12 m and 0.01 m
side margins) and has never caused a collision: in both run-2 collisions the log shows
"lidar saw no bay - falling back to the map" and no approach re-scan line at all, so
those runs were map-only. The stolen-slot manoeuvre colliding twice in a row is new
compared with the 21 Aug park-check (which passed it in ground-truth mode) and is being
checked with a map-only control run on today's stack. The bus-stop gate has not yet been
exercised live (run 1 of the fourth exam drew a different bay).

**Control run (11:44-11:54, map slots only, stack 57b547a): PASS, FAIL (collision with the
parked Tesla in the stolen-slot run), PASS.** Same failure without the lidar in the loop, so
the collision is the stack's own: with the chosen slot taken and no free slot ahead it kept
the taken slot as its target and relied on camera-mode obstacle logic to hold, which did not.
Fix e4f025f: the van re-targets to a hold point in the lane 5 m short of the taken bay
("Held short of the taken bay" at completion) - a safe failure of the parking task instead
of a collision. The proper answer (reverse gear or a kerb-side fallback slot) stays open.
