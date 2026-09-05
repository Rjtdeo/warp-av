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

### Fifth exam (12:00-12:09, stack e4f025f: gate + stop-short): 0 of 3

Run 1 collided with a street prop 58 s into the mission (before any parking logic;
log being read). Run 2 (slot stolen): the approach re-scan moved the target **10.12 m
along the bay** onto a lidar slot behind the parked cars and the van sat 180 s behind
one of them - the gate allowed 12 m along; it now allows 4 m unless the map's target
is itself taken. Run 3 fell back to the map (no re-scan fired) and finished 0.17 m over
the slot's side line - the camera-mode parking stutter seen in every map-slot run today.

**Where the whole-loop stands after five lidar park-checks and one control.** The bay
finder is a sound component (probe: 36 of 40 spots, 0 slots on a parked vehicle, median
side error ~0). End to end it has parked the van in a bay it found twice (0.12 m and
0.01 m side margins) and has caused no collision. But the three-run park-check on
today's stack is dominated by the stack's own camera-mode parking behaviour (finishing
0.1-0.4 m over the side line; driving into a bay it knows is taken; a street-prop
collision en route) and by luck of the random destination, so three runs cannot show
the lidar's effect. The clean experiment is to run map-vs-lidar bays in ground-truth
perception mode, where the parker itself is known good (21 Aug: +0.17 / -0.02 m), over
more than three runs.

### The clean experiment (4 Sep, from 12:17): map bays vs lidar bays, ground-truth perception

Same 3-run park-check twice over (6 runs), the van's obstacle perception on ground truth
in both arms so the parker itself is the known-good one, and only the bay source differs.
Perception mode and bay source are set by the rig before every run (`--perception-mode`,
`--parking-source`, `--repeat`).

**Arm A - map bays:** 3 of 6. Run 1 slipped through in camera mode (the stack had just
booted; the mode switch did not take for that run) and stalled 0.8 m from the spot. In
ground-truth mode the three plain parkings all finished inside (side margins -0.03 to
-0.10 m within the 0.10 m tolerance, headings 1-3 deg); both stolen-slot runs failed - one
sat behind the thief until the clock ran out (the stop-short rule fired too late, now fixed
to re-check from 30 m out), one hit a decorative car on the approach.

**Arm B - lidar bays:** 4 of 6 by the rig's count - but the stack's own log shows the lidar
path engaged properly in only one of the six runs ("4 slots from the LIDAR", parked, run 6);
the others fell back to the map at mission start and the approach re-scan then reported
"0 kerb-height points to the right", "no straight line fits (curve or clutter)", or - the
gate doing its job once - "lidar slot rejected: 6.0 m off the reference bay's line". The two
stolen-slot runs failed exactly as in arm A (waited behind the thief; hit a decorative car).

**So the honest reading of the experiment:** the three plain parkings scored 3 of 3 in both
arms, but arm B is not a test of lidar bays - it is mostly map bays with the lidar declining
to answer. At the sweep's destinations the finder, tuned on straight lane-side spots, often
meets a set-back bay (6.25 m right of the lane, its kerb near the 8 m edge of the search
window), a curved kerb, or no raised kerb at all. Two things follow: widen and curve-tolerate
the kerb fit (search out to ~10 m, fit a gentle arc or piecewise line, accept a flush edge from
the road-surface boundary); and fix the parker's stolen-slot handling (re-check from 30 m,
commit 91ba2d9, deployed after this experiment). Then the whole-loop comparison is worth
re-running. The component result stands: 36 of 40 probe spots, 0 slots on a parked car, and
two end-to-end parks in bays the van found itself.

### Finder v2 (4 Sep, ~12:50): set-back bays, cambered roads, curving kerbs

Search widened to 11 m; heights taken from a road plane fitted to the lidar instead of an
assumed flat road; the kerb edge fitted as a gentle curve (radius > ~40 m) with the slot
heading from the local tangent; and the edge taken as the nearest CLUSTER of raised points
per strip rather than the nearest single point (road noise a few centimetres high was
winning once the kerb was 9 m out). Probe on 60 fresh roadside spots: kerb at 45, a bay at 42,
**0 on a parked vehicle**, side error mean 0.44 m (17 of 42 beyond 0.5 m; one 4 m outlier that
the stack's gate would reject), heading mean 2.7 deg. What the rig calls a set-back bay is the
van driving one lane further left on a two-lane road; the probe gains --from-left-lane to test
that geometry directly.

**Probe from the left lane (the rig's set-back geometry), 39 spots:** on the 35 set-back spots
the finder found the kerb at 26 and a bay at 26, **0 on a parked vehicle**, side error mean
0.32 m (worst 1.57 m), heading mean 3.0 deg (worst 13.3 deg). Before v2 it reported "0
kerb-height points" at these. Nine misses and the two large headings remain on the list;
the stack's gate rejects anything more than 1.5 m or 15 deg off the map's line.

### The learned parker inside the van (4 Sep, 13:02-13:18): first park, then a reset bug

`rl_parker.RLParker` takes the wheel for the last 16 m of a slot parking when `parker = rl`
(`/api/parking/parker`), with round 6's brain and its exact training action mapping, and
hands back on overshoot, wander or 45 s. Arm C - lidar bays, ground-truth perception,
learned parker, six runs:

| run | result | what the van's log says |
|---|---|---|
| 1 | PASS | **"learned parker took the wheel 16.3 m from the slot" ... "INSIDE slot #1: YES (0.47 m front/back, 0.21 m side), 1 deg"** - the trained brain's first park inside the van software |
| 2 | PASS | a re-scan moved the target behind the van and the parker reported an "83 m overshoot"; the rules finished the park |
| 3, 4 | PASS, FAIL (pole) | driven by the RULES: the parker's done flag never cleared between missions |
| 5 | GAP | rules again (the parker switch timed out after a restart); gridlock |
| 6 | FAIL | learned parker held 180 s behind a stopped vehicle, then "timed out" |

Fixed (c87854f): every one-shot parking flag and the parker now reset at every mission
start (the stop-short and lidar-retry flags were not resetting either); overshoot only
counts after the van has actually reached the bay; the 45 s counts time at the wheel; lidar
slots sit 0.6 m off the kerb (run 4's pole); the rig waits 40 s for the brain to load.
Re-running.

**Arm C, second pass (13:25-13:41, c87854f): 3 of 6.** Run 1: the learned parker took the wheel
at 16.3 m and parked inside (0.13 m side, 2 deg) - its second park inside the van. The rest
exposed two more of my own bugs: the per-mission reset sat on a branch of start_mission and
did not always run (a parker left engaged by a stopped mission judged the next mission's slot
"183 m off the slot line" and gave up at once), and with finder v2 seeing further, FIND PARKING
at mission start began taking lidar bays beside the START and bending the route onto them (run
3 drove into a traffic light at speed on that route). Both fixed in 3376829: resets run
unconditionally at the top of start_mission; lidar slots at mission start count only within
40 m of the destination. Third pass running.

### The learned parker inside the van, third pass (4 Sep, 13:57-14:08): 5 of 6

Stack 3376829. Lidar bays, ground-truth perception, round 6's brain driving the last 16 m.
The brain took the wheel in **all six** runs (every log: "learned parker took the wheel
16.x m from the slot").

| run | slot | result |
|---|---|---|
| 1 | map | parked, INSIDE (0.33 / 0.15 m), 1 deg |
| 2 | map, stolen -> re-targeted | parked, 4 cm over the side line (within tolerance), 1 deg |
| 3 | map | parked, 7 cm over (within tolerance), 1 deg |
| 4 | map | **collision with a decorative parked car beside the bay** |
| 5 | **lidar** (re-scan moved the target 0.47 m), stolen -> re-targeted | parked, INSIDE (0.40 / 0.14 m), 2 deg |
| 6 | map | parked, INSIDE (0.18 m side), 1.5 deg |

Run 5 is the full chain: a bay the lidar found, the trained brain driving into it. Against
the hand-written parker on the same six runs this morning (arm B: 3 of 6, both stolen-slot
runs failed), the learned parker handled both stolen-slot runs. The one failure is the known
gap: round 6 has no obstacle inputs, and in ground-truth mode the van's own perception does
not see map decoration either, so nothing stopped it. Obstacle awareness (round 7's unfinished
work) is now the next target, with a reverse gear as the honest way to park between two cars.

## Round 9 plan (from Waymo's published recipes, 2026-09-04)
Copy first, then practise. A rule-based teacher (hold the lane, turn in late, reverse if a car sits
right behind) records demonstrations from perturbed starts; the brain is trained to copy them, tidied
with DAgger rounds, then fine-tuned with today's rewards plus a fading stay-close-to-teacher term.
Also from the reading: auto-label lidar clusters with CARLA ground truth to train a tiny classifier
(test with truth off); bucket the scoreboard by difficulty and re-spend runs where it fails; a
decoy-car robustness exam. Full digest: docs/RESEARCH_DIGEST.md.

## Round 8 result (2026-09-05, 00:56 LA)
Obstacle awareness learned: empty bay 30/30, car two bays back 29/30, car three bays back 30/30
(round 6 on the two-bays-back test: 0/30). Car right behind the bay still 0/20 (needs the
reverse manoeuvre; round 9). The road there: 8a-8d were set-up flaws (van born beside or
behind the car, then the early swing-in habit), 8e's lane-hold charge cracked it, 8f/8g spread
the starts, 8g's saved brain collapsed in its last minutes (lesson: rolling snapshots, cap
unsolvable stages, lower the learning rate late), 8h redid the last stretch cleanly.
Brain: rl/models/parking_ppo_round8.zip (9 inputs incl. feelers, 3 controls incl. reverse).
- 2026-09-05 01:39: sensor test with the round-8 brain, 6 runs, 0 collisions, 4 PASS; the brain parked
  from lidar + camera in run 1; two integration fixes pushed (aligned hand-over, stop-override rule),
  to be re-tested; open: camera-mode "obstacle at 0.0 m" route block. Details: rl/REPORT_CARD.md.
