# RL parking report card

Brain: round 6, checkpoint at 687,628 steps (`parking_ppo_round6.zip`), trained
2026-09-03 15:25-19:05. Every exam below starts from the full ~16 m lane distance
(`--p 0.0`), 30 attempts, deterministic actions. Raw rows in `rl/exams/`.

| exam | arena | parked | crashes | side clearance mean / worst | heading mean / worst |
|---|---|---|---|---|---|
| this morning, round-3 brain, seed 123 | as found | **0/30** | 0 (30 timeouts) | - | - |
| round 6, seed 123 | as found | 27/30 | 3 | 0.21 / 0.19 m | 0.9 / 1.2 deg |
| round 6, seed 2026 | as found | 21/30 | 9 | 0.21 / 0.18 m | 0.9 / 1.2 deg |
| round 6, seed 123 | clean bays | **30/30** | 0 | 0.20 / 0.04 m | 0.9 / 2.0 deg |
| round 6, seed 2026 | clean bays | **30/30** | 0 | 0.20 / 0.04 m | 0.9 / 1.9 deg |

**What "as found" vs "clean bays" means.** With hit logging, all 12 misses in the
"as found" exams were `static.car` / `static.motorcycle`: Town10's decorative
parked vehicles, baked into the map, not actors. One miss had the van parked
dead centre, 0.3 deg off, at 0.02 m/s, on top of a motorcycle. The arena now
drops any bay with such a vehicle in it, in the 18 m behind it, or within 4 m
either side (64 of 216 bays dropped; 48 such vehicles on the map). The August
sweep found the same objects defeating the main stack's occupancy check.

**Read this honestly.** 60/60 is on bays with nothing parked near them. The
student's five observations contain no obstacle information at all, so it
cannot avoid a neighbour it cannot see; that is the next thing to build, not a
detail. It also still runs on CARLA ground truth for its own pose and the bay's
position (see KNOWN_ISSUES.md for the main stack's equivalent caveat). What this
card does show: from 0/30 this morning, the pull-in and stop from 16 m is
learned, straighter and better centred than the hand-coded parker (Aug-21
park-check: side margins +0.17 / -0.02 m, heading 1-3 deg). Worst side
clearance across the 60 clean parks was 0.04 m: inside the box, but a note for
round 7.

Ground: 7 m x 2.5 m slot, 4.70 m x 1.96 m van, so 0.27 m side clearance when
perfectly centred and 1.15 m at each end.

## Harder exams, same brain (2026-09-03 20:00-20:04)

Clean bays, seed 2026, 30 attempts each, deterministic actions. Nothing retrained.
Each row changes ONE thing about the start or about what the student sees; the
last row changes all four. Rows in `rl/exams/2026-09-03_round6_harder_*.csv`.

| exam | what changed | parked | side clearance mean / worst | heading mean / worst | start distance |
|---|---|---|---|---|---|
| baseline | nothing | 30/30 | 0.20 / 0.04 m | 0.9 / 1.9 deg | 15.3-16.3 m |
| far | lane start 24 m instead of 16 - farther than it ever trained | **30/30** | 0.21 / 0.08 m | 0.8 / 2.8 deg | 22.4-25.2 m |
| crooked | start heading up to +/-15 deg off (trained +/-3) | **30/30** | 0.20 / 0.04 m | 0.9 / 1.9 deg | 15.3-16.3 m |
| off-centre | start up to +/-1 m off the lane centre (trained 0) | **30/30** | 0.21 / 0.18 m | 0.9 / 1.2 deg | 15.1-16.4 m |
| noisy eyes | gaussian noise on what it sees, every step: 0.25 m position, 3 deg heading, 0.2 m/s speed | **30/30** | 0.15 / 0.03 m | 1.6 / 5.1 deg | 15.2-16.5 m |
| everything | all four at once | **30/30** | 0.15 / 0.03 m | 1.6 / 3.3 deg | 21.4-25.7 m |

Read: distance, crookedness and offset at the start cost it nothing it can
measure. Noisy eyes cost it precision, not success - centring drops from 0.20
to 0.15 m and the worst heading rises from 1.9 to 5.1 deg. That is the first
evidence on the sensor question: the skill tolerates 25 cm / 3 deg of jitter in
its inputs without retraining, while parking less neatly. Neither the noise nor
the harder starts change the two caveats above: no obstacles are observed, and
the inputs are still ground truth with noise added rather than a real sensor.

## The two final exams (2026-09-03 20:20-20:31): finding the edge

Pass marks were set BEFORE running. Clean bays, new seeds, same brain, no retraining.
Rows in `rl/exams/2026-09-03_round6_final_*.csv` (now also recording the spawn heading
error and sideways offset actually drawn).

| tier | start | what it sees | pass mark | parked | failures | side clearance mean / worst | heading mean / worst |
|---|---|---|---|---|---|---|---|
| harder | 27-35 m back, up to +/-25 deg crooked, up to +/-1.5 m off the lane centre | noise 0.5 m / 6 deg / 0.4 m/s | 90%+, no collisions | **42/60 (70%) - FAIL** | 11 timeouts, 4 collisions, 3 wandered off | 0.10 / **0.00** m | 1.6 / 4.4 deg |
| hardest | 29-43 m back, up to +/-40 deg, up to +/-2 m off | noise 1 m / 10 deg / 0.6 m/s, view frozen 1 step in 5, everything 0.3 s late | none - built to break it | **3/100** | 40 collisions, 37 timeouts, 20 wandered off | 0.10 / 0.05 m | 1.9 / 2.5 deg |

**Verdict, stated as an envelope rather than a yes/no.** The skill is learnt and it
generalises well past its training: 150/150 at 1.5x the distance, 5x the start
angle, 1 m off-centre and 0.25 m / 3 deg of noise on its inputs. Doubling all of
that at once drops it to 70% with parks that touch the line; quadrupling the
noise and adding sensor freezes and 0.3 s of delay collapses it to 3%. Failures in
the harder tier do not correlate with start angle (11 deg failed vs 12 deg parked)
or start distance (31.7 vs 31.7 m), so it is the combined load, not one knob.
Which knob matters most is not answerable from these two runs - the hardest tier
changed six things at once by design; single-factor runs at these levels are the
next step if the envelope needs to be mapped.

Two things to keep honest. The collisions at 32-40 m are mostly vegetation,
poles, kerbs and buildings - the van driving off the road under large heading
errors and noisy inputs, which is a real failure. One (harder #32) was a
decorative car 30 m back, outside the 18 m keep-clear zone the arena checks;
starts beyond ~24 m are past what the arena was cleaned for. And the student
still observes no obstacles at all; nothing in these exams changes that.

**Harder tier, repeated (20:36-20:43).** Exact re-run with the same seed: 42/60 again,
identical outcome on all 60 attempts - the exam is deterministic given its seed
(the same seed drives the same start draws and the same noise draws), so 70% is
not measurement jitter. A fresh set of 60 starts (seed 888): 47/60, with 5
timeouts, 3 collisions, 5 wandered off, worst side clearance again 0.00 m.
All three harder runs together: 131/180 = 73%. The 90% mark stands failed.

## Round 7 (overnight 3-4 Sep): eyes for obstacles - did not learn avoidance

Round 7 gave the student four "feelers" (nearest obstacle per sector) and put a
real parked car in a neighbouring bay during practice. Three legs overnight:
7a from zero froze (a car 7 m ahead on the easiest rung turned overshoots into
crashes; it learned to sit still). 7b warm-started from round 6 kept its parking
but the car ONE bay behind was geometrically impossible to nose past (parallel
bays, no reverse gear): 0/460 with cars both sides. 7c moved the car TWO bays
back, added a too-close warning (then made it 5x stronger), then widened the side
feelers to the rear quarters. With-car parking stayed at 22-28% in every block
for 16,445 attempts; no-car parking stayed at 99%.

Morning exams, clean bays, seed 2026, 30 attempts from the full distance, a car
parked two bays back on every attempt (rows in `rl/exams/2026-09-04_*.csv`):

| brain | with a car two bays back | on empty bays |
|---|---|---|
| round 6 (cannot see the car) | 3/30, 27 collisions | 30/30 (yesterday) |
| round 7 (four feelers, 5.5 h of practice with the car) | **3/30, 27 collisions** | **30/30**, 0.24 m side clearance, 0.3 deg |

Identical, attempt for attempt - the same 27 starts crash into the same cars.
Round 7 learned nothing about the neighbour; it did keep, and slightly neaten,
round 6's parking.

**Why, as best I can tell.** The per-step "closer to the bay centre" pay rewards
turning in early, which is exactly the path that clips the car; a warning that
fires late and a crash at the end never outweighed it, and the avoiding path
(drive straight past, then turn in late and sharp) is far from the learned one,
so exploration never found it. Three sensing/warning fixes in one night changed
nothing, which points at the scoring's shape, not the sensing.

**Next, to be done in daylight:** pay approach progress along the bay axis rather
than towards the centre, so cutting in early is not rewarded; and/or a reverse
gear, which is how parallel parking is actually done. Round 6 remains the
deliverable for empty bays.

### Round 7 on round 6's full exam battery (4 Sep, 10:26-10:45)

Same seeds, same settings, same clean bays. Round 7 kept round 6's skill on clean
information and is slightly straighter there (0.2-0.3 deg vs 0.8-0.9 deg), but it
is markedly worse whenever what it sees is noisy: the misses are timeouts, not
crashes. Overnight it was taught to be wary when a feeler reads "close"; jittered
inputs make the feelers flicker, and it hesitates until the clock runs out. Round 6,
which has no feelers and no such rule, cannot be spooked that way.

| exam | round 6 | round 7 | round 7 failures |
|---|---|---|---|
| 24 m start | 30/30 | **30/30** | - |
| up to 15 deg crooked | 30/30 | **30/30** | - |
| up to 1 m off-centre | 30/30 | **30/30** | - |
| noisy eyes | 30/30 | **24/30** | 6 timeout |
| all four at once | 30/30 | **20/30** | 10 timeout |
| HARDER tier (2x) | 42/60 | **28/60** | 27 timeout, 4 wandered, 1 crash |
| HARDEST tier | 3/100 | **10/100** | 37 timeout, 27 crash, 26 wandered |

Round 6 remains the better brain for anything with imperfect inputs; round 7 is
marginally the better one on clean, empty bays. Rows in .

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

## Round 8g mid-training exam — 2026-09-04 19:05 LA (practice arena, simulator positions, no camera/lidar)

Brain frozen at 830k round-8 steps (`parking_ppo_round8g_test.zip`), 30 attempts each, seed 2026, clean bays, reverse gear allowed.

| exam | start | parked | notes |
|---|---|---|---|
| empty bay (`r8g_empty`) | 16 m | **30/30** | margins 1.08 m / 0.16 m, as round 6 |
| car three bays back (`r8g_three_back`) | 29 m | **29/30** | the one crash was at birth against a 7 m minibus placed as the "car" (its tail reached the start point) |
| car two bays back (`r8g_two_back`) | 22 m | **1/30** | 29 crashes, all into the car's flank/rear at 14-18 m before the bay: the van still swerves right within its first metres from a 22 m start |
| car right behind (`r8g_right_behind`) | 16 m | **0/30** | all crashes at ~12 m, nose into the car's rear; stage 2 (pull past, reverse in) was never reached in training |

Reading: the "stay in your lane" lesson (round 8e) is real and holds for a car three or four bays back. Two bays back fails because the brain's first move is welded to the 16-22 m start distances it trained at; round 8g spreads the starts to unlearn that. Raw rows: `rl/exams/2026-09-04_r8g_*.csv` (hit, ax, ay per crash).

## Round 8 final exam — 2026-09-05 00:56 LA (brain `rl/models/parking_ppo_round8.zip`)

Which brain: round 8h's final. Round 8g had learned the same skills but its last five minutes
collapsed (every attempt timing out) and the only saved copy was written after that; 8h redid
the last stretch from the 19:02 snapshot with rolling snapshots, the unsolvable "car right
behind" stage capped out, and half the learning rate (74 min, 2,276 attempts, no collapse).
Probe before the exams: 10/10. Exams: seed 2026, clean bays, reverse gear allowed.

| exam | start | round 8 | round 6 (no feelers) |
|---|---|---|---|
| empty bay | 16 m | **30/30** | 30/30 (earlier exams) |
| car two bays back | 22 m | **29/30** (one crash at 12 m) | **0/30** (hits the car at 19 m, every time) |
| car three bays back | 29 m | **30/30** (24 with the car placed) | — |
| car right behind the bay | 16 m | 0/20 (all at 12 m, nose into the car) | 0/30 (earlier exam) |

Reading: the van now stays in its lane past a parked car and turns in late, then parks with the
same margins as on an empty street (1.08 m front/back, 0.16 m sides). The remaining case, a car
right behind the bay, needs the pull-past-and-reverse manoeuvre: never trained (the stage was
capped after it caused the 8g collapse); round 9's teacher demonstrates it. Raw rows:
`rl/exams/2026-09-05_r8_*.csv`, `rl/exams/2026-09-05_r6_two_back.csv`.

## Sensor test with the round-8 brain — 2026-09-05 01:12–01:39 LA (full van software)

Camera + lidar perception, bays found by the lidar (map fallback), the learned parker = round-8
brain (9 inputs incl. feelers, reverse gear) taking the wheel at 30 m. Park-check plan twice:
six runs, two of them with a car in the chosen slot. Sweep output: backup folder `sweep_r8_sensor`.

| run | scenario | verdict | brain drove? | notes |
|---|---|---|---|---|
| 1 | empty bay | PASS | yes | lidar found 6 slots; brain parked inside (0.96 m / 0.09 m), feelers live |
| 2 | slot taken | FAIL (stuck 201 s) | yes, at 15.5 m after re-target | behaviour layer's "vehicle blocking path" stop for the parked car 8 m ahead froze the van while the brain held the wheel |
| 3 | empty bay | PASS | took over at 29.8 m, gave up at once | handed over on a bend, 10.5 m off the bay's line; rules parker parked (0.72 / 0.15 m) |
| 4 | empty bay | FAIL (timeout) | never reached | route blocked for minutes by an "obstacle at 0.0 m" 33 m before the destination: a camera/lidar perception false positive, not the parker |
| 5 | slot taken | PASS | no | rules parker after re-scan (0.92 / −0.09 m) |
| 6 | empty bay | PASS | no | rules parker (0.86 / 0.07 m) |

Collisions: 0 in 6. Verdict: the plug-in works end to end (run 1 is the first sensor-driven
park by the obstacle-aware brain), but the hand-over is inconsistent and the behaviour layer's
stop fights the brain near parked cars. Fixes (commit 3e093fb, not yet re-tested): hand over only
when lined up with the bay (≤ 4.5 m off its line, ≤ 25° off its heading, not yet past it); while the
brain drives, a behaviour stop wins only for a pedestrian, anything moving, anything within 2 m,
or when perception cannot say. Open: the "obstacle at 0.0 m" route block in camera mode.

## Sensor re-test with the fixes — 2026-09-05 01:56–02:22 LA

Same six runs, van software 6720a23 (aligned hand-over, stop-override rule). **5 PASS, 1 FAIL, 0 collisions.**

| run | scenario | verdict | brain drove? | notes |
|---|---|---|---|---|
| 1 | empty bay | PASS | took over at 7.0 m, gave up after 45 s | rules parker finished 0.33 m past the box front |
| 2 | slot taken | PASS | no | rules parker after re-target (last night: stuck 200 s) |
| 3 | empty bay | PASS | no | rules parker (last night: brain handed over on a bend and quit) |
| 4 | empty bay | PASS | no | rules parker (last night: route blocked before the bay) |
| 5 | slot taken | PASS | **yes, at 7.4 m after the re-target** | **parked by the learned parker in 3 s: 0.82 m front/back, 0.21 m side, 0.5°** |
| 6 | empty bay | FAIL (stuck 212 s) | yes, later | camera-mode "route blocked by obstacle at 1.2 m" held the van for 3.5 min before the bay (perception false block, open); once released the brain took over at 8.7 m and parked in 5 s (0.63 / 0.21 m, 0.3°) |

Reading: the two integration fixes did their job (no more freezing beside a parked car, no more hand-over
on a bend), and run 5 is the target manoeuvre end to end: sensors find the bay, a car sits in the
chosen slot, the van re-targets, the brain drives and parks. Two things to tune: the alignment gate
now delays the hand-over to 7–9 m (the approach roads curve until late; consider 35° / 5.5 m), and the
brain stalled once for 45 s (run 1). Open stack issue: camera-mode route blocks at 0–1.2 m.
