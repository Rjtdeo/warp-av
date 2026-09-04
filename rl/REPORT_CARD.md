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
