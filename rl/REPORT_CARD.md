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
