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
