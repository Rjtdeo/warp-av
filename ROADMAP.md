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

4. **Prediction layer.** React to where things WILL be, not where they are — start with
   straight-line extrapolation of every tracked object (the missing Waymo layer:
   perceive → **predict** → plan → act).
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
