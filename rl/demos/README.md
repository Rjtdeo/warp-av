# Round 9 recordings

- `demos.npz` — 1,200 instructor drives in the arena (374,885 moments): observation (9 numbers), the instructor's
  action (steer, pedal, gear), reward, episode id, done. Instructor parked 1,119/1,200 (93%): empty 95%, car
  ahead 96%, three bays back 95%, two bays back 95%, right behind 85%. Per-drive rows in `demos_episodes.csv`,
  console in `record_out.txt`.
- `dagger_1.npz` — DAgger round 1: 400 drives by the copied brain with the instructor's label at every moment
  (116,184 moments). Student before the refit: empty 92%, car ahead 85%, three back 95%, two back 0%, right behind 0%.

Relabel for the gear-aware brain: `python rl/relabel_demos.py rl/demos/demos.npz rl/demos/dagger_1.npz --suffix _g --student-driven rl/demos/dagger_1.npz`.
