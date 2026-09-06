# Brains

| file | what it is | inputs / controls | verified result |
|---|---|---|---|
| `parking_ppo_round6.zip` | round 6: empty-bay parker (the van software's default) | 5 / 2 | empty bay 30/30 from 16 m; hits a car two bays back 30/30 |
| `parking_ppo_round8.zip` | round 8h final: obstacle-aware parker | 9 (4 feelers) / 3 (reverse gear) | empty 30/30, car two bays back 29/30, three back 30/30, right behind 0/20; sensor test 5/6, 0 collisions |
| `parking_ppo_round9_bc0.zip` | round 9: pure copy of the instructor (before any correction) | 9 / 3 | empty 18/20, two back 0/15, right behind 0/15 |
| `parking_ppo_round9_bc.zip` | round 9: copy after one DAgger round | 9 / 3 | empty 18/20, two back 0/15 (stalls), right behind 0/15 (stalls) |

The round-9 brains stall in reverse manoeuvres because their speed input was unsigned; the
gear-aware (10-input) refit is the next step (`rl/relabel_demos.py`, `pretrain_bc.py --mirror`).
Use in the van software: `POST /api/parking/parker {"parker": "rl", "brain": "rl/models/<file>", "handover_m": 30}`.
