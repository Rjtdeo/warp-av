# Static-truth recordings (Town10HD, CARLA 0.9.15, 2026-09-10)

One row per LiDAR blob the van's own pipeline found, with CARLA's answer key attached:
what the van MEASURED (height, length, width, share of points above 2 m, how far it sits
from the nearest lane a vehicle can use) next to what the thing REALLY IS (the majority
class of its points, and the top-3 class counts, so a person glued to a wall is visible).

| file | what | blobs |
|---|---|---|
| `town10_seed7.npz` | the recording the rules were learned on (118 viewpoints) | 8,678 |
| `town10_seed23_fresh.npz` | a fresh recording the rules were never tuned on | 9,150 |

Made with `tools/record_static_truth.py`; the rules in `src/warp_av/perception/motion_class.py`
come from `tools/learn_static_rules.py town10_seed7.npz --check town10_seed23_fresh.npz`.
`tests/test_planning_static_dynamic.py` re-checks them against both files.
