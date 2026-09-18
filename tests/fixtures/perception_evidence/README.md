# Perception evidence recordings (not scored)

Seven LiDAR + camera recordings taken on the CARLA rig (Town10HD, 2026-09-17) with
`tools/record_fixture.py --no-spawn --seconds 1.2`, the van teleported to a recorded pose of a
live run and standing still. They were made to trace every perception stage on the two
parked-car failures of the validation programme:

| fixture | van pose (x, y, yaw) | what it looks at |
|---|---|---|
| rc1_merc_A12 / B8 / C5 / D3 | the four RC-1 approach poses of run WAV-0001 | the Mercedes at (108.5, 112.2), 12 / 8 / 5 / 3 m away (V3A, P-B02: the 2.56 m box) |
| rc1_must_25 / 20 / 15 | (82.5, 137.2, −23.9) / (87.4, 134.5, −31.0) / (91.5, 131.4, −38.8) | the Mustang at (106.3, 117.5), 25 / 20 / 15 m away (V3B, P-B01: the far line) |

They live here, not in `tests/fixtures/perception/`, on purpose: the scored replay harness
(`tests/test_replay_harness.py`, `tools/replay_score.py`) replays every folder in that directory
against an `expected.json` answer key, and these have none — they are 1.2 s recordings for stage
tracing (`scratch/v3a/stage_trace.py`), not two-second scored baselines. `warp_av.perception.replay.load_fixture(path)`
reads them by path like any other fixture.
