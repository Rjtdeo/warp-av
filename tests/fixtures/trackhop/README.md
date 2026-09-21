# Track-hop fixtures (2026-09-21)

Four pieces cut from the sighting log of one live batch on the rig (13 runs on 400f99d, Town10HD, the log switched on
with `logs/track_obs.on`; see `src/warp_av/perception/sighting_log.py`). That log holds every sighting the tracker
was fed, at full precision, and replaying it through the tracker gave the live tracks back exactly in 8152 of 8152
frames. So these are not rebuilt or imagined inputs: they are what the van's tracker really saw.

Each file is the stretch around ONE "will cross our path" warning: `frames` (time from the first frame, the van's
pose `[x, y, yaw, speed]`, the sightings), the `route` nearby, what the van recorded live (`live_record`), what the
tracker and the prediction made of it before the change (`before_the_change`), and what CARLA said was really
there (`truth_for_scoring_only` -- never an input).

Kept in a cut: every sighting within 10 m of the warned-about track, frame by frame, from 3 s before that track was
born; and every sighting taken by any track that came that near, wherever it was. Every cut was checked when it was
made: replayed alone through the tracker as it was at 400f99d, it raises the same warning about the same thing on the
same frames as the full log does (scratch/trackhop/cut_fixture.py).

| file | run | what was there (CARLA) | what the van did live |
|---|---|---|---|
| `false_young_track_reach.json` | WAV-REACH-SHOULDER | nothing within 81 m | YIELD "obstacle will cross our path 14m ahead in 3.5s", brake 1.0 |
| `false_at_speed_0166.json` | WAV-0166 | nothing within 171 m | YIELD "obstacle will cross our path 6m ahead in 2.0s", brake 1.0, from 4.8 m/s |
| `true_crossing_car_0145.json` | WAV-0145 | a traffic car 0.4 m from the track, doing 8.3 m/s | YIELD "vehicle will cross our path 11m ahead in 1.5s", brake 1.0 |
| `true_young_fast_car_0148.json` | WAV-0148 | a traffic car 0.95 m from the track, doing 7.7 m/s; the track was 2.0 s old | YIELD, brake 1.0 |

`tests/test_track_hop_false_motion.py` runs each through the real tracker, the real prediction and the real
behaviour rule: the two false ones must reproduce the recorded warning with the new test switched off and raise
nothing with it on; the two true ones must raise the same warning on the same frame.
